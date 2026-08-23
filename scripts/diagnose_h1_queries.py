from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from losses.h1_set_loss import box_cxcywh_to_xyxy, box_iou
from models.build import build_model
from scripts.train_mobile_adas3d import (
    build_criterion,
    build_dataloader,
    move_targets_to_device,
)
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose matched and background H1 queries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _summary(values: List[float]) -> Dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    if not tensor.numel():
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    if config["model"]["name"] != "MobileADAS3D-H1":
        raise RuntimeError("H1 query diagnostics require MobileADAS3D-H1")
    if config.get("loss", {}).get("classification_mode") != "implicit_background_softmax":
        raise RuntimeError("Diagnostics require the H1-v2 implicit background objective")

    device = get_device(config["training"].get("device", "auto"))
    loader = build_dataloader(
        config=config,
        split_name=args.split,
        batch_size=int(config["validation"]["batch_size"]),
        num_workers=0,
        shuffle=False,
        device=device,
    )
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_criterion(config).to(device)

    matched_scores: List[float] = []
    unmatched_scores: List[float] = []
    matched_ious: List[float] = []
    predicted_counts: List[int] = []
    gt_counts: List[int] = []

    with torch.inference_mode():
        for batch in loader:
            targets = move_targets_to_device(batch["targets"], device)
            outputs = model(batch["images"].to(device))
            matches = criterion._match(outputs, targets)
            background = torch.zeros_like(outputs["class_logits"][..., :1])
            class_scores = torch.cat(
                (outputs["class_logits"], background), dim=-1
            ).softmax(dim=-1)[..., :-1].max(dim=-1).values
            scores = class_scores * outputs["quality"].sigmoid().squeeze(-1)
            for batch_index, (query_indices, target_indices) in enumerate(matches):
                matched_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=device)
                matched_mask[query_indices] = True
                matched_scores.extend(scores[batch_index, matched_mask].float().cpu().tolist())
                unmatched_scores.extend(scores[batch_index, ~matched_mask].float().cpu().tolist())
                predicted_counts.append(int((scores[batch_index] >= args.score_threshold).sum()))
                valid_targets = targets["object_mask"][batch_index]
                gt_counts.append(int(valid_targets.sum()))
                if query_indices.numel():
                    predicted_boxes = outputs["box2d_cxcywh"][batch_index, query_indices]
                    target_boxes = targets["box2d"][batch_index, valid_targets][target_indices]
                    ious = box_iou(
                        box_cxcywh_to_xyxy(predicted_boxes.float()),
                        box_cxcywh_to_xyxy(target_boxes.float()),
                    ).diag()
                    matched_ious.extend(ious.cpu().tolist())

    matched = _summary(matched_scores)
    unmatched = _summary(unmatched_scores)
    iou = _summary(matched_ious)
    mean_predictions = sum(predicted_counts) / max(len(predicted_counts), 1)
    mean_gt = sum(gt_counts) / max(len(gt_counts), 1)
    gates = {
        "matched_score_median_ge_0_50": matched["median"] >= 0.50,
        "unmatched_score_p95_le_0_10": unmatched["p95"] <= 0.10,
        "matched_iou_mean_ge_0_70": iou["mean"] >= 0.70,
        "mean_count_error_le_1": abs(mean_predictions - mean_gt) <= 1.0,
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "samples": len(loader.dataset),
        "score_threshold": args.score_threshold,
        "matched_scores": matched,
        "unmatched_scores": unmatched,
        "matched_box_iou": iou,
        "mean_predictions_per_image": mean_predictions,
        "mean_ground_truth_per_image": mean_gt,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"H1-v2 tiny-overfit gate failed; see {report_path}")


if __name__ == "__main__":
    main()
