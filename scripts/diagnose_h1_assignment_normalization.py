from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from losses.h1_set_loss import box_cxcywh_to_xyxy, box_iou
from models.build import build_model
from scripts.train_mobile_adas3d import build_criterion, build_dataloader, move_targets_to_device
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_checkpoint(value: str) -> Tuple[int, Path]:
    try:
        step_text, path_text = value.split("=", 1)
        step = int(step_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("checkpoint must use STEP=PATH") from error
    if step <= 0 or not path_text:
        raise argparse.ArgumentTypeError("checkpoint step and path must be non-empty/positive")
    return step, Path(path_text)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if not tensor.numel():
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose H1 assignment churn and BatchNorm mode sensitivity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def set_mode(model: nn.Module, mode: str) -> int:
    model.eval()
    batch_norm_count = 0
    if mode == "batch_stats":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
                batch_norm_count += 1
    elif mode != "eval":
        raise ValueError(mode)
    return batch_norm_count


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    score_threshold: float,
    collect_assignments: bool,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    matched_scores: List[float] = []
    unmatched_scores: List[float] = []
    matched_ious: List[float] = []
    predicted_counts: List[int] = []
    gt_counts: List[int] = []
    assignments: Dict[str, Dict[str, float]] = {}
    with torch.inference_mode():
        for batch in loader:
            targets = move_targets_to_device(batch["targets"], device)
            outputs = model(batch["images"].to(device))
            matches = criterion._match(outputs, targets)
            background = torch.zeros_like(outputs["class_logits"][..., :1])
            class_scores = torch.cat((outputs["class_logits"], background), -1).softmax(-1)[..., :-1].max(-1).values
            scores = class_scores * outputs["quality"].sigmoid().squeeze(-1)
            for batch_index, (query_indices, target_indices) in enumerate(matches):
                matched_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=device)
                matched_mask[query_indices] = True
                matched_scores.extend(scores[batch_index, matched_mask].float().cpu().tolist())
                unmatched_scores.extend(scores[batch_index, ~matched_mask].float().cpu().tolist())
                predicted_counts.append(int((scores[batch_index] >= score_threshold).sum()))
                valid_mask = targets["object_mask"][batch_index]
                valid_padded = torch.nonzero(valid_mask, as_tuple=False).flatten()
                gt_counts.append(int(valid_mask.sum()))
                if not query_indices.numel():
                    continue
                predicted_boxes = outputs["box2d_cxcywh"][batch_index, query_indices]
                target_boxes = targets["box2d"][batch_index, valid_mask][target_indices]
                ious = box_iou(box_cxcywh_to_xyxy(predicted_boxes.float()), box_cxcywh_to_xyxy(target_boxes.float())).diag()
                matched_ious.extend(ious.cpu().tolist())
                if collect_assignments:
                    sample_id = batch["metadata"][batch_index]["sample_id"]
                    for local_index, query_index in enumerate(query_indices):
                        padded_index = int(valid_padded[target_indices[local_index]])
                        key = f"{sample_id}:{padded_index}"
                        assignments[key] = {
                            "query_id": int(query_index),
                            "iou_2d": float(ious[local_index]),
                            "score": float(scores[batch_index, query_index]),
                            "class_id": int(targets["class_ids"][batch_index, padded_index]),
                        }
    mean_predictions = sum(predicted_counts) / max(len(predicted_counts), 1)
    mean_gt = sum(gt_counts) / max(len(gt_counts), 1)
    metrics: Dict[str, object] = {
        "matched_scores": summarize(matched_scores),
        "unmatched_scores": summarize(unmatched_scores),
        "matched_box_iou": summarize(matched_ious),
        "mean_predictions_per_image": mean_predictions,
        "mean_ground_truth_per_image": mean_gt,
        "mean_count_error": abs(mean_predictions - mean_gt),
    }
    return metrics, assignments


def assignment_summary(by_step: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    steps = sorted(by_step)
    common = set.intersection(*(set(by_step[step]) for step in steps))
    adjacent_total = adjacent_same = 0
    unique_counts: List[int] = []
    per_object = []
    for key in sorted(common):
        queries = [int(by_step[step][key]["query_id"]) for step in steps]
        ious = [float(by_step[step][key]["iou_2d"]) for step in steps]
        unique_counts.append(len(set(queries)))
        adjacent_same += sum(left == right for left, right in zip(queries, queries[1:]))
        adjacent_total += max(len(queries) - 1, 0)
        per_object.append({"object_key": key, "query_ids": queries, "ious": ious})
    fully_stable = sum(item["query_ids"].count(item["query_ids"][0]) == len(steps) for item in per_object)
    return {
        "steps": steps,
        "common_objects": len(common),
        "adjacent_same_query_rate": adjacent_same / max(adjacent_total, 1),
        "fully_stable_object_rate": fully_stable / max(len(common), 1),
        "unique_queries_per_object": summarize(unique_counts),
        "per_object": per_object,
    }


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoint)
    if len({step for step, _ in checkpoints}) != len(checkpoints):
        raise RuntimeError("Checkpoint steps must be unique")
    for _, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = apply_runtime_overrides(load_config(args.config), profile=args.profile, dataset_root=args.dataset_root, split_dir=args.split_dir, output_dir=args.output_dir)
    device = get_device(config["training"].get("device", "auto"))
    loader = build_dataloader(config, args.split, int(config["validation"]["batch_size"]), 0, False, device)
    criterion = build_criterion(config).to(device)
    results = []
    eval_assignments: Dict[int, Dict[str, Dict[str, float]]] = {}
    for step, checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mode_results = {}
        for mode in ("eval", "batch_stats"):
            model = build_model(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            batch_norm_count = set_mode(model, mode)
            metrics, assignments = evaluate(model, criterion, loader, device, args.score_threshold, collect_assignments=(mode == "eval"))
            metrics["batch_norm_layers_using_batch_stats"] = batch_norm_count
            mode_results[mode] = metrics
            if mode == "eval":
                eval_assignments[step] = assignments
        results.append({"step": step, "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "modes": mode_results})
    report = {
        "schema_version": 1,
        "complete": True,
        "split": args.split,
        "samples": len(loader.dataset),
        "score_threshold": args.score_threshold,
        "checkpoints": results,
        "eval_assignment_stability": assignment_summary(eval_assignments),
        "interpretation_rules": {
            "batchnorm_mismatch_signal": "batch_stats materially improves score separation, IoU, or count error over eval",
            "assignment_churn_signal": "low adjacent_same_query_rate together with weak localization progression",
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
