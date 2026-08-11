from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.geometry import scale_p2_for_resize
from data.kitti_parser import parse_kitti_label_file
from data.kitti_r40 import evaluate_kitti_r40
from data.split_resolver import get_split_file
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI BEV and 3D AP_R40 on the labeled validation split."
    )
    parser.add_argument("--config", default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Write only metric CSV/JSON artifacts, not per-frame KITTI prediction txt files.",
    )
    return parser.parse_args()


def _scale_prediction_to_original(
    prediction: Dict[str, Any],
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> Dict[str, Any]:
    scaled = dict(prediction)
    x1, y1, x2, y2 = prediction["bbox_2d"]
    scale_x = original_width / float(input_width)
    scale_y = original_height / float(input_height)
    scaled["bbox_2d"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
    return scaled


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _write_kitti_predictions(
    output_dir: Path,
    sample_ids: List[str],
    predictions: Dict[str, List[Dict[str, Any]]],
) -> None:
    label_dir = output_dir / "predictions_kitti_format"
    label_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        lines = []
        for pred in predictions.get(sample_id, []):
            x, y, z = pred["location_3d"]
            h, w, length = pred["dimensions_3d_hwl"]
            yaw = float(pred["yaw"])
            alpha = _wrap_angle(yaw - math.atan2(float(x), float(z)))
            x1, y1, x2, y2 = pred["bbox_2d"]
            lines.append(
                f"{pred['class_name']} 0.00 0 {alpha:.6f} "
                f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                f"{h:.6f} {w:.6f} {length:.6f} "
                f"{x:.6f} {y:.6f} {z:.6f} {yaw:.6f} {pred['score']:.8f}"
            )
        (label_dir / f"{sample_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
    )
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][profile]["root_dir"]
    split_file = get_split_file(config, args.split)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
        class_mapping=dataset_cfg.get("class_mapping"),
    )
    limit = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    selected_indices = list(range(limit))

    device = get_device(config["training"].get("device", "auto"))
    # The checkpoint replaces every parameter; avoid an unnecessary ImageNet
    # download while constructing the evaluation model.
    model_cfg["pretrained"] = False
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])
    ground_truth: Dict[str, List[Dict[str, Any]]] = {}
    predictions: Dict[str, List[Dict[str, Any]]] = {}

    print(f"Evaluating {limit}/{len(dataset)} images on {device}")
    with torch.inference_mode():
        for output_index, dataset_index in enumerate(selected_indices, start=1):
            sample = dataset[dataset_index]
            image = F.interpolate(
                sample["image"].unsqueeze(0),
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            ).to(device)
            P2_model = scale_p2_for_resize(
                P2=sample["P2"],
                orig_w=int(sample["original_size"]["width"]),
                orig_h=int(sample["original_size"]["height"]),
                input_w=input_width,
                input_h=input_height,
            )
            decoded = decode_mobile_adas3d_outputs(
                outputs=model(image),
                classes=dataset_cfg["classes"],
                class_mean_dims=target_cfg["class_mean_dims"],
                input_height=input_height,
                input_width=input_width,
                score_threshold=args.score_threshold,
                topk=args.topk,
                nms_iou_threshold=args.nms_iou_threshold,
                P2=P2_model,
                location_source=model_cfg.get("location_source", "loc_xy"),
                score_mode=model_cfg.get(
                    "score_mode",
                    config.get("inference", {}).get("score_mode", "class"),
                ),
                quality_score_power=float(
                    config.get("inference", {}).get("quality_score_power", 1.0)
                ),
            )[0]
            sample_id = sample["sample_id"]
            predictions[sample_id] = [
                _scale_prediction_to_original(
                    pred,
                    original_width=int(sample["original_size"]["width"]),
                    original_height=int(sample["original_size"]["height"]),
                    input_width=input_width,
                    input_height=input_height,
                )
                for pred in decoded
            ]
            label_path = (
                Path(root_dir) / dataset_cfg["label_dir"] / f"{sample_id}.txt"
            )
            ground_truth[sample_id] = [
                asdict(obj) for obj in parse_kitti_label_file(label_path)
            ]
            if output_index % 100 == 0 or output_index == limit:
                print(f"  {output_index}/{limit}")

    results = evaluate_kitti_r40(
        ground_truth=ground_truth,
        predictions=predictions,
        classes=dataset_cfg["classes"],
    )
    rows = [result.to_dict() for result in results]
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (
        checkpoint_path.parent / f"kitti_r40_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "kitti_r40_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": "KITTI AP_R40 local validation",
        "split_protocol": dataset_cfg["splits"].get("protocol", "unknown"),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "evaluated_images": limit,
        "split_images": len(dataset),
        "complete_split": limit == len(dataset),
        "score_threshold": args.score_threshold,
        "topk": args.topk,
        "nms_iou_threshold": args.nms_iou_threshold,
        "metrics": rows,
    }
    (output_dir / "kitti_r40_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_predictions:
        _write_kitti_predictions(
            output_dir=output_dir,
            sample_ids=[dataset[index]["sample_id"] for index in selected_indices],
            predictions=predictions,
        )

    print("\nKITTI AP_R40 (%):")
    for metric in ("bev", "3d"):
        print(f"  {metric.upper()}")
        for class_name in dataset_cfg["classes"]:
            class_results = [
                result for result in results
                if result.metric == metric and result.class_name == class_name
            ]
            values = " / ".join(f"{result.ap_r40:.2f}" for result in class_results)
            print(f"    {class_name}: {values}  (easy/moderate/hard)")
    if limit != len(dataset):
        print("WARNING: partial split; these numbers are diagnostic, not reportable.")
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
