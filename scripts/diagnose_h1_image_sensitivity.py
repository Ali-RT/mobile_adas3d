from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.collate import resize_image_tensor
from data.kitti_dataset import KITTIDataset
from models.build import build_model
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure H1 output sensitivity to image content.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-a", required=True)
    parser.add_argument("--sample-b", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    if model_cfg["name"] not in {"MobileADAS3D-H1", "MobileADAS3D-H2"}:
        raise RuntimeError("Image sensitivity requires MobileADAS3D-H1/H2")
    root = dataset_cfg["profiles"][dataset_cfg["active_profile"]]["root_dir"]
    dataset = KITTIDataset(
        root_dir=root,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        sample_ids=[args.sample_a, args.sample_b],
        class_mapping=dataset_cfg.get("class_mapping"),
    )
    if dataset.sample_ids[0] == dataset.sample_ids[1]:
        raise RuntimeError("Sensitivity samples must be different")
    device = get_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_architecture = checkpoint.get(
        "architecture", checkpoint.get("config", {}).get("model", {}).get("name")
    )
    if checkpoint_architecture != model_cfg["name"]:
        raise RuntimeError(
            f"Refusing {checkpoint_architecture} checkpoint for {model_cfg['name']}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])

    images = torch.stack(
        [
            resize_image_tensor(
                dataset[index]["image"],
                int(model_cfg["input_height"]),
                int(model_cfg["input_width"]),
            )
            for index in range(2)
        ]
    ).to(device)
    with torch.inference_mode():
        output_a = model(images[:1])
        output_a_repeat = model(images[:1])
        output_b = model(images[1:])

    deltas = {}
    repeat_max = 0.0
    for name in output_a:
        content_delta = (output_a[name] - output_b[name]).float().abs()
        repeat_delta = (output_a[name] - output_a_repeat[name]).float().abs()
        repeat_max = max(repeat_max, float(repeat_delta.max()))
        deltas[name] = {
            "mean_abs": float(content_delta.mean()),
            "max_abs": float(content_delta.max()),
        }
    gates = {
        "deterministic_repeat_max_le_1e_6": repeat_max <= 1e-6,
        "class_logits_mean_delta_ge_0_02": deltas["class_logits"]["mean_abs"] >= 0.02,
        "box_mean_delta_ge_0_01": deltas["box2d_cxcywh"]["mean_abs"] >= 0.01,
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "checkpoint": str(args.checkpoint),
        "sample_a": dataset.sample_ids[0],
        "sample_b": dataset.sample_ids[1],
        "repeat_max_abs_delta": repeat_max,
        "content_deltas": deltas,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"H1 image-sensitivity gate failed; see {report_path}")


if __name__ == "__main__":
    main()
