from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from models.decode import decode_mobile_adas3d_outputs
from models.torchscript_wrapper import tuple_outputs_to_dict
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchScript MobileADAS3D end-to-end inference latency."
    )

    parser.add_argument("--config", type=str, default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--torchscript-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=100)

    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)

    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--benchmark-iters", type=int, default=300)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    return parser.parse_args()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0

    values_sorted = sorted(values)

    if len(values_sorted) == 1:
        return values_sorted[0]

    pos = (len(values_sorted) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values_sorted) - 1)
    weight = pos - lower

    return values_sorted[lower] * (1.0 - weight) + values_sorted[upper] * weight


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {path}")


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def get_memory_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    if device.type == "cuda":
        info["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        info["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        info["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        info["cuda_max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)

    return info


def prepare_input(
    image_tensor: torch.Tensor,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> torch.Tensor:
    image = image_tensor.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    image = image.to(device=device, dtype=torch.float32, non_blocking=True)

    return image


def run_one_iteration(
    model: torch.jit.ScriptModule,
    sample: Dict[str, Any],
    device: torch.device,
    input_height: int,
    input_width: int,
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    score_threshold: float,
    topk: int,
    nms_iou_threshold: float,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    image = prepare_input(
        image_tensor=sample["image"],
        input_height=input_height,
        input_width=input_width,
        device=device,
    )
    synchronize_if_needed(device)
    preprocess_end = time.perf_counter()

    forward_start = time.perf_counter()

    with torch.no_grad():
        tuple_outputs = model(image)

    synchronize_if_needed(device)
    forward_end = time.perf_counter()

    decode_start = time.perf_counter()

    outputs = tuple_outputs_to_dict(tuple_outputs)

    predictions = decode_mobile_adas3d_outputs(
        outputs=outputs,
        classes=classes,
        class_mean_dims=class_mean_dims,
        input_height=input_height,
        input_width=input_width,
        score_threshold=score_threshold,
        topk=topk,
        nms_iou_threshold=nms_iou_threshold,
    )[0]

    synchronize_if_needed(device)
    decode_end = time.perf_counter()

    total_end = time.perf_counter()

    return {
        "sample_id": sample["sample_id"],
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "forward_ms": (forward_end - forward_start) * 1000.0,
        "decode_ms": (decode_end - decode_start) * 1000.0,
        "total_ms": (total_end - total_start) * 1000.0,
        "num_predictions": len(predictions),
    }


def main() -> None:
    args = parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"Set torch num threads: {args.num_threads}")

    config = load_config(args.config)
    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=None,
    )

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    device = get_device(args.device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]
    split_file = get_split_file(config, args.split)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    num_images = min(args.max_images, len(dataset))

    if num_images <= 0:
        raise ValueError("max-images must be > 0 for benchmarking.")

    print("Preloading samples into memory to exclude disk I/O from benchmark.")
    samples = [dataset[i] for i in range(num_images)]

    print("Loading TorchScript model...")
    model = torch.jit.load(args.torchscript_path, map_location=device)
    model.to(device)
    model.eval()

    print("\nTorchScript end-to-end benchmark config:")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Samples preloaded: {num_images}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"TopK: {args.topk}")
    print(f"NMS IoU threshold: {args.nms_iou_threshold}")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Benchmark iters: {args.benchmark_iters}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"TorchScript path: {args.torchscript_path}")

    print("\nWarmup full pipeline...")
    for idx in range(args.warmup_iters):
        sample = samples[idx % num_images]

        _ = run_one_iteration(
            model=model,
            sample=sample,
            device=device,
            input_height=input_height,
            input_width=input_width,
            classes=dataset_cfg["classes"],
            class_mean_dims=target_cfg["class_mean_dims"],
            score_threshold=args.score_threshold,
            topk=args.topk,
            nms_iou_threshold=args.nms_iou_threshold,
        )

    print("Benchmarking full pipeline...")
    rows: List[Dict[str, Any]] = []

    for idx in range(args.benchmark_iters):
        sample = samples[idx % num_images]

        row = run_one_iteration(
            model=model,
            sample=sample,
            device=device,
            input_height=input_height,
            input_width=input_width,
            classes=dataset_cfg["classes"],
            class_mean_dims=target_cfg["class_mean_dims"],
            score_threshold=args.score_threshold,
            topk=args.topk,
            nms_iou_threshold=args.nms_iou_threshold,
        )

        row["iteration"] = idx
        row["device"] = str(device)
        row["score_threshold"] = args.score_threshold
        row["topk"] = args.topk
        rows.append(row)

    preprocess_values = [r["preprocess_ms"] for r in rows]
    forward_values = [r["forward_ms"] for r in rows]
    decode_values = [r["decode_ms"] for r in rows]
    total_values = [r["total_ms"] for r in rows]
    prediction_counts = [r["num_predictions"] for r in rows]

    preprocess_summary = summarize(preprocess_values)
    forward_summary = summarize(forward_values)
    decode_summary = summarize(decode_values)
    total_summary = summarize(total_values)

    memory_info = get_memory_info(device)

    report = {
        "torchscript_path": args.torchscript_path,
        "config": args.config,
        "profile": args.profile,
        "device": str(device),
        "split": args.split,
        "num_preloaded_images": num_images,
        "benchmark_iters": args.benchmark_iters,
        "input_height": input_height,
        "input_width": input_width,
        "output_stride": int(model_cfg["output_stride"]),
        "score_threshold": args.score_threshold,
        "topk": args.topk,
        "nms_iou_threshold": args.nms_iou_threshold,
        "num_threads": torch.get_num_threads(),
        "preprocess_latency": preprocess_summary,
        "forward_latency": forward_summary,
        "decode_latency": decode_summary,
        "total_latency": total_summary,
        "fps_mean_total": 1000.0 / total_summary["mean_ms"] if total_summary["mean_ms"] > 0 else 0.0,
        "fps_p50_total": 1000.0 / total_summary["p50_ms"] if total_summary["p50_ms"] > 0 else 0.0,
        "num_predictions_mean": statistics.mean(prediction_counts) if prediction_counts else 0.0,
        "num_predictions_p50": statistics.median(prediction_counts) if prediction_counts else 0.0,
        "memory": memory_info,
    }

    json_path = output_dir / f"benchmark_torchscript_e2e_{device.type}_bs1.json"
    csv_path = output_dir / f"benchmark_torchscript_e2e_samples_{device.type}_bs1.csv"

    save_json(report, json_path)
    save_csv(rows, csv_path)

    print("\nTorchScript End-to-End Latency Summary")
    print(f"Device: {device}")
    print(f"Input: {input_width} x {input_height}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"TopK: {args.topk}")
    print(f"Mean predictions/image: {report['num_predictions_mean']:.2f}")

    print("\nPreprocess:")
    print(f"  mean={preprocess_summary['mean_ms']:.2f} ms")
    print(f"  p50 ={preprocess_summary['p50_ms']:.2f} ms")
    print(f"  p95 ={preprocess_summary['p95_ms']:.2f} ms")

    print("\nForward:")
    print(f"  mean={forward_summary['mean_ms']:.2f} ms")
    print(f"  p50 ={forward_summary['p50_ms']:.2f} ms")
    print(f"  p95 ={forward_summary['p95_ms']:.2f} ms")

    print("\nDecode + NMS:")
    print(f"  mean={decode_summary['mean_ms']:.2f} ms")
    print(f"  p50 ={decode_summary['p50_ms']:.2f} ms")
    print(f"  p95 ={decode_summary['p95_ms']:.2f} ms")

    print("\nTotal:")
    print(f"  mean={total_summary['mean_ms']:.2f} ms")
    print(f"  p50 ={total_summary['p50_ms']:.2f} ms")
    print(f"  p90 ={total_summary['p90_ms']:.2f} ms")
    print(f"  p95 ={total_summary['p95_ms']:.2f} ms")
    print(f"  p99 ={total_summary['p99_ms']:.2f} ms")
    print(f"  FPS mean={report['fps_mean_total']:.2f}")
    print(f"  FPS p50 ={report['fps_p50_total']:.2f}")

    if memory_info:
        print("\nMemory:")
        for key, value in memory_info.items():
            print(f"  {key}: {value:.2f} MB")

    print("\nBenchmark files:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()