from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from models.build import build_model
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MobileADAS3D pure model-forward latency."
    )

    parser.add_argument("--config", type=str, default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps, or auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--use-fp16", action="store_true")
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


def load_checkpoint_if_needed(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint.")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"Metric value: {checkpoint.get('metric_value', checkpoint.get('loss', 'unknown'))}")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

    # MPS synchronize exists in recent PyTorch versions.
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def benchmark_forward(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
    use_amp: bool,
) -> Dict[str, Any]:
    model.eval()

    print("\nWarmup...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

        synchronize_if_needed(device)

    print("Benchmarking pure model forward...")
    latencies_ms: List[float] = []

    with torch.no_grad():
        for _ in range(benchmark_iters):
            synchronize_if_needed(device)
            start = time.perf_counter()

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            synchronize_if_needed(device)
            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)

    return {
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": median_ms,
        "latency_p90_ms": percentile(latencies_ms, 90),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_p99_ms": percentile(latencies_ms, 99),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "fps_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "fps_p50": 1000.0 / median_ms if median_ms > 0 else 0.0,
        "num_samples": len(latencies_ms),
        "latencies_ms": latencies_ms,
    }


def get_memory_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    if device.type == "cuda":
        info["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        info["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        info["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        info["cuda_max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)

    return info


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {output_path}")


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def main() -> None:
    args = parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"Set torch num threads: {args.num_threads}")

    config = load_config(args.config)
    config = apply_runtime_overrides(config=config, profile=args.profile, run_name=None)

    model_cfg = config["model"]
    training_cfg = config["training"]

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    device_str = args.device if args.device is not None else training_cfg.get("device", "auto")
    device = get_device(device_str)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config)
    load_checkpoint_if_needed(model, args.checkpoint, device=device)

    model.to(device)

    if args.use_fp16:
        if device.type == "cuda":
            print("Using CUDA AMP fp16.")
        elif device.type == "mps":
            print("Converting model/input to fp16 for MPS.")
            model = model.half()
        else:
            print("Warning: fp16 on CPU is usually slower or unsupported. Keeping fp32.")

    dtype = torch.float16 if args.use_fp16 and device.type == "mps" else torch.float32

    dummy_input = torch.randn(
        args.batch_size,
        3,
        input_height,
        input_width,
        device=device,
        dtype=dtype,
    )

    result = benchmark_forward(
        model=model,
        dummy_input=dummy_input,
        device=device,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        use_amp=args.use_fp16,
    )

    memory_info = get_memory_info(device)

    report = {
        "config": args.config,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "output_stride": int(model_cfg["output_stride"]),
        "feature_height": input_height // int(model_cfg["output_stride"]),
        "feature_width": input_width // int(model_cfg["output_stride"]),
        "use_fp16": bool(args.use_fp16),
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "num_threads": torch.get_num_threads(),
        "latency": {
            key: value
            for key, value in result.items()
            if key != "latencies_ms"
        },
        "memory": memory_info,
    }

    json_path = output_dir / f"benchmark_latency_{device.type}_bs{args.batch_size}.json"
    csv_path = output_dir / f"benchmark_latency_samples_{device.type}_bs{args.batch_size}.csv"

    save_json(report, json_path)

    rows = [
        {
            "iteration": idx,
            "latency_ms": value,
            "device": str(device),
            "batch_size": args.batch_size,
            "use_fp16": bool(args.use_fp16),
        }
        for idx, value in enumerate(result["latencies_ms"])
    ]

    save_csv(rows, csv_path)

    print("\nForward Latency Summary")
    print(f"Device: {device}")
    print(f"Input: {input_width} x {input_height}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.use_fp16}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Mean latency: {result['latency_mean_ms']:.2f} ms")
    print(f"P50 latency:  {result['latency_p50_ms']:.2f} ms")
    print(f"P90 latency:  {result['latency_p90_ms']:.2f} ms")
    print(f"P95 latency:  {result['latency_p95_ms']:.2f} ms")
    print(f"P99 latency:  {result['latency_p99_ms']:.2f} ms")
    print(f"Mean FPS:     {result['fps_mean']:.2f}")
    print(f"P50 FPS:      {result['fps_p50']:.2f}")

    if memory_info:
        print("\nMemory:")
        for key, value in memory_info.items():
            print(f"{key}: {value:.2f} MB")

    print("\nBenchmark files:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()