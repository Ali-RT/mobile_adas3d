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

from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchScript model forward latency."
    )

    parser.add_argument("--config", type=str, default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--torchscript-path",
        "--torchscript",
        dest="torchscript_path",
        type=str,
        required=True,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
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

    model_cfg = config["model"]

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

    model = torch.jit.load(args.torchscript_path, map_location=device)
    model.to(device)
    model.eval()

    x = torch.randn(
        args.batch_size,
        3,
        input_height,
        input_width,
        device=device,
        dtype=torch.float32,
    )

    print("\nTorchScript latency benchmark")
    print(f"Device: {device}")
    print(f"Input: {input_width} x {input_height}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Benchmark iters: {args.benchmark_iters}")
    print(f"TorchScript path: {args.torchscript_path}")

    with torch.no_grad():
        print("\nWarmup...")
        for _ in range(args.warmup_iters):
            _ = model(x)

        synchronize_if_needed(device)

        print("Benchmarking...")
        latencies_ms: List[float] = []

        for _ in range(args.benchmark_iters):
            synchronize_if_needed(device)
            start = time.perf_counter()

            _ = model(x)

            synchronize_if_needed(device)
            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p90_ms = percentile(latencies_ms, 90)
    p95_ms = percentile(latencies_ms, 95)
    p99_ms = percentile(latencies_ms, 99)

    memory_info = get_memory_info(device)

    report = {
        "torchscript_path": args.torchscript_path,
        "device": str(device),
        "batch_size": args.batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "num_threads": torch.get_num_threads(),
        "latency": {
            "mean_ms": mean_ms,
            "p50_ms": p50_ms,
            "p90_ms": p90_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
            "fps_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "fps_p50": 1000.0 / p50_ms if p50_ms > 0 else 0.0,
        },
        "memory": memory_info,
    }

    json_path = output_dir / f"benchmark_torchscript_latency_{device.type}_bs{args.batch_size}.json"
    csv_path = output_dir / f"benchmark_torchscript_latency_samples_{device.type}_bs{args.batch_size}.csv"

    rows = [
        {
            "iteration": idx,
            "latency_ms": value,
            "device": str(device),
            "batch_size": args.batch_size,
        }
        for idx, value in enumerate(latencies_ms)
    ]

    save_json(report, json_path)
    save_csv(rows, csv_path)

    print("\nTorchScript Forward Latency Summary")
    print(f"Mean latency: {mean_ms:.2f} ms")
    print(f"P50 latency:  {p50_ms:.2f} ms")
    print(f"P90 latency:  {p90_ms:.2f} ms")
    print(f"P95 latency:  {p95_ms:.2f} ms")
    print(f"P99 latency:  {p99_ms:.2f} ms")
    print(f"Mean FPS:     {report['latency']['fps_mean']:.2f}")
    print(f"P50 FPS:      {report['latency']['fps_p50']:.2f}")

    if memory_info:
        print("\nMemory:")
        for key, value in memory_info.items():
            print(f"{key}: {value:.2f} MB")

    print("\nFiles:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
