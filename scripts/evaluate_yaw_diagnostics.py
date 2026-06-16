from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze yaw/orientation errors from matched_3d_metrics CSV."
    )
    parser.add_argument(
        "--matched-csv",
        type=str,
        required=True,
        help="Path to matched_3d_metrics_<split>.csv from evaluate_3d_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to matched CSV parent / yaw_diagnostics.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name override. If omitted, inferred from CSV if possible.",
    )
    parser.add_argument(
        "--flip-threshold-deg",
        type=float,
        default=120.0,
        help="Standard yaw error above this may indicate front/back flip.",
    )
    parser.add_argument(
        "--axis-good-threshold-deg",
        type=float,
        default=30.0,
        help="Axis-aware yaw error below this supports front/back flip interpretation.",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=50,
        help="Number of worst yaw rows to save.",
    )
    return parser.parse_args()


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No rows to save: {output_path}")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {output_path}")


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    w = pos - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def summarize_group(rows: List[Dict[str, Any]], group_name: str, group_key: str | None) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if group_key is None:
        groups["ALL"] = rows
    else:
        for row in rows:
            groups[str(row[group_key])].append(row)

    out: List[Dict[str, Any]] = []
    for group_value, group_rows in sorted(groups.items(), key=lambda x: x[0]):
        std = [r["yaw_standard_error_deg"] for r in group_rows]
        axis = [r["yaw_axis_error_deg"] for r in group_rows]
        flips = [r["is_front_back_flip_candidate"] for r in group_rows]
        scores = [r["score"] for r in group_rows]
        ious = [r["iou_2d"] for r in group_rows]
        out.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "count": len(group_rows),
                "score_mean": mean(scores),
                "iou_2d_mean": mean(ious),
                "yaw_standard_mean_deg": mean(std),
                "yaw_standard_p50_deg": percentile(std, 50),
                "yaw_standard_p75_deg": percentile(std, 75),
                "yaw_standard_p90_deg": percentile(std, 90),
                "yaw_standard_p95_deg": percentile(std, 95),
                "yaw_axis_mean_deg": mean(axis),
                "yaw_axis_p50_deg": percentile(axis, 50),
                "yaw_axis_p75_deg": percentile(axis, 75),
                "yaw_axis_p90_deg": percentile(axis, 90),
                "yaw_axis_p95_deg": percentile(axis, 95),
                "axis_improvement_mean_deg": mean([s - a for s, a in zip(std, axis)]),
                "front_back_flip_candidates": int(sum(flips)),
                "front_back_flip_candidate_rate": float(sum(flips) / max(len(group_rows), 1)),
                "std_gt_30deg_rate": float(sum(v > 30.0 for v in std) / max(len(std), 1)),
                "std_gt_60deg_rate": float(sum(v > 60.0 for v in std) / max(len(std), 1)),
                "std_gt_90deg_rate": float(sum(v > 90.0 for v in std) / max(len(std), 1)),
                "axis_gt_30deg_rate": float(sum(v > 30.0 for v in axis) / max(len(axis), 1)),
                "axis_gt_60deg_rate": float(sum(v > 60.0 for v in axis) / max(len(axis), 1)),
            }
        )
    return out


def main() -> None:
    args = parse_args()

    matched_csv = Path(args.matched_csv)
    if not matched_csv.exists():
        raise FileNotFoundError(matched_csv)

    output_dir = Path(args.output_dir) if args.output_dir else matched_csv.parent / "yaw_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(matched_csv)
    if df.empty:
        raise ValueError(f"Matched CSV is empty: {matched_csv}")

    required = [
        "split",
        "sample_id",
        "class_name",
        "score",
        "iou_2d",
        "distance_bucket",
        "size_bucket",
        "gt_yaw_rad",
        "pred_yaw_rad",
        "yaw_abs_error_deg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {matched_csv}: {missing}")

    split = args.split or str(df["split"].iloc[0])

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        standard_error = float(row["yaw_abs_error_deg"])
        # Axis-aware: yaw and yaw + 180 degrees are treated as equivalent.
        axis_error = min(standard_error, abs(180.0 - standard_error))

        is_flip = (
            standard_error >= float(args.flip_threshold_deg)
            and axis_error <= float(args.axis_good_threshold_deg)
        )

        rows.append(
            {
                "split": split,
                "sample_id": row["sample_id"],
                "class_name": row["class_name"],
                "score": float(row["score"]),
                "iou_2d": float(row["iou_2d"]),
                "distance_bucket": row["distance_bucket"],
                "size_bucket": row["size_bucket"],
                "gt_depth_m": float(row.get("gt_depth_m", 0.0)),
                "gt_box_height_px": float(row.get("gt_box_height_px", 0.0)),
                "gt_box_width_px": float(row.get("gt_box_width_px", 0.0)),
                "gt_yaw_rad": float(row["gt_yaw_rad"]),
                "pred_yaw_rad": float(row["pred_yaw_rad"]),
                "yaw_standard_error_deg": standard_error,
                "yaw_axis_error_deg": axis_error,
                "axis_improvement_deg": standard_error - axis_error,
                "is_front_back_flip_candidate": int(is_flip),
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    summary_rows.extend(summarize_group(rows, "ALL", None))
    summary_rows.extend(summarize_group(rows, "class_name", "class_name"))
    summary_rows.extend(summarize_group(rows, "distance_bucket", "distance_bucket"))
    summary_rows.extend(summarize_group(rows, "size_bucket", "size_bucket"))

    # Class + distance summary.
    class_distance_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        class_distance_groups[(row["class_name"], row["distance_bucket"])].append(row)

    class_distance_rows: List[Dict[str, Any]] = []
    for (class_name, dist), group_rows in sorted(class_distance_groups.items(), key=lambda x: x[0]):
        tmp = summarize_group(group_rows, "class_distance", None)[0]
        tmp["class_name"] = class_name
        tmp["distance_bucket"] = dist
        class_distance_rows.append(tmp)

    # Worst rows by standard error and by front/back-flip-likelihood.
    worst_standard = sorted(rows, key=lambda r: r["yaw_standard_error_deg"], reverse=True)[: args.worst_k]
    worst_axis = sorted(rows, key=lambda r: r["yaw_axis_error_deg"], reverse=True)[: args.worst_k]
    flip_candidates = [r for r in rows if r["is_front_back_flip_candidate"]]
    flip_candidates = sorted(flip_candidates, key=lambda r: r["yaw_standard_error_deg"], reverse=True)[: args.worst_k]

    save_csv(rows, output_dir / f"yaw_diagnostic_rows_{split}.csv")
    save_csv(summary_rows, output_dir / f"yaw_diagnostic_summary_{split}.csv")
    save_csv(class_distance_rows, output_dir / f"yaw_diagnostic_class_distance_{split}.csv")
    save_csv(worst_standard, output_dir / f"yaw_worst_standard_{split}.csv")
    save_csv(worst_axis, output_dir / f"yaw_worst_axis_{split}.csv")
    save_csv(flip_candidates, output_dir / f"yaw_front_back_flip_candidates_{split}.csv")

    print("\nYaw Diagnostic Summary")
    for row in summary_rows:
        if row["group_name"] == "ALL":
            print(
                f"ALL count={row['count']} "
                f"standard_mean={row['yaw_standard_mean_deg']:.2f}deg "
                f"standard_p50={row['yaw_standard_p50_deg']:.2f}deg "
                f"standard_p90={row['yaw_standard_p90_deg']:.2f}deg "
                f"axis_mean={row['yaw_axis_mean_deg']:.2f}deg "
                f"axis_p50={row['yaw_axis_p50_deg']:.2f}deg "
                f"axis_p90={row['yaw_axis_p90_deg']:.2f}deg "
                f"flip_rate={row['front_back_flip_candidate_rate']:.3f}"
            )

    print("\nPer-class yaw summary")
    for row in summary_rows:
        if row["group_name"] == "class_name":
            print(
                f"{row['group_value']}: count={row['count']} "
                f"std_mean={row['yaw_standard_mean_deg']:.2f}deg "
                f"std_p50={row['yaw_standard_p50_deg']:.2f}deg "
                f"std_p90={row['yaw_standard_p90_deg']:.2f}deg "
                f"axis_mean={row['yaw_axis_mean_deg']:.2f}deg "
                f"axis_p50={row['yaw_axis_p50_deg']:.2f}deg "
                f"axis_p90={row['yaw_axis_p90_deg']:.2f}deg "
                f"flip_rate={row['front_back_flip_candidate_rate']:.3f} "
                f"std>90_rate={row['std_gt_90deg_rate']:.3f}"
            )

    print("\nPer-distance yaw summary")
    for row in summary_rows:
        if row["group_name"] == "distance_bucket":
            print(
                f"{row['group_value']}: count={row['count']} "
                f"std_mean={row['yaw_standard_mean_deg']:.2f}deg "
                f"std_p90={row['yaw_standard_p90_deg']:.2f}deg "
                f"axis_mean={row['yaw_axis_mean_deg']:.2f}deg "
                f"axis_p90={row['yaw_axis_p90_deg']:.2f}deg "
                f"flip_rate={row['front_back_flip_candidate_rate']:.3f}"
            )

    print(f"\nOutput dir: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
