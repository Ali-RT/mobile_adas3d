from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    path: Path
    sort_key: tuple[int, int, str]


def parse_epoch_from_checkpoint(path: Path) -> Optional[int]:
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def collect_checkpoint_candidates(
    checkpoint_dir: Path,
    checkpoint_glob: str = "epoch_*.pt",
    include_best_latest: bool = True,
) -> List[CheckpointCandidate]:
    candidates: List[CheckpointCandidate] = []

    for path in sorted(checkpoint_dir.glob(checkpoint_glob)):
        if not path.is_file():
            continue
        epoch = parse_epoch_from_checkpoint(path)
        if epoch is None:
            continue
        candidates.append(
            CheckpointCandidate(
                name=path.stem,
                path=path,
                sort_key=(0, epoch, path.name),
            )
        )

    if include_best_latest:
        for rank, filename in enumerate(("best.pt", "latest.pt"), start=1):
            path = checkpoint_dir / filename
            if path.is_file():
                candidates.append(
                    CheckpointCandidate(
                        name=path.stem,
                        path=path,
                        sort_key=(rank, 0, path.name),
                    )
                )

    return sorted(candidates, key=lambda item: item.sort_key)


def metric_column_name(metric: str, class_name: str, difficulty: str) -> str:
    def safe_token(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")

    return f"ap_{safe_token(metric)}_{safe_token(class_name)}_{safe_token(difficulty)}"


def summarize_checkpoint_metrics(
    checkpoint: CheckpointCandidate,
    evaluation_summary: Dict[str, Any],
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.path),
        "evaluated_images": evaluation_summary.get("evaluated_images"),
        "split_images": evaluation_summary.get("split_images"),
        "complete_split": evaluation_summary.get("complete_split"),
        "score_threshold": evaluation_summary.get("score_threshold"),
        "topk": evaluation_summary.get("topk"),
        "nms_iou_threshold": evaluation_summary.get("nms_iou_threshold"),
    }

    if checkpoint_metadata is not None:
        row.update(
            {
                "epoch": checkpoint_metadata.get("epoch"),
                "global_step": checkpoint_metadata.get("global_step"),
                "metric_value": checkpoint_metadata.get("metric_value"),
                "best_metric": checkpoint_metadata.get("best_metric"),
            }
        )

    metrics = evaluation_summary.get("metrics", [])

    for metric_row in metrics:
        column = metric_column_name(
            metric=str(metric_row["metric"]),
            class_name=str(metric_row["class_name"]),
            difficulty=str(metric_row["difficulty"]),
        )
        row[column] = float(metric_row["ap_r40"])

    for metric_name in ("3d", "bev"):
        values = [
            float(metric_row["ap_r40"])
            for metric_row in metrics
            if metric_row["metric"] == metric_name
            and metric_row["difficulty"] == "moderate"
        ]
        row[f"mean_{metric_name}_moderate"] = (
            sum(values) / len(values) if values else None
        )

    row["selection_score"] = row.get("ap_3d_Car_moderate")

    return row


def flatten_metric_rows(
    checkpoint: CheckpointCandidate,
    evaluation_summary: Dict[str, Any],
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metadata = checkpoint_metadata or {}

    for metric_row in evaluation_summary.get("metrics", []):
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint.path),
                "epoch": metadata.get("epoch"),
                "global_step": metadata.get("global_step"),
                "metric_value": metadata.get("metric_value"),
                "best_metric": metadata.get("best_metric"),
                **metric_row,
            }
        )

    return rows


def best_rows_by_metric(
    rows: Sequence[Dict[str, Any]],
    metric_columns: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}

    for column in metric_columns:
        valid_rows = [
            row
            for row in rows
            if row.get(column) is not None
        ]
        if valid_rows:
            best[column] = max(valid_rows, key=lambda row: float(row[column]))

    return best


def to_jsonable_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - only relevant in minimal envs.
        return {"metadata_error": f"torch import failed: {error}"}

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        return {"metadata_error": str(error)}

    if not isinstance(checkpoint, dict):
        return {}

    return {
        "epoch": to_jsonable_scalar(checkpoint.get("epoch")),
        "global_step": to_jsonable_scalar(checkpoint.get("global_step")),
        "metric_value": to_jsonable_scalar(
            checkpoint.get("metric_value", checkpoint.get("loss"))
        ),
        "best_metric": to_jsonable_scalar(checkpoint.get("best_metric")),
    }


def write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    preferred = [
        "checkpoint",
        "epoch",
        "global_step",
        "metric_value",
        "best_metric",
        "selection_score",
        "mean_3d_moderate",
        "mean_bev_moderate",
        "ap_3d_Car_moderate",
        "ap_bev_Car_moderate",
        "checkpoint_path",
        "evaluated_images",
        "split_images",
        "complete_split",
        "score_threshold",
        "topk",
        "nms_iou_threshold",
    ]
    all_fields = sorted({key for row in rows for key in row})
    fieldnames = [field for field in preferred if field in all_fields]
    fieldnames.extend(field for field in all_fields if field not in fieldnames)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(
    args: argparse.Namespace,
    checkpoint: CheckpointCandidate,
    output_dir: Path,
) -> None:
    command: List[str] = [
        args.python,
        "scripts/evaluate_kitti_r40.py",
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint.path),
        "--split",
        args.split,
        "--score-threshold",
        str(args.score_threshold),
        "--topk",
        str(args.topk),
        "--nms-iou-threshold",
        str(args.nms_iou_threshold),
        "--output-dir",
        str(output_dir),
    ]

    if args.profile is not None:
        command.extend(["--profile", args.profile])
    if args.dataset_root is not None:
        command.extend(["--dataset-root", args.dataset_root])
    if args.split_dir is not None:
        command.extend(["--split-dir", args.split_dir])
    if args.max_images >= 0:
        command.extend(["--max-images", str(args.max_images)])
    if not args.keep_predictions:
        command.append("--skip-predictions")

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI AP_R40 across saved checkpoints in a run directory."
    )
    parser.add_argument("--config", default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-glob", default="epoch_*.pt")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument(
        "--no-best-latest",
        action="store_true",
        help="Only evaluate checkpoint-glob matches, not best.pt/latest.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_dir is None and args.run_dir is None:
        raise ValueError("Pass --run-dir or --checkpoint-dir")

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else Path(args.run_dir) / "checkpoints"
    )
    run_dir = Path(args.run_dir) if args.run_dir is not None else checkpoint_dir.parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / f"checkpoint_ap_sweep_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = collect_checkpoint_candidates(
        checkpoint_dir=checkpoint_dir,
        checkpoint_glob=args.checkpoint_glob,
        include_best_latest=not args.no_best_latest,
    )

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    summary_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Sweep output dir: {output_dir}")
    print(f"Checkpoints: {len(checkpoints)}")

    for index, checkpoint in enumerate(checkpoints, start=1):
        checkpoint_output_dir = output_dir / checkpoint.name
        summary_path = checkpoint_output_dir / "kitti_r40_summary.json"

        print(
            f"\n[{index}/{len(checkpoints)}] {checkpoint.name}: {checkpoint.path}",
            flush=True,
        )

        try:
            if args.force or not summary_path.is_file():
                run_evaluation(
                    args=args,
                    checkpoint=checkpoint,
                    output_dir=checkpoint_output_dir,
                )
            else:
                print(f"Reusing existing summary: {summary_path}")

            evaluation_summary = load_json(summary_path)
            metadata = load_checkpoint_metadata(checkpoint.path)
            summary_rows.append(
                summarize_checkpoint_metrics(
                    checkpoint=checkpoint,
                    evaluation_summary=evaluation_summary,
                    checkpoint_metadata=metadata,
                )
            )
            long_rows.extend(
                flatten_metric_rows(
                    checkpoint=checkpoint,
                    evaluation_summary=evaluation_summary,
                    checkpoint_metadata=metadata,
                )
            )
        except Exception as error:
            error_row = {
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint.path),
                "error": str(error),
            }
            errors.append(error_row)
            print(f"ERROR: {error}", flush=True)
            if not args.continue_on_error:
                raise

    write_csv(summary_rows, output_dir / "checkpoint_ap_summary.csv")
    write_csv(long_rows, output_dir / "checkpoint_ap_metrics_long.csv")
    if errors:
        write_csv(errors, output_dir / "checkpoint_ap_errors.csv")

    best_by = best_rows_by_metric(
        summary_rows,
        metric_columns=[
            "ap_3d_Car_moderate",
            "mean_3d_moderate",
            "ap_bev_Car_moderate",
            "mean_bev_moderate",
        ],
    )
    json_summary = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "num_checkpoints": len(checkpoints),
        "num_completed": len(summary_rows),
        "num_errors": len(errors),
        "best_by": best_by,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        "errors": errors,
    }
    (output_dir / "checkpoint_ap_summary.json").write_text(
        json.dumps(json_summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("\nSweep complete.")
    print(f"Summary CSV: {output_dir / 'checkpoint_ap_summary.csv'}")
    print(f"Long CSV: {output_dir / 'checkpoint_ap_metrics_long.csv'}")
    print(f"Summary JSON: {output_dir / 'checkpoint_ap_summary.json'}")

    for metric_name, row in best_by.items():
        print(
            f"Best {metric_name}: {row['checkpoint']} = {float(row[metric_name]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
