from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CHECKPOINT_RE = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            tail.append(line.rstrip())
            tail = tail[-80:]
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Command exited {return_code}; log={log_path}\n" + "\n".join(tail)
        )


def metric_value(summary: dict[str, Any], metric: str, class_name: str) -> float:
    matches = [
        row
        for row in summary["metrics"]
        if row["metric"] == metric
        and row["class_name"] == class_name
        and row["difficulty"] == "moderate"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {metric}/{class_name}/moderate metric, found {len(matches)}"
        )
    return float(matches[0]["ap_r40"])


def collect_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in run_dir.glob("checkpoint_epoch_*.pth"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_file():
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Sweep MonoDETR R0 checkpoints with the frozen product taxonomy."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--product-config", default="configs/kitti_mobileadas3d_s1.yaml"
    )
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--source-name-prefix", default="MonoDETR_R0")
    parser.add_argument("--epochs", type=int, nargs="*")
    args = parser.parse_args()

    monodetr_repo = args.monodetr_repo.resolve()
    mobile_repo = args.mobile_repo.resolve()
    training_config = args.training_config.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    for required in (monodetr_repo, mobile_repo, run_dir, args.dataset_root.resolve()):
        if not required.exists():
            raise FileNotFoundError(required)
    if not training_config.is_file():
        raise FileNotFoundError(training_config)

    checkpoints = collect_checkpoints(run_dir)
    if args.epochs:
        requested = set(args.epochs)
        checkpoints = [item for item in checkpoints if item[0] in requested]
        missing = requested - {epoch for epoch, _ in checkpoints}
        if missing:
            raise FileNotFoundError(f"Requested checkpoint epochs missing: {sorted(missing)}")
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, (epoch, checkpoint) in enumerate(checkpoints, start=1):
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        summary_path = epoch_dir / "kitti_r40_summary.json"
        if summary_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("complete_split"):
                print(f"[{index}/{len(checkpoints)}] epoch {epoch}: cached", flush=True)
                summary = cached
            else:
                summary_path.unlink()
                summary = None
        else:
            summary = None

        if summary is None:
            print(f"[{index}/{len(checkpoints)}] epoch {epoch}: inference", flush=True)
            eval_config = dict(base_config)
            eval_config["tester"] = dict(base_config["tester"])
            eval_config["trainer"] = dict(base_config["trainer"])
            eval_config["tester"].update(
                {"mode": "single", "checkpoint": epoch, "threshold": args.score_threshold, "topk": args.topk}
            )
            eval_config["trainer"].update(
                {"save_all": True, "pretrain_model": None, "resume_model": False}
            )
            eval_config_path = monodetr_repo / "configs/monodetr_r0_product_sweep.yaml"
            eval_config_path.write_text(
                yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8"
            )
            run_logged(
                [sys.executable, "-u", "tools/train_val.py", "--config", str(eval_config_path), "--evaluate_only"],
                monodetr_repo,
                epoch_dir / "monodetr_inference.log",
            )
            prediction_dir = run_dir / "outputs/data"
            prediction_files = list(prediction_dir.glob("*.txt"))
            if len(prediction_files) != 3769:
                raise RuntimeError(
                    f"Epoch {epoch} produced {len(prediction_files)}/3769 prediction files"
                )
            run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_kitti_prediction_dir.py",
                    "--config",
                    args.product_config,
                    "--profile",
                    args.profile,
                    "--dataset-root",
                    str(args.dataset_root),
                    "--split-dir",
                    str(args.split_dir),
                    "--prediction-dir",
                    str(prediction_dir),
                    "--split",
                    "val",
                    "--classes",
                    "Vehicle",
                    "Pedestrian",
                    "--source-name",
                    f"{args.source_name_prefix}_epoch_{epoch:03d}",
                    "--output-dir",
                    str(epoch_dir),
                ],
                mobile_repo,
                epoch_dir / "product_evaluation.log",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        vehicle_3d = metric_value(summary, "3d", "Vehicle")
        pedestrian_3d = metric_value(summary, "3d", "Pedestrian")
        vehicle_bev = metric_value(summary, "bev", "Vehicle")
        pedestrian_bev = metric_value(summary, "bev", "Pedestrian")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "vehicle_3d_moderate": vehicle_3d,
                "pedestrian_3d_moderate": pedestrian_3d,
                "mean_3d_moderate": (vehicle_3d + pedestrian_3d) / 2.0,
                "vehicle_bev_moderate": vehicle_bev,
                "pedestrian_bev_moderate": pedestrian_bev,
                "mean_bev_moderate": (vehicle_bev + pedestrian_bev) / 2.0,
                "complete_split": bool(summary["complete_split"]),
            }
        )

    if len(rows) != len(checkpoints) or not all(row["complete_split"] for row in rows):
        raise RuntimeError("R0 sweep is incomplete")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["mean_3d_moderate"],
            row["vehicle_3d_moderate"],
            row["mean_bev_moderate"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    best = ranked[0]
    best["checkpoint_sha256"] = sha256_file(Path(best["checkpoint"]))
    fieldnames = list(ranked[0])
    with (output_dir / "r0_product_checkpoint_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)
    selection = {
        "schema_version": 1,
        "complete": True,
        "selection_rule": "mean Vehicle/Pedestrian moderate 3D AP_R40; tie Vehicle 3D then mean BEV",
        "evaluated_checkpoints": len(rows),
        "score_threshold": args.score_threshold,
        "topk": args.topk,
        "selected": best,
        "ranked": ranked,
    }
    (output_dir / "r0_product_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    selected_link = output_dir / "SELECTED_CHECKPOINT_PATH.txt"
    selected_link.write_text(str(best["checkpoint"]) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
