"""CPU-only check that the two patched upstream loaders encode the same view."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from m61_common import (NATIVE_CLASSES, TARGET_KEYS, array_hash, input_fingerprint,
                        read_json, write_json)
from prepare_m61_distillation import make_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-repo", type=Path, required=True)
    p.add_argument("--teacher-repo", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--role", choices=("student", "teacher"))
    args = p.parse_args()
    if args.role:
        # The upstream dataset imports an unused CUDA evaluator at module load.
        # Simulation is scoped to this loader-only subprocess; no model or IoU
        # evaluation runs here. Real cache/smoke/training processes never set it.
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
        repo = (args.student_repo if args.role == "student" else args.teacher_repo).resolve()
        sys.path.insert(0, str(repo))
        from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
        name = "monodetr" if args.role == "student" else "monodgp"
        base = yaml.safe_load((repo / f"configs/{name}.yaml").read_text())
        cfg = make_config(base, args.dataset_root.resolve(), args.role, args.output.parent, repo)
        dataset = KITTI_Dataset(args.split, cfg["dataset"])
        dataset.data_augmentation = False
        if dataset.cls2id != NATIVE_CLASSES or np.any(dataset.cls_mean_size):
            raise RuntimeError("Unexpected class/dimension encoding")
        rows = []
        for i in range(min(args.limit, len(dataset))):
            image, calib, raw, info = dataset[i]
            target = {k: raw[k][raw["mask_2d"]] for k in TARGET_KEYS}
            rows.append(dict(image_id=f"{int(info['img_id']):06d}",
                             input_sha256=input_fingerprint(image, calib, raw["img_size"]),
                             target_sha256=array_hash(target), objects=len(target["labels"])))
        write_json(args.output, rows)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role in ("teacher", "student"):
        path = args.output.with_name(args.output.stem + f"_{role}.json")
        paths[role] = path
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--student-repo", str(args.student_repo.resolve()),
                        "--teacher-repo", str(args.teacher_repo.resolve()),
                        "--dataset-root", str(args.dataset_root.resolve()),
                        "--split", args.split, "--limit", str(args.limit),
                        "--output", str(path.resolve()), "--role", role], check=True)
    teacher, student = read_json(paths["teacher"]), read_json(paths["student"])
    report = dict(schema_version=1, complete=teacher == student, samples=len(teacher), split=args.split,
                  input_and_target_hashes_identical=teacher == student, rows=teacher,
                  model_inference_performed=False, unused_evaluator_import_simulated=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if teacher != student:
        raise RuntimeError("Dataset views differ; inspect the per-role probe reports before caching")


if __name__ == "__main__":
    main()
