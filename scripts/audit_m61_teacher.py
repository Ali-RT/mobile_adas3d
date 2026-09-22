"""Compare train-only teacher/student geometry and freeze reliable Vehicle targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from m61_common import (COMPONENTS, OUTPUT_KEYS, check_split, load_manifest, npz_read,
                        npz_write, sha256, validate_cache, write_json)


def xyxy(box):
    x, y, left, right, top, bottom = box
    return np.array([x - left, y - top, x + right, y + bottom])


def iou(a, b):
    a, b = xyxy(a), xyxy(b)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
    return float(intersection / max(float(union), 1e-12))


def alpha(angles):
    bin_id = int(np.argmax(angles[:12]))
    return bin_id * (2 * np.pi / 12) + float(angles[12 + bin_id])


def center_xyz(box, depth, calibration, image_size):
    u, v = box[:2] * image_size
    p = calibration
    return np.array([(u * depth - p[0, 2] * depth - p[0, 3]) / p[0, 0],
                     (v * depth - p[1, 2] * depth - p[1, 3]) / p[1, 1], depth])


def errors(sample, query, gt):
    depth = float(sample["out_pred_depth"][query, 0])
    true_depth = float(sample["tgt_depth"][gt, 0])
    dim = sample["out_pred_3d_dim"][query]
    true_dim = sample["tgt_size_3d"][gt]
    true_alpha = float(sample["tgt_heading_bin"][gt, 0]) * (2 * np.pi / 12) + float(sample["tgt_heading_res"][gt, 0])
    predicted_alpha = alpha(sample["out_pred_angle"][query])
    angular_error = abs((predicted_alpha - true_alpha + np.pi) % (2 * np.pi) - np.pi)
    center = center_xyz(sample["out_pred_boxes"][query], depth, sample["calibration"], sample["image_size"])
    true_center = center_xyz(sample["tgt_boxes_3d"][gt], true_depth, sample["calibration"], sample["image_size"])
    return dict(depth=abs(depth - true_depth), dimensions=float(np.abs(dim - true_dim).mean()),
                center=float(np.linalg.norm(center - true_center)), angle=float(np.degrees(angular_error)))


def audit_sample(teacher, student, policy):
    for key in ("input_sha256", "target_sha256", "manifest_sha256"):
        if str(teacher[key]) != str(student[key]):
            raise RuntimeError(f"Teacher/student {key} mismatch; do not distill different image views/GT")
    n = len(teacher["tgt_labels"])
    aligned = {k: np.zeros((n,) + teacher["out_" + k].shape[1:], np.float32) for k in OUTPUT_KEYS}
    masks = np.zeros((n, len(COMPONENTS)), dtype=bool)
    tq = {int(g): int(q) for q, g in zip(teacher["query_indices"], teacher["gt_indices"])}
    sq = {int(g): int(q) for q, g in zip(student["query_indices"], student["gt_indices"])}
    comparisons = []
    for gt in sorted(tq.keys() & sq.keys()):
        q, s = tq[gt], sq[gt]
        if int(teacher["tgt_labels"][gt]) != 1:
            continue
        true_depth = float(teacher["tgt_depth"][gt, 0])
        if not 2 <= true_depth < policy["max_depth_exclusive"]:
            continue
        logits = teacher["out_pred_logits"][q]
        score = float(1 / (1 + np.exp(-np.clip(logits[1], -80, 80))))
        overlap = iou(teacher["out_pred_boxes"][q], teacher["tgt_boxes_3d"][gt])
        if (np.argmax(logits) != 1 or score < policy["min_score"] or overlap < policy["min_iou"]
                or teacher["out_pred_depth"][q, 0] <= 0
                or np.any(teacher["out_pred_3d_dim"][q] <= 0)):
            continue
        te, se = errors(teacher, q, gt), errors(student, s, gt)
        if not all(np.isfinite(list(te.values()))) or not all(np.isfinite(list(se.values()))):
            raise RuntimeError("Non-finite geometry in train audit")
        for key in OUTPUT_KEYS:
            aligned[key][gt] = teacher["out_" + key][q]
        for j, component in enumerate(COMPONENTS):
            masks[gt, j] = te[component] < policy["individual_error_ratio_max"] * se[component]
        comparisons.append(dict(gt=gt, teacher=te, student=se))
    return aligned, masks, comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    m = load_manifest(args.manifest)
    output = Path(m["output_dir"])
    for role in ("teacher", "student"):
        validate_cache(output / "cache" / role, m, role)
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    sums = {c: dict(teacher=[], student=[], approved=0) for c in COMPONENTS}
    files = {}
    for i, image_id in enumerate(ids):
        teacher = npz_read(output / "cache/teacher" / f"{image_id}.npz")
        student = npz_read(output / "cache/student" / f"{image_id}.npz")
        aligned, masks, rows = audit_sample(teacher, student, m["audit_policy"])
        for j, component in enumerate(COMPONENTS):
            sums[component]["teacher"].extend(r["teacher"][component] for r in rows)
            sums[component]["student"].extend(r["student"][component] for r in rows)
            sums[component]["approved"] += int(masks[:, j].sum())
        path = output / "approved_targets" / f"{image_id}.npz"
        arrays = {**aligned, "masks": masks, "labels": teacher["tgt_labels"],
                  "image_size": teacher["image_size"],
                  **{k: teacher[k] for k in ("input_sha256", "target_sha256", "manifest_sha256")}}
        if path.exists():
            previous = npz_read(path)
            if set(previous) != set(arrays) or any(not np.array_equal(previous[k], v) for k, v in arrays.items()):
                raise RuntimeError(f"Existing approved targets changed: {image_id}")
        else:
            npz_write(path, arrays)
        files[image_id] = sha256(path)
        if (i + 1) % 500 == 0:
            print(f"Audited {i + 1}/{len(ids)} training images", flush=True)
    metrics = {}
    for c, values in sums.items():
        t = float(np.mean(values["teacher"])) if values["teacher"] else None
        s = float(np.mean(values["student"])) if values["student"] else None
        metrics[c] = dict(teacher_mean_error=t, student_mean_error=s,
                          paired_objects=len(values["teacher"]), approved_objects=values["approved"],
                          enabled=t is not None and t < s and values["approved"] >= m["audit_policy"]["min_component_pairs"])
    enabled = [c for c in COMPONENTS if metrics[c]["enabled"]]
    report = dict(schema_version=1, complete=True, manifest_sha256=m["manifest_sha256"],
                  split="train", samples=len(ids), components=metrics, enabled_components=enabled,
                  units=dict(depth="m", dimensions="m mean H/W/L", center="m geometric-center L2",
                             angle="degrees wrapped observation alpha"),
                  source_cache_sha256={r: sha256(output / "cache" / r / "cache_manifest.json") for r in ("teacher", "student")},
                  approved_files=files, pilot_authorized=bool(enabled), full_run_authorized=False)
    write_json(output / "m61_teacher_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "approved_files"}, indent=2))
    if not enabled:
        raise RuntimeError("No geometry component passed the train-only teacher audit; stop before training")


if __name__ == "__main__":
    main()
