"""Bind reviewed M62 evidence and materialize exactly 7014 depth-only targets."""
import argparse
import copy
from pathlib import Path
import numpy as np
from audit_m61_teacher import audit_sample
from m61_common import npz_write
from m63_common import (REVISION, code_hash, reviewed_m62, json_hash, sha256,
    read_json, write_json, npz_read, check_split)

def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    old = reviewed_m62(root)
    report = read_json(root / "m62_diagnostic.json")
    if not report["complete"] or report["proposed_component_for_review"] != "depth":
        raise RuntimeError("Reviewed M62 depth proposal missing")
    ids = check_split(Path(old["dataset_root"]) / "ImageSets/train.txt", "train")
    val_ids = check_split(Path(old["dataset_root"]) / "ImageSets/val.txt", "val")
    for image_id in ids + val_ids:
        for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
            path = Path(old["dataset_root"]) / "training" / folder / (image_id + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
    model = copy.deepcopy(old["models"]["student"])
    model["repo"] = old["repo"]
    m = dict(schema_version=1, revision=REVISION, code_sha256=code_hash(),
        m62_output=str(root), output_dir=str(output), dataset_root=old["dataset_root"],
        environment=old["environment"], student=model, seed=20268, epochs=10,
        evaluation_epochs=[10], batch_size=4, learning_rate=1e-5, overall_kd_weight=0.25,
        loss="Vehicle final-depth L1 in metres; GT losses unchanged", augmentation=False,
        full_run_authorized=False, phone_deployment_authorized=False,
        variants={v: {"run_dir": str(output / "runs" / v)}
                  for v in ("control", "vehicle_kd")})
    m["manifest_sha256"] = json_hash(m)
    manifest_path = output / "m63_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != m:
        raise RuntimeError("Existing M63 identity differs; preserve it")
    write_json(manifest_path, m)
    completed = {}
    for role in ("teacher", "student"):
        path = root / "cache" / role / "complete.json"
        if sha256(path) != report["source_cache_sha256"][role]:
            raise RuntimeError(f"Changed M62 cache report: {role}")
        c = read_json(path)
        if (not c["complete"] or set(c["files"]) != set(ids)
                or c["manifest_sha256"] != old["manifest_sha256"]
                or c["environment"] != old["environment"] or c["role"] != role):
            raise RuntimeError("Cache identity mismatch")
        completed[role] = c
    files, rows, count = {}, [], 0
    for i, image_id in enumerate(ids):
        samples = {}
        for role in completed:
            path = root / "cache" / role / (image_id + ".npz")
            if sha256(path) != completed[role]["files"][image_id]:
                raise RuntimeError(f"Cache changed: {path}")
            samples[role] = npz_read(path)
            if str(samples[role]["manifest_sha256"]) != old["manifest_sha256"]:
                raise RuntimeError("Cache sample identity mismatch")
        teacher = samples["teacher"]
        aligned, masks, matched = audit_sample(teacher, samples["student"], old["policy"])
        for row in matched:
            row.update(image_id=image_id, depth=float(teacher["tgt_depth"][row["gt"], 0]))
        rows.extend(matched)
        masks[:, 1:] = False
        count += int(masks[:, 0].sum())
        arrays = {**aligned, "masks": masks, "labels": teacher["tgt_labels"],
                  "image_size": teacher["image_size"], "input_sha256": teacher["input_sha256"],
                  "target_sha256": teacher["target_sha256"],
                  "manifest_sha256": np.asarray(m["manifest_sha256"])}
        path = output / "approved_targets" / (image_id + ".npz")
        if path.exists():
            saved = npz_read(path)
            if set(saved) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise RuntimeError(f"Changed approved targets: {path}")
        else:
            npz_write(path, arrays)
        files[image_id] = sha256(path)
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"Verified {i+1}/{len(ids)}; approved depth objects={count}", flush=True)
    if count != 7014 or json_hash(rows) != json_hash(read_json(root / "m62_geometry_pairs.json")):
        raise RuntimeError("Regenerated M62 evidence differs; do not train")
    audit = dict(complete=True, manifest_sha256=m["manifest_sha256"], samples=len(ids),
        enabled_components=["depth"], approved_depth_objects=count, approved_files=files,
        source_cache_sha256=report["source_cache_sha256"], pilot_authorized=True,
        full_run_authorized=False)
    write_json(output / "m63_teacher_audit.json", audit)
    print(f"Prepared {manifest_path}; next: fresh baseline, CUDA smoke, paired pilot.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m62-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.m62_output, a.output)
