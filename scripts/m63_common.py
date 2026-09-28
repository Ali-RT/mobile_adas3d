"""M63 frozen R0-to-A2 depth pilot; M61/M62 evidence remains unchanged."""
from pathlib import Path
from m61_common import (AP_GATES, COMPONENTS, ROOT, array_hash, build_runtime,
    check_split, environment, input_fingerprint, json_hash, pack_targets,
    read_json, seed_all, sha256, write_json)
from diagnose_m62_r0_a2 import load as load_m62, retry_read as npz_read

REVISION = "M63-2026-09-28-r1"
EVIDENCE = {
    "m62_manifest.json": "ed85f7c98652ac4c666ac0830d79ab9f65876cce55ee1ac1ad401dfe252e3880",
    "m62_diagnostic.json": "53f630b01e95b3194a480ea880af0b57b7fe88e2bcd3fca4a5a74b0d6a6ebce0",
    "m62_geometry_pairs.json": "e2310737b5fb3d4fe9a40ce6c81a4c23a80fd3de8c3a5f0a0dfeac079d41a5e3",
}
BASELINE = dict(vehicle_3d_moderate=15.457290707384603,
    pedestrian_3d_moderate=7.532768186994984, mean_3d_moderate=11.495029447189793,
    vehicle_bev_moderate=21.37498561663574, pedestrian_bev_moderate=8.489171811764757,
    vehicle_near_recall=0.8829344841114162, pedestrian_near_recall=0.6922398589065256)

def code_hash():
    paths = ["scripts/m63_common.py", "scripts/prepare_m63_depth_pilot.py",
             "scripts/train_m63_student.py", "scripts/evaluate_m63_pilot.py",
             "third_party/monodetr/m63_depth_loss.py",
             "scripts/evaluate_kitti_prediction_dir.py",
             "scripts/audit_product_prediction_geometry.py",
             "configs/kitti_mobileadas3d_s1.yaml"]
    return json_hash({p: sha256(ROOT / p) for p in paths})

def reviewed_m62(root):
    root = Path(root)
    for name, digest in EVIDENCE.items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Not the reviewed M62 evidence: {root / name}")
    return load_m62(root / "m62_manifest.json")

def baseline_checks(metrics):
    return {k: abs(metrics[k] - v) <= (0.01 if "recall" in k else 0.15)
            for k, v in BASELINE.items()}

def require_baseline(m):
    path = Path(m["output_dir"]) / "m63_baseline_metrics.json"
    b = read_json(path)
    if (b.get("manifest_sha256") != m["manifest_sha256"]
            or b.get("checkpoint_sha256") != m["student"]["checkpoint_sha256"]
            or not all(baseline_checks(b).values())):
        raise RuntimeError("Fresh A2 baseline differs; review before training")
    return sha256(path)

def load_manifest(path):
    m = read_json(path)
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if (json_hash(body) != m["manifest_sha256"] or m["revision"] != REVISION
            or m["code_sha256"] != code_hash()):
        raise RuntimeError("M63 manifest or implementation changed")
    previous = reviewed_m62(m["m62_output"])
    if m["environment"] != environment() or previous["environment"] != m["environment"]:
        raise RuntimeError("Restore the M62 GPU/software environment; do not mix runs")
    expected_student = dict(previous["models"]["student"], repo=previous["repo"])
    if m["student"] != expected_student or m["dataset_root"] != previous["dataset_root"]:
        raise RuntimeError("M63 student or dataset differs from reviewed M62")
    check_split(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    return m
