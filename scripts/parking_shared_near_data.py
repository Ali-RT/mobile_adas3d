"""CPU-only trainval package for near-range shared-depth training.

Inherit frozen recording-log roles, never move held-out data into training.
Readiness uses completed point audits; preparation rechecks selected source
bytes and reprojection. A failed support gate produces no training archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_shared_depth_data as pilot_data
from scripts import prepare_parking_lidar_depth as gt

CONFIG = ROOT / "configs/parking_shared_depth_near_v1.json"
BASE = ROOT / "outputs/parking_nuscenes_trainval_20261027"


def cell_masks(points, mapping):
    """Match audited [2,4) and [4,6] conservative class/range cells."""
    import numpy as np
    result = {}
    for cls in ["Vehicle", "Pedestrian"]:
        for lo, hi in [(2, 4), (4, 6)]:
            upper = points["z"] <= hi if hi == 6 else points["z"] < hi
            result[f"{cls}:{lo}-{hi}m"] = (points["conservative"] & np.isin(points["semantic"], mapping[cls])
                                            & (points["z"] >= lo) & upper)
    return result


def validate_manifest(m):
    pilot_data.validate_manifest(m)
    if (m["config"] != json.loads(CONFIG.read_text()) or m["scenes"]["test"]
            or any(s["role"] not in ["train", "dev"] for s in m["samples"])
            or not m["readiness"]["training_data_ready"]):
        raise ValueError("Only the reviewed train/dev recipe is allowed; held-out images are excluded")
    if len({s["sample_token"] for s in m["samples"]}) != len(m["samples"]):
        raise ValueError("Use one camera per LiDAR keyframe")
    for role in ["train", "dev"]:
        if sum(s["role"] == role for s in m["samples"]) > m["config"][f"maximum_{role}_images"]:
            raise ValueError("Prepared input count exceeds the declared budget")
    for s in m["samples"]:
        if set(s["counts"]) != set(m["config"]["critical_cells"]) or any(type(n) is not int or n < 0 for n in s["counts"].values()):
            raise ValueError("Invalid audited cell counts")
    logs = {role: {s["recording_log"] for s in m["samples"] if s["role"] == role} for role in ["train", "dev"]}
    forbidden = set(m["excluded_recording_logs"])
    if logs["train"] & logs["dev"] or (logs["train"] | logs["dev"]) & forbidden:
        raise ValueError("Recording-log leakage")
    if readiness(m["samples"], m["config"]) != m["readiness"]:
        raise ValueError("Development/training support changed")
    init = m["initialization"]
    if (init["sha256"] != m["config"]["initial_checkpoint_sha256"]
            or init["run_identity"] != m["config"]["initial_run_identity"]):
        raise ValueError("Initialization changed")
    for key in ["file", "sidecar_file"]:
        pilot_data.safe_member(init[key])
    return m


def read_manifest(folder):
    return validate_manifest(json.loads((Path(folder) / "input_manifest.json").read_text()))


def readiness(rows, config):
    result = {}
    for role in ["train", "dev"]:
        cells = {}
        chosen = [r for r in rows if r["role"] == role]
        for cell in config["critical_cells"]:
            eligible = [r for r in chosen if r["counts"][cell] >= config["minimum_points_in_cell_image"]]
            frames = {r["sample_token"] for r in eligible}
            logs = {r["recording_log"] for r in eligible}
            n = sum(r["counts"][cell] for r in eligible)
            cells[cell] = {"unique_keyframes": len(frames), "recording_logs": len(logs), "points": n,
                "passed": len(frames) >= config[f"minimum_{role}_keyframes_per_cell"]
                    and len(logs) >= config[f"minimum_{role}_logs_per_cell"] and n >= config["minimum_points_per_cell"]}
        result[role] = {"images": len(chosen), "cells": cells}
    return {"roles": result, "training_data_ready": all(c["passed"] for r in result.values() for c in r["cells"].values())}


def select(rows, maximum, config):
    """Deterministic GT enrichment, one camera per LiDAR keyframe."""
    if {r["role"] for r in rows} - {"train", "dev"}:
        raise ValueError("Never select held-out inputs for this package")
    ranked = sorted(rows, key=lambda r: (common.sha(f"{config['seed']}:{r['camera_token']}".encode()), r["camera_token"]))
    selected, used = [], set()
    while len(selected) < maximum:
        added = False
        for cell in config["critical_cells"]:
            row = next((r for r in ranked if r["sample_token"] not in used and r["counts"][cell] >= config["minimum_points_in_cell_image"]), None)
            if row is not None:
                selected.append(row); used.add(row["sample_token"]); added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
    for row in ranked:
        if len(selected) == maximum:
            break
        if row["sample_token"] not in used:
            selected.append(row); used.add(row["sample_token"])
    return selected


def collect(parts, acquisition):
    """Do not reopen thousands of raw files just to report a failed gate."""
    sources, all_rows, used = [], [], set()
    scene_info = {s["scene"]: s for s in acquisition["per_scene"]}
    frozen_roles = {s: r["role"] for s, r in scene_info.items()}
    logs_by_role = {}
    for scene in scene_info.values():
        logs_by_role.setdefault(scene["role"], set()).add(scene["log_token"])
    forbidden = set().union(*(v for k, v in logs_by_role.items() if k not in ["train", "dev"]))
    if logs_by_role["train"] & logs_by_role["dev"] or (logs_by_role["train"] | logs_by_role["dev"]) & forbidden:
        raise ValueError("Frozen acquisition roles share a recording log")
    for folder, coverage in parts:
        folder, coverage = Path(folder), Path(coverage)
        m = json.loads((folder / "part_inputs.json").read_text())
        identity = json.loads((folder / "prepare_identity.json").read_text())
        report = json.loads((coverage / "point_coverage.json").read_text())
        audited = json.loads((coverage / "audit_identity.json").read_text())
        if (not m["complete"] or m["manifest_sha256"] != common.identity({k: v for k, v in m.items() if k != "manifest_sha256"})
                or common.identity(identity) != m["identity_sha256"] or identity["acquisition_sha256"] != common.identity(acquisition)
                or common.identity(audited) != report["identity_sha256"] or not report["complete"]
                or audited["input_manifest_sha256"] != m["manifest_sha256"]
                or any(frozen_roles.get(scene) != role for scene, role in m["scene_roles"].items())
                or audited["source_sha256"]["prepare_parking_lidar_depth.py"] != common.file_sha(Path(gt.__file__))):
            raise ValueError("Use matching completed source manifests and point audits")
        if sources and (audited["geometry"] != sources[0]["audit"]["geometry"] or audited["class_mapping"] != sources[0]["audit"]["class_mapping"]):
            raise ValueError("Source parts use different depth geometry/class mapping")
        index = len(sources)
        sources.append({"folder": folder, "coverage": coverage, "manifest": m, "audit": audited})
        keyframes = {k["sample_token"]: k for k in m["keyframes"]}
        for row in report["camera_rows"]:
            if row["role"] not in ["train", "dev"]:
                continue
            if row["camera_token"] in used:
                raise ValueError("Repeated camera across acquisition parts")
            used.add(row["camera_token"])
            k = keyframes[row["sample_token"]]
            info = scene_info[row["scene"]]
            if row["role"] != info["role"] or k["role"] != row["role"] or k["scene"] != row["scene"]:
                raise ValueError("Source row role changed")
            all_rows.append({**row, "recording_log": info["log_token"], "source_index": index})
    return sources, all_rows, sorted(forbidden)


def prepare(a):
    config = json.loads(a.config.read_text())
    acquisition = json.loads(a.acquisition.read_text())
    parts = a.part or [(BASE / "part01_inputs_r1", BASE / "part01_coverage_r1")]
    if any(a.output.resolve().is_relative_to(Path(x).resolve()) for pair in parts for x in pair):
        raise ValueError("Keep outputs separate from source parts")
    sources, rows, forbidden = collect(parts, acquisition)
    selected = sum((select([r for r in rows if r["role"] == role], config[f"maximum_{role}_images"], config) for role in ["train", "dev"]), [])
    status = readiness(selected, config)
    identity = {"config": config, "acquisition_sha256": common.file_sha(a.acquisition),
        "implementation_sha256": common.file_sha(Path(__file__)),
        "sources": [{"manifest_sha256": s["manifest"]["manifest_sha256"],
                     "coverage_sha256": common.file_sha(s["coverage"] / "point_coverage.json")} for s in sources]}
    common.immutable(a.output / "prepare_identity.json", common.encoded(identity))
    common.immutable(a.output / "readiness.json", common.encoded(status))
    print(json.dumps(status, indent=2), flush=True)
    if not status["training_data_ready"]:
        present = {r["scene"] for r in rows}
        candidates = sorted([s for s in acquisition["per_scene"] if s["role"] == "dev" and s["scene"] not in present],
            key=lambda s: (-min(s["candidate_keyframes"]["Pedestrian:2-4m"], s["candidate_keyframes"]["Pedestrian:4-6m"]), s["scene"]))[:12]
        common.immutable(a.output / "additional_dev_candidates.json", common.encoded({
            "scope": "Annotation-centre acquisition proxies only; audit actual points after downloading",
            "candidates": candidates, "do_not_move_heldout_roles": True}))
        if a.stage == "prepare":
            raise RuntimeError("Train/dev support is insufficient. No training archive created. Add audited keyframe parts and use a new output folder; preserve frozen roles.")
        return
    if a.stage == "readiness":
        return
    import numpy as np
    def one(row):
        source = sources[row["source_index"]]; m = source["manifest"]
        k = next(k for k in m["keyframes"] if k["sample_token"] == row["sample_token"])
        camera = k["sensors"][row["channel"]]; lidar = k["sensors"]["LIDAR_TOP"]
        root = source["folder"] / "data"
        for name in [camera["filename"], lidar["filename"], k["labels_file"]]:
            if common.file_sha(gt.safe_file(root, name)) != m["files"][name]["sha256"]:
                raise ValueError("Selected source bytes changed")
        raw = np.fromfile(gt.safe_file(root, lidar["filename"]), dtype="<f4")
        semantic = np.fromfile(gt.safe_file(root, k["labels_file"]), dtype=np.uint8)
        if raw.size % 5 or not raw.size or raw.size // 5 != len(semantic):
            raise ValueError("Malformed LiDAR/semantic arrays")
        cs = m["calibrated_sensor"][camera["calibrated_sensor_token"]]
        uv, z, indices, _ = gt.camera_projection(raw.reshape(-1, 5)[:, :3],
            m["calibrated_sensor"][lidar["calibrated_sensor_token"]], m["ego_pose"][lidar["ego_pose_token"]],
            m["ego_pose"][camera["ego_pose_token"]], cs, camera["width"], camera["height"])
        points = gt.rasterize(uv, z, semantic[indices], np.zeros(len(z), bool), camera["width"], camera["height"], source["audit"]["geometry"])
        counts = {key: int(mask.sum()) for key, mask in cell_masks(points, source["audit"]["class_mapping"]).items()}
        if counts != row["counts"] or int(points["conservative"].sum()) < config["minimum_total_conservative_points_per_image"]:
            raise ValueError("Reprojected GT differs from audited counts")
        image_file = f"images/{camera['token']}.jpg"; truth_file = f"ground_truth/{camera['token']}.npz"
        common.immutable(a.output / image_file, gt.safe_file(root, camera["filename"]).read_bytes())
        buffer = io.BytesIO(); np.savez_compressed(buffer, **{k: v for k, v in points.items() if k != "raster_points_before_range_semantic_filter"})
        common.immutable(a.output / truth_file, buffer.getvalue())
        return {k: v for k, v in {**row, "image_file": image_file, "image_sha256": m["files"][camera["filename"]]["sha256"],
            "ground_truth_file": truth_file, "ground_truth_sha256": common.sha(buffer.getvalue()),
            "width": camera["width"], "height": camera["height"], "intrinsics": cs["camera_intrinsic"]}.items() if k != "source_index"}
    samples = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, sample in enumerate(pool.map(one, selected), 1):
            samples.append(sample)
            if i % 64 == 0:
                print(f"CPU projected/staged {i}/{len(selected)} images", flush=True)
    checkpoint = a.initial_checkpoint
    sidecar = checkpoint.with_suffix(".json")
    if common.file_sha(checkpoint) != config["initial_checkpoint_sha256"] or json.loads(sidecar.read_text())["sha256"] != common.file_sha(checkpoint):
        raise ValueError("Use the frozen Mini epoch-4 initialization")
    init = {"file": "initialization/epoch_004.pth", "sha256": common.file_sha(checkpoint),
            "sidecar_file": "initialization/epoch_004.json", "sidecar_sha256": common.file_sha(sidecar),
            "run_identity": config["initial_run_identity"]}
    common.immutable(a.output / init["file"], checkpoint.read_bytes())
    common.immutable(a.output / init["sidecar_file"], sidecar.read_bytes())
    m = {"complete": True, "revision": config["revision"], "config": config,
        "prepare_identity_sha256": common.identity(identity), "mapping": sources[0]["audit"]["class_mapping"],
        "geometry": sources[0]["audit"]["geometry"], "initialization": init,
        "readiness": status, "excluded_recording_logs": forbidden,
        "scenes": {role: sorted({s["scene"] for s in samples if s["role"] == role}) for role in ["train", "dev", "test"]},
        "samples": samples, "files": {s[f"{k}_file"]: s[f"{k}_sha256"] for s in samples for k in ["image", "ground_truth"]}}
    m["manifest_sha256"] = common.identity(m); validate_manifest(m)
    common.immutable(a.output / "input_manifest.json", common.encoded(m)); publish(a.output, m)


def payload(m):
    init = m["initialization"]
    return {**m["files"], init["file"]: init["sha256"], init["sidecar_file"]: init["sidecar_sha256"]}


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        if any(i.file_size > 32 * 1024 ** 2 for i in z.infolist()) or sum(i.file_size for i in z.infolist()) > 8 * 1024 ** 3:
            raise ValueError("Training archive exceeds the bounded package size")
        for name in names:
            pilot_data.safe_member(name)
        m = validate_manifest(json.loads(z.read("input_manifest.json")))
        if set(names) != set(payload(m)) | {"input_manifest.json"}:
            raise ValueError("Unexpected training archive inventory")
        for name, digest in payload(m).items():
            if common.sha(z.read(name)) != digest:
                raise ValueError("Training payload changed")
    return m


def publish(folder, m):
    archive = folder / "shared_near_inputs.zip"
    if archive.exists():
        if verify_archive(archive) != m:
            raise ValueError("Preserve conflicting archive")
        common.immutable(folder / "archive_receipt.json", common.encoded({"file": archive.name,
            "sha256": common.file_sha(archive), "manifest_sha256": m["manifest_sha256"]}))
        return
    temporary = folder / ".shared_near_inputs.partial.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for name, digest in sorted(payload(m).items()):
            raw = gt.safe_file(folder, name).read_bytes()
            if common.sha(raw) != digest:
                raise ValueError("Prepared payload changed")
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
        z.writestr("input_manifest.json", common.encoded(m))
    temporary.replace(archive)
    common.immutable(folder / "archive_receipt.json", common.encoded({"file": archive.name,
        "sha256": common.file_sha(archive), "manifest_sha256": m["manifest_sha256"]}))
    print(f"CPU package ready: {archive}", flush=True)


def restore(archive, output):
    m = verify_archive(archive)
    def one(item):
        name, digest = item
        destination = output / pilot_data.safe_member(name)
        if destination.is_symlink() or not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("Staging path escapes output")
        if destination.exists():
            if common.file_sha(destination) != digest:
                raise ValueError("Preserve changed local input")
            return
        with zipfile.ZipFile(archive) as z:
            raw = z.read(name)
        common.immutable(destination, raw)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(one, payload(m).items()), 1):
            if i % 128 == 0:
                print(f"CPU restored/verified {i}/{len(payload(m))} members", flush=True)
    common.immutable(output / "input_manifest.json", common.encoded(m))
    print("CPU staging complete; train/dev support passed; no held-out images in package", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["readiness", "prepare", "restore"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--part", nargs=2, type=Path, action="append", metavar=("INPUTS", "COVERAGE"))
    p.add_argument("--acquisition", type=Path, default=BASE / "acquisition_metadata_r1/metadata_shortlist.json")
    p.add_argument("--initial-checkpoint", type=Path, default=ROOT / "outputs/parking_shared_depth_training_20261027/pilot_mps_r2/checkpoints/epoch_004.pth")
    p.add_argument("--archive", type=Path)
    p.add_argument("--config", type=Path, default=CONFIG)
    a = p.parse_args()
    if a.output.is_symlink() or any(a.output.resolve().is_relative_to(Path(x).resolve()) for pair in (a.part or []) for x in pair):
        p.error("Keep outputs separate from immutable source parts")
    if a.stage == "restore":
        if a.archive is None:
            p.error("restore requires --archive")
        restore(a.archive, a.output)
    else:
        prepare(a)


if __name__ == "__main__":
    main()
