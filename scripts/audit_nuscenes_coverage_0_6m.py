"""CPU-only full nuScenes coverage census, not model evaluation or training.

Read immutable Drive archives; stage one part locally; publish resumable,
compressed 64-keyframe chunks. Never change frozen scene/log assignments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import nuscenes_coverage_data as data
from scripts import parking_near_field_geometry as geometry

CONFIG = ROOT / "configs/nuscenes_coverage_0_6m_v1.json"
ROLES = ("train", "dev", "heldout_eval", "excluded_previously_used_mini_log")
LIMITATIONS = [
    "Distance is positive camera-axis Z, not Euclidean camera range or nearest ultrasonic surface range.",
    "No model inference, training, accuracy threshold, pass/fail accuracy test, or phone update is performed.",
    "Sparse measured support is not complete image visibility, complete 2D annotation, or metric accuracy qualification.",
    "Centre-in-image counts are annotation proxies, not proof of unoccluded camera visibility.",
    "Sensor ego poses compensate camera/LiDAR timing; object motion within the allowed 50ms gap is not modelled.",
    "Surface-supported objects require five exact-semantic, unambiguous oriented-box-associated conservative returns in one camera/band.",
    "Six cameras and overlapping bands reuse objects; unique annotation observations and instance tracks are unions, not sums.",
    "Instance tracks are not independent statistical samples. Keyframes repeat tracks over time.",
    "Original scene/log roles are unchanged. Excluded Mini logs remain excluded from training.",
    "Previously inspected evaluation logs are flagged. The uninspected-log slice is not proof of absence from model pretraining.",
    "A coverage result does not authorize training, certify a benchmark, or establish V0/V1 accuracy.",
]


def signed(value, field="manifest_sha256"):
    return {**value, field: common.identity(value)}


def check_signed(value, field="manifest_sha256"):
    if value.get(field) != common.identity({k: v for k, v in value.items() if k != field}):
        raise ValueError("Coverage record digest differs; preserve the run")
    return value


def load_config(path):
    c = json.loads(Path(path).read_text())
    if (c["parts"] != list(range(1, 11)) or c["range_bands_m"] != [[0, 1], [1, 2], [0, 2], [2, 4], [4, 6], [0, 6]]
            or c["accuracy_threshold_m"] is not None or c["accuracy_pass_fail_enabled"] is not False
            or c["training_authorized"] is not False or c["keyframes_per_chunk"] < 1):
        raise ValueError("Use the coverage-only contract with all near-field bands")
    return c


def initialize(args):
    c = load_config(args.config)
    roles = data.verified_roles(ROOT / c["frozen_roles_file"], c["expected_scenes"])
    dataset, output, local = (p.resolve() for p in (args.dataset, args.output, args.local))
    for a, b in [(dataset, output), (dataset, local), (output, local)]:
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("Dataset, results and helper-owned local scratch must be separate non-nested directories")
    if str(local).startswith("/content/drive/"):
        raise ValueError("Local scratch must be on Colab temporary disk, not Drive")
    for p in (args.dataset, args.output, args.local):
        if p.is_symlink() or p.parent.is_symlink():
            raise ValueError("Preserve symlinked workflow roots")
    manifest, receipts = data.verified_downloads(dataset, c, ROOT / c["download_sources_file"])
    runtime = {"python": ".".join(platform.python_version_tuple()[:2])}
    for package, expected in c["runtime_packages"].items():
        runtime[package] = importlib.metadata.version(package)
        if runtime[package] != expected:
            raise ValueError(f"Coverage task dependency {package}: expected {expected}, found {runtime[package]}; rerun notebook setup")
    code = [Path(__file__), Path(data.__file__), Path(geometry.__file__),
            ROOT / "scripts/prepare_parking_lidar_depth.py", ROOT / "scripts/parking_edge_common.py",
            ROOT / "scripts/download_nuscenes_to_drive.py"]
    identity = signed({"config": c, "roles_sha256": roles["manifest_sha256"],
                       "download_manifest_sha256": manifest["manifest_sha256"],
                       "runtime": runtime, "code_sha256": {p.name: common.file_sha(p) for p in code}})
    common.immutable(output / "run_identity.json", common.encoded(identity))
    if local.exists() and any(local.iterdir()) and not (local / "owner.json").is_file():
        raise ValueError("Preserve nonempty unowned local scratch")
    common.immutable(local / "owner.json", common.encoded(identity))
    for name in ["metadata", "lidarseg", "archives", "parts"]:
        data.safe_target(local, name)
        if (local / name).is_symlink():
            raise ValueError("Preserve symlinked scratch subdirectory")
    return c, roles, dataset, output, local, receipts, identity


def prepare_support(c, roles, dataset, local, receipts, identity):
    reserve = int(c["local_disk_reserve_gb"] * 1e9)
    for name, kind in [("v1.0-trainval_meta.tgz", "metadata"),
                       ("nuScenes-lidarseg-all-v1.0.tar.bz2", "lidarseg")]:
        archive = data.copy_archive(dataset, local, receipts[name], reserve)
        print(f"Check/extract local {kind}; no image files are written to Drive", flush=True)
        data.extract_support(archive, local / kind, receipts[name], c, roles, kind)
    return data.build_index(local, local / "metadata", local / "lidarseg", c, roles,
                            {"run_sha256": identity["manifest_sha256"]})


@lru_cache(maxsize=1)
def index_constants(database):
    with data.connect_index(database) as db:
        calibration = {r[0]: json.loads(r[1]) for r in db.execute("SELECT * FROM calibration")}
        semantic = {r["name"]: r["index"] for r in json.loads(db.execute("SELECT value FROM info WHERE key='semantic_categories'").fetchone()[0])}
    return calibration, semantic


def audit_frame(frame, part_root, labels_root, database, c, roles):
    import numpy as np
    calibration, semantic = index_constants(database)
    with data.connect_index(database) as db:
        poses = {}
        for row in frame["sensors"].values():
            token = row["ego_pose_token"]
            poses[token] = json.loads(db.execute("SELECT body FROM poses WHERE token=?", (token,)).fetchone()[0])
        anns = [json.loads(r[0]) for r in db.execute("SELECT body FROM annotations WHERE sample=? ORDER BY token", (frame["sample_token"],))]
        lidar = frame["sensors"]["LIDAR_TOP"]
        label = db.execute("SELECT filename FROM labels WHERE token=?", (lidar["token"],)).fetchone()[0]
    raw = np.fromfile(data.safe_target(part_root, lidar["local_file"]), dtype="<f4")
    sem = np.fromfile(data.safe_target(labels_root, label), dtype=np.uint8)
    if not len(raw) or raw.size % 5 or raw.size // 5 != len(sem) or sem.max() > 31 or not np.isfinite(raw).all():
        raise ValueError("Malformed LiDAR/semantic arrays")
    mapping = {cls: [semantic[n] for n in names] for cls, names in c["classes"].items()}
    ls, lp = calibration[lidar["calibrated_sensor_token"]], poses[lidar["ego_pose_token"]]
    role = roles["scenes"][frame["scene"]]
    result = []; excluded = []; owner = None
    for channel, camera in sorted(frame["sensors"].items()):
        if channel == "LIDAR_TOP":
            continue
        gap = abs(camera["timestamp"] - lidar["timestamp"]) / 1000
        if gap > c["maximum_camera_lidar_gap_ms"]:
            excluded.append({"camera_token": camera["token"], "reason": "camera_lidar_gap_over_limit"})
            continue
        cs, cp = calibration[camera["calibrated_sensor_token"]], poses[camera["ego_pose_token"]]
        width, height = camera["width"], camera["height"]
        uv, z, ids, world = geometry.project_positive(raw.reshape(-1, 5)[:, :3], ls, lp, cp, cs, width, height)
        if owner is None:
            owner = geometry.associate_objects(world, sem, anns)
        near = geometry.sparse_near_raster(uv, z, sem[ids], ids, width, height, c["boundary_guard"])
        counts = {f"{cls}:{geometry.band_name(lo, hi)}": int((near["conservative"] & np.isin(near["semantic"], indices)
                  & geometry.band_mask(near["z"], lo, hi)).sum())
                  for cls, indices in mapping.items() for lo, hi in c["range_bands_m"]}
        ownership = owner[near["source_indices"]]; objects = []
        for index, ann in enumerate(anns):
            mask = near["conservative"] & (ownership == index)
            surface = {geometry.band_name(lo, hi): int((mask & geometry.band_mask(near["z"], lo, hi)).sum())
                       for lo, hi in c["range_bands_m"]}
            centre_z, in_image = geometry.centre_in_camera(ann, cp, cs, width, height)
            if surface["0-6m"] or (in_image and 0 < centre_z <= 6):
                objects.append({"annotation_token": ann["token"], "instance_token": ann["instance_token"],
                    "class_name": ann["class_name"], "source_class": ann["source_class"],
                    "global_visibility_token": ann["visibility_token"], "centre_z_m": centre_z,
                    "centre_in_image": in_image, "surface_points": surface})
        result.append({"camera_token": camera["token"], "sample_token": frame["sample_token"],
            "scene": frame["scene"], "recording_log": role["log_token"], "role": role["role"],
            "channel": channel, "point_counts": counts, "objects": objects,
            "ambiguous_near_returns_excluded_from_object_support": int((near["conservative"] & (ownership == -2)).sum())})
    return signed({"sample_token": frame["sample_token"], "scene": frame["scene"],
                   "lidar_token": lidar["token"], "camera_rows": result, "excluded": excluded}, "record_sha256")


def validate_records(records, tokens, roles):
    if [r["sample_token"] for r in records] != tokens:
        raise ValueError("Chunk keyframe inventory differs")
    for record in records:
        check_signed(record, "record_sha256")
        role = roles["scenes"][record["scene"]]
        cameras = [r["camera_token"] for r in record["camera_rows"]] + [r["camera_token"] for r in record["excluded"]]
        if len(cameras) != 6 or len(set(cameras)) != 6:
            raise ValueError("Chunk lacks six distinct accounted camera views")
        for row in record["camera_rows"]:
            if (row["sample_token"] != record["sample_token"] or row["scene"] != record["scene"]
                    or row["role"] != role["role"] or row["recording_log"] != role["log_token"]):
                raise ValueError("Chunk changed frozen role/log assignment")


def read_chunk(path, part_identity, tokens, roles, expected_sha=None):
    raw = common.retry_read(path)
    if expected_sha is not None and common.sha(raw) != expected_sha:
        raise ValueError("Completed chunk bytes changed")
    value = check_signed(json.loads(gzip.decompress(raw)))
    if value["part_identity"] != part_identity:
        raise ValueError("Chunk belongs to a different source/run")
    validate_records(value["records"], tokens, roles)
    return value["records"]


def part_contract(number, receipt, identity):
    return {"part": number, "archive_sha256": receipt["sha256"], "run_sha256": identity["manifest_sha256"]}


def completed_part(folder, contract, roles):
    value = check_signed(json.loads((folder / "result.json").read_text()))
    if value["part_identity"] != contract or value["complete"] is not True:
        raise ValueError("Completed part identity differs")
    all_tokens = []
    for chunk in value["chunks"]:
        path = data.safe_target(folder, chunk["file"])
        read_chunk(path, contract, chunk["sample_tokens"], roles, chunk["sha256"])
        all_tokens.extend(chunk["sample_tokens"])
    if all_tokens != value["sample_tokens"] or len(set(all_tokens)) != len(all_tokens):
        raise ValueError("Completed part repeats/misses keyframes")
    return value


def process_part(number, receipt, dataset, output, local, database, c, roles, identity, workers, keep_local=False):
    folder = data.safe_target(output, f"parts/part{number:02d}")
    contract = part_contract(number, receipt, identity)
    if (folder / "result.json").exists():
        value = completed_part(folder, contract, roles)
        print(f"Part {number:02d}: verified complete Drive results; no archive restaging", flush=True)
        return value
    archive = data.copy_archive(dataset, local, receipt, int(c["local_disk_reserve_gb"] * 1e9))
    part_root = local / "parts" / f"part{number:02d}"
    staged = data.stage_part(archive, receipt, part_root, database, c, contract)
    frames = staged["keyframes"]; chunks = []; start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(frames), c["keyframes_per_chunk"]):
            batch = frames[offset:offset + c["keyframes_per_chunk"]]
            tokens = [f["sample_token"] for f in batch]
            relative = f"chunks/{offset // c['keyframes_per_chunk']:05d}.json.gz"
            path = data.safe_target(folder, relative)
            if path.exists():
                records = read_chunk(path, contract, tokens, roles)
            else:
                records = list(pool.map(lambda frame: audit_frame(frame, part_root, local / "lidarseg", database, c, roles), batch))
                validate_records(records, tokens, roles)
                raw = gzip.compress(common.encoded(signed({"part_identity": contract, "records": records})), mtime=0)
                common.immutable(path, raw)
                read_chunk(path, contract, tokens, roles, common.sha(raw))
            chunks.append({"file": relative, "sha256": common.file_sha(path), "sample_tokens": tokens})
            done = offset + len(batch)
            print(f"Part {number:02d}: {done}/{len(frames)} keyframes durable; elapsed {(time.monotonic()-start)/60:.1f} min", flush=True)
    value = signed({"part_identity": contract, "complete": True, "chunks": chunks,
        "sample_tokens": [f["sample_token"] for f in frames], "scenes": sorted({f["scene"] for f in frames}),
        "rgb_headers_checked": staged["rgb_headers_checked"], "source_archive_sha256": receipt["sha256"]})
    common.immutable(folder / "result.json", common.encoded(value))
    completed_part(folder, contract, roles)
    if not keep_local:
        data.release_part(local, part_root, archive, contract)
    return value


class Census:
    """Streaming unions distinguish camera appearances, observations and tracks."""
    def __init__(self, c):
        self.c = c; self.frames = set(); self.cameras = set(); self.scenes = set(); self.logs = set()
        self.excluded = Counter(); self.cells = {}
        for cls in c["classes"]:
            for lo, hi in c["range_bands_m"]:
                self.cells[f"{cls}:{geometry.band_name(lo, hi)}"] = {
                    "semantic_surface_points": 0, "camera_views_with_minimum_points": set(),
                    "surface_supported_object_camera_appearances": 0,
                    "surface_supported_keyframes": set(), "surface_supported_annotation_observations": set(),
                    "surface_supported_instance_tracks": set(), "surface_supported_scenes": set(),
                    "surface_supported_recording_logs": set(), "centre_in_image_object_camera_appearances": 0,
                    "centre_in_image_annotation_observations": set(), "centre_in_image_instance_tracks": set()}

    def add(self, record, role):
        self.frames.add(record["sample_token"]); self.scenes.add(record["scene"]); self.logs.add(role["log_token"])
        self.excluded.update(r["reason"] for r in record["excluded"])
        for row in record["camera_rows"]:
            camera = row["camera_token"]; self.cameras.add(camera)
            for key, points in row["point_counts"].items():
                cell = self.cells[key]; cell["semantic_surface_points"] += points
                if points >= self.c["minimum_surface_points_per_image_band"]:
                    cell["camera_views_with_minimum_points"].add(camera)
            for obj in row["objects"]:
                for lo, hi in self.c["range_bands_m"]:
                    name = geometry.band_name(lo, hi); cell = self.cells[f"{obj['class_name']}:{name}"]
                    if obj["surface_points"][name] >= self.c["minimum_surface_points_per_object_camera_band"]:
                        cell["surface_supported_object_camera_appearances"] += 1
                        for key, value in [("keyframes", record["sample_token"]), ("annotation_observations", obj["annotation_token"]),
                            ("instance_tracks", obj["instance_token"]), ("scenes", record["scene"]), ("recording_logs", role["log_token"])]:
                            cell["surface_supported_" + key].add(value)
                    z = obj["centre_z_m"]
                    if obj["centre_in_image"] and z > 0 and z >= lo and (z <= hi if hi == 6 else z < hi):
                        cell["centre_in_image_object_camera_appearances"] += 1
                        cell["centre_in_image_annotation_observations"].add(obj["annotation_token"])
                        cell["centre_in_image_instance_tracks"].add(obj["instance_token"])

    def result(self):
        cells = {key: {k: len(v) if isinstance(v, set) else v for k, v in cell.items()}
                 for key, cell in self.cells.items()}
        for key, cell in self.cells.items():
            cls = key.split(":")[0]
            cells[key]["centre_observations_without_five_surface_points_in_any_view_0_6m"] = len(
                cell["centre_in_image_annotation_observations"] - self.cells[cls + ":0-6m"]["surface_supported_annotation_observations"])
        return {"keyframes": len(self.frames), "camera_views_audited": len(self.cameras),
                "scenes": len(self.scenes), "recording_logs": len(self.logs), "camera_exclusions": dict(self.excluded),
                "cells": cells}


def aggregate(output, c, roles, receipts, identity):
    groups = {role: Census(c) for role in ROLES}
    groups["heldout_uninspected_logs"] = Census(c)
    scenes = {}; logs = {}; frames = set(); cameras = set(); lidars = set(); parts = []
    for number in c["parts"]:
        folder = output / "parts" / f"part{number:02d}"
        receipt = receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"]
        contract = part_contract(number, receipt, identity)
        part = completed_part(folder, contract, roles); parts.append(part["manifest_sha256"])
        for chunk in part["chunks"]:
            for record in read_chunk(folder / chunk["file"], contract, chunk["sample_tokens"], roles, chunk["sha256"]):
                token = record["sample_token"]; lidar = record["lidar_token"]
                views = {r["camera_token"] for r in record["camera_rows"] + record["excluded"]}
                if token in frames or lidar in lidars or cameras & views:
                    raise ValueError("Global sensor/keyframe inventory repeats across parts")
                frames.add(token); lidars.add(lidar); cameras.update(views)
                role = roles["scenes"][record["scene"]]
                groups[role["role"]].add(record, role)
                if role["role"] == "heldout_eval" and role["log_token"] not in roles["previously_inspected_evaluation_logs"]:
                    groups["heldout_uninspected_logs"].add(record, role)
                scenes.setdefault(record["scene"], Census(c)).add(record, role)
                logs.setdefault(role["log_token"], Census(c)).add(record, role)
        print(f"Summary unions: {number}/10 parts, {len(frames)} unique keyframes", flush=True)
    if (len(frames) != c["expected_keyframes"] or len(scenes) != c["expected_scenes"]
            or set(scenes) != set(roles["scenes"]) or len(cameras) != 6 * len(frames)):
        raise ValueError("Full trainval keyframe/scene/camera inventory is incomplete")
    # Counts alone cannot establish exact metadata membership; compare the complete set.
    # The sensor archive stage checked every filename and sample against the index.
    report = signed({"complete": True, "identity_sha256": identity["manifest_sha256"],
        "parts_sha256": parts, "unique_keyframes": len(frames), "unique_scenes": len(scenes),
        "rgb_headers_checked": len(cameras), "range_bands_m": c["range_bands_m"],
        "groups": {k: v.result() for k, v in groups.items()},
        "previously_inspected_evaluation_logs": roles["previously_inspected_evaluation_logs"],
        "model_inference_performed": False, "training_performed": False, "splits_changed": False,
        "accuracy_threshold_m": None, "accuracy_pass_fail_enabled": False,
        "phone_app_changed": False, "training_authorized": False, "limitations": LIMITATIONS})
    log_roles = {r["log_token"]: r["role"] for r in roles["scenes"].values()}
    for filename, value in [("coverage_summary.json", report),
        ("coverage_by_scene.json", {k: {**roles["scenes"][k], **v.result()} for k, v in sorted(scenes.items())}),
        ("coverage_by_log.json", {k: {"role": log_roles[k], **v.result()} for k, v in sorted(logs.items())})]:
        common.immutable(output / filename, common.encoded(value))
    return report


def bundle(output):
    check_signed(json.loads((output / "coverage_summary.json").read_text()))
    names = ["run_identity.json", "coverage_summary.json", "coverage_by_scene.json", "coverage_by_log.json"]
    names += [f"parts/part{i:02d}/result.json" for i in range(1, 11)]
    # Deterministic ZIP supports a rerun without replacing a completed artifact.
    import io
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, (2026, 10, 28, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (output / name).read_bytes())
    target = output / "nuscenes_0_6m_coverage_results.zip"
    common.immutable(target, stream.getvalue())
    print(f"Return this small summary ZIP: {target} ({target.stat().st_size / 1e6:.1f} MB); no images or raw points", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bundle"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--keep-local-part", action="store_true")
    args = parser.parse_args()
    c, roles, dataset, output, local, receipts, identity = initialize(args)
    # Same-runtime lock only. Never run two notebooks against one Drive RUN_ID.
    from scripts.download_nuscenes_to_drive import transfer_lock
    with transfer_lock(output, {"name": "coverage_run"}):
        if args.stage == "bundle":
            aggregate(output, c, roles, receipts, identity); bundle(output); return
        database = prepare_support(c, roles, dataset, local, receipts, identity)
        if args.stage == "prepare":
            print("CPU preflight ready. Run the coverage cell next; no GPU is needed.", flush=True); return
        for number in c["parts"]:
            process_part(number, receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"], dataset, output,
                         local, database, c, roles, identity, args.workers, args.keep_local_part)
        with data.connect_index(database) as db:
            expected = {r[0] for r in db.execute("SELECT token FROM samples")}
        found = {t for i in c["parts"] for t in json.loads((output / f"parts/part{i:02d}/result.json").read_text())["sample_tokens"]}
        if found != expected:
            raise ValueError("Processed keyframes differ from the complete metadata inventory")
        aggregate(output, c, roles, receipts, identity)


if __name__ == "__main__":
    main()
