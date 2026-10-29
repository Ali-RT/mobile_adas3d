"""CPU-only frozen V0/V1 sparse-depth benchmark; no models or training.

Read the completed coverage chunks. Freeze GT-only selection before opening
sensor archives. Export original RGB and exact sparse masks into bounded,
immutable Drive shards, one locally verified source archive at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_nuscenes_coverage_0_6m as coverage
from scripts import nuscenes_coverage_data as data
from scripts import parking_edge_common as common
from scripts import parking_near_field_geometry as geometry
from scripts import parking_v0_v1_metrics as metrics

CONFIG = ROOT / "configs/parking_v0_v1_benchmark_v1.json"


def load_protocol(path=CONFIG):
    p = json.loads(Path(path).read_text())
    if (p["classes"] != ["Vehicle", "Pedestrian"] or p["range_bands_m"] != [[0, 1], [1, 2], [0, 2], [2, 4], [4, 6], [0, 6]]
            or p["roles"] != ["dev", "heldout_eval"] or p["accuracy_threshold_m"] is not None
            or p["accuracy_pass_fail_enabled"] or p["training_authorized"] or p["model_inference_authorized"]
            or not 1 <= p["images_per_shard"] <= 64 or not 1 <= p["maximum_images_per_role"] <= 2048):
        raise ValueError("Use the frozen, depth-only dev/held-out benchmark contract")
    return p


def validate_coverage(folder, protocol):
    """Small published ZIP plus exact matching on-Drive JSON, never extraction."""
    path = folder / "nuscenes_0_6m_coverage_results.zip"
    if common.file_sha(path) != protocol["reviewed_coverage_zip_sha256"]:
        raise ValueError("Coverage result ZIP differs from the reviewed complete run")
    expected = {"run_identity.json", "coverage_summary.json", "coverage_by_scene.json", "coverage_by_log.json"}
    expected |= {f"parts/part{i:02d}/result.json" for i in range(1, 11)}
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Coverage summary ZIP inventory/CRC differs")
        for name in sorted(expected):
            raw = archive.read(name)
            if raw != common.retry_read(data.safe_target(folder, name)):
                raise ValueError("Coverage JSON differs from the returned reviewed ZIP")
    identity = coverage.check_signed(json.loads((folder / "run_identity.json").read_text()))
    summary = coverage.check_signed(json.loads((folder / "coverage_summary.json").read_text()))
    if (identity["manifest_sha256"] != protocol["reviewed_coverage_identity_sha256"]
            or summary["manifest_sha256"] != protocol["reviewed_coverage_summary_sha256"]
            or summary["identity_sha256"] != identity["manifest_sha256"] or not summary["complete"]
            or summary["unique_keyframes"] != 34149 or summary["unique_scenes"] != 850
            or identity["config"]["revision"] != protocol["coverage_revision"]):
        raise ValueError("Coverage run identity/inventory differs")
    for name, sha in identity["code_sha256"].items():
        if Path(name).name != name or common.file_sha(ROOT / "scripts" / name) != sha:
            raise ValueError("Reviewed coverage implementation changed; do not mix source versions")
    return identity, summary


def initialize(a):
    protocol = load_protocol(a.config)
    paths = [p.resolve() for p in (a.dataset, a.coverage, a.output, a.local)]
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Source dataset, coverage run, output and local scratch must be separate non-nested paths")
    dataset, cov, output, local = paths
    if str(local).startswith("/content/drive/"):
        raise ValueError("Use temporary Colab disk for local scratch, not Drive")
    for p in (a.dataset, a.coverage, a.output, a.local):
        if p.is_symlink() or p.parent.is_symlink():
            raise ValueError("Preserve symlinked workflow roots")
    cov_id, summary = validate_coverage(cov, protocol)
    cov_config = cov_id["config"]
    roles = data.verified_roles(ROOT / cov_config["frozen_roles_file"], cov_config["expected_scenes"])
    if roles["manifest_sha256"] != cov_id["roles_sha256"]:
        raise ValueError("Frozen scene roles changed")
    manifest, receipts = data.verified_downloads(dataset, cov_config, ROOT / cov_config["download_sources_file"])
    if manifest["manifest_sha256"] != cov_id["download_manifest_sha256"]:
        raise ValueError("Download identity differs from the completed coverage run")
    runtime = {"python": ".".join(platform.python_version_tuple()[:2])}
    for name, expected in cov_config["runtime_packages"].items():
        runtime[name] = importlib.metadata.version(name)
        if runtime[name] != expected:
            raise ValueError(f"Task dependency {name} differs; rerun notebook CPU setup")
    sources = [Path(__file__), Path(metrics.__file__)]
    identity = coverage.signed({"protocol": protocol, "coverage_identity_sha256": cov_id["manifest_sha256"],
        "coverage_summary_sha256": summary["manifest_sha256"], "runtime": runtime,
        "implementation_sha256": {p.name: common.file_sha(p) for p in sources},
        "reviewed_geometry_sha256": cov_id["code_sha256"], "roles_sha256": roles["manifest_sha256"]})
    common.immutable(output / "benchmark_identity.json", common.encoded(identity))
    if local.exists() and any(local.iterdir()) and not (local / "owner.json").exists():
        raise ValueError("Preserve nonempty unowned local scratch")
    common.immutable(local / "owner.json", common.encoded(identity))
    for name in ["metadata", "lidarseg", "archives", "parts"]:
        target = data.safe_target(local, name)
        if target.is_symlink():
            raise ValueError("Preserve symlinked local scratch")
    return protocol, cov_id, roles, receipts, identity, paths


def supported_counts(row, protocol):
    return {cell: sum(obj["surface_points"][cell.split(":")[1]]
                     for obj in row["objects"] if obj["class_name"] == cell.split(":")[0]
                     and obj["surface_points"][cell.split(":")[1]] >= protocol["minimum_points_per_object_camera_cell"])
            for cell in metrics.cell_names(protocol)}


def candidates_from_chunks(folder, cov_id, roles, receipts, protocol):
    seen = set(); candidates = []; record_tokens = set(); digests = []
    forbidden = set(roles["previously_inspected_evaluation_logs"])
    for number in range(1, 11):
        part_folder = data.safe_target(folder, f"parts/part{number:02d}")
        contract = coverage.part_contract(number, receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"], cov_id)
        part = coverage.check_signed(json.loads((part_folder / "result.json").read_text()))
        if part["part_identity"] != contract or part["complete"] is not True:
            raise ValueError("Coverage part identity differs")
        flat = [token for chunk in part["chunks"] for token in chunk["sample_tokens"]]
        if flat != part["sample_tokens"] or len(set(flat)) != len(flat):
            raise ValueError("Coverage part chunk inventory differs")
        digests.append(part["manifest_sha256"])
        for chunk in part["chunks"]:
            for record in coverage.read_chunk(data.safe_target(part_folder, chunk["file"]), contract, chunk["sample_tokens"], roles, chunk["sha256"]):
                if record["sample_token"] in record_tokens:
                    raise ValueError("Coverage keyframe repeats across parts")
                record_tokens.add(record["sample_token"])
                for row in record["camera_rows"]:
                    if row["camera_token"] in seen:
                        raise ValueError("Coverage camera repeats")
                    seen.add(row["camera_token"])
                    role = roles["scenes"][row["scene"]]
                    if row["role"] not in protocol["roles"] or role["log_token"] in forbidden:
                        continue
                    counts = supported_counts(row, protocol)
                    if not any(counts[cell] for cell in protocol["selection_cells"]):
                        continue
                    candidates.append({"camera_token": row["camera_token"], "sample_token": row["sample_token"],
                        "scene": row["scene"], "role": row["role"], "recording_log": row["recording_log"],
                        "source_part": number, "coverage_row_sha256": common.identity(row),
                        "coverage_row": row, "supported_counts": counts})
        print(f"GT-only candidate scan {number}/10: {len(candidates)} eligible dev/held-out views", flush=True)
    if len(record_tokens) != cov_id["config"]["expected_keyframes"]:
        raise ValueError("Coverage keyframe inventory is incomplete")
    return candidates, digests


def select_views(rows, protocol):
    """Preserve rare views; fair deterministic fill without model predictions."""
    rank = lambda r: (common.sha((protocol["selection_seed"] + r["camera_token"]).encode()), r["camera_token"])
    ranked = sorted(rows, key=rank)
    if len({r["camera_token"] for r in ranked}) != len(ranked) or len({r["role"] for r in ranked}) > 1:
        raise ValueError("Selection accepts distinct cameras from one original role")
    maximum = protocol["maximum_images_per_role"]
    selected = [r for r in ranked if any(r["supported_counts"][cell] > 0 for cell in protocol["rare_retention_cells"])]
    if len(selected) > maximum:
        raise ValueError("Rare-view retention exceeds the frozen budget; review a new protocol rather than silently dropping rare inputs")
    used = {r["camera_token"] for r in selected}; frames = {r["sample_token"] for r in selected}
    log_counts = Counter(r["recording_log"] for r in selected)
    while len(selected) < maximum:
        added = False
        for cell in protocol["selection_cells"]:
            eligible = [r for r in ranked if r["camera_token"] not in used and r["sample_token"] not in frames and r["supported_counts"][cell] > 0]
            if not eligible:
                continue
            row = min(eligible, key=lambda r: (log_counts[r["recording_log"]], rank(r)))
            selected.append(row); used.add(row["camera_token"]); frames.add(row["sample_token"])
            log_counts[row["recording_log"]] += 1; added = True
            if len(selected) == maximum:
                break
        if not added:
            break
    return sorted(selected, key=lambda r: (r["source_part"], r["camera_token"]))


def selection_inventory(rows, protocol):
    result = {}
    for role in protocol["roles"]:
        chosen = [r for r in rows if r["role"] == role]; cells = {}
        for cell in metrics.cell_names(protocol):
            cls, band = cell.split(":"); observations = set(); tracks = set(); logs = set()
            for row in chosen:
                for obj in row["coverage_row"]["objects"]:
                    if obj["class_name"] == cls and obj["surface_points"][band] >= protocol["minimum_points_per_object_camera_cell"]:
                        observations.add(obj["annotation_token"]); tracks.add(obj["instance_token"]); logs.add(row["recording_log"])
            cells[cell] = {"images": sum(r["supported_counts"][cell] > 0 for r in chosen),
                "eligible_reference_points": sum(r["supported_counts"][cell] for r in chosen),
                "annotation_observations": len(observations), "instance_tracks": len(tracks), "recording_logs": len(logs)}
        result[role] = {"images": len(chosen), "unique_keyframes": len({r["sample_token"] for r in chosen}),
            "scenes": len({r["scene"] for r in chosen}), "recording_logs": len({r["recording_log"] for r in chosen}), "cells": cells}
    return result


def plan(cov, output, cov_id, roles, receipts, protocol, identity):
    path = output / "selection_manifest.json"
    if path.exists():
        value = coverage.check_signed(json.loads(common.retry_read(path)))
        if value["benchmark_identity_sha256"] != identity["manifest_sha256"]:
            raise ValueError("Selection belongs to a different benchmark")
        validate_selection(value, protocol, roles)
        return value
    candidates, digests = candidates_from_chunks(cov, cov_id, roles, receipts, protocol)
    selected = sum((select_views([r for r in candidates if r["role"] == role], protocol) for role in protocol["roles"]), [])
    value = coverage.signed({"benchmark_identity_sha256": identity["manifest_sha256"],
        "coverage_parts_sha256": digests, "samples": selected,
        "available_pool": selection_inventory(candidates, protocol), "selected_pool": selection_inventory(selected, protocol),
        "selection_uses_predictions": False, "training_authorized": False, "model_inference_performed": False})
    validate_selection(value, protocol, roles)
    common.immutable(path, common.encoded(value))
    print(json.dumps(value["selected_pool"], indent=2), flush=True)
    return value


def validate_selection(value, protocol, roles):
    samples = value["samples"]; seen = set(); logs_by_role = defaultdict(set)
    for row in samples:
        if row["camera_token"] in seen or row["role"] not in protocol["roles"]:
            raise ValueError("Repeated camera or forbidden benchmark role")
        seen.add(row["camera_token"])
        frozen = roles["scenes"][row["scene"]]
        if (row["role"] != frozen["role"] or row["recording_log"] != frozen["log_token"]
                or any(row[k] != row["coverage_row"][k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log"])
                or row["recording_log"] in roles["previously_inspected_evaluation_logs"]
                or common.identity(row["coverage_row"]) != row["coverage_row_sha256"]
                or row["supported_counts"] != supported_counts(row["coverage_row"], protocol)):
            raise ValueError("Frozen selection source or scene/log role differs")
        logs_by_role[row["role"]].add(row["recording_log"])
    if logs_by_role["dev"] & logs_by_role["heldout_eval"]:
        raise ValueError("Recording-log leakage")
    for role in protocol["roles"]:
        chosen = [r for r in samples if r["role"] == role]
        if not chosen or len(chosen) > protocol["maximum_images_per_role"]:
            raise ValueError("Missing role or exceeded image budget")
    if selection_inventory(samples, protocol) != value["selected_pool"]:
        raise ValueError("Selection counts changed")


def stage_selected(archive_path, receipt, root, database, rows, cov_config, contract):
    """Stream one local archive; extract only selected RGB and referenced scans."""
    from PIL import Image
    common.immutable(root / "owner.json", common.encoded(contract))
    wanted = {}; sensors = {}; scans = {}
    with data.connect_index(database) as db:
        for row in rows:
            camera = json.loads(db.execute("SELECT body FROM sensor_data WHERE token=?", (row["camera_token"],)).fetchone()[0])
            lidar = json.loads(db.execute("SELECT body FROM sensor_data WHERE sample=? AND channel='LIDAR_TOP'", (row["sample_token"],)).fetchone()[0])
            if camera["sample_token"] != row["sample_token"]:
                raise ValueError("Selected camera/keyframe metadata differs")
            sensors[camera["token"]] = camera; scans[row["sample_token"]] = lidar
            for sensor in [camera, lidar]:
                wanted[sensor["filename"]] = sensor
    marker = root / "selected_staging.json"
    if marker.exists():
        saved = coverage.check_signed(json.loads(marker.read_text()))
        if saved["contract"] != contract or set(saved["files"]) != set(wanted):
            raise ValueError("Local selection staging differs")
        for name, info in saved["files"].items():
            p = data.safe_target(root / "data", name)
            if p.stat().st_size != info["size_bytes"] or common.file_sha(p) != info["sha256"]:
                raise ValueError("Preserve changed staged benchmark source")
        return sensors, scans, saved["files"]
    seen = set(); files = {}
    data.require_space(root, sum(20 * 1024**2 if s["channel"] != "LIDAR_TOP" else 4 * 1024**2 for s in wanted.values()), int(cov_config["local_disk_reserve_gb"] * 1e9))
    with data.bounded_tar(archive_path, receipt) as archive:
        for member in archive:
            name = data.member_info(member, seen, cov_config["maximum_archive_member_bytes"])
            if name not in wanted or not member.isfile():
                continue
            sensor = wanted[name]
            if sensor["channel"] != "LIDAR_TOP" and member.size > 20 * 1024**2:
                raise ValueError("Oversized selected RGB file")
            destination = data.safe_target(root / "data", name)
            info = data.extract_file(archive, member, destination)
            if sensor["channel"] == "LIDAR_TOP":
                if info["size_bytes"] == 0 or info["size_bytes"] % 20:
                    raise ValueError("Malformed selected LiDAR scan")
            else:
                with Image.open(destination) as image:
                    if image.size != (sensor["width"], sensor["height"]) or image.mode != "RGB":
                        raise ValueError("Selected RGB header differs")
                    image.verify()
            files[name] = info
            if len(files) % 64 == 0:
                print(f"Selected source staging: {len(files)}/{len(wanted)} files", flush=True)
    if set(files) != set(wanted):
        raise ValueError("Source archive lacks a frozen selected RGB/LiDAR member")
    common.immutable(marker, common.encoded(coverage.signed({"contract": contract, "files": files})))
    return sensors, scans, files


def deterministic_zip(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.endswith(".jpg") else zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    return stream.getvalue()


def encode_npz(points):
    import numpy as np
    members = {}
    for name, array in points.items():
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
        members[name + ".npy"] = stream.getvalue()
    return deterministic_zip(members)


def reference_for_view(row, camera, lidar, root, labels, database, cov_config, protocol):
    import numpy as np
    calibration, semantic = coverage.index_constants(database)
    with data.connect_index(database) as db:
        ls = calibration[lidar["calibrated_sensor_token"]]; cs = calibration[camera["calibrated_sensor_token"]]
        lp = json.loads(db.execute("SELECT body FROM poses WHERE token=?", (lidar["ego_pose_token"],)).fetchone()[0])
        cp = json.loads(db.execute("SELECT body FROM poses WHERE token=?", (camera["ego_pose_token"],)).fetchone()[0])
        annotations = [json.loads(r[0]) for r in db.execute("SELECT body FROM annotations WHERE sample=? ORDER BY token", (row["sample_token"],))]
        label = db.execute("SELECT filename FROM labels WHERE token=?", (lidar["token"],)).fetchone()[0]
    if abs(camera["timestamp"] - lidar["timestamp"]) / 1000 > cov_config["maximum_camera_lidar_gap_ms"]:
        raise ValueError("Selected reference exceeds reviewed sensor timing gate")
    raw = np.fromfile(data.safe_target(root / "data", lidar["filename"]), dtype="<f4")
    sem = np.fromfile(data.safe_target(labels, label), dtype=np.uint8)
    if not len(raw) or raw.size % 5 or raw.size // 5 != len(sem) or sem.max() > 31 or not np.isfinite(raw).all():
        raise ValueError("Malformed selected LiDAR/semantic arrays")
    uv, z, ids, world = geometry.project_positive(raw.reshape(-1, 5)[:, :3], ls, lp, cp, cs, camera["width"], camera["height"])
    owner = geometry.associate_objects(world, sem, annotations)
    near = geometry.sparse_near_raster(uv, z, sem[ids], ids, camera["width"], camera["height"], cov_config["boundary_guard"])
    mapping = {cls: [semantic[name] for name in names] for cls, names in cov_config["classes"].items()}
    actual = {f"{cls}:{geometry.band_name(lo, hi)}": int((near["conservative"] & np.isin(near["semantic"], indices)
        & geometry.band_mask(near["z"], lo, hi)).sum()) for cls, indices in mapping.items() for lo, hi in protocol["range_bands_m"]}
    if actual != row["coverage_row"]["point_counts"]:
        raise ValueError("Selected raw reprojection does not reproduce reviewed coverage counts")
    ownership = owner[near["source_indices"]]
    expected_objects = {o["annotation_token"]: o for o in row["coverage_row"]["objects"]}
    for index, ann in enumerate(annotations):
        surface = {geometry.band_name(lo, hi): int((near["conservative"] & (ownership == index) & geometry.band_mask(near["z"], lo, hi)).sum())
                   for lo, hi in protocol["range_bands_m"]}
        if surface["0-6m"] and (ann["token"] not in expected_objects or surface != expected_objects[ann["token"]]["surface_points"]):
            raise ValueError("Selected object surface support differs from reviewed coverage")
    selected = near["conservative"] & (ownership >= 0)
    points = {"xy": near["xy"][selected].astype(np.int32), "z": near["z"][selected].astype(np.float32),
              "semantic": near["semantic"][selected].astype(np.uint8), "object_index": ownership[selected].astype(np.int32)}
    objects = [{k: ann[k] for k in ["token", "instance_token", "class_name", "source_class", "semantic_index", "visibility_token"]} for ann in annotations]
    objects = [{"annotation_token": obj.pop("token"), **obj} for obj in objects]
    points["cell_masks"] = metrics.make_masks(points, objects, protocol)
    sample = {k: row[k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log", "source_part", "coverage_row_sha256"]}
    sample.update(width=camera["width"], height=camera["height"], intrinsics=cs["camera_intrinsic"],
        timestamp_us=camera["timestamp"], camera_lidar_gap_ms=abs(camera["timestamp"] - lidar["timestamp"]) / 1000,
        objects=objects, cell_names=metrics.cell_names(protocol), eligible_reference_counts=row["supported_counts"],
        image_member=f"images/{camera['token']}.jpg", ground_truth_member=f"ground_truth/{camera['token']}.npz")
    metrics.validate_gt(points, sample, protocol)
    if {cell: int(mask.sum()) for cell, mask in zip(sample["cell_names"], points["cell_masks"])} != row["supported_counts"]:
        raise ValueError("Frozen eligible mask counts differ from selected support")
    return sample, encode_npz(points)


def verify_shard(path, rows, identity, protocol, expected_sha=None):
    import numpy as np
    raw = common.retry_read(path)
    if expected_sha is not None and common.sha(raw) != expected_sha:
        raise ValueError("Completed benchmark shard bytes changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        value = coverage.check_signed(json.loads(archive.read("shard_manifest.json")))
        if value["benchmark_identity_sha256"] != identity["manifest_sha256"] or [s["camera_token"] for s in value["samples"]] != [r["camera_token"] for r in rows]:
            raise ValueError("Shard source/run/selection identity differs")
        expected = {"shard_manifest.json"}
        for sample, row in zip(value["samples"], rows):
            if any(sample[k] != row[k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log", "source_part", "coverage_row_sha256"]):
                raise ValueError("Shard changed selected source/split")
            for name, sha in [(sample["image_member"], sample["image_sha256"]), (sample["ground_truth_member"], sample["ground_truth_sha256"])]:
                data.safe_name(name); expected.add(name)
                if common.sha(archive.read(name)) != sha:
                    raise ValueError("Shard RGB/GT bytes changed")
            with np.load(io.BytesIO(archive.read(sample["ground_truth_member"])), allow_pickle=False) as values:
                points = {k: values[k] for k in values.files}
            metrics.validate_gt(points, sample, protocol)
            if {cell: int(mask.sum()) for cell, mask in zip(metrics.cell_names(protocol), points["cell_masks"])} != row["supported_counts"]:
                raise ValueError("Shard mask inventory changed")
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Unexpected/missing/duplicate benchmark shard members")
    return value["samples"]


def shard_batches(selection, number, protocol):
    for role in protocol["roles"]:
        rows = [r for r in selection["samples"] if r["source_part"] == number and r["role"] == role]
        for offset in range(0, len(rows), protocol["images_per_shard"]):
            yield f"shards/{role}/part{number:02d}_{offset // protocol['images_per_shard']:03d}.zip", rows[offset:offset + protocol["images_per_shard"]]


def prepare_part(number, selection, dataset, output, local, database, receipts, cov_config, protocol, identity, workers):
    batches = list(shard_batches(selection, number, protocol))
    if not batches:
        return []
    missing = [(name, rows) for name, rows in batches if not data.safe_target(output, name).exists()]
    contract = {"part": number, "archive_sha256": receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"]["sha256"],
                "benchmark_identity_sha256": identity["manifest_sha256"], "selection_sha256": selection["manifest_sha256"]}
    if missing:
        receipt = receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"]
        archive = data.copy_archive(dataset, local, receipt, int(cov_config["local_disk_reserve_gb"] * 1e9))
        root = data.safe_target(local, f"parts/part{number:02d}")
        # Stage all this part's selected rows so the staging contract survives interrupted shard publication.
        rows = [r for _, batch in batches for r in batch]
        sensors, scans, files = stage_selected(archive, receipt, root, database, rows, cov_config, contract)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, batch in missing:
                generated = list(pool.map(lambda row: reference_for_view(row, sensors[row["camera_token"]], scans[row["sample_token"]],
                    root, local / "lidarseg", database, cov_config, protocol), batch))
                members = {}; samples = []
                for row, (sample, gt_raw) in zip(batch, generated):
                    camera = sensors[row["camera_token"]]; image = data.safe_target(root / "data", camera["filename"]).read_bytes()
                    sample.update(image_sha256=common.sha(image), ground_truth_sha256=common.sha(gt_raw))
                    if sample["image_sha256"] != files[camera["filename"]]["sha256"]:
                        raise ValueError("Selected RGB changed during preparation")
                    members[sample["image_member"]] = image; members[sample["ground_truth_member"]] = gt_raw; samples.append(sample)
                members["shard_manifest.json"] = common.encoded(coverage.signed({"benchmark_identity_sha256": identity["manifest_sha256"], "samples": samples}))
                target = data.safe_target(output, name); raw = deterministic_zip(members)
                common.immutable(target, raw); verify_shard(target, batch, identity, protocol, common.sha(raw))
                print(f"Durable benchmark shard: {name}, {len(batch)} images", flush=True)
    records = []
    for name, rows in batches:
        path = data.safe_target(output, name); samples = verify_shard(path, rows, identity, protocol)
        records.append({"file": name, "sha256": common.file_sha(path), "size_bytes": path.stat().st_size, "samples": samples})
    result = coverage.signed({"contract": contract, "shards": records})
    common.immutable(output / f"parts/part{number:02d}.json", common.encoded(result))
    if missing:
        # Same ownership contract as the coverage cleanup, plus this benchmark's source/selection binding.
        data.release_part(local, root, archive, contract)
    else:
        print(f"Part {number:02d}: verified complete shards; no source archive restaged", flush=True)
    return records


def finalize(selection, output, protocol, identity):
    shards = []; samples = []; selected = {r["camera_token"]: r for r in selection["samples"]}
    for number in range(1, 11):
        expected_batches = list(shard_batches(selection, number, protocol))
        if not expected_batches:
            continue
        value = coverage.check_signed(json.loads((output / f"parts/part{number:02d}.json").read_text()))
        if value["contract"]["benchmark_identity_sha256"] != identity["manifest_sha256"] or value["contract"]["selection_sha256"] != selection["manifest_sha256"]:
            raise ValueError("Part completion receipt belongs to another benchmark")
        if [s["file"] for s in value["shards"]] != [name for name, _ in expected_batches]:
            raise ValueError("Part shard inventory differs from frozen plan")
        for shard, (name, rows) in zip(value["shards"], expected_batches):
            path = data.safe_target(output, name)
            actual = verify_shard(path, rows, identity, protocol, shard["sha256"])
            if actual != shard["samples"] or path.stat().st_size != shard["size_bytes"]:
                raise ValueError("Part receipt sample inventory differs from shard")
            shards.append({k: shard[k] for k in ["file", "sha256", "size_bytes"]})
            samples.extend({**sample, "shard_file": name, "shard_sha256": shard["sha256"]} for sample in actual)
    tokens = [s["camera_token"] for s in samples]
    if len(set(tokens)) != len(tokens) or set(tokens) != set(selected):
        raise ValueError("Benchmark package misses/repeats frozen selected views")
    order = {r["camera_token"]: index for index, r in enumerate(selection["samples"])}
    samples.sort(key=lambda s: order[s["camera_token"]])
    manifest = coverage.signed({"complete": True, "revision": protocol["revision"],
        "benchmark_identity_sha256": identity["manifest_sha256"], "selection_sha256": selection["manifest_sha256"],
        "metric_protocol": protocol, "samples": samples, "shards": shards,
        "selected_pool": selection["selected_pool"], "original_roles_changed": False,
        "training_images_included": False, "model_artifacts_bound": False,
        "model_inference_performed": False, "training_performed": False, "phone_app_changed": False})
    common.immutable(output / "benchmark_manifest.json", common.encoded(manifest))
    warnings = []
    for role in protocol["roles"]:
        for cell in protocol["exploratory_cells"] + protocol["secondary_cells"]:
            support = selection["selected_pool"][role]["cells"][cell]
            warnings.append({"role": role, "cell": cell, **support,
                "interpretation": "Descriptive support only. Small/correlated counts do not qualify product performance; 0-1m remains exploratory."})
    report = coverage.signed({"complete": True, "benchmark_manifest_sha256": manifest["manifest_sha256"],
        "benchmark_identity_sha256": identity["manifest_sha256"], "images": len(samples), "shards": len(shards),
        "shard_bytes": sum(s["size_bytes"] for s in shards), "available_pool": selection["available_pool"],
        "selected_pool": selection["selected_pool"], "scarcity_notes": warnings,
        "accuracy_threshold_m": None, "accuracy_pass_fail_enabled": False,
        "model_artifacts_bound": False, "model_inference_performed": False, "training_performed": False,
        "next_step": "Review package support, then register the exact unchanged V0 model and preprocessing identity before running inference. Register V1 separately after authorized training; evaluate both against this unchanged manifest.",
        "limitations": coverage.LIMITATIONS + [protocol["scope"], protocol["confidence"],
            "GT-enriched bounded selection is not the full-data prevalence distribution.",
            "Multiple mandatory rare views can share a keyframe. Scoring averages views inside annotation observations and reports track/log support.",
            "The package freezes data/metrics, not V0/V1 weights. It provides no measured model scores yet."]})
    common.immutable(output / "preparation_report.json", common.encoded(report))
    template = {"benchmark_manifest_sha256": manifest["manifest_sha256"],
        "V0": {"artifact_sha256": None, "architecture_sha256": None, "preprocessing_sha256": None, "output_sampling_sha256": None, "role": "unchanged_baseline"},
        "V1": {"artifact_sha256": None, "architecture_sha256": None, "preprocessing_sha256": None, "output_sampling_sha256": None, "role": "later_trained_candidate"},
        "status": "UNBOUND: no model evaluation or training was performed by this preparation workflow"}
    common.immutable(output / "model_registration_template.json", common.encoded(template))
    return report


def bundle(output, protocol):
    report = coverage.check_signed(json.loads((output / "preparation_report.json").read_text()))
    if not report["complete"]:
        raise ValueError("Preparation has not completed")
    names = ["benchmark_identity.json", "selection_manifest.json", "benchmark_manifest.json", "preparation_report.json", "model_registration_template.json"]
    names += sorted(p.relative_to(output).as_posix() for p in (output / "parts").glob("part??.json"))
    members = {name: data.safe_target(output, name).read_bytes() for name in names}
    members["frozen_metric_protocol.json"] = common.encoded(protocol)
    target = output / "parking_v0_v1_benchmark_results.zip"
    common.immutable(target, deterministic_zip(members))
    print(f"Return this summary ZIP: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("RGB/GT shards stay on Drive. No V0/V1 inference or training has started.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "prepare", "bundle"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    protocol, cov_id, roles, receipts, identity, paths = initialize(args)
    dataset, cov, output, local = paths
    from scripts.download_nuscenes_to_drive import transfer_lock
    with transfer_lock(output, {"name": "v0_v1_benchmark"}):
        selection = plan(cov, output, cov_id, roles, receipts, protocol, identity)
        if args.stage == "plan":
            print("Selection frozen. Next: CPU-only RGB/sparse-GT preparation.", flush=True); return
        if args.stage == "prepare":
            database = coverage.prepare_support(cov_id["config"], roles, dataset, local, receipts, identity)
            for number in range(1, 11):
                prepare_part(number, selection, dataset, output, local, database, receipts, cov_id["config"], protocol, identity, args.workers)
        report = finalize(selection, output, protocol, identity)
        print(json.dumps({k: report[k] for k in ["complete", "images", "shards", "shard_bytes", "model_artifacts_bound", "model_inference_performed", "training_performed"]}, indent=2), flush=True)
        if args.stage == "bundle":
            bundle(output, protocol)


if __name__ == "__main__":
    main()
