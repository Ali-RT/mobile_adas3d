"""Scene-separated sparse GT inputs for decoder-only shared-depth training.

CPU preparation is separate from GPU use. Images/GT travel in one reusable zip;
restore verifies hashes and uses four copy workers. Existing validation caches
and source data are read-only. This is a mini-data pilot, not dense metrology GT.
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
from scripts import prepare_parking_lidar_depth as gt

CONFIG = ROOT / "configs/parking_shared_depth_pilot_v1.json"


def scene_split(train_scenes, test_scenes, config):
    if len(set(train_scenes)) != 8 or len(set(test_scenes)) != 2 or set(train_scenes) & set(test_scenes):
        raise ValueError("Expected disjoint official eight/two mini scene split")
    ranked = sorted(train_scenes, key=lambda s: common.sha(f"{config['seed']}:{s}".encode()))
    return {"train": sorted(ranked[config["dev_scenes"]:]),
            "dev": sorted(ranked[:config["dev_scenes"]]), "test": sorted(test_scenes)}


def validate_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != common.identity(body) or not manifest.get("complete"):
        raise ValueError("Training input manifest changed")
    roles = manifest["scenes"]
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if set(roles[a]) & set(roles[b]):
            raise ValueError("Scene leakage between data roles")
    samples = manifest["samples"]
    if len({s["camera_token"] for s in samples}) != len(samples):
        raise ValueError("Repeated camera sample")
    grouped = {role: {s["sample_token"] for s in samples if s["role"] == role} for role in roles}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if grouped[a] & grouped[b]:
            raise ValueError("Keyframe leakage between data roles")
    expected = {}
    for sample in samples:
        if sample["role"] not in roles or sample["scene"] not in roles[sample["role"]]:
            raise ValueError("Sample scene/role mismatch")
        for name in ["image", "ground_truth"]:
            member, digest = sample[f"{name}_file"], sample[f"{name}_sha256"]
            safe_member(member)
            if member in expected and expected[member] != digest:
                raise ValueError("Conflicting data member")
            expected[member] = digest
    if expected != manifest["files"]:
        raise ValueError("Unexpected training archive inventory")
    return manifest


def safe_member(name):
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("Unsafe training archive member")
    return p


def read_manifest(root):
    return validate_manifest(json.loads((Path(root) / "input_manifest.json").read_text()))


def point_grid(xy, width, height, metadata):
    """Original integer pixel centres -> align_corners=False normalized grid.

    Resize uses OpenCV's half-pixel convention; padding is never labelled. This
    samples the coarse prediction at GT locations, not resized/densified GT.
    """
    import numpy as np
    xy = np.asarray(xy)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
            or (xy < 0).any() or (xy[:, 0] >= width).any() or (xy[:, 1] >= height).any()):
        raise ValueError("Invalid original-image GT coordinates")
    nw, nh = metadata["resized_size"]; left, top = metadata["padding_left_top"]
    iw, ih = metadata["input_size"]
    centres = (xy.astype(np.float64) + .5) * [nw / width, nh / height] + [left, top]
    return (2 * centres / [iw, ih] - 1).astype(np.float32)


def coverage(samples, points_by_token, mapping):
    import numpy as np
    result = {}
    for role in ["train", "dev", "test"]:
        selected = [s for s in samples if s["role"] == role]
        cells = {}
        for cls, indices in mapping.items():
            for low, high in [(2, 4), (4, 6), (2, 6)]:
                counts = []
                for sample in selected:
                    p = points_by_token[sample["camera_token"]]
                    mask = p["conservative"] & np.isin(p["semantic"], indices) & (p["z"] >= low)
                    mask &= p["z"] <= high if high == 6 else p["z"] < high
                    counts.append(int(mask.sum()))
                cells[f"{cls}:{low}-{high}m"] = {"points": sum(counts), "images": sum(n > 0 for n in counts)}
        result[role] = {"images": len(selected), "scene_counts": dict(Counter(s["scene"] for s in selected)), "cells": cells}
    return result


def publish_archive(output, manifest):
    destination = output / "shared_depth_inputs.zip"
    if destination.exists():
        verify_archive(destination, manifest)
        return
    # ZipFile streams files instead of building a multi-GB BytesIO object.
    temporary = output / ".shared_depth_inputs.partial.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(manifest["files"]):
            info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (output / member).read_bytes())
        archive.writestr("input_manifest.json", common.encoded(manifest))
    temporary.replace(destination)


def verify_archive(path, manifest=None):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for name in names:
            safe_member(name)
        saved = validate_manifest(json.loads(archive.read("input_manifest.json")))
        if manifest is not None and saved != manifest:
            raise ValueError("Archive/manifest differs")
        if set(names) != set(saved["files"]) | {"input_manifest.json"}:
            raise ValueError("Unexpected archive files")
        for member, digest in saved["files"].items():
            if common.sha(archive.read(member)) != digest:
                raise ValueError("Training archive member bytes changed")
    return saved


def restore(archive, output):
    manifest = verify_archive(archive)
    def one(item):
        name, digest = item
        target = output / safe_member(name)
        if target.is_symlink() or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError("Training staging path escapes root")
        if target.is_file():
            if common.file_sha(target) != digest:
                raise ValueError("Preserve changed staged data")
        else:
            with zipfile.ZipFile(archive) as source:
                raw = source.read(name)
            if common.sha(raw) != digest:
                raise ValueError("Training input bytes changed")
            common.immutable(target, raw)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(one, manifest["files"].items()), 1):
            if i % 128 == 0: print(f"Staged/verified {i}/{len(manifest['files'])} members", flush=True)
    common.immutable(output / "input_manifest.json", common.encoded(manifest))
    print("Reusable input archive staged; no GPU used", flush=True)


def prepare(a):
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_train, mini_val
    from nuscenes.utils.geometry_utils import points_in_box
    from scripts.evaluate_parking_lidar_depth import read_manifest as read_test, load_points
    config = json.loads(a.config.read_text()); geometry = json.loads(gt.CONFIG.read_text())
    root, output = a.dataset.resolve(), a.output.resolve()
    if output.is_relative_to(root) or output.is_relative_to(a.test_inputs.resolve()) or a.output.is_symlink():
        raise ValueError("Keep new preparation separate from read-only data/test cache")
    test = read_test(a.test_inputs)
    if Path(test["dataset_root"]).resolve() != root:
        raise ValueError("Use the same read-only mini dataset as the reviewed held-out screen")
    split = scene_split(mini_train, mini_val, config)
    identity = {"config": config, "geometry_config": geometry, "source_dataset": str(root),
                "heldout_manifest_sha256": common.file_sha(a.test_inputs / "input_manifest.json"),
                "reviewed_gt_prepare_identity_sha256": test["prepare_identity_sha256"],
                "implementation_sha256": {p.name: common.file_sha(p) for p in [Path(__file__), ROOT / "scripts/prepare_parking_lidar_depth.py"]}}
    common.immutable(output / "prepare_identity.json", common.encoded(identity))
    if (output / "input_manifest.json").exists():
        manifest = read_manifest(output)
        if manifest["prepare_identity_sha256"] != common.identity(identity):
            raise ValueError("Preserve different prepared run")
        for member, digest in manifest["files"].items():
            if common.file_sha(output / member) != digest: raise ValueError("Prepared bytes changed")
        publish_archive(output, manifest); print("Prepared cache verified", flush=True); return
    nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    mapping = test["mapping"]; rows = []; points_cache = {}
    samples = sorted(nusc.sample, key=lambda s: s["token"])
    for i, sample in enumerate(samples):
        scene = nusc.get("scene", sample["scene_token"])["name"]
        role = next((k for k in ["train", "dev"] if scene in split[k]), None)
        if role is None: continue
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ls = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"]); lp = nusc.get("ego_pose", lidar["ego_pose_token"])
        label = nusc.get("lidarseg", lidar["token"])
        raw = np.fromfile(gt.safe_file(root, lidar["filename"]), dtype="<f4")
        if raw.size % 5: raise ValueError("Malformed LiDAR")
        xyz = raw.reshape(-1, 5)[:, :3].copy(); sem = np.fromfile(gt.safe_file(root, label["filename"]), dtype=np.uint8)
        if len(xyz) != len(sem) or not len(xyz) or sem.max() > 31: raise ValueError("LiDAR semantics mismatch")
        world = gt.rotation(lp["rotation"]) @ (gt.rotation(ls["rotation"]) @ xyz.T + np.asarray(ls["translation"])[:, None]) + np.asarray(lp["translation"])[:, None]
        static = np.zeros(len(xyz), dtype=bool)
        for token in sample["anns"]:
            ann = nusc.get("sample_annotation", token)
            attrs = {nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]}
            cls = next((k for k, v in geometry["classes"].items() if ann["category_name"] in v), None)
            if cls and attrs.intersection(geometry["static_attributes"]):
                static |= points_in_box(nusc.get_box(token), world) & np.isin(sem, mapping[cls])
        for channel in geometry["camera_channels"]:
            camera = nusc.get("sample_data", sample["data"][channel])
            if abs(camera["timestamp"] - lidar["timestamp"]) / 1000 > geometry["max_camera_lidar_timestamp_difference_ms"]: continue
            cs = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"]); cp = nusc.get("ego_pose", camera["ego_pose_token"])
            uv, z, indices, _ = gt.camera_projection(xyz, ls, lp, cp, cs, camera["width"], camera["height"])
            points = gt.rasterize(uv, z, sem[indices], static[indices], camera["width"], camera["height"], geometry)
            points.pop("raster_points_before_range_semantic_filter")
            points["static"] &= np.linalg.norm(np.asarray(cp["translation"]) - lp["translation"]) <= geometry["max_ego_translation_between_sensor_poses_for_static_subset_m"]
            if int(points["conservative"].sum()) < config["minimum_points_per_image"]: continue
            row = dict(camera_token=camera["token"], sample_token=sample["token"], scene=scene, role=role,
                       source_image_file=camera["filename"], width=camera["width"], height=camera["height"], intrinsics=cs["camera_intrinsic"])
            rows.append(row); points_cache[camera["token"]] = points
        if (i + 1) % 20 == 0: print(f"Indexed {i+1}/{len(samples)} mini keyframes", flush=True)
    selected = []
    for role in ["train", "dev"]:
        ranked = sorted([r for r in rows if r["role"] == role], key=lambda r: common.sha(f"{config['seed']}:{r['camera_token']}".encode()))
        selected += ranked[:config[f"maximum_{role}_images"]]
        if len(ranked) < config[f"maximum_{role}_images"]: raise ValueError(f"Insufficient {role} images")
    for row in test["samples"]:
        selected.append({k: row[k] for k in ["camera_token", "sample_token", "scene", "width", "height", "intrinsics"]} | {"role": "test", "source_image_file": row["image_file"]})
        points_cache[row["camera_token"]] = load_points(a.test_inputs, row)
    files = {}
    def stage(row):
        row = dict(row); token = row["camera_token"]
        source = gt.safe_file(root, row.pop("source_image_file")); image = f"images/{token}{source.suffix}"
        raw_image = common.retry_read(source); common.immutable(output / image, raw_image)
        buffer = io.BytesIO(); np.savez_compressed(buffer, **points_cache[token]); raw_points = buffer.getvalue()
        member = f"ground_truth/{token}.npz"; common.immutable(output / member, raw_points)
        row.update(image_file=image, image_sha256=common.sha(raw_image), ground_truth_file=member, ground_truth_sha256=common.sha(raw_points))
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        staged = list(pool.map(stage, selected))
    for row in staged:
        for name in ["image", "ground_truth"]: files[row[f"{name}_file"]] = row[f"{name}_sha256"]
    manifest = dict(schema_version=1, complete=True, revision=config["revision"], config=config, geometry_config=geometry,
                    prepare_identity_sha256=common.identity(identity), scenes=split, mapping=mapping, samples=staged,
                    coverage=coverage(staged, points_cache, mapping), files=files,
                    heldout_manifest_sha256=identity["heldout_manifest_sha256"],
                    GT_scope="Sparse camera-axis Z from single LiDAR sweep; cross-sensor occlusion/timing uncertainty remains; not 10cm metrology GT",
                    pretrained_dataset_overlap="Unknown", training_performed=False)
    manifest["manifest_sha256"] = common.identity(manifest); validate_manifest(manifest)
    common.immutable(output / "input_manifest.json", common.encoded(manifest)); publish_archive(output, manifest)
    print(json.dumps(manifest["coverage"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="stage", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--test-inputs", type=Path, required=True)
    q.add_argument("--config", type=Path, default=CONFIG); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("restore"); q.add_argument("--archive", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); prepare(a) if a.stage == "prepare" else restore(a.archive, a.output)


if __name__ == "__main__": main()
