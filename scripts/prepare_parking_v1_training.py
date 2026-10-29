"""CPU-only, train-log-only 0-6m V1 inputs with archive-first resumability.

No model dependencies. Frozen benchmark and source helpers remain unchanged.
One-time coverage metadata is cached in a ZIP. Selected RGB/near GT/broad GT
travel in one immutable archive per source part, not loose Drive files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_nuscenes_coverage_0_6m as coverage
from scripts import nuscenes_coverage_data as data
from scripts import parking_edge_common as common
from scripts import parking_near_field_geometry as geometry
from scripts import parking_v0_v1_metrics as metrics
from scripts import prepare_parking_v0_v1_benchmark as reference

CONFIG = ROOT / "configs/parking_v1_training_inputs_v1.json"
SUMMARY_NAME = "nuscenes_0_6m_coverage_results.zip"
DOWNLOAD_NAME = "download_manifest_keyframes_all.json"


def configuration(path=CONFIG):
    c = json.loads(path.read_text())
    if (c["candidate_role"] != "train" or c["maximum_train_images"] != 4096 or c["images_per_shard"] != 32
            or c["selection_cells"] != ["Vehicle:0-2m", "Pedestrian:0-2m", "Vehicle:2-4m", "Pedestrian:2-4m", "Vehicle:4-6m", "Pedestrian:4-6m"]
            or c["training_authorized"] or c["model_inference_authorized"] or c["heldout_evaluation_authorized"]
            or c["phone_app_changed"] or c["accuracy_threshold_m"] is not None
            or common.file_sha(ROOT / "configs/parking_v1_near_field_proposal_v1.json") != c["proposal_config_sha256"]):
        raise ValueError("Use the reviewed preparation-only V1 input contract")
    return c


def read_once(path):
    return Path(path).read_bytes()


def present(path):
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False


def atomic_copy(source, target, expected_sha=None, maximum=2**31):
    """Sequential transfer; completed outputs are never overwritten."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink() or source.stat().st_size > maximum:
        raise ValueError("Unexpected transfer path/size")
    if present(target):
        digest = expected_sha or common.file_sha(source)
        if common.file_sha(target) != digest:
            raise ValueError("Preserve conflicting complete archive: " + str(target))
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".v1_transfer_", delete=False) as output:
        temporary = Path(output.name)
        try:
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                while block := stream.read(8 * 1024**2):
                    output.write(block); digest.update(block)
            output.flush()
            actual = digest.hexdigest()
            if expected_sha is not None and actual != expected_sha:
                raise ValueError("Archive transfer SHA256 differs")
            if present(target):
                raise ValueError("Concurrent archive publisher detected")
            temporary.replace(target)
            return actual
        finally:
            temporary.unlink(missing_ok=True)


def publication_record(source, target, local):
    digest = common.file_sha(source)
    marker = local / "published" / (common.sha(str(target).encode()) + ".json")
    receipt = {"target": str(target), "sha256": digest, "size_bytes": source.stat().st_size}
    return marker, receipt


def acknowledge_verified_restore(source, target, local):
    # A verified local copy was just read from this completed remote archive.
    # Do not immediately read that same remote archive a second time.
    marker, receipt = publication_record(source, target, local)
    common.immutable(marker, common.encoded(receipt))


def publish(source, target, local, maximum=2**31):
    marker, receipt = publication_record(source, target, local)
    if present(marker):
        if json.loads(read_once(marker)) != receipt:
            raise ValueError("Local publication acknowledgement changed")
        return
    atomic_copy(source, target, receipt["sha256"], maximum)
    common.immutable(marker, common.encoded(receipt))


def summary_catalog(raw, c):
    if common.sha(raw) != c["reviewed_coverage_zip_sha256"]:
        raise ValueError("Use the reviewed complete coverage summary ZIP")
    expected = {"run_identity.json", "coverage_summary.json", "coverage_by_scene.json", "coverage_by_log.json"}
    expected |= {f"parts/part{n:02d}/result.json" for n in range(1, 11)}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if (len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected
                or sum(i.file_size for i in archive.infolist()) > 100_000_000 or archive.testzip() is not None):
            raise ValueError("Coverage summary inventory/CRC differs")
        identity = coverage.check_signed(json.loads(archive.read("run_identity.json")))
        summary = coverage.check_signed(json.loads(archive.read("coverage_summary.json")))
        parts = {n: coverage.check_signed(json.loads(archive.read(f"parts/part{n:02d}/result.json"))) for n in range(1, 11)}
    if (identity["manifest_sha256"] != c["reviewed_coverage_identity_sha256"]
            or summary["manifest_sha256"] != c["reviewed_coverage_summary_sha256"]
            or not summary["complete"] or summary["identity_sha256"] != identity["manifest_sha256"]):
        raise ValueError("Coverage identity differs")
    for name, digest in identity["code_sha256"].items():
        if Path(name).name != name or common.file_sha(ROOT / "scripts" / name) != digest:
            raise ValueError("Reviewed coverage geometry/source changed: " + name)
    return identity, summary, parts


def download_catalog(raw, c):
    if common.sha(raw) != c["reviewed_download_manifest_file_sha256"]:
        raise ValueError("Use the reviewed complete keyframe download manifest")
    m = coverage.check_signed(json.loads(raw))
    from scripts.download_nuscenes_to_drive import load_sources
    _, sources = load_sources(ROOT / "configs/nuscenes_drive_sources_v1.json", "keyframes")
    if (not m["complete"] or m["variant"] != "keyframes" or m["parts"] != list(range(1, 11))
            or [r["file"] for r in m["archives"]] != [s["name"] for s in sources]
            or any(r["source"] != s or r["size_bytes"] != s["size_bytes"] for r, s in zip(m["archives"], sources))):
        raise ValueError("Download manifest source inventory differs")
    return m, {r["file"]: r for r in m["archives"]}


def expected_chunks(parts, cov_id, receipts):
    result = {}; tokens = set()
    for number, part in sorted(parts.items()):
        contract = coverage.part_contract(number, receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"], cov_id)
        flat = [t for chunk in part["chunks"] for t in chunk["sample_tokens"]]
        if (not part["complete"] or part["part_identity"] != contract or flat != part["sample_tokens"]
                or len(flat) != len(set(flat)) or tokens.intersection(flat)):
            raise ValueError("Coverage part/keyframe inventory differs")
        tokens.update(flat)
        for chunk in part["chunks"]:
            name = f"parts/part{number:02d}/" + data.safe_name(chunk["file"])
            if name in result or not chunk["file"].startswith("chunks/"):
                raise ValueError("Repeated or unexpected coverage chunk")
            result[name] = (number, chunk, contract)
    if len(tokens) != cov_id["config"]["expected_keyframes"]:
        raise ValueError("Incomplete coverage keyframes")
    return result


class UploadedCoverage:
    """Read selected suffixes from one or more Drive website folder ZIPs."""
    def __init__(self, paths, maximum):
        self.archives = []; self.entries = {}
        try:
            total = 0
            for path in paths:
                if path.stat().st_size > maximum:
                    raise ValueError("Coverage upload is too large; do not upload sensor datasets here")
                archive = zipfile.ZipFile(path); self.archives.append(archive)
                for info in archive.infolist():
                    data.safe_name(info.filename)
                    total += info.file_size
                    if total > maximum:
                        raise ValueError("Oversized decoded coverage upload")
                    if not info.is_dir():
                        self.entries.setdefault(info.filename, []).append((archive, info))
        except BaseException:
            self.close(); raise

    def read(self, suffix, maximum):
        hits = [x for name, entries in self.entries.items() if name == suffix or name.endswith("/"+suffix) for x in entries]
        if len(hits) != 1 or hits[0][1].file_size > maximum:
            raise ValueError("Missing, repeated or oversized uploaded member: " + suffix)
        return hits[0][0].read(hits[0][1])

    def close(self):
        for archive in self.archives: archive.close()


def inspect_snapshot(path, c):
    if path.stat().st_size > c["maximum_coverage_snapshot_bytes"]:
        raise ValueError("Oversized coverage snapshot")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or len(names) > 2048
                or sum(i.file_size for i in archive.infolist()) > c["maximum_coverage_snapshot_bytes"]):
            raise ValueError("Unexpected coverage snapshot inventory")
        for name in names: data.safe_name(name)
        if (archive.getinfo(SUMMARY_NAME).file_size > 100_000_000
                or archive.getinfo(DOWNLOAD_NAME).file_size > 10_000_000
                or archive.getinfo("snapshot_inventory.json").file_size > 2_000_000):
            raise ValueError("Oversized snapshot control record")
        cov_id, summary, parts = summary_catalog(archive.read(SUMMARY_NAME), c)
        manifest, receipts = download_catalog(archive.read(DOWNLOAD_NAME), c)
        chunks = expected_chunks(parts, cov_id, receipts)
        expected = {SUMMARY_NAME, DOWNLOAD_NAME, "snapshot_inventory.json"} | set(chunks)
        index = json.loads(archive.read("snapshot_inventory.json"))
        if set(names) != expected or set(index["files"]) != expected - {"snapshot_inventory.json"}:
            raise ValueError("Coverage snapshot has missing/extra members")
        for name, digest in index["files"].items():
            info = archive.getinfo(name)
            if (name in chunks and info.file_size > c["maximum_chunk_bytes"]):
                raise ValueError("Oversized coverage chunk")
            raw = archive.read(name)
            if common.sha(raw) != digest or (name in chunks and digest != chunks[name][1]["sha256"]):
                raise ValueError("Coverage chunk differs from reviewed summary")
    if manifest["manifest_sha256"] != cov_id["download_manifest_sha256"]:
        raise ValueError("Coverage/download identity mismatch")
    return cov_id, summary, receipts, chunks


def snapshot(a, c):
    target = a.local / "source_state.zip"
    remote = a.output / "source_state.zip"
    restored = False
    if not present(target) and present(remote):
        print("Restoring one coverage-state archive, not individual Drive chunks", flush=True)
        atomic_copy(remote, target, maximum=c["maximum_coverage_snapshot_bytes"])
        restored = True
    if not present(target):
        if not a.coverage_archive and not a.allow_one_time_small_drive_reads:
            raise FileNotFoundError("Upload the completed nuscenes_full_0_6m_r1 coverage folder ZIP from drive.google.com to /content, set COVERAGE_UPLOADS, then rerun. This is NOT the sensor dataset and NOT just nuscenes_0_6m_coverage_results.zip. Alternatively explicitly allow the one-time coverage chunk reads after Drive recovers.")
        uploaded = UploadedCoverage(a.coverage_archive, c["maximum_coverage_snapshot_bytes"]) if a.coverage_archive else None
        try:
            reader = (lambda name, limit: uploaded.read(name, limit)) if uploaded else (lambda name, limit: read_once(a.coverage/name))
            summary_raw = reader(SUMMARY_NAME, 100_000_000)
            download_raw = read_once(a.download_manifest or a.dataset/DOWNLOAD_NAME)
            cov_id, _, parts = summary_catalog(summary_raw, c)
            _, receipts = download_catalog(download_raw, c)
            chunks = expected_chunks(parts, cov_id, receipts)
            if not uploaded:
                print(f"Explicit one-time staging of {len(chunks)} small Drive chunks; later stages use only this local snapshot", flush=True)
            temporary = a.local / ".source_state.building.zip"
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                hashes = {}
                for name, raw in [(SUMMARY_NAME, summary_raw), (DOWNLOAD_NAME, download_raw)]:
                    archive.writestr(zipfile.ZipInfo(name, (2026,10,29,0,0,0)), raw); hashes[name] = common.sha(raw)
                size = len(summary_raw) + len(download_raw)
                for i, (name, (_, chunk, _)) in enumerate(sorted(chunks.items()), 1):
                    raw = reader(name, c["maximum_chunk_bytes"])
                    size += len(raw)
                    if len(raw) > c["maximum_chunk_bytes"] or size > c["maximum_coverage_snapshot_bytes"] or common.sha(raw) != chunk["sha256"]:
                        raise ValueError("Coverage chunk bytes/size differ")
                    archive.writestr(zipfile.ZipInfo(name, (2026,10,29,0,0,0)), raw); hashes[name] = common.sha(raw)
                    if i % 25 == 0 or i == len(chunks): print(f"Local coverage snapshot: {i}/{len(chunks)} chunks", flush=True)
                archive.writestr(zipfile.ZipInfo("snapshot_inventory.json", (2026,10,29,0,0,0)), common.encoded({"files": hashes}))
            inspect_snapshot(temporary, c)
            temporary.replace(target)
        finally:
            if uploaded: uploaded.close()
    catalog = inspect_snapshot(target, c)
    if restored: acknowledge_verified_restore(target, remote, a.local)
    publish(target, remote, a.local, c["maximum_coverage_snapshot_bytes"])
    return target, catalog


def data_protocol(c):
    p = reference.load_protocol()
    return {**p, "roles": ["train"], "maximum_images_per_role": c["maximum_train_images"],
        "images_per_shard": c["images_per_shard"], "selection_seed": c["selection_seed"],
        "selection_cells": c["selection_cells"], "rare_retention_cells": c["rare_retention_cells"]}


def validate_row(row, c, roles, p):
    frozen = roles["scenes"].get(row["scene"], {})
    if (row["role"] != "train" or frozen.get("role") != "train"
            or row["recording_log"] != frozen.get("log_token")
            or row["recording_log"] in roles["previously_inspected_evaluation_logs"]
            or any(row[k] != row["coverage_row"][k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log"])
            or common.identity(row["coverage_row"]) != row["coverage_row_sha256"]
            or row["supported_counts"] != reference.supported_counts(row["coverage_row"], p)
            or row["coverage_row"]["channel"] not in data.CHANNELS - {"LIDAR_TOP"}):
        raise ValueError("Forbidden role or changed training candidate")


def select_train(rows, c):
    if not rows or any(r["role"] != "train" for r in rows) or len({r["camera_token"] for r in rows}) != len(rows):
        raise ValueError("Select unique training cameras only")
    ranks = {r["camera_token"]: (common.sha((c["selection_seed"]+r["camera_token"]).encode()), r["camera_token"]) for r in rows}
    rank = lambda r: ranks[r["camera_token"]]
    ranked = sorted(rows, key=rank)
    selected = [r for r in ranked if any(r["supported_counts"][cell] for cell in c["rare_retention_cells"])]
    if len(selected) > c["maximum_train_images"]:
        raise ValueError("Mandatory rare views exceed budget; do not silently drop them")
    used = {r["camera_token"] for r in selected}; frames = {r["sample_token"] for r in selected}
    logs = Counter(r["recording_log"] for r in selected)
    cameras = Counter((r["recording_log"], r["coverage_row"]["channel"]) for r in selected)
    while len(selected) < c["maximum_train_images"]:
        added = False
        for cell in c["selection_cells"]:
            eligible = [r for r in ranked if r["camera_token"] not in used and r["sample_token"] not in frames and r["supported_counts"][cell]]
            if not eligible: continue
            row = min(eligible, key=lambda r: (logs[r["recording_log"]], cameras[(r["recording_log"],r["coverage_row"]["channel"])], rank(r)))
            selected.append(row); used.add(row["camera_token"]); frames.add(row["sample_token"])
            logs[row["recording_log"]] += 1; cameras[(row["recording_log"],row["coverage_row"]["channel"])] += 1
            added = True
            if len(selected) == c["maximum_train_images"]: break
        if not added: break
    return sorted(selected, key=lambda r: (r["source_part"], r["camera_token"]))


def support(rows, c, p):
    result = reference.selection_inventory(rows, p)["train"]
    result["by_camera"] = {ch: reference.selection_inventory([r for r in rows if r["coverage_row"]["channel"] == ch], p)["train"]
        for ch in sorted(data.CHANNELS - {"LIDAR_TOP"})}
    result["cell_data_diversity_checks"] = {cell: result["cells"][cell]["instance_tracks"] >= c["readiness_minimum_tracks_per_cell"]
        and result["cells"][cell]["recording_logs"] >= c["readiness_minimum_logs_per_cell"] for cell in c["selection_cells"]}
    result["six_cell_data_ready"] = all(result["cell_data_diversity_checks"].values())
    return result


def validate_selection(selection, identity, c, roles, p):
    coverage.check_signed(selection)
    rows = selection["samples"]
    if (selection["preparation_identity_sha256"] != identity["manifest_sha256"] or not rows
            or len(rows) > c["maximum_train_images"] or len({r["camera_token"] for r in rows}) != len(rows)):
        raise ValueError("Training selection identity/inventory differs")
    for row in rows: validate_row(row, c, roles, p)
    if support(rows, c, p) != selection["selected_pool"]:
        raise ValueError("Training support changed")


def plan(a, c, state, catalog, identity, roles, p):
    path = a.local / "selection_manifest.json"
    portable = a.local / "selection_state.zip"; remote = a.output / portable.name
    if not present(path) and (present(portable) or present(remote)):
        restored = not present(portable)
        if restored: atomic_copy(remote, portable, maximum=c["maximum_coverage_snapshot_bytes"])
        with zipfile.ZipFile(portable) as archive:
            if archive.namelist() != ["selection_manifest.json"]:
                raise ValueError("Unexpected selection archive")
            raw = archive.read("selection_manifest.json")
        value = json.loads(raw); validate_selection(value, identity, c, roles, p)
        common.immutable(path, raw)
        if restored: acknowledge_verified_restore(portable, remote, a.local)
    if present(path):
        value = json.loads(read_once(path)); validate_selection(value, identity, c, roles, p)
        if not present(portable):
            common.immutable(portable, reference.deterministic_zip({path.name: read_once(path)}))
        publish(portable, remote, a.local, c["maximum_coverage_snapshot_bytes"])
        print("Reused frozen training selection; no coverage chunks rescanned", flush=True)
        return value
    cov_id, _, _, chunks = catalog
    rows = []; frames = set(); cameras = set()
    with zipfile.ZipFile(state) as archive:
        for i, (name, (number, chunk, contract)) in enumerate(sorted(chunks.items()), 1):
            raw = archive.read(name)
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as decoded:
                payload = decoded.read(c["maximum_decoded_chunk_bytes"]+1)
            if len(payload) > c["maximum_decoded_chunk_bytes"]:
                raise ValueError("Oversized decoded coverage chunk")
            value = coverage.check_signed(json.loads(payload))
            if value["part_identity"] != contract: raise ValueError("Coverage chunk source changed")
            coverage.validate_records(value["records"], chunk["sample_tokens"], roles)
            for record in value["records"]:
                if record["sample_token"] in frames: raise ValueError("Repeated coverage keyframe")
                frames.add(record["sample_token"])
                for row in record["camera_rows"]:
                    if row["camera_token"] in cameras: raise ValueError("Repeated coverage camera")
                    cameras.add(row["camera_token"])
                    if row["role"] != "train": continue
                    counts = reference.supported_counts(row, p)
                    if not any(counts[k] for k in c["selection_cells"]): continue
                    candidate = {k: row[k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log"]}
                    candidate.update(source_part=number, coverage_row=row, coverage_row_sha256=common.identity(row), supported_counts=counts)
                    validate_row(candidate, c, roles, p); rows.append(candidate)
            if i % 25 == 0 or i == len(chunks): print(f"Local GT-only selection scan: {i}/{len(chunks)} chunks; {len(rows)} training candidates", flush=True)
    if len(frames) != cov_id["config"]["expected_keyframes"]: raise ValueError("Incomplete source coverage scan")
    selected = select_train(rows, c)
    value = coverage.signed({"preparation_identity_sha256": identity["manifest_sha256"], "samples": selected,
        "available_pool": support(rows, c, p), "selected_pool": support(selected, c, p),
        "selection_uses_predictions": False, "training_performed": False})
    validate_selection(value, identity, c, roles, p)
    common.immutable(path, common.encoded(value))
    common.immutable(portable, reference.deterministic_zip({path.name: read_once(path)}))
    publish(portable, remote, a.local, c["maximum_coverage_snapshot_bytes"])
    return value


def broad_reference(camera, lidar, root, labels, database, cov_config):
    import numpy as np
    calibration, _ = coverage.index_constants(database)
    with data.connect_index(database) as db:
        lp = json.loads(db.execute("SELECT body FROM poses WHERE token=?", (lidar["ego_pose_token"],)).fetchone()[0])
        cp = json.loads(db.execute("SELECT body FROM poses WHERE token=?", (camera["ego_pose_token"],)).fetchone()[0])
        label = db.execute("SELECT filename FROM labels WHERE token=?", (lidar["token"],)).fetchone()[0]
    xyz = np.fromfile(data.safe_target(root / "data", lidar["filename"]), dtype="<f4").reshape(-1, 5)[:, :3]
    sem = np.fromfile(data.safe_target(labels, label), dtype=np.uint8)
    uv, z, ids, _ = geometry.project_positive(xyz, calibration[lidar["calibrated_sensor_token"]], lp, cp,
        calibration[camera["calibrated_sensor_token"]], camera["width"], camera["height"])
    broad = geometry.original_gt.rasterize(uv, z, sem[ids], np.zeros(len(z), bool), camera["width"], camera["height"],
        {"gt_boundary_guard": cov_config["boundary_guard"], "depth_min_m": 2, "depth_max_m": 80, "invalid_semantic_indices": [0,31]})
    keep = broad["conservative"]
    return {"xy": broad["xy"][keep].astype(np.int32), "z": broad["z"][keep].astype(np.float32)}


def validate_broad(points, sample):
    import numpy as np
    xy, z = points["xy"], points["z"]
    if (set(points) != {"xy", "z"} or xy.shape != (len(z),2) or xy.dtype != np.int32 or z.dtype != np.float32
            or z.ndim != 1 or not np.isfinite(z).all() or not ((z >= 2) & (z <= 80)).all()
            or not ((xy[:,0] >= 0) & (xy[:,0] < sample["width"])).all()
            or not ((xy[:,1] >= 0) & (xy[:,1] < sample["height"])).all() or len(np.unique(xy, axis=0)) != len(z)):
        raise ValueError("Malformed broad sparse GT")


def verify_train_shard(raw, rows, identity, p):
    import numpy as np
    from PIL import Image
    if len(raw) > 100_000_000: raise ValueError("Unexpectedly large training shard")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        value = coverage.check_signed(json.loads(archive.read("shard_manifest.json")))
        if value["preparation_identity_sha256"] != identity["manifest_sha256"] or len(value["samples"]) != len(rows):
            raise ValueError("Training shard identity/count differs")
        expected = {"shard_manifest.json"}
        for sample, row in zip(value["samples"], rows):
            if sample["role"] != "train" or any(sample[k] != row[k] for k in ["camera_token", "sample_token", "scene", "role", "recording_log", "source_part", "coverage_row_sha256"]):
                raise ValueError("Training shard changed selection or role")
            for kind in ["image", "ground_truth", "broad"]:
                name = data.safe_name(sample[kind+"_member"]); expected.add(name)
                if common.sha(archive.read(name)) != sample[kind+"_sha256"]: raise ValueError("Training member bytes differ")
            with Image.open(io.BytesIO(archive.read(sample["image_member"]))) as image:
                if image.mode != "RGB" or image.size != (sample["width"], sample["height"]): raise ValueError("RGB header changed")
            with np.load(io.BytesIO(archive.read(sample["ground_truth_member"])), allow_pickle=False) as arrays:
                points = {k: arrays[k] for k in arrays.files}
            metrics.validate_gt(points, sample, p)
            counts = {cell: int(mask.sum()) for cell, mask in zip(metrics.cell_names(p), points["cell_masks"])}
            if counts != row["supported_counts"] or sample["eligible_reference_counts"] != counts:
                raise ValueError("Frozen near mask support differs")
            with np.load(io.BytesIO(archive.read(sample["broad_member"])), allow_pickle=False) as arrays:
                points = {k: arrays[k] for k in arrays.files}
            validate_broad(points, sample)
            if len(points["z"]) != sample["broad_reference_points"]: raise ValueError("Broad count changed")
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Training shard inventory/CRC differs")
    return value["samples"]


def part_batches(selection, number, c):
    rows = [r for r in selection["samples"] if r["source_part"] == number]
    for offset in range(0, len(rows), c["images_per_shard"]):
        yield f"shards/train/part{number:02d}_{offset // c['images_per_shard']:03d}.zip", rows[offset:offset+c["images_per_shard"]]


def inspect_part(path, selection, number, c, identity, p):
    if path.stat().st_size > c["maximum_part_archive_bytes"]: raise ValueError("Oversized train part")
    batches = list(part_batches(selection, number, c))
    with zipfile.ZipFile(path) as archive:
        value = coverage.check_signed(json.loads(archive.read("part_receipt.json")))
        expected = {name for name, _ in batches} | {"part_receipt.json"}
        if (value["preparation_identity_sha256"] != identity["manifest_sha256"]
                or value["selection_sha256"] != selection["manifest_sha256"] or value["part"] != number
                or len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected
                or [s["file"] for s in value["shards"]] != [name for name, _ in batches]
                or sum(i.file_size for i in archive.infolist()) > c["maximum_part_archive_bytes"]):
            raise ValueError("Training part inventory/identity differs")
        for shard, (name, rows) in zip(value["shards"], batches):
            raw = archive.read(name)
            if len(raw) != shard["size_bytes"] or common.sha(raw) != shard["sha256"]:
                raise ValueError("Training part member bytes differ")
            if verify_train_shard(raw, rows, identity, p) != shard["samples"]:
                raise ValueError("Training part sample metadata differs")
    return value


def prepare_part(a, number, selection, c, identity, p, receipts, cov_config, database):
    batches = list(part_batches(selection, number, c))
    if not batches: return None
    final = a.local / "train_parts" / f"train_part{number:02d}.zip"
    remote = a.output / "train_parts" / final.name
    source = receipts[f"v1.0-trainval{number:02d}_keyframes.tgz"]
    contract = {"part": number, "archive_sha256": source["sha256"], "preparation_identity_sha256": identity["manifest_sha256"],
        "selection_sha256": selection["manifest_sha256"]}
    root = a.local / "parts" / f"part{number:02d}"
    restored = False
    if not present(final) and present(remote):
        print(f"Restore completed train part {number:02d}: one archive read, no sensor restaging", flush=True)
        atomic_copy(remote, final, maximum=c["maximum_part_archive_bytes"])
        restored = True
    if present(final):
        value = inspect_part(final, selection, number, c, identity, p)
        if restored: acknowledge_verified_restore(final, remote, a.local)
        publish(final, remote, a.local, c["maximum_part_archive_bytes"])
        leftover = a.local/"archives"/source["file"]
        if present(root) and present(leftover):
            data.release_part(a.local, root, leftover, contract)
        return value
    archive_path = data.copy_archive(a.dataset, a.local, source, int(cov_config["local_disk_reserve_gb"]*1e9))
    rows = [r for _, batch in batches for r in batch]
    sensors, scans, files = reference.stage_selected(archive_path, source, root, database, rows, cov_config, contract)
    def generate(row):
        camera = sensors[row["camera_token"]]; lidar = scans[row["sample_token"]]
        sample, near = reference.reference_for_view(row, camera, lidar, root, a.local/"lidarseg", database, cov_config, p)
        broad = broad_reference(camera, lidar, root, a.local/"lidarseg", database, cov_config)
        validate_broad(broad, sample); broad_raw = reference.encode_npz(broad)
        image = read_once(data.safe_target(root/"data", camera["filename"]))
        if common.sha(image) != files[camera["filename"]]["sha256"]: raise ValueError("RGB changed during preparation")
        sample.update(channel=row["coverage_row"]["channel"], image_sha256=common.sha(image), ground_truth_sha256=common.sha(near),
            broad_member=f"broad/{camera['token']}.npz", broad_sha256=common.sha(broad_raw), broad_reference_points=len(broad["z"]))
        return sample, {sample["image_member"]: image, sample["ground_truth_member"]: near, sample["broad_member"]: broad_raw}
    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for name, batch in batches:
            target = data.safe_target(root, name)
            if not present(target):
                members = {}; samples = []
                for sample, content in pool.map(generate, batch): members.update(content); samples.append(sample)
                members["shard_manifest.json"] = common.encoded(coverage.signed({"preparation_identity_sha256": identity["manifest_sha256"], "samples": samples}))
                raw = reference.deterministic_zip(members); verify_train_shard(raw, batch, identity, p)
                common.immutable(target, raw)
            raw = read_once(target); samples = verify_train_shard(raw, batch, identity, p)
            records.append({"file": name, "sha256": common.sha(raw), "size_bytes": len(raw), "samples": samples})
            print(f"Prepared local train shard {name}: {len(samples)} images", flush=True)
    value = coverage.signed({"part": number, "preparation_identity_sha256": identity["manifest_sha256"],
        "selection_sha256": selection["manifest_sha256"], "source_archive_sha256": source["sha256"], "shards": records})
    final.parent.mkdir(parents=True, exist_ok=True)
    building = final.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in records:
            archive.writestr(zipfile.ZipInfo(record["file"], (2026,10,29,0,0,0)), read_once(data.safe_target(root, record["file"])))
        archive.writestr(zipfile.ZipInfo("part_receipt.json", (2026,10,29,0,0,0)), common.encoded(value))
    inspect_part(building, selection, number, c, identity, p)
    building.replace(final); publish(final, remote, a.local, c["maximum_part_archive_bytes"])
    print(f"Durable train part {number:02d}: {final.stat().st_size/1e6:.1f} MB; source scratch can now be released", flush=True)
    data.release_part(a.local, root, archive_path, contract)
    return value


def finalize(a, selection, c, identity, p):
    samples = []; parts = []
    for number in range(1, 11):
        if not list(part_batches(selection, number, c)): continue
        path = a.local/"train_parts"/f"train_part{number:02d}.zip"
        if not present(path): raise FileNotFoundError("Incomplete train package; resume prepare: " + str(path))
        value = inspect_part(path, selection, number, c, identity, p)
        relative = "train_parts/"+path.name
        for shard in value["shards"]:
            samples.extend({**s, "part_archive": relative, "shard_member": shard["file"], "shard_sha256": shard["sha256"]} for s in shard["samples"])
        parts.append({"file": relative, "sha256": common.file_sha(path), "size_bytes": path.stat().st_size,
            "images": sum(len(s["samples"]) for s in value["shards"]), "shards": len(value["shards"])})
    if (len({s["camera_token"] for s in samples}) != len(samples) or any(s["role"] != "train" for s in samples)
            or {s["camera_token"] for s in samples} != {s["camera_token"] for s in selection["samples"]}):
        raise ValueError("Training package does not exactly cover the frozen train selection")
    broad_points = sum(s["broad_reference_points"] for s in samples)
    manifest = coverage.signed({"complete": True, "revision": c["revision"], "config": c,
        "preparation_identity_sha256": identity["manifest_sha256"], "selection_sha256": selection["manifest_sha256"],
        "frozen_benchmark_manifest_sha256": c["frozen_benchmark_manifest_sha256"], "near_mask_protocol": p,
        "samples": samples, "parts": parts, "selected_pool": selection["selected_pool"],
        "broad_reference_points": broad_points, "training_data_ready": selection["selected_pool"]["six_cell_data_ready"] and broad_points > 0,
        "training_performed": False, "model_inference_performed": False, "heldout_sensor_samples_exported": 0,
        "dev_sensor_samples_exported": 0, "phone_app_changed": False})
    common.immutable(a.local/"training_input_manifest.json", common.encoded(manifest))
    warnings = ["0-1m remains exploratory; rare retention is not evidence of adequate data.",
        "No accuracy threshold or product qualification. Camera-Z is not ultrasonic nearest-surface range.",
        "Six cameras/overlapping bands reuse objects; do not sum track counts across cells."]
    for channel, row in selection["selected_pool"]["by_camera"].items():
        for cell in c["selection_cells"]:
            if not row["cells"][cell]["instance_tracks"]: warnings.append(f"Missing selected training support: {channel} {cell}")
    report = coverage.signed({"complete": True, "revision": c["revision"], "training_input_manifest_sha256": manifest["manifest_sha256"],
        "images": len(samples), "parts": len(parts), "shards": sum(r["shards"] for r in parts),
        "archive_bytes": sum(r["size_bytes"] for r in parts), "selected_pool": selection["selected_pool"],
        "broad_reference_points": broad_points, "training_data_ready": manifest["training_data_ready"],
        "warnings": warnings, "training_performed": False, "optimizer_steps": 0, "model_inference_performed": False,
        "dev_or_heldout_sensor_samples_exported": 0, "phone_app_changed": False,
        "next_step": "Return this preparation report for review. Training remains unauthorized until the frozen package and prospective recipe are reviewed; do not activate a GPU yet."})
    common.immutable(a.local/"preparation_report.json", common.encoded(report))
    names = ["preparation_identity.json", "selection_manifest.json", "training_input_manifest.json", "preparation_report.json"]
    common.immutable(a.local/"parking_v1_training_inputs_results.zip", reference.deterministic_zip({name: read_once(a.local/name) for name in names}))
    publish(a.local/"parking_v1_training_inputs_results.zip", a.output/"parking_v1_training_inputs_results.zip", a.local, c["maximum_part_archive_bytes"])
    print(json.dumps({k: report[k] for k in ["complete", "images", "parts", "archive_bytes", "training_data_ready", "training_performed"]}, indent=2), flush=True)
    return report


def initialize_owner(a, c):
    paths = [x.resolve() for x in [a.dataset, a.coverage, a.output, a.local]]
    if any(x.is_symlink() for x in [a.dataset, a.coverage, a.output, a.local]) or str(a.local.resolve()).startswith("/content/drive/"):
        raise ValueError("Use separate non-symlink local scratch, not mounted Drive")
    if any(left.is_relative_to(right) or right.is_relative_to(left) for i,left in enumerate(paths) for right in paths[i+1:]):
        raise ValueError("Dataset, coverage, new output and local scratch must be separate")
    owner = {"revision": c["revision"], "config_sha256": common.identity(c), "output": str(a.output.resolve())}
    if present(a.local) and any(a.local.iterdir()) and not present(a.local/"owner.json"):
        raise ValueError("Preserve nonempty unowned local scratch")
    common.immutable(a.local/"owner.json", common.encoded(owner))


def initialize(a, c):
    initialize_owner(a, c)
    for package, expected in c["runtime_packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("CPU dependency differs; rerun setup: " + package)
    roles = data.verified_roles(ROOT/c["frozen_roles_file"], 850)
    if (roles["manifest_sha256"] != c["frozen_roles_sha256"]
            or len({r["log_token"] for r in roles["scenes"].values() if r["role"] == "train"}) != c["expected_training_logs"]):
        raise ValueError("Frozen training log roles differ")
    state, catalog = snapshot(a, c)
    cov_id = catalog[0]
    if cov_id["roles_sha256"] != roles["manifest_sha256"]: raise ValueError("Coverage and frozen roles differ")
    p = data_protocol(c)
    identity = coverage.signed({"config": c, "coverage_identity_sha256": cov_id["manifest_sha256"],
        "roles_sha256": roles["manifest_sha256"], "data_protocol": p,
        "producer_sha256": {str(path.relative_to(ROOT)): common.file_sha(path) for path in
            [Path(__file__), CONFIG, reference.CONFIG, Path(reference.__file__), Path(metrics.__file__), Path(geometry.__file__)]},
        "reviewed_coverage_sources": cov_id["code_sha256"]})
    common.immutable(a.local/"preparation_identity.json", common.encoded(identity))
    return state, catalog, identity, roles, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["snapshot", "plan", "prepare", "bundle"])
    for name in ["dataset", "coverage", "output", "local"]: parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--coverage-archive", type=Path, action="append", default=[])
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--allow-one-time-small-drive-reads", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1,5), default=4)
    a = parser.parse_args(); c = configuration()
    from scripts.download_nuscenes_to_drive import transfer_lock
    try:
        initialize_owner(a, c)
        with transfer_lock(a.local/"locks", {"name": "parking_v1_training_inputs"}):
            state, catalog, identity, roles, p = initialize(a, c)
            if a.stage == "snapshot": print("Verified local source snapshot is ready; no GPU or sensor samples used."); return
            selection = plan(a, c, state, catalog, identity, roles, p)
            print(json.dumps(selection["selected_pool"], indent=2), flush=True)
            if a.stage == "plan": return
            if not selection["selected_pool"]["six_cell_data_ready"]:
                raise RuntimeError("Selected training data lacks minimum six-cell diversity; inspect the plan. Do not stage sensors or start training.")
            if a.stage == "prepare":
                cov_id, _, receipts, _ = catalog
                pending = [n for n in range(1,11) if list(part_batches(selection,n,c))
                    and not present(a.local/"train_parts"/f"train_part{n:02d}.zip") and not present(a.output/"train_parts"/f"train_part{n:02d}.zip")]
                database = coverage.prepare_support(cov_id["config"], roles, a.dataset, a.local, receipts, identity) if pending else None
                for number in range(1,11): prepare_part(a, number, selection, c, identity, p, receipts, cov_id["config"], database)
            finalize(a, selection, c, identity, p)
    except OSError as error:
        if error.errno == 5:
            raise RuntimeError("Input/output error: Drive quota or mount failure is possible, not confirmed. No automatic retry/remount. Preserve local/source archives; wait for recovery or use website-downloaded archives. Resume the same stage and RUN_ID; completed part archives are reusable.") from error
        raise


if __name__ == "__main__": main()
