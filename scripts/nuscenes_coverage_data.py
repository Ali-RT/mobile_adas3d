"""Bounded, local-only staging/indexing for the full nuScenes coverage audit.

Drive contains immutable source archives and compressed result chunks, never
an extracted forest of image files. A local SQLite index avoids loading the
multi-GB metadata tables into RAM. RGB headers are checked but pixels are not
saved: this workflow counts coverage, not image quality or detection accuracy.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
import tempfile

from scripts import parking_edge_common as common
from scripts import download_nuscenes_to_drive as downloader

CHANNELS = {"LIDAR_TOP", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
META_TABLES = ["scene", "sample", "sensor", "calibrated_sensor", "sample_data",
               "ego_pose", "category", "instance", "sample_annotation"]


def stream_rows(path):
    import ijson
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, "item", use_float=True)


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise ValueError("Unsafe archive member")
    return str(p)


def safe_target(root, name):
    target = Path(root) / safe_name(name)
    if target.is_symlink() or not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Staging path escapes helper-owned local scratch")
    return target


def verified_roles(path, expected_scenes):
    roles = json.loads(Path(path).read_text())
    if (roles.get("manifest_sha256") != common.identity({k: v for k, v in roles.items() if k != "manifest_sha256"})
            or len(roles["scenes"]) != expected_scenes):
        raise ValueError("Frozen role map differs")
    log_roles = {}
    allowed = {"train", "dev", "heldout_eval", "excluded_previously_used_mini_log"}
    for scene, row in roles["scenes"].items():
        if row["role"] not in allowed:
            raise ValueError("Unknown frozen role")
        prior = log_roles.setdefault(row["log_token"], row["role"])
        if prior != row["role"]:
            raise ValueError("Frozen roles share a recording log")
    if not set(roles["previously_inspected_evaluation_logs"]) <= {k for k, v in log_roles.items() if v == "heldout_eval"}:
        raise ValueError("Previously inspected evaluation log is not held out")
    return roles


def verified_downloads(dataset, config, sources_config):
    manifest_path = Path(dataset) / "download_manifest_keyframes_all.json"
    if common.file_sha(manifest_path) != config["reviewed_download_manifest_file_sha256"]:
        raise ValueError("Use the reviewed complete download manifest; preserve changed downloads")
    manifest = json.loads(manifest_path.read_text())
    source_config, sources = downloader.load_sources(sources_config, "keyframes")
    if (manifest.get("manifest_sha256") != common.identity({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            or manifest.get("complete") is not True or manifest.get("variant") != "keyframes"
            or manifest.get("parts") != list(range(1, 11))
            or manifest.get("config_revision") != source_config["revision"]
            or [r["file"] for r in manifest["archives"]] != [s["name"] for s in sources]):
        raise ValueError("Download manifest contract differs")
    receipts = {}
    for record, source in zip(manifest["archives"], sources):
        actual = downloader.read_receipt(Path(dataset) / "receipts" / (source["name"] + ".json"), source)
        if actual != record or actual["source"] != source:
            raise ValueError("Download receipt differs from reviewed manifest")
        archive = Path(dataset) / "archives" / source["name"]
        if archive.is_symlink() or archive.stat().st_size != record["size_bytes"]:
            raise ValueError("Drive archive size/path differs")
        if source["observed_sha256"] and record["sha256"] != source["observed_sha256"]:
            raise ValueError("Previously observed reference SHA256 differs")
        receipts[source["name"]] = record
    return manifest, receipts


def require_space(folder, needed, reserve):
    folder = Path(folder)
    while not folder.exists():
        folder = folder.parent
    if shutil.disk_usage(folder).free < needed + reserve:
        raise RuntimeError(f"Colab local disk needs {(needed + reserve) / 1e9:.1f} GB free for this stage; Drive quota is separate")


def copy_archive(dataset, local, receipt, reserve):
    """Resume an owned local copy; verify its whole SHA256 before use."""
    destination = Path(local) / "archives" / receipt["file"]
    identity = {"file": receipt["file"], "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}
    owner = destination.with_name(destination.name + ".copy.json")
    partial = destination.with_name(destination.name + ".partial")
    for path in [destination, partial, owner]:
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Preserve symlinked local copy")
    if (destination.exists() or partial.exists()) and not owner.exists():
        raise ValueError("Preserve unowned local archive")
    common.immutable(owner, common.encoded(identity))
    print(f"Verify/stage local copy: {receipt['file']}", flush=True)
    if destination.exists():
        if destination.stat().st_size != receipt["size_bytes"] or common.file_sha(destination) != receipt["sha256"]:
            raise ValueError("Local archive changed; preserve it")
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > receipt["size_bytes"]:
        raise ValueError("Local partial archive exceeds expected size")
    require_space(local, receipt["size_bytes"] - offset, reserve)
    source = Path(dataset) / "archives" / receipt["file"]
    full = hashlib.sha256()
    with source.open("rb") as incoming:
        if offset:
            with partial.open("rb") as existing:
                remaining = offset
                while remaining:
                    n = min(8 * 1024 ** 2, remaining)
                    old = existing.read(n); raw = incoming.read(n)
                    if len(raw) != n or raw != old:
                        raise ValueError("Local partial prefix differs from Drive; preserve it")
                    full.update(raw); remaining -= n
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("ab") as outgoing:
            copied = offset
            while copied < receipt["size_bytes"]:
                raw = incoming.read(min(8 * 1024 ** 2, receipt["size_bytes"] - copied))
                if not raw:
                    raise OSError("Drive archive ended early; rerun to resume the local copy")
                if outgoing.write(raw) != len(raw):
                    raise OSError("Short local copy write")
                full.update(raw); copied += len(raw)
                if copied // (512 * 1024 ** 2) != (copied - len(raw)) // (512 * 1024 ** 2):
                    print(f"Local copy {receipt['file']}: {copied / 1e9:.2f}/{receipt['size_bytes'] / 1e9:.2f} GB", flush=True)
    if full.hexdigest() != receipt["sha256"]:
        raise ValueError("Local copy SHA256 differs; preserve the partial")
    partial.replace(destination)
    return destination


class LimitedReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.count += len(value)
        if self.count > self.limit:
            raise ValueError("Archive exceeds its reviewed decoded size")
        return value


@contextmanager
def bounded_tar(path, receipt):
    opener = gzip.open if str(path).endswith(".tgz") else __import__("bz2").open
    with opener(path, "rb") as decoded:
        stream = LimitedReader(decoded, receipt["decoded_bytes_checked_not_saved"])
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            yield archive
        while stream.read(4 * 1024 ** 2):
            pass
        if stream.count != receipt["decoded_bytes_checked_not_saved"]:
            raise ValueError("Decoded archive length differs from the reviewed receipt")


def member_info(member, seen, maximum):
    if member.isdir() and member.name in {".", "./"}:
        return "."
    name = safe_name(member.name)
    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
        raise ValueError("Unsafe/special archive member")
    if member.isdir():
        return name
    if name in seen or not 0 <= member.size <= maximum:
        raise ValueError("Duplicate or oversized archive member")
    seen.add(name)
    return name


def extract_file(archive, member, destination, expected_sha=None):
    """Stream large metadata files; never extractall or overwrite conflicts."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.extractfile(member) as incoming:
        full = hashlib.sha256(); copied = 0
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".coverage_extract_", delete=False) as out:
            temp = Path(out.name)
            try:
                while block := incoming.read(4 * 1024 ** 2):
                    copied += len(block); full.update(block); out.write(block)
                out.flush()
                if copied != member.size or (expected_sha is not None and full.hexdigest() != expected_sha):
                    raise ValueError("Extracted member size/SHA256 differs")
                if destination.exists():
                    if common.file_sha(destination) != full.hexdigest():
                        raise ValueError("Preserve changed extracted file")
                else:
                    temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
    return {"sha256": full.hexdigest(), "size_bytes": copied}


def extract_support(path, destination, receipt, config, roles, kind):
    marker = destination / "extraction.json"
    source_id = {"source_sha256": receipt["sha256"], "kind": kind}
    if marker.exists():
        m = json.loads(marker.read_text())
        if m["identity"] != source_id or m["manifest_sha256"] != common.identity({k: v for k, v in m.items() if k != "manifest_sha256"}):
            raise ValueError("Support extraction identity changed")
        for name, info in m["files"].items():
            p = safe_target(destination, name)
            if p.stat().st_size != info["size_bytes"] or common.file_sha(p) != info["sha256"]:
                raise ValueError("Support extraction bytes changed")
        return m
    reserve = int(config["local_disk_reserve_gb"] * 1e9)
    require_space(destination, receipt["decoded_bytes_checked_not_saved"], reserve)
    files = {}; seen = set()
    wanted = {f"v1.0-trainval/{t}.json" for t in META_TABLES} if kind == "metadata" else None
    with bounded_tar(path, receipt) as archive:
        for member in archive:
            name = member_info(member, seen, config["maximum_archive_member_bytes"])
            if not member.isfile():
                continue
            keep = name in wanted if wanted is not None else (
                name in {"v1.0-trainval/lidarseg.json", "v1.0-trainval/category.json"}
                or (name.startswith("lidarseg/v1.0-trainval/") and name.endswith("_lidarseg.bin")))
            if keep:
                expected = roles["metadata_table_sha256"][Path(name).stem] if kind == "metadata" else None
                files[name] = extract_file(archive, member, safe_target(destination, name), expected)
                if kind == "metadata" or len(files) % 5000 == 0:
                    print(f"Local {kind}: {len(files)} selected files checked/extracted" +
                          (f" ({Path(name).name})" if kind == "metadata" else ""), flush=True)
    if wanted is not None and set(files) != wanted:
        raise ValueError("Metadata tables are incomplete")
    if kind == "lidarseg" and not {"v1.0-trainval/lidarseg.json", "v1.0-trainval/category.json"} <= set(files):
        raise ValueError("LiDAR semantic tables are incomplete")
    m = {"identity": source_id, "files": files}; m["manifest_sha256"] = common.identity(m)
    common.immutable(marker, common.encoded(m))
    return m


def connect_index(path):
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def build_index(local, metadata, labels, config, roles, identity):
    """Stream JSON into a disposable SQLite index; no SDK or Torch required."""
    database = local / "metadata.sqlite"
    receipt_path = local / "metadata_index.json"
    identity_sha = common.identity(identity)
    if database.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity_sha256"] != identity_sha or receipt["sha256"] != common.file_sha(database):
            raise ValueError("Preserve changed local metadata index; use a new local scratch directory")
        return database
    building = local / "metadata.building.sqlite"
    owner = local / "metadata_build.identity.json"
    if building.exists() and not owner.exists():
        raise ValueError("Preserve unowned partial database")
    common.immutable(owner, common.encoded(identity))
    if database.exists():
        with connect_index(database) as db:
            saved = db.execute("SELECT value FROM info WHERE key='identity_sha256'").fetchone()
            if saved is None or saved[0] != identity_sha or db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Incomplete/conflicting final local database")
        common.immutable(receipt_path, common.encoded({"identity_sha256": identity_sha, "sha256": common.file_sha(database)}))
        return database
    # A partial DB under this matching helper-owned build identity is disposable.
    building.unlink(missing_ok=True)
    db = sqlite3.connect(building)
    db.executescript("""
        PRAGMA journal_mode=OFF;
        CREATE TABLE info(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE samples(token TEXT PRIMARY KEY,scene TEXT);
        CREATE TABLE calibration(token TEXT PRIMARY KEY,body TEXT);
        CREATE TABLE poses(token TEXT PRIMARY KEY,body TEXT);
        CREATE TABLE sensors(token TEXT PRIMARY KEY,channel TEXT,modality TEXT);
        CREATE TABLE sensor_data(token TEXT PRIMARY KEY,filename TEXT UNIQUE,sample TEXT,channel TEXT,body TEXT);
        CREATE UNIQUE INDEX sample_channel ON sensor_data(sample,channel);
        CREATE TABLE annotations(token TEXT PRIMARY KEY,sample TEXT,body TEXT);
        CREATE INDEX sample_annotations ON annotations(sample);
        CREATE TABLE labels(token TEXT PRIMARY KEY,filename TEXT);
    """)
    meta = metadata / "v1.0-trainval"
    scenes = {s["token"]: s for s in stream_rows(meta / "scene.json")}
    if {s["name"] for s in scenes.values()} != set(roles["scenes"]):
        raise ValueError("Metadata scene inventory differs from frozen roles")
    for s in scenes.values():
        if s["log_token"] != roles["scenes"][s["name"]]["log_token"]:
            raise ValueError("Metadata recording log changed")
    db.executemany("INSERT INTO samples VALUES (?,?)", ((s["token"], scenes[s["scene_token"]]["name"]) for s in stream_rows(meta / "sample.json")))
    db.executemany("INSERT INTO sensors VALUES (?,?,?)", ((s["token"], s["channel"], s["modality"]) for s in stream_rows(meta / "sensor.json")))
    sensors = {r[0]: (r[1], r[2]) for r in db.execute("SELECT * FROM sensors")}
    calibration = {r["token"]: r for r in stream_rows(meta / "calibrated_sensor.json")}
    db.executemany("INSERT INTO calibration VALUES (?,?)", ((t, json.dumps(v)) for t, v in calibration.items()))
    needed_poses = set(); count = 0
    for row in stream_rows(meta / "sample_data.json"):
        if not row["is_key_frame"]:
            continue
        channel, _ = sensors[calibration[row["calibrated_sensor_token"]]["sensor_token"]]
        if channel not in CHANNELS:
            continue
        row["channel"] = channel
        db.execute("INSERT INTO sensor_data VALUES (?,?,?,?,?)", (row["token"], safe_name(row["filename"]), row["sample_token"], channel, json.dumps(row)))
        needed_poses.add(row["ego_pose_token"]); count += 1
        if count % 20000 == 0:
            db.commit(); print(f"Metadata index: {count} keyframe sensor records", flush=True)
    found_poses = 0
    for row in stream_rows(meta / "ego_pose.json"):
        if row["token"] in needed_poses:
            db.execute("INSERT INTO poses VALUES (?,?)", (row["token"], json.dumps(row))); found_poses += 1
    if found_poses != len(needed_poses):
        raise ValueError("Missing referenced ego pose")
    category = {r["token"]: r["name"] for r in stream_rows(meta / "category.json")}
    instances = {r["token"]: category[r["category_token"]] for r in stream_rows(meta / "instance.json")}
    semantic_rows = list(stream_rows(labels / "v1.0-trainval/category.json"))
    semantic = {r["name"]: r["index"] for r in semantic_rows}
    product = {name: cls for cls, names in config["classes"].items() for name in names}
    for row in stream_rows(meta / "sample_annotation.json"):
        name = instances[row["instance_token"]]
        if name not in product:
            continue
        row.update(class_name=product[name], source_class=name, semantic_index=semantic[name])
        db.execute("INSERT INTO annotations VALUES (?,?,?)", (row["token"], row["sample_token"], json.dumps(row)))
    db.executemany("INSERT INTO labels VALUES (?,?)", ((r["token"], safe_name(r["filename"])) for r in stream_rows(labels / "v1.0-trainval/lidarseg.json")))
    sample_count = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    bad_groups = db.execute("SELECT sample FROM sensor_data GROUP BY sample HAVING COUNT(*)!=7").fetchall()
    if (sample_count != config["expected_keyframes"] or count != sample_count * 7 or bad_groups
            or db.execute("SELECT COUNT(*) FROM sensor_data sd LEFT JOIN samples s ON s.token=sd.sample WHERE s.token IS NULL").fetchone()[0]
            or db.execute("SELECT COUNT(*) FROM sensor_data sd LEFT JOIN labels l ON l.token=sd.token WHERE sd.channel='LIDAR_TOP' AND l.token IS NULL").fetchone()[0]):
        raise ValueError("Full keyframe sensor/label inventory is incomplete")
    db.execute("INSERT INTO info VALUES ('identity_sha256',?)", (identity_sha,))
    db.execute("INSERT INTO info VALUES ('semantic_categories',?)", (json.dumps(semantic_rows),))
    db.commit()
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("Generated metadata database failed integrity check")
    db.close(); building.replace(database)
    common.immutable(receipt_path, common.encoded({"identity_sha256": identity_sha, "sha256": common.file_sha(database)}))
    print(f"Metadata ready: {sample_count} keyframes; all frozen scene/log assignments preserved", flush=True)
    return database


def stage_part(archive_path, receipt, part_root, database, config, part_identity):
    """Check all six RGB headers, save only LiDAR scans on local scratch."""
    from PIL import Image
    marker = part_root / "staging.json"
    identity_sha = common.identity(part_identity)
    common.immutable(part_root / "owner.json", common.encoded(part_identity))
    if marker.exists():
        value = json.loads(marker.read_text())
        if value["identity_sha256"] != identity_sha or value["manifest_sha256"] != common.identity({k: v for k, v in value.items() if k != "manifest_sha256"}):
            raise ValueError("Local part staging identity changed")
        for name, info in value["lidar_files"].items():
            p = safe_target(part_root, name)
            if p.stat().st_size != info["size_bytes"] or common.file_sha(p) != info["sha256"]:
                raise ValueError("Staged LiDAR bytes changed")
        return value
    require_space(part_root, receipt["decoded_bytes_checked_not_saved"], int(config["local_disk_reserve_gb"] * 1e9))
    seen = set(); frames = {}; lidar_files = {}; rgb_headers = 0
    with connect_index(database) as db, bounded_tar(archive_path, receipt) as archive:
        for member in archive:
            name = member_info(member, seen, config["maximum_archive_member_bytes"])
            if not member.isfile() or not (name.startswith("samples/CAM_") or name.startswith("samples/LIDAR_TOP/")):
                continue
            result = db.execute("SELECT sd.body,s.scene FROM sensor_data sd JOIN samples s ON s.token=sd.sample WHERE sd.filename=?", (name,)).fetchone()
            if result is None:
                raise ValueError("Archive camera/LiDAR file is absent from the frozen keyframe metadata")
            row = json.loads(result[0]); sample = row["sample_token"]
            frame = frames.setdefault(sample, {"sample_token": sample, "scene": result[1], "sensors": {}})
            channel = row["channel"]
            if channel in frame["sensors"]:
                raise ValueError("Repeated keyframe sensor in archive")
            if channel == "LIDAR_TOP":
                local_name = "lidar/" + row["token"] + ".bin"
                info = extract_file(archive, member, safe_target(part_root, local_name))
                if info["size_bytes"] % 20:
                    raise ValueError("Raw LiDAR scan is not five-channel Float32")
                lidar_files[local_name] = info; row["local_file"] = local_name
            else:
                if member.size > 20 * 1024 ** 2:
                    raise ValueError("Unexpected camera image size")
                with archive.extractfile(member) as image_bytes:
                    raw = image_bytes.read()
                if len(raw) != member.size:
                    raise ValueError("Camera archive member ended early")
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (row["width"], row["height"]) or image.mode != "RGB":
                        raise ValueError("Camera header differs from metadata")
                    image.verify()
                row["image_sha256"] = common.sha(raw); rgb_headers += 1
                if rgb_headers % 1000 == 0:
                    print(f"Sensor archive: {rgb_headers} RGB headers checked; no RGB files saved", flush=True)
            frame["sensors"][channel] = row
    if (not frames or any(set(k["sensors"]) != CHANNELS for k in frames.values())
            or len({k["scene"] for k in frames.values()}) != config["expected_scenes_per_part"]):
        raise ValueError("Sensor archive lacks complete seven-sensor keyframes or expected scenes")
    result = {"identity_sha256": identity_sha, "keyframes": [frames[t] for t in sorted(frames)],
              "lidar_files": lidar_files, "rgb_headers_checked": rgb_headers,
              "rgb_pixels_saved": False, "source_archive_sha256": receipt["sha256"]}
    result["manifest_sha256"] = common.identity(result)
    common.immutable(marker, common.encoded(result))
    print(f"Part staging complete: {len(frames)} keyframes, {rgb_headers} RGB headers; only local LiDAR files saved", flush=True)
    return result


def release_part(local, part_root, archive, part_identity):
    """Delete only this completed helper-owned local part, never Drive data."""
    local = Path(local).resolve(); part_root = Path(part_root)
    if (part_root.is_symlink() or part_root.resolve().parent != local / "parts"
            or json.loads((part_root / "owner.json").read_text()) != part_identity):
        raise ValueError("Refuse cleanup of unowned local scratch")
    archive = Path(archive)
    owner = archive.with_name(archive.name + ".copy.json")
    if (archive.is_symlink() or not archive.resolve().is_relative_to(local / "archives")
            or json.loads(owner.read_text())["sha256"] != part_identity["archive_sha256"]):
        raise ValueError("Refuse cleanup of unowned local archive copy")
    shutil.rmtree(part_root)
    archive.unlink(); owner.unlink()
