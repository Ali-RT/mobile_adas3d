"""CPU-only, resumable official nuScenes archive downloads into Google Drive.

No extraction or model dependencies. Own only new archive partials/sidecars;
never overwrite unrelated files. Recheck complete chunks after interruption,
require honest HTTP Range responses, and validate read-back SHA256 plus the
compression CRC before publishing a completion receipt. ETags are pinned
object identities, not assumed to be plain file checksums.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import bz2
import fcntl
import gzip
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common

CONFIG = ROOT / "configs/nuscenes_drive_sources_v1.json"
CHUNK_BYTES = 64 * 1024 ** 2
ALLOWED_HOSTS = {"motional-nuscenes.s3.amazonaws.com", "d36yt3mvayqw5m.cloudfront.net"}


def validate_url(url):
    u = urllib.parse.urlsplit(url)
    if (u.scheme != "https" or u.hostname not in ALLOWED_HOSTS or u.username or u.password
            or u.query or u.fragment or u.port not in (None, 443)):
        raise ValueError("Use the configured public HTTPS sources; never paste credentials or signed URLs into this workflow")


def load_sources(config, variant="keyframes", parts=None):
    c = json.loads(Path(config).read_text())
    if variant not in {"keyframes", "full"} or set(c["variants"]) != {"keyframes", "full"}:
        raise ValueError("Choose keyframes or full sensor archives")
    if parts is not None and (len(parts) != len(set(parts)) or not set(parts) <= set(range(1, 11))):
        raise ValueError("Parts must be distinct integers from 1 through 10")
    selected = c["common_archives"] + [r for i, r in enumerate(c["variants"][variant], 1) if parts is None or i in parts]
    if len(c["variants"][variant]) != 10 or len({s["name"] for s in selected}) != len(selected):
        raise ValueError("Archive inventory changed")
    for s in selected:
        validate_url(s["url"])
        if (Path(s["name"]).name != s["name"] or "\\" in s["name"]
                or urllib.parse.urlsplit(s["url"]).path.rsplit("/", 1)[-1] != s["name"]
                or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(tgz|tar\.bz2)", s["name"])
                or type(s["size_bytes"]) is not int or not 0 < s["size_bytes"] < 64 * 1024 ** 3
                or not isinstance(s["etag"], str) or not s["etag"]
                or (s["observed_sha256"] is not None and not re.fullmatch(r"[a-f0-9]{64}", s["observed_sha256"]))):
            raise ValueError("Invalid pinned archive record")
    return c, selected


def remote_probe(source):
    validate_url(source["url"])
    request = urllib.request.Request(source["url"], method="HEAD", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=45) as response:
        validate_url(response.geturl())
        if (response.status != 200 or int(response.headers.get("Content-Length", -1)) != source["size_bytes"]
                or response.headers.get("ETag") != source["etag"]):
            raise RuntimeError(f"Remote archive identity changed: {source['name']}; preserve existing files")
        return {"size_bytes": source["size_bytes"], "etag": source["etag"],
                "accept_ranges": response.headers.get("Accept-Ranges"),
                "last_modified": response.headers.get("Last-Modified")}


def owned_paths(root, name):
    root = Path(root)
    if root.is_symlink():
        raise RuntimeError("Preserve symlinked dataset root")
    paths = {"archive": root / "archives" / name,
             "partial": root / "archives" / (name + ".partial"),
             "owner": root / "download_state" / (name + ".identity.json"),
             "state": root / "download_state" / (name + ".chunks.json"),
             "receipt": root / "receipts" / (name + ".json")}
    for p in paths.values():
        if p.is_symlink() or p.parent.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Preserve symlinked or escaping download paths")
    return paths


def source_identity(source):
    return {"revision": "NUSCENES-DOWNLOAD-TRANSFER-r1", "source": source, "chunk_bytes": CHUNK_BYTES}


def replace_state(path, value):
    """Replace only a transfer-owned checkpoint; never an input/archive."""
    if path.is_symlink():
        raise RuntimeError("Preserve symlinked transfer checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".nuscenes_state_", delete=False) as f:
        temp = Path(f.name)
        f.write(common.encoded(value))
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def transfer_lock(root, source):
    lock_root = Path(tempfile.gettempdir()) / "mobile_adas3d_nuscenes_locks"
    lock_root.mkdir(exist_ok=True)
    token = common.identity([str(Path(root).resolve()), source["name"]])
    with (lock_root / (token + ".lock")).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This runtime already has a downloader for this archive") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_state(path, owner_id, total):
    if not path.exists():
        return {"owner_sha256": owner_id, "chunks": []}
    value = json.loads(path.read_text())
    if value.get("owner_sha256") != owner_id or not isinstance(value.get("chunks"), list):
        raise RuntimeError("Partial download owner changed; preserve it")
    chunks = value["chunks"]
    if len(chunks) > math.ceil(total / CHUNK_BYTES):
        raise RuntimeError("Invalid partial chunk count")
    for i, chunk in enumerate(chunks):
        if (chunk.get("size_bytes") != min(CHUNK_BYTES, total - i * CHUNK_BYTES)
                or not re.fullmatch(r"[a-f0-9]{64}", chunk.get("sha256", ""))):
            raise RuntimeError("Invalid partial chunk receipt")
    return value


def verified_prefix(partial, state):
    full = hashlib.sha256(); verified = 0
    if not partial.exists():
        if state["chunks"]:
            raise RuntimeError("Recorded partial download disappeared")
        return full, 0
    if partial.stat().st_size < sum(x["size_bytes"] for x in state["chunks"]):
        raise RuntimeError("Partial download is shorter than verified chunks")
    with partial.open("rb") as f:
        for i, chunk in enumerate(state["chunks"]):
            raw = f.read(chunk["size_bytes"])
            if len(raw) != chunk["size_bytes"] or common.sha(raw) != chunk["sha256"]:
                raise RuntimeError(f"Partial chunk {i} changed; preserve this download")
            full.update(raw); verified += len(raw)
    return full, verified


def validate_response(response, source, offset):
    validate_url(response.geturl())
    expected = source["size_bytes"] - offset
    if response.headers.get("ETag") != source["etag"] or int(response.headers.get("Content-Length", -1)) != expected:
        raise RuntimeError("Download response identity/length changed")
    if response.status == 206:
        if response.headers.get("Content-Range") != f"bytes {offset}-{source['size_bytes'] - 1}/{source['size_bytes']}":
            raise RuntimeError("Server returned an incorrect HTTP range")
    elif response.status != 200 or offset != 0:
        raise RuntimeError("Server ignored resume range; no data appended")


def validate_payload(path, source, transfer_sha=None):
    if path.stat().st_size != source["size_bytes"]:
        raise RuntimeError("Completed archive length differs")
    print(f"Read-back SHA256: {source['name']}", flush=True)
    digest = common.file_sha(path)
    expected = source.get("observed_sha256")
    if (transfer_sha is not None and digest != transfer_sha) or (expected is not None and digest != expected):
        raise RuntimeError("Completed archive SHA256 differs; preserve the file")
    print(f"Compression CRC check (no extraction): {source['name']}", flush=True)
    opener = gzip.open if source["name"].endswith(".tgz") else bz2.open
    decoded = 0
    with opener(path, "rb") as stream:
        while block := stream.read(8 * 1024 ** 2):
            decoded += len(block)
    return {"sha256": digest, "size_bytes": source["size_bytes"],
            "compression_crc_validated": True, "decoded_bytes_checked_not_saved": decoded,
            "matched_previously_observed_sha256": expected is not None,
            "publisher_signed_checksum_available": False}


def read_receipt(path, source):
    receipt = json.loads(path.read_text())
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != common.identity(body) or not receipt.get("complete")
            or receipt.get("owner_sha256") != common.identity(source_identity(source))
            or receipt.get("file") != source["name"] or not receipt.get("compression_crc_validated")
            or receipt.get("size_bytes") != source["size_bytes"]
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("sha256", ""))):
        raise RuntimeError("Completed receipt differs; preserve the existing dataset")
    return receipt


def download_one(root, source, retries=3, retry_seconds=2):
    p = owned_paths(root, source["name"])
    with transfer_lock(root, source):
        remote_probe(source)
        identity = source_identity(source); owner_id = common.identity(identity)
        if p["archive"].exists() and p["receipt"].exists():
            receipt = read_receipt(p["receipt"], source)
            if p["archive"].stat().st_size != source["size_bytes"] or common.file_sha(p["archive"]) != receipt["sha256"]:
                raise RuntimeError("Existing complete archive changed; preserve it")
            print(f"Reused verified complete archive: {source['name']}", flush=True)
            return receipt
        if p["receipt"].exists() and not p["archive"].exists():
            raise RuntimeError("Receipt exists but completed archive disappeared")
        if (p["partial"].exists() or p["archive"].exists() or p["state"].exists()) and not p["owner"].exists():
            raise RuntimeError("Unowned existing file; preserve it and choose another dataset root")
        common.immutable(p["owner"], common.encoded(identity))
        state = read_state(p["state"], owner_id, source["size_bytes"])
        # Recover a process killed between publishing its archive and receipt.
        candidate = p["archive"] if p["archive"].exists() else p["partial"]
        if p["archive"].exists() and p["partial"].exists():
            raise RuntimeError("Both final and partial archives exist; preserve them")
        full, offset = verified_prefix(candidate, state)
        if candidate.exists() and candidate.stat().st_size > source["size_bytes"]:
            raise RuntimeError("Owned partial exceeds source length; preserve it")
        if p["archive"].exists() and offset != source["size_bytes"]:
            raise RuntimeError("Final archive lacks a complete owned chunk record")
        p["partial"].parent.mkdir(parents=True, exist_ok=True)
        failure = 0
        while offset < source["size_bytes"]:
            print(f"Resume {source['name']}: {offset:,}/{source['size_bytes']:,} bytes", flush=True)
            # Only an uncommitted tail of this helper-owned partial is discarded.
            with p["partial"].open("r+b" if p["partial"].exists() else "w+b") as f:
                f.truncate(offset)
            headers = {"Accept-Encoding": "identity", "If-Range": source["etag"], "Range": f"bytes={offset}-"}
            try:
                request = urllib.request.Request(source["url"], headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    validate_response(response, source, offset)
                    with p["partial"].open("ab") as outgoing:
                        while offset < source["size_bytes"]:
                            needed = min(CHUNK_BYTES, source["size_bytes"] - offset)
                            pieces = []; received = 0
                            while received < needed:
                                block = response.read(min(8 * 1024 ** 2, needed - received))
                                if not block:
                                    raise ConnectionError("Archive transfer ended early")
                                pieces.append(block); received += len(block)
                            block = b"".join(pieces)
                            if outgoing.write(block) != len(block):
                                raise OSError("Short Drive write")
                            outgoing.flush()
                            chunk = {"size_bytes": len(block), "sha256": common.sha(block)}
                            next_state = {"owner_sha256": owner_id, "chunks": state["chunks"] + [chunk]}
                            replace_state(p["state"], next_state)
                            state = next_state; full.update(block); offset += len(block)
                            print(f"{source['name']}: {100 * offset / source['size_bytes']:.1f}% ({offset:,} bytes verified in transfer checkpoints)", flush=True)
                break
            except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError, TimeoutError, OSError) as exc:
                failure += 1
                if failure > retries:
                    raise RuntimeError(f"Transfer interrupted for {source['name']}; rerun this cell to resume verified chunks") from exc
                print(f"Transfer interrupted; retry {failure}/{retries} from the last committed chunk", flush=True)
                time.sleep(retry_seconds)
        candidate = p["archive"] if p["archive"].exists() else p["partial"]
        checked = validate_payload(candidate, source, full.hexdigest())
        if not p["archive"].exists():
            candidate.replace(p["archive"])
        receipt = {"complete": True, "owner_sha256": owner_id, "file": source["name"],
                   "source": source, **checked, "model_inference_performed": False, "training_performed": False}
        receipt["receipt_sha256"] = common.identity(receipt)
        common.immutable(p["receipt"], common.encoded(receipt))
        print(f"Complete and verified: {source['name']}", flush=True)
        return receipt


def make_plan(root, sources, available_drive_gb, reserve_gb):
    if not math.isfinite(available_drive_gb) or available_drive_gb < 0 or not math.isfinite(reserve_gb) or reserve_gb < 0:
        raise ValueError("Drive free-space/reserve values must be finite and non-negative")
    remaining = 0; status = []
    for source in sources:
        p = owned_paths(root, source["name"])
        present = 0
        if p["archive"].exists() and p["receipt"].exists():
            read_receipt(p["receipt"], source)
            if p["archive"].stat().st_size != source["size_bytes"]:
                raise RuntimeError("Existing archive size differs")
            present = source["size_bytes"]
        elif p["partial"].exists() or p["archive"].exists() or p["state"].exists():
            if not p["owner"].exists() or json.loads(p["owner"].read_text()) != source_identity(source):
                raise RuntimeError("Preserve an unowned/conflicting download")
            read_state(p["state"], common.identity(source_identity(source)), source["size_bytes"])
            file = p["archive"] if p["archive"].exists() else p["partial"]
            present = min(file.stat().st_size if file.exists() else 0, source["size_bytes"])
        remaining += source["size_bytes"] - present
        status.append({"name": source["name"], "size_bytes": source["size_bytes"], "present_bytes_not_yet_reverified": present})
    # Conservatively count a partial tail as needing download again. It already
    # consumes Drive quota, but at most one more chunk of scratch is reserved.
    required = remaining + CHUNK_BYTES + int(reserve_gb * 1e9)
    plan = {"files": status, "archive_count": len(sources), "total_compressed_bytes": sum(s["size_bytes"] for s in sources),
            "remaining_download_bytes": remaining, "free_drive_bytes_reported": int(available_drive_gb * 1e9),
            "reserve_bytes": int(reserve_gb * 1e9), "required_free_bytes_now": required,
            "space_check_passed": available_drive_gb * 1e9 >= required,
            "scope": "Archive storage only; prepared training shards require additional space. Drive quota is distinct from Colab VM disk.",
            "gpu_required": False, "archive_extraction_performed": False}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "download", "verify"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=["keyframes", "full"], default="keyframes")
    parser.add_argument("--parts", nargs="+", type=int)
    parser.add_argument("--available-drive-gb", type=float)
    parser.add_argument("--reserve-gb", type=float, default=20)
    a = parser.parse_args()
    config, sources = load_sources(a.config, a.variant, a.parts)
    if a.stage in {"plan", "download"}:
        if a.available_drive_gb is None:
            parser.error("Report actual free Drive space with --available-drive-gb; filesystem disk_usage is not a Drive quota check")
        plan = make_plan(a.root, sources, a.available_drive_gb, a.reserve_gb)
        print(json.dumps(plan, indent=2), flush=True)
        if not plan["space_check_passed"]:
            raise RuntimeError("Insufficient reported Drive space; no downloads started")
        if a.stage == "plan":
            for source in sources:
                remote_probe(source)
            print("All source identities and storage preflight passed; no dataset downloaded", flush=True)
            return
    receipts = []
    for source in sources:
        if a.stage == "download":
            receipts.append(download_one(a.root, source))
        else:
            p = owned_paths(a.root, source["name"])
            receipt = read_receipt(p["receipt"], source)
            checked = validate_payload(p["archive"], source, receipt["sha256"])
            if checked != {k: receipt[k] for k in checked}:
                raise RuntimeError("Completed verification differs from its receipt")
            receipts.append(receipt)
    manifest = {"complete": True, "config_revision": config["revision"], "variant": a.variant,
                "parts": a.parts or list(range(1, 11)), "archives": receipts,
                "training_performed": False, "archive_extraction_performed": False,
                "benchmark_data_prepared": False, "accuracy_threshold_m": None}
    manifest["manifest_sha256"] = common.identity(manifest)
    token = "all" if a.parts is None else "_".join(str(i) for i in a.parts)
    destination = a.root / f"download_manifest_{a.variant}_{token}.json"
    common.immutable(destination, common.encoded(manifest))
    print(f"All requested archives verified: {destination}. Do not allocate a GPU yet; coverage preparation is separate.", flush=True)


if __name__ == "__main__":
    main()
