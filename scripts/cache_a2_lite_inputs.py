"""CPU-prepared, reusable KITTI input archive; local restore for GPU runs.

Only reviewed images, labels, calibration and explicit Chen splits are
cached. No model, teacher predictions, training state or experiment output.
PNG files are already compressed: ZIP_STORED avoids a compression pass.
Archive transfers use verified chunks and can resume their owned partials.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_a2_lite_data as staging

REVISION = "A2-LITE-INPUT-CACHE-2026-10-14-r1"
ARCHIVE = "kitti_chen_7481.zip"
RECEIPT = "kitti_input_cache.json"
CHUNK_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 ** 3


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def signature(value):
    return staging.digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"Preserve symlinked cache file: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError(f"Preserve conflicting cache record: {path}; use a new cache folder")


def read_at(handle, offset, count):
    for attempt in range(3):
        try:
            handle.seek(offset)
            return handle.read(count)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


def free_space(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def oracle(pilot):
    ids, expected = staging.reviewed_data(pilot)
    return ids, expected, {"train": ids[:3712], "val": ids[3712:]}


def separate_paths(args):
    dataset, shared = args.dataset.resolve(), args.cache.resolve()
    if dataset.is_relative_to(shared) or shared.is_relative_to(dataset):
        raise RuntimeError("Keep the reusable cache separate from the local dataset")
    if args.pilot.is_dir():
        protected = args.pilot.resolve()
        outputs = [dataset, shared]
        if args.local_archive is not None: outputs.append(args.local_archive.resolve())
        if any(p.is_relative_to(protected) for p in outputs):
            raise RuntimeError("Reviewed pilot is read-only; use separate cache/data/archive paths")


def info_for(ids, expected, splits):
    if (len(ids) != len(set(ids)) or set(ids) != set(expected)
            or splits["train"] + splits["val"] != ids or set(splits["train"]) & set(splits["val"])):
        raise RuntimeError("Cache requires disjoint, unchanged train/val identities")
    return dict(schema_version=1, revision=REVISION, oracle_sha256=staging.PILOT_MANIFEST_SHA,
                inventory_sha256=signature(expected), train_frames=len(splits["train"]),
                val_frames=len(splits["val"]), input_files=len(ids) * 3,
                split_sha256={k: staging.digest(("\n".join(v) + "\n").encode()) for k, v in splits.items()},
                storage="ZIP_STORED", includes_models=False, includes_teacher_predictions=False)


def archive_record(path):
    full = hashlib.sha256(); chunks = []; size = 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            full.update(block); chunks.append(staging.digest(block)); size += len(block)
    return dict(file=ARCHIVE, sha256=full.hexdigest(), size_bytes=size,
                chunk_bytes=CHUNK_BYTES, chunk_sha256=chunks)


def read_receipt(cache, info):
    value = json.loads((cache / RECEIPT).read_text())
    saved = value.pop("receipt_sha256", None)
    if saved != signature(value) or not value.get("complete") or value.get("info") != info or value.get("cache_id") != signature(info):
        raise RuntimeError("Cache provenance differs; preserve it and use the reviewed input cache")
    a = value["archive"]
    if (set(a) != {"file", "sha256", "size_bytes", "chunk_bytes", "chunk_sha256"}
            or a["file"] != ARCHIVE or not isinstance(a["size_bytes"], int)
            or not 0 < a["size_bytes"] <= MAX_ARCHIVE_BYTES or a["chunk_bytes"] != CHUNK_BYTES
            or len(a["chunk_sha256"]) != math.ceil(a["size_bytes"] / CHUNK_BYTES)
            or any(not isinstance(s, str) or not re.fullmatch(r"[a-f0-9]{64}", s)
                   for s in [a["sha256"], *a["chunk_sha256"]])):
        raise RuntimeError("Malformed cache archive identity")
    return value


@contextmanager
def transfer_lock(destination):
    # Local advisory locks release automatically if the process is killed.
    # The shared cache is still guarded by immutable identity/hash checks.
    folder = Path(tempfile.gettempdir()) / "mobile_adas3d_input_locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (staging.digest(str(destination.resolve()).encode()) + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def copy_archive(source, destination, record):
    """One sequential transfer; verify/resume only helper-owned chunk data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with transfer_lock(destination):
        if destination.is_symlink():
            raise RuntimeError(f"Preserve symlinked archive: {destination}")
        if destination.exists():
            if archive_record(destination) != record:
                raise RuntimeError(f"Preserve conflicting archive: {destination}")
            return dict(copied_bytes=0, resumed_chunks=len(record["chunk_sha256"]), reused_complete=True)
        if source.stat().st_size != record["size_bytes"]:
            raise RuntimeError("Source archive size changed")
        partial = destination.with_name(destination.name + ".partial")
        owner = partial.with_name(partial.name + ".json")
        if partial.is_symlink() or owner.is_symlink():
            raise RuntimeError("Preserve symlinked partial archive/identity")
        if partial.exists() and not owner.exists():
            raise RuntimeError("Unowned partial archive; preserve it and choose a new local archive path")
        write_once(owner, json_bytes(record))
        existing_size = partial.stat().st_size if partial.exists() else 0
        if free_space(destination.parent) < max(0, record["size_bytes"] - existing_size) + 64 * 1024 ** 2:
            raise RuntimeError("Insufficient space for remaining archive transfer; preserve verified partial chunks")
        mode = "r+b" if partial.exists() else "w+b"
        started = time.monotonic(); copied = resumed = 0
        with source.open("rb") as incoming, partial.open(mode) as outgoing:
            existing = outgoing.seek(0, os.SEEK_END)
            if existing > record["size_bytes"]:
                raise RuntimeError("Owned partial archive exceeds its frozen size; preserve it")
            for index, expected in enumerate(record["chunk_sha256"]):
                offset = index * CHUNK_BYTES; count = min(CHUNK_BYTES, record["size_bytes"] - offset)
                if existing >= offset + count:
                    outgoing.seek(offset); block = outgoing.read(count)
                    if staging.digest(block) != expected:
                        raise RuntimeError(f"Changed complete partial chunk {index}; preserve it")
                    resumed += 1
                else:
                    # Only the unverified tail of this owned transfer may be
                    # truncated after interruption; complete chunks are kept.
                    outgoing.truncate(offset)
                    block = read_at(incoming, offset, count)
                    if len(block) != count or staging.digest(block) != expected:
                        raise RuntimeError(f"Archive source chunk {index} changed; no bad chunk written")
                    outgoing.seek(offset); outgoing.write(block); outgoing.flush(); copied += count
                elapsed = max(time.monotonic() - started, .001)
                print(f"Archive transfer {offset + count:,}/{record['size_bytes']:,} bytes; "
                      f"{resumed} chunks reused, {copied / 1024 ** 2 / elapsed:.1f} MiB/s copied", flush=True)
        if archive_record(partial) != record:
            raise RuntimeError("Completed archive hash failed; preserve the partial")
        if destination.exists():
            if archive_record(destination) != record:
                raise RuntimeError("Concurrent destination differs; preserve both files")
            partial.unlink()
        else:
            partial.rename(destination)
        owner.unlink()
        return dict(copied_bytes=copied, resumed_chunks=resumed, reused_complete=False)


def zip_entry(archive, name, data):
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_STORED; entry.external_attr = 0o100600 << 16
    archive.writestr(entry, data)


def build_archive(dataset, local_archive, ids, expected, splits, info):
    if local_archive.exists():
        verify_archive(local_archive, dataset=None, ids=ids, expected=expected, splits=splits, info=info)
        return archive_record(local_archive)
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    required = sum((dataset / "training" / kind / (s + suffix)).stat().st_size for s in ids for kind, suffix in staging.KINDS)
    if shutil.disk_usage(local_archive.parent).free < required + 64 * 1024 ** 2:
        raise RuntimeError("Insufficient local space for archive; need another dataset-sized allocation plus64 MiB")
    temporary = local_archive.with_name(local_archive.name + ".building")
    if temporary.exists():
        raise RuntimeError("Preserve interrupted archive build; choose a fresh local archive path")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            zip_entry(archive, "cache_info.json", json_bytes(info))
            for split, values in splits.items():
                zip_entry(archive, f"ImageSets/{split}.txt", ("\n".join(values) + "\n").encode())
            for index, sample in enumerate(ids):
                for kind, suffix in staging.KINDS:
                    name = f"training/{kind}/{sample}{suffix}"
                    data = staging.retry_read(dataset / name)
                    if staging.digest(data) != expected[sample][kind]:
                        raise RuntimeError(f"Local data changed before packaging: {name}")
                    zip_entry(archive, name, data)
                if (index + 1) % 500 == 0 or index + 1 == len(ids):
                    print(f"Packaged {index + 1}/{len(ids)} verified frames; no PNG recompression", flush=True)
        temporary.rename(local_archive)
    except BaseException:
        # This is a new helper-owned build file, never the shared archive.
        temporary.unlink(missing_ok=True)
        raise
    return archive_record(local_archive)


def verify_archive(local_archive, dataset, ids, expected, splits, info):
    names = {"cache_info.json", "ImageSets/train.txt", "ImageSets/val.txt"}
    names.update(f"training/{kind}/{s}{suffix}" for s in ids for kind, suffix in staging.KINDS)
    copied = reused = 0
    if dataset is not None:
        if (not dataset.name.startswith("kitti_a2_lite_accuracy_") or dataset.is_symlink()
                or dataset.resolve().is_relative_to(Path("/content/drive"))):
            raise RuntimeError("Restore into a private local kitti_a2_lite_accuracy_<RUN_ID> folder")
        for folder in ("ImageSets", *[f"training/{kind}" for kind, _ in staging.KINDS]):
            if not (dataset / folder).resolve().is_relative_to(dataset.resolve()):
                raise RuntimeError("Preserve extraction directory pointing outside local dataset")
    with zipfile.ZipFile(local_archive) as archive:
        records = archive.infolist()
        if (len(records) != len(names) or {r.filename for r in records} != names
                or any(r.compress_type != zipfile.ZIP_STORED or r.file_size > 64 * 1024 ** 2
                       or (r.external_attr >> 16) & 0o170000 == 0o120000 for r in records)):
            raise RuntimeError("Unexpected/duplicate/unsafe archive members; no files extracted")
        if json.loads(archive.read("cache_info.json")) != info:
            raise RuntimeError("Embedded archive provenance differs")
        if dataset is not None:
            required = sum(r.file_size for r in records if r.filename != "cache_info.json"
                           and not (dataset / r.filename).exists())
            if free_space(dataset) < required + 64 * 1024 ** 2:
                raise RuntimeError("Insufficient local space for remaining extracted data; no input files copied")
        for split, values in splits.items():
            name = f"ImageSets/{split}.txt"; data = ("\n".join(values) + "\n").encode()
            if archive.read(name) != data:
                raise RuntimeError("Cache train/val split changed; no images extracted")
            if dataset is not None: staging.atomic_copy(dataset / name, data)
        for index, sample in enumerate(ids):
            for kind, suffix in staging.KINDS:
                name = f"training/{kind}/{sample}{suffix}"
                destination = dataset / name if dataset is not None else None
                if destination is not None and destination.exists():
                    if destination.is_symlink() or staging.digest(staging.retry_read(destination)) != expected[sample][kind]:
                        raise RuntimeError(f"Preserve conflicting restored file: {destination}")
                    reused += 1
                    continue
                data = archive.read(name)
                if staging.digest(data) != expected[sample][kind]:
                    raise RuntimeError(f"Archive input bytes differ from reviewed data: {name}")
                if destination is not None: copied += int(staging.atomic_copy(destination, data))
            if (index + 1) % 500 == 0 or index + 1 == len(ids):
                print(f"Archive inputs verified {index + 1}/{len(ids)} frames; {copied} files restored, {reused} reused", flush=True)
    return dict(restored_files=copied, reused_files=reused, verified_frames=len(ids))


def prepare(args):
    separate_paths(args)
    ids, expected, splits = oracle(args.pilot); info = info_for(ids, expected, splits)
    args.cache.mkdir(parents=True, exist_ok=True)
    if (args.cache / RECEIPT).exists():
        receipt = read_receipt(args.cache, info)
        if archive_record(args.cache / ARCHIVE) != receipt["archive"]:
            raise RuntimeError("Shared completed cache archive changed; preserve it")
        print("Existing verified input cache reused; no per-image Drive reads repeated.", flush=True)
        return
    staging.stage(args.dataset, args.candidate, ids, expected, args.workers)
    local = args.local_archive or args.dataset.parent / "a2_lite_archive_build" / ARCHIVE
    record = build_archive(args.dataset, local, ids, expected, splits, info)
    if record["size_bytes"] > MAX_ARCHIVE_BYTES:
        raise RuntimeError("Unexpected archive size; no shared cache published")
    transfer = copy_archive(local, args.cache / ARCHIVE, record)
    receipt = dict(complete=True, cache_id=signature(info), info=info, archive=record)
    receipt["receipt_sha256"] = signature(receipt)
    write_once(args.cache / RECEIPT, json_bytes(receipt))
    print(json.dumps(dict(cache=str(args.cache), archive_size_bytes=record["size_bytes"],
                         cache_id=receipt["cache_id"], transfer=transfer, gpu_used=False), indent=2), flush=True)


def restore(args):
    separate_paths(args)
    ids, expected, splits = oracle(args.pilot); info = info_for(ids, expected, splits)
    receipt = read_receipt(args.cache, info)
    local = args.local_archive or args.dataset.parent / "a2_lite_archive_local" / ARCHIVE
    if local.resolve().is_relative_to(args.cache.resolve()) or local.resolve().is_relative_to(Path("/content/drive")):
        raise RuntimeError("Copy the archive to local disk first; do not extract directly from Drive")
    transfer = copy_archive(args.cache / ARCHIVE, local, receipt["archive"])
    verified = verify_archive(local, args.dataset, ids, expected, splits, info)
    report = dict(complete=True, revision=REVISION, cache_id=receipt["cache_id"], dataset=str(args.dataset),
                  local_archive=str(local), archive_sha256=receipt["archive"]["sha256"],
                  transfer=transfer, extraction=verified, train_frames=len(splits["train"]),
                  val_frames=len(splits["val"]), every_input_hash_verified=True,
                  training_recipe_changed=False, gpu_used=False)
    # A local operational receipt; never edits experiment manifests/checkpoints.
    destination = args.dataset / "a2_lite_archive_restore.json"
    with tempfile.NamedTemporaryFile(dir=args.dataset, prefix=".cache_receipt_", delete=False) as handle:
        temporary = Path(handle.name); handle.write(json_bytes(report))
    temporary.replace(destination)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "restore"))
    for name in ("pilot", "cache", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--local-archive", type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(REVISION, flush=True)
    {"prepare": prepare, "restore": restore}[args.stage](args)


if __name__ == "__main__":
    main()
