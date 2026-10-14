"""Faster I/O-only KITTI staging; no torch, GPU, or experiment mutations.

Use the exact reviewed512 manifest as the byte-level data oracle. Scan each
candidate directory once, copy/hash with a bounded thread pool, and reuse
only hash-verified local files. The unchanged geometry prepare stage still
verifies the complete staged dataset before it authorizes training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

REVISION = "A2-LITE-IO-STAGING-2026-10-14-r1"
PILOT_MANIFEST_SHA = "4d3a4fde2fc6ad2327c5089bc621508e1d49560cb29d3b2460e2be9072cf2d0f"
KINDS = (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reviewed_data(source):
    if source.is_dir():
        raw = (source / "recovery_manifest.json").read_bytes()
    else:
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n == "recovery_manifest.json" or n.endswith("/recovery_manifest.json")]
            if len(names) != 1:
                raise RuntimeError("Expected one original512 pilot manifest")
            raw = archive.read(names[0])
    if digest(raw) != PILOT_MANIFEST_SHA:
        raise RuntimeError("Use the original reviewed512 pilot, not full-recovery results")
    value = json.loads(raw)
    ids = value["train_ids"] + value["val_ids"]
    expected = value["data_file_sha256"]
    if (len(ids) != 7481 or len(set(ids)) != 7481 or set(expected) != set(ids)
            or any(not re.fullmatch(r"[0-9]{6}", sample) for sample in ids)
            or any(set(record) != {k for k, _ in KINDS}
                   or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in record.values())
                   for record in expected.values())):
        raise RuntimeError("Invalid reviewed train3712 + val3769 data inventory")
    return ids, expected


def retry_read(path):
    for attempt in range(3):
        try:
            return path.read_bytes()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


def atomic_copy(destination, data):
    """Never overwrite conflicting data; partial writes stay temporary."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"Preserve symlinked staging destination: {destination}")
    if destination.exists():
        if retry_read(destination) != data:
            raise RuntimeError(f"Preserve conflicting staged file: {destination}")
        return False
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".staging_", delete=False) as handle:
            temporary = Path(handle.name); handle.write(data)
        # Linking is an atomic, no-clobber publication on local disk. Unlike
        # replace(), it cannot overwrite a file created by another process.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if retry_read(destination) != data:
                raise RuntimeError(f"Preserve conflicting staged file: {destination}")
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inventories(dataset, candidates):
    result = {kind: [] for kind, _ in KINDS}
    seen = set()
    for root in (dataset, *candidates):
        for kind, _ in KINDS:
            aliases = (kind, kind[:-1] + "02") if kind in ("image_2", "label_2") else (kind,)
            for alias in aliases:
                directory = root / "training" / alias
                key = str(directory)
                if key in seen or not directory.is_dir():
                    continue
                seen.add(key)
                print(f"Indexing {directory}", flush=True)
                # Directory names are cheap to collect once; opening/hashing is
                # the verification. Avoid thousands of per-file Drive stats.
                with os.scandir(directory) as entries:
                    names = {entry.name for entry in entries}
                result[kind].append((directory, names))
    return result


def stage(dataset, candidates, ids, expected, workers=4):
    if not dataset.name.startswith("kitti_a2_lite_accuracy_"):
        raise RuntimeError("Use a private kitti_a2_lite_accuracy_<RUN_ID> folder")
    if dataset.is_symlink() or dataset.resolve().is_relative_to(Path("/content/drive")):
        raise RuntimeError("Stage onto local disk, not a symlink or Google Drive")
    if not 1 <= workers <= 8:
        raise ValueError("Use 1–8 I/O workers (default 4)")
    for kind, _ in KINDS:
        parent = dataset / "training" / kind
        if not parent.resolve().is_relative_to(dataset.resolve()):
            raise RuntimeError(f"Preserve staging directory pointing outside local dataset: {parent}")
    sources = inventories(dataset, candidates)
    missing = [(sample, kind) for sample in ids for kind, suffix in KINDS
               if not any(sample + suffix in names for _, names in sources[kind])]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} input files; none copied. First missing: {missing[:5]}")
    started = time.monotonic()
    print(f"Staging {len(ids)} frames / {len(ids) * 3} files with {workers} I/O workers; GPU is not used.", flush=True)

    def one(sample):
        copied = reused = size = 0
        for kind, suffix in KINDS:
            destination = dataset / "training" / kind / (sample + suffix)
            if destination.is_symlink():
                raise RuntimeError(f"Preserve symlinked staged file: {destination}")
            if destination.exists():
                data = retry_read(destination)
                if digest(data) != expected[sample][kind]:
                    raise RuntimeError(f"Preserve conflicting staged file: {destination}; no overwrite")
                reused += 1
                continue
            matches = [directory / (sample + suffix) for directory, names in sources[kind]
                       if sample + suffix in names]
            found = False
            for candidate in matches:
                data = retry_read(candidate)
                if digest(data) == expected[sample][kind]:
                    changed = atomic_copy(destination, data)
                    copied += int(changed); reused += int(not changed)
                    size += len(data) if changed else 0
                    found = True
                    break
            if not found:
                raise RuntimeError(f"No byte-identical reviewed source for {sample}/{kind}; no bad data copied")
        return copied, reused, size

    copied = reused = size = completed = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [pool.submit(one, sample) for sample in ids]
        for future in as_completed(futures):
            c, r, b = future.result(); copied += c; reused += r; size += b; completed += 1
            if completed % 100 == 0 or completed == len(ids):
                elapsed = max(time.monotonic() - started, .001)
                remaining = elapsed * (len(ids) - completed) / completed
                print(f"Staged/verified {completed}/{len(ids)} frames; {copied} files copied, {reused} reused; "
                      f"{completed / elapsed:.1f} frames/s; estimated remaining {remaining / 60:.1f} min", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return dict(complete=True, revision=REVISION, frames=len(ids), files=len(ids) * 3,
                copied_files=copied, reused_files=reused, copied_bytes=size,
                elapsed_seconds=time.monotonic() - started, workers=workers,
                every_file_sha256_verified=True, gpu_used=False,
                pilot_manifest_sha256=PILOT_MANIFEST_SHA, training_recipe_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ids, expected = reviewed_data(args.pilot)
    report = stage(args.dataset, args.candidate, ids, expected, args.workers)
    # Operational receipt only; the unchanged model prepare stage separately
    # checks the exact data hashes before signing its training manifest.
    raw = (json.dumps(report, indent=2) + "\n").encode()
    destination = args.dataset / "a2_lite_io_staging.json"
    with tempfile.NamedTemporaryFile(dir=args.dataset, prefix=".receipt_", delete=False) as handle:
        temporary = Path(handle.name); handle.write(raw)
    temporary.replace(destination)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
