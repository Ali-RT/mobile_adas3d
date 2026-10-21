"""Frozen inputs and reusable small archives for parking candidate screens.

Standard library only: prepare the same 64 VAL frames before allocating a GPU.
This is an intentionally enriched engineering subset, not a full KITTI score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/parking_edge_candidates_v1.json"
PRODUCT = {"Car": "Vehicle", "Van": "Vehicle", "Truck": "Vehicle", "Tram": "Vehicle",
           "Pedestrian": "Pedestrian", "Person_sitting": "Pedestrian"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def identity(value):
    return sha(encoded(value))


def immutable(path, raw):
    """Publish atomically; never replace a conflicting completed artifact."""
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"Preserve symlinked output: {path}")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"Preserve changed output; use a new RUN_ID: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".parking_", delete=False) as f:
        temporary = Path(f.name)
        f.write(raw)
    try:
        # Drive FUSE does not consistently implement hard links. A single
        # notebook owns each run; recheck immediately before atomic rename.
        if path.exists():
            if path.read_bytes() != raw:
                raise RuntimeError(f"Conflicting concurrent publication: {path}")
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retry_read(path):
    for n in range(3):
        try:
            return Path(path).read_bytes()
        except OSError:
            if n == 2:
                raise
            time.sleep(.5 * (n + 1))


def load_config(path=CONFIG):
    c = json.loads(Path(path).read_text())
    if c["classes"] != ["Vehicle", "Pedestrian"] or c["frames"] != 64:
        raise ValueError("Use the separate fixed64 parking screen contract")
    return c


def band(z):
    if not math.isfinite(z) or not 2 <= z <= 6:
        return None
    return min(int(z), 5)


def targets(raw):
    rows = []
    for number, line in enumerate(raw.decode().splitlines(), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 15:
            raise ValueError(f"KITTI label row {number}: expected 15 fields")
        if p[0] not in PRODUCT:
            continue
        v = [float(x) for x in p[1:]]
        if not all(math.isfinite(x) for x in v) or v[5] <= v[3] or v[6] <= v[4]:
            raise ValueError(f"Invalid KITTI target row {number}")
        rows.append(dict(class_name=PRODUCT[p[0]], source_class=p[0], bbox=v[3:7],
                         camera_center_z_m=v[12], truncated=v[0], occluded=int(v[1]), row=number))
    return rows


def select(ids, labels, c):
    rank = lambda s: (sha((c["seed"] + s).encode()), s)
    ranked = sorted(ids, key=rank)
    buckets = {(cls, b): [] for cls in c["classes"] for b in range(2, 6)}
    for sample in ranked:
        cells = {(r["class_name"], band(r["camera_center_z_m"])) for r in targets(labels[sample])}
        for cell in cells:
            if cell in buckets:
                buckets[cell].append(sample)
    chosen = []
    for _ in range(c["frames_per_class_band"]):
        for values in buckets.values():
            next_id = next((s for s in values if s not in chosen), None)
            if next_id is not None:
                chosen.append(next_id)
    for sample in ranked:
        if len(chosen) >= c["frames"]:
            break
        if sample not in chosen:
            chosen.append(sample)
    if len(chosen) != c["frames"]:
        raise ValueError("Not enough unique validation frames for the fixed screen")
    return chosen, {f"{cls}_{b}_{b+1}m": len(v) for (cls, b), v in buckets.items()}


def verify_manifest(m, c):
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if (m.get("manifest_sha256") != identity(body) or m.get("config_sha256") != identity(c)
            or len(m.get("ids", [])) != c["frames"] or len(set(m["ids"])) != c["frames"]
            or any(not re.fullmatch(r"[0-9]{6}", s) for s in m["ids"])):
        raise RuntimeError("Input cache identity changed; preserve it and use a new cache folder")
    expected = {f"training/{kind}/{s}{suffix}" for s in m["ids"]
                for kind, suffix in (("image_2", ".png"), ("label_2", ".txt"))}
    if set(m["files"]) != expected:
        raise RuntimeError("Unexpected cache file inventory")
    # Record the producing helper's hash, but do not make an I/O-only bugfix
    # invalidate reusable image bytes. The frozen config, member inventory and
    # each byte hash are revalidated on every restore. Model runs separately
    # bind their own implementation and runtime identities.
    return m


def restore(cache, dataset, c):
    m = verify_manifest(json.loads((cache / "input_manifest.json").read_text()), c)
    raw = retry_read(cache / "inputs64.zip")
    if sha(raw) != m["archive_sha256"]:
        raise RuntimeError("Input archive changed; preserve it")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(m["files"]):
            raise RuntimeError("Unexpected or duplicate archive member")
        for name, expected in m["files"].items():
            value = archive.read(name)
            if sha(value) != expected:
                raise RuntimeError(f"Changed archive member: {name}")
            destination = dataset / name
            if not destination.resolve().is_relative_to(dataset.resolve()):
                raise RuntimeError("Staging path escapes the local dataset")
            immutable(destination, value)
    immutable(dataset / "input_manifest.json", encoded(m))
    print(f"Restored/verified {len(m['ids'])} frames from one reusable archive; no GPU used.", flush=True)
    return m


def prepare(args):
    c = load_config(args.config)
    if args.local_data.is_symlink() or args.cache.is_symlink():
        raise ValueError("Preserve symlinked data or cache folders; choose explicit private folders")
    cache, dataset = args.cache.resolve(), args.local_data.resolve()
    if dataset.is_relative_to(cache) or cache.is_relative_to(dataset) or dataset.is_relative_to(Path("/content/drive")):
        raise ValueError("Keep local data off Drive and separate from the shared cache")
    source = args.dataset_root.resolve()
    if cache.is_relative_to(source) or dataset.is_relative_to(source):
        raise ValueError("The source dataset is read-only; use separate local data and cache folders")
    if (cache / "input_manifest.json").exists():
        return restore(cache, dataset, c)
    split_raw = retry_read(args.split)
    ids = split_raw.decode().split()
    if (len(ids) != c["split_images"] or len(ids) != len(set(ids))
            or sha(split_raw) != c["split_sha256"]
            or any(not re.fullmatch(r"[0-9]{6}", s) for s in ids)):
        raise ValueError("Expected the unchanged Chen val3769 split")
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1 to 8 I/O workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        labels = {}
        paths = [args.dataset_root / "training/label_2" / (s + ".txt") for s in ids]
        for n, (sample, raw) in enumerate(zip(ids, pool.map(retry_read, paths)), 1):
            labels[sample] = raw
            if n % 500 == 0 or n == len(ids):
                print(f"Indexed labels {n}/{len(ids)}", flush=True)
        selected, capacity = select(ids, labels, c)
        def stage_one(sample):
            result = {}
            for kind, suffix in (("image_2", ".png"), ("label_2", ".txt")):
                name = f"training/{kind}/{sample}{suffix}"
                raw = labels[sample] if kind == "label_2" else retry_read(args.dataset_root / name)
                destination = dataset / name
                if not destination.resolve().is_relative_to(dataset):
                    raise RuntimeError("Staging symlink escapes local data")
                immutable(destination, raw)
                result[name] = sha(raw)
            return result
        hashes = {}
        for n, values in enumerate(pool.map(stage_one, selected), 1):
            hashes.update(values)
            if n % 16 == 0:
                print(f"Staged/verified {n}/{len(selected)} images and labels", flush=True)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(hashes):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0)), (dataset / name).read_bytes())
    raw = archive_bytes.getvalue()
    m = dict(schema_version=1, revision=c["revision"], config_sha256=identity(c),
             implementation_sha256=file_sha(Path(__file__)), split_sha256=sha(split_raw),
             label_inventory_sha256=identity({s: sha(labels[s]) for s in sorted(labels)}),
             ids=selected, available_frames_per_cell=capacity, files=hashes,
             archive_sha256=sha(raw), selection=c["selection"],
             representative_full_validation=False, gpu_used=False, training_performed=False)
    m["manifest_sha256"] = identity(m)
    immutable(cache / "inputs64.zip", raw)
    immutable(cache / "input_manifest.json", encoded(m))
    immutable(dataset / "input_manifest.json", encoded(m))
    print(f"Input cache ready: {cache}; 64 enriched validation frames, not a full-VAL benchmark.", flush=True)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--local-data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--workers", type=int, default=4)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
