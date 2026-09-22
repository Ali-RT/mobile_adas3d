"""Cache one frozen model on all train images, in its own upstream process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from m61_common import (OUTPUT_KEYS, array_hash, build_runtime, check_split, environment,
                        input_fingerprint, load_manifest, npz_read, npz_write,
                        pack_targets, read_json, seed_all, sha256, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--role", choices=("teacher", "student"), required=True)
    args = p.parse_args()
    m = load_manifest(args.manifest)
    seed_all(m["seed"])
    model, criterion, dataset = build_runtime(m, args.role)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    directory = Path(m["output_dir"]) / "cache" / args.role
    directory.mkdir(parents=True, exist_ok=True)
    provenance = dict(manifest_sha256=m["manifest_sha256"], role=args.role, environment=environment())
    identity_path = directory / "identity.json"
    if identity_path.exists() and read_json(identity_path) != provenance:
        raise RuntimeError("Existing cache identity/environment differs; use a fresh experiment directory")
    write_json(identity_path, provenance)
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    previous_report = read_json(directory / "cache_manifest.json") if (directory / "cache_manifest.json").exists() else None
    files = {}
    for index, (images, calibs, raw, info) in enumerate(loader):
        image_id = f"{int(info['img_id'][0]):06d}"
        if image_id != ids[index]:
            raise RuntimeError("Dataset order differs from train split")
        target = pack_targets(raw, "cpu")[0]
        target_hash = array_hash(target)
        input_hash = input_fingerprint(images[0], calibs[0], raw["img_size"][0])
        path = directory / f"{image_id}.npz"
        existing = npz_read(path) if path.exists() else None
        if existing is not None:
            if previous_report and previous_report["files"].get(image_id) != sha256(path):
                raise RuntimeError(f"Previously completed cache was changed: {image_id}")
            if (str(existing["input_sha256"]) != input_hash
                    or str(existing["target_sha256"]) != target_hash
                    or str(existing["manifest_sha256"]) != m["manifest_sha256"]):
                raise RuntimeError(f"Changed cache input/target/provenance: {image_id}")
        else:
            gpu_target = [{k: v.cuda() for k, v in target.items()}]
            with torch.no_grad():
                outputs = model(images.cuda(), calibs.cuda(), gpu_target, raw["img_size"].cuda(), dn_args=None)
                if any(not torch.isfinite(outputs[k]).all() for k in OUTPUT_KEYS):
                    raise RuntimeError(f"Non-finite {args.role} outputs: {image_id}")
                qi, gi = criterion.matcher(outputs, gpu_target, group_num=1)[0]
            arrays = {"out_" + k: outputs[k][0].cpu().numpy() for k in OUTPUT_KEYS}
            arrays.update({"tgt_" + k: v.numpy() for k, v in target.items()})
            arrays.update(input_sha256=np.asarray(input_hash), target_sha256=np.asarray(target_hash),
                          manifest_sha256=np.asarray(m["manifest_sha256"]),
                          query_indices=qi.cpu().numpy(), gt_indices=gi.cpu().numpy(),
                          calibration=calibs[0].numpy(), image_size=raw["img_size"][0].numpy())
            npz_write(path, arrays)
        files[image_id] = sha256(path)
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"{args.role}: cached/verified {index + 1}/{len(ids)}", flush=True)
    if {p.stem for p in directory.glob("*.npz")} != set(ids):
        raise RuntimeError("Unexpected cached IDs (validation leakage or stale files)")
    report = dict(schema_version=1, complete=True, split="train", samples=len(ids),
                  optimizer_steps=0, augmentation=False, files=files, **provenance)
    write_json(directory / "cache_manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
