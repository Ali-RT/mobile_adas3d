"""M62: frozen R0/A2 train-only diagnosis; no training or deployment authorization."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from audit_m61_teacher import audit_sample
from m61_common import (COMPONENTS, NATIVE_CLASSES, OUTPUT_KEYS, ROOT, STUDENT_COMMIT,
                        array_hash, check_split, environment, input_fingerprint,
                        json_hash, npz_read, npz_write, pack_targets, read_json,
                        seed_all, sha256, source_hash, write_json)
from prepare_m61_distillation import make_config
from prepare_monodetr_a2_student import resolve_r0_selection
from prepare_monodetr_a2g_vehicle_geometry_gate import resolve_a2_checkpoint

REVISION = "M62-2026-09-28-r1"
POLICY = dict(min_score=0.3, min_iou=0.5, max_depth_exclusive=60.0,
              individual_error_ratio_max=0.95)
TIE_ORDER = ("depth", "center", "dimensions", "angle")


def implementation_hash():
    paths = [Path(__file__), ROOT / "scripts/audit_m61_teacher.py",
             ROOT / "scripts/m61_common.py", ROOT / "scripts/prepare_m61_distillation.py",
             ROOT / "scripts/prepare_monodetr_a2_student.py",
             ROOT / "scripts/prepare_monodetr_a2g_vehicle_geometry_gate.py"]
    return json_hash({p.name: sha256(p) for p in paths})


def require_identity(saved, current):
    if saved != current:
        changes = {k: dict(saved=saved.get(k), current=current.get(k))
                   for k in sorted(set(saved) | set(current)) if saved.get(k) != current.get(k)}
        raise RuntimeError("Saved identity differs; preserve existing files and restore the environment "
                           "or choose a new output folder:\n" + json.dumps(changes, indent=2))


def retry_read(path):
    for attempt in range(3):
        try:
            return npz_read(path)
        except OSError as exc:
            if exc.errno != 5 or attempt == 2:
                raise RuntimeError(f"Cannot read cache {path}; preserve it for inspection: {exc}") from exc
            print(f"Temporary I/O error: {path}; retry {attempt + 1}/2", flush=True)
            time.sleep(attempt + 1)


def prepare(args):
    repo, data, output = args.repo.resolve(), args.dataset_root.resolve(), args.output.resolve()
    if environment()["gpu"] is None or environment()["timm"] != "1.0.20":
        raise RuntimeError("Use CUDA and timm==1.0.20")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != STUDENT_COMMIT:
        raise RuntimeError("Wrong pinned MonoDETR source commit")
    ids = {s: check_split(data / "ImageSets" / f"{s}.txt", s) for s in ("train", "val")}
    if set(ids["train"]) & set(ids["val"]):
        raise RuntimeError("Training/validation overlap")
    for image_id in ids["train"]:
        for folder, suffix in (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt")):
            p = data / "training" / folder / (image_id + suffix)
            if not p.is_file():
                raise FileNotFoundError(p)
    teacher, _ = resolve_r0_selection(args.r0_selection)
    student, _ = resolve_a2_checkpoint(args.a2_selection)
    base = yaml.safe_load((repo / "configs/monodetr.yaml").read_text())
    models = {}
    for role, checkpoint in (("teacher", teacher), ("student", student)):
        cfg = make_config(base, data, role, output, repo)
        models[role] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), config=cfg)
    manifest = dict(schema_version=1, revision=REVISION, repo=str(repo), dataset_root=str(data),
                    output=str(output), implementation_sha256=implementation_hash(),
                    source_sha256=source_hash(repo), environment=environment(), models=models,
                    policy=POLICY, split="train", samples=3712, optimizer_steps=0,
                    training_authorized=False, deployment_authorized=False,
                    component_rule=dict(min_improved_objects=100, min_objects_per_distance_bin=25,
                                        min_distance_bins=2, tie_order=list(TIE_ORDER)))
    manifest["manifest_sha256"] = json_hash(manifest)
    path = output / "m62_manifest.json"
    if path.exists():
        require_identity(read_json(path), manifest)
    write_json(path, manifest)
    print(json.dumps(manifest, indent=2))


def load(path):
    m = read_json(path)
    signature = m.pop("manifest_sha256")
    if signature != json_hash(m) or m["revision"] != REVISION:
        raise RuntimeError("Manifest changed")
    m["manifest_sha256"] = signature
    if m["implementation_sha256"] != implementation_hash() or m["source_sha256"] != source_hash(m["repo"]):
        raise RuntimeError("Frozen code/source changed; do not mix runs")
    require_identity(m["environment"], environment())
    for entry in m["models"].values():
        if sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
    check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    return m


def runtime(m, role):
    import torch
    repo = Path(m["repo"])
    sys.path[:0] = [str(repo), str(repo / "lib/models/monodetr/ops")]
    from lib.helpers.model_helper import build_model
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    cfg = m["models"][role]["config"]
    dataset = KITTI_Dataset("train", cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != NATIVE_CLASSES or np.any(dataset.cls_mean_size):
        raise RuntimeError("Class IDs or dimension encoding changed")
    seed_all(20268)
    model, criterion = build_model(cfg["model"])
    # Only the known checkpoint whose exact SHA was checked by load().
    payload = torch.load(m["models"][role]["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    return model.cuda().eval(), criterion.cuda().eval(), dataset


def probe(m):
    import torch
    model, _, dataset = runtime(m, "student")
    images, calibs, raw, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    targets = pack_targets(raw, "cuda")
    images, calibs, sizes = images.cuda(), calibs.cuda(), raw["img_size"].cuda()
    durations = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(images, calibs, targets, sizes, dn_args=None)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if any(not bool(torch.isfinite(outputs[k]).all()) for k in OUTPUT_KEYS):
                raise RuntimeError("A2 native forward has non-finite outputs")
            if i >= 5:
                durations.append(elapsed)
    custom = [dict(name=n, type=type(module).__name__) for n, module in model.named_modules()
              if "MSDeformAttn" in type(module).__name__]
    report = dict(schema_version=1, complete=True, manifest_sha256=m["manifest_sha256"],
                  model="A2 epoch130", environment=m["environment"], batch_size=1,
                  input_shape=list(images.shape), warmups=5, measured_runs=10,
                  native_cuda_model_only_ms=dict(mean=float(np.mean(durations)),
                                                  median=float(np.median(durations)),
                                                  p95=float(np.percentile(durations, 95))),
                  parameters=sum(p.numel() for p in model.parameters()),
                  parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  output_shapes={k: list(outputs[k].shape) for k in OUTPUT_KEYS},
                  native_custom_attention_modules=custom,
                  export_blocker="Native custom CUDA attention requires a verified portable implementation"
                  if custom else "Module inventory is not proof of export compatibility",
                  coreml_conversion_tested=False, iphone_latency_measured=False,
                  iphone_latency_estimate_ms=None, optimizer_steps=0, training_authorized=False)
    write_json(Path(m["output"]) / "m62_a2_compatibility.json", report)
    print(json.dumps(report, indent=2))


def cache(m, role):
    import torch
    directory = Path(m["output"]) / "cache" / role
    identity = dict(manifest_sha256=m["manifest_sha256"], role=role, environment=m["environment"])
    if (directory / "identity.json").exists():
        require_identity(read_json(directory / "identity.json"), identity)
    write_json(directory / "identity.json", identity)
    previous = read_json(directory / "complete.json") if (directory / "complete.json").exists() else None
    model, criterion, dataset = runtime(m, role)
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    files = {}
    for i, (images, calibs, raw, info) in enumerate(torch.utils.data.DataLoader(dataset, batch_size=1)):
        image_id = f"{int(info['img_id'][0]):06d}"
        if image_id != ids[i]:
            raise RuntimeError("Unexpected dataset order")
        target = pack_targets(raw, "cpu")[0]
        metadata = dict(input_sha256=input_fingerprint(images[0], calibs[0], raw["img_size"][0]),
                        target_sha256=array_hash(target), manifest_sha256=m["manifest_sha256"])
        path = directory / f"{image_id}.npz"
        if path.exists():
            old = retry_read(path)
            require_identity({k: str(old[k]) for k in metadata}, metadata)
            if previous and previous["files"].get(image_id) != sha256(path):
                raise RuntimeError(f"Completed cache file changed: {path}")
        else:
            gt = [{k: v.cuda() for k, v in target.items()}]
            with torch.no_grad():
                out = model(images.cuda(), calibs.cuda(), gt, raw["img_size"].cuda(), dn_args=None)
                if any(not bool(torch.isfinite(out[k]).all()) for k in OUTPUT_KEYS):
                    raise RuntimeError(f"Non-finite {role} output: {image_id}")
                qi, gi = criterion.matcher(out, gt, group_num=1)[0]
            arrays = {"out_" + k: out[k][0].cpu().numpy() for k in OUTPUT_KEYS}
            arrays.update({"tgt_" + k: v.numpy() for k, v in target.items()})
            arrays.update({k: np.asarray(v) for k, v in metadata.items()})
            arrays.update(query_indices=qi.cpu().numpy(), gt_indices=gi.cpu().numpy(),
                          calibration=calibs[0].numpy(), image_size=raw["img_size"][0].numpy())
            npz_write(path, arrays)
        files[image_id] = sha256(path)
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"{role}: cached/verified {i + 1}/{len(ids)}", flush=True)
    if {p.stem for p in directory.glob('*.npz')} != set(ids):
        raise RuntimeError("Unexpected cache IDs")
    write_json(directory / "complete.json", dict(complete=True, files=files, **identity))


def summarize(rows):
    result = {}
    for component in COMPONENTS:
        t = np.asarray([r["teacher"][component] for r in rows])
        s = np.asarray([r["student"][component] for r in rows])
        win = t < POLICY["individual_error_ratio_max"] * s
        bins = {}
        for lo, hi in ((2, 20), (20, 40), (40, 60)):
            selected = np.asarray([lo <= r["depth"] < hi for r in rows], dtype=bool)
            bins[f"{lo}_{hi}m"] = dict(paired=int(selected.sum()), improved=int((win & selected).sum()))
        eligible = int(win.sum()) >= 100 and sum(b["improved"] >= 25 for b in bins.values()) >= 2
        result[component] = dict(paired_objects=len(rows), improved_objects=int(win.sum()),
                                 teacher_mean_error=float(t.mean()) if len(t) else None,
                                 a2_mean_error=float(s.mean()) if len(s) else None,
                                 teacher_median_error=float(np.median(t)) if len(t) else None,
                                 a2_median_error=float(np.median(s)) if len(s) else None,
                                 distance_bins=bins, eligible_for_review=eligible)
    eligible = [c for c in TIE_ORDER if result[c]["eligible_for_review"]]
    proposal = max(eligible, key=lambda c: result[c]["improved_objects"]) if eligible else None
    return result, proposal


def analyze(m):
    root = Path(m["output"])
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    reports = {}
    for role in ("teacher", "student"):
        directory = root / "cache" / role
        r = read_json(directory / "complete.json")
        if not r["complete"] or set(r["files"]) != set(ids) or {p.stem for p in directory.glob('*.npz')} != set(ids):
            raise RuntimeError("Incomplete/non-training cache")
        require_identity({k: r[k] for k in ("role", "environment", "manifest_sha256")},
                         dict(role=role, environment=m["environment"], manifest_sha256=m["manifest_sha256"]))
        reports[role] = r
    rows = []
    total_vehicle = 0
    for i, image_id in enumerate(ids):
        samples = {}
        for role in reports:
            path = root / "cache" / role / f"{image_id}.npz"
            if sha256(path) != reports[role]["files"][image_id]:
                raise RuntimeError(f"Cache hash changed: {path}")
            samples[role] = retry_read(path)
        _, _, matched = audit_sample(samples["teacher"], samples["student"], POLICY)
        total_vehicle += int((samples['teacher']['tgt_labels'] == 1).sum())
        for row in matched:
            row.update(image_id=image_id, depth=float(samples['teacher']['tgt_depth'][row['gt'], 0]))
        rows.extend(matched)
        if (i + 1) % 500 == 0:
            print(f"Diagnosed {i + 1}/{len(ids)}", flush=True)
    components, proposal = summarize(rows)
    report = dict(schema_version=1, complete=True, revision=REVISION,
                  manifest_sha256=m["manifest_sha256"], split="train", samples=len(ids),
                  total_usable_vehicle_gt=total_vehicle, quality_filtered_paired_objects=len(rows),
                  components=components, proposed_component_for_review=proposal,
                  rule=m["component_rule"], global_teacher_mean_superiority_required=False,
                  units=dict(depth="m", dimensions="mean absolute H/W/L m", center="XYZ L2 m", angle="wrapped alpha degrees"),
                  training_authorized=False, deployment_authorized=False, optimizer_steps=0,
                  source_cache_sha256={role: sha256(root / 'cache' / role / 'complete.json') for role in reports},
                  limitation="Train-set teacher wins identify candidate supervision, not proof of validation benefit")
    write_json(root / "m62_geometry_pairs.json", rows)
    report['geometry_pairs_sha256'] = sha256(root / "m62_geometry_pairs.json")
    write_json(root / "m62_diagnostic.json", report)
    print(json.dumps(report, indent=2))
    print("STOP for review. No distillation training is authorized by this report.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "probe", "cache", "analyze"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--r0-selection", type=Path)
    p.add_argument("--a2-selection", type=Path)
    p.add_argument("--role", choices=("teacher", "student"))
    args = p.parse_args()
    if args.action == "prepare":
        if not all((args.repo, args.dataset_root, args.output, args.r0_selection, args.a2_selection)):
            p.error("prepare requires repo, dataset-root, output, r0-selection, and a2-selection")
        prepare(args)
    else:
        if args.manifest is None or (args.action == "cache" and args.role is None):
            p.error("manifest required; cache also requires role")
        m = load(args.manifest)
        if args.action == "cache":
            cache(m, args.role)
        elif args.action == "probe":
            probe(m)
        else:
            analyze(m)


if __name__ == "__main__":
    main()
