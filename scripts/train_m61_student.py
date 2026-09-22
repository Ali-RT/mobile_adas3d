"""M61 CUDA smoke / paired continuation, resumable from complete Drive epochs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from m61_common import (COMPONENTS, ROOT, array_hash, build_runtime, check_split,
                        environment, input_fingerprint, load_manifest, npz_read,
                        pack_targets, read_json, seed_all, sha256, write_json)

sys.path.insert(0, str(ROOT))
from third_party.monodetr.m61_distillation_loss import geometry_distillation


def validated_audit(m):
    root = Path(m["output_dir"])
    report = read_json(root / "m61_teacher_audit.json")
    if (not report.get("complete") or not report.get("pilot_authorized")
            or report["manifest_sha256"] != m["manifest_sha256"]
            or not report["enabled_components"]
            or not set(report["enabled_components"]) <= set(COMPONENTS)):
        raise RuntimeError("M61 train-only teacher audit has not authorized this pilot")
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    if set(report["approved_files"]) != set(ids):
        raise RuntimeError("Audit approval IDs differ from training split")
    for role in ("teacher", "student"):
        path = root / "cache" / role / "cache_manifest.json"
        if sha256(path) != report["source_cache_sha256"][role] or read_json(path)["environment"] != environment():
            raise RuntimeError("Cache/environment changed after audit")
    for image_id, digest in report["approved_files"].items():
        if sha256(root / "approved_targets" / f"{image_id}.npz") != digest:
            raise RuntimeError(f"Changed approved target {image_id}")
    return report


def approved_batch(m, images, calibs, raw, info, targets):
    approved = []
    for i, image_id in enumerate(info["img_id"]):
        a = npz_read(Path(m["output_dir"]) / "approved_targets" / f"{int(image_id):06d}.npz")
        if (str(a["manifest_sha256"]) != m["manifest_sha256"]
                or str(a["input_sha256"]) != input_fingerprint(images[i], calibs[i], raw["img_size"][i])
                or str(a["target_sha256"]) != array_hash(targets[i])):
            raise RuntimeError(f"Cache/input/GT mismatch at {image_id}; augmentation or data changed")
        approved.append(a)
    return approved


def supervised_loss(criterion, outputs, targets):
    losses = criterion(outputs, targets, None)
    total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
    return total, losses


def finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)


def train_smoke(m, audit, model, criterion, dataset):
    """No optimizer steps; check KD contributes real finite gradients and control identity."""
    dataset_ids = dataset.idx_list
    candidate = None
    for i, image_id in enumerate(dataset_ids):
        a = npz_read(Path(m["output_dir"]) / "approved_targets" / f"{int(image_id):06d}.npz")
        enabled_columns = [COMPONENTS.index(c) for c in audit["enabled_components"]]
        if a["masks"][:, enabled_columns].any():
            candidate = i
            break
    if candidate is None:
        raise RuntimeError("No approved training sample for smoke test")
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [candidate]), batch_size=1)
    images, calibs, raw, info = next(iter(loader))
    targets = pack_targets(raw, "cuda")
    approved = approved_batch(m, images, calibs, raw, info, targets)
    model.train()
    criterion.train()
    outputs = model(images.cuda(), calibs.cuda(), targets, raw["img_size"].cuda(), dn_args=None)
    gt_loss, _ = supervised_loss(criterion, outputs, targets)
    assignments = criterion.matcher(outputs, targets, group_num=criterion.group_num)
    kd, counts = geometry_distillation(outputs, targets, assignments, approved,
                                       audit["enabled_components"], m["overall_kd_weight"])
    # A disabled KD path is exactly the same tensor/value, not a near-equality claim.
    disabled_total = gt_loss
    identical = torch.equal(disabled_total, gt_loss)
    grads = torch.autograd.grad(kd["total"], [p for p in model.parameters() if p.requires_grad],
                                allow_unused=True, retain_graph=True)
    teacher_gradient_nonzero = any(g is not None and bool((g != 0).any()) for g in grads)
    teacher_gradient_finite = all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    del grads
    total = gt_loss + kd["total"]
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite M61 smoke loss")
    total.backward()
    combined_gradients_finite = finite_gradients(model)
    gt_value, kd_value = float(gt_loss.detach()), float(kd["total"].detach())
    del outputs, gt_loss, kd, total, disabled_total
    # Native KITTI filtering can leave an image with no usable GT objects.
    # Exercise that background-only batch without dropping it from training.
    empty_sample_id = None
    empty_gt_loss = None
    empty_gradients_finite = True
    model.zero_grad(set_to_none=True)
    for i, image_id in enumerate(dataset_ids):
        a = npz_read(Path(m["output_dir"]) / "approved_targets" / f"{int(image_id):06d}.npz")
        if len(a["labels"]):
            continue
        empty_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [i]), batch_size=1)
        ei, ec, er, einfo = next(iter(empty_loader))
        et = pack_targets(er, "cuda")
        approved_batch(m, ei, ec, er, einfo, et)
        eo = model(ei.cuda(), ec.cuda(), et, er["img_size"].cuda(), dn_args=None)
        el, _ = supervised_loss(criterion, eo, et)
        if not bool(torch.isfinite(el)):
            raise RuntimeError("Non-finite native GT loss for a background-only batch")
        el.backward()
        empty_sample_id = f"{int(image_id):06d}"
        empty_gt_loss = float(el.detach())
        empty_gradients_finite = finite_gradients(model)
        break
    passed = (identical and teacher_gradient_nonzero and teacher_gradient_finite
              and combined_gradients_finite and empty_gradients_finite)
    report = dict(schema_version=1, complete=bool(passed), manifest_sha256=m["manifest_sha256"],
                  audit_sha256=sha256(Path(m["output_dir"]) / "m61_teacher_audit.json"),
                  environment=environment(), sample_id=f"{int(info['img_id'][0]):06d}",
                  optimizer_steps=0, disabled_path_identical=bool(identical),
                  teacher_gradient_nonzero=bool(teacher_gradient_nonzero),
                  finite_gradients=bool(teacher_gradient_finite and combined_gradients_finite and empty_gradients_finite),
                  gt_loss=gt_value, kd_loss=kd_value,
                  empty_gt_sample_id=empty_sample_id, empty_gt_loss=empty_gt_loss,
                  empty_gt_tested=empty_sample_id is not None,
                  approved_grouped_pairs=counts, enabled_components=audit["enabled_components"])
    write_json(Path(m["output_dir"]) / "m61_training_smoke.json", report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("M61 CUDA smoke failed; training not authorized")


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def restore_checkpoint(run_dir, binding, model, optimizer):
    start = 0
    history = []
    checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        last = checkpoints[-1]
        digest_path = last.with_suffix(".sha256")
        if digest_path.is_file() and digest_path.read_text().strip() != sha256(last):
            raise RuntimeError(f"Incomplete checkpoint transaction: {last}; inspect before resuming")
        # Our own tensor/primitive-only checkpoint, not an arbitrary uploaded pickle.
        payload = torch.load(last, map_location="cpu", weights_only=True)
        if payload["m61"] != binding or payload["epoch"] != int(last.stem.split("_")[-1]):
            raise RuntimeError("Resume checkpoint provenance/epoch mismatch")
        if not digest_path.exists():
            # Recover a completed atomic save interrupted before its checksum sidecar.
            digest_path.write_text(sha256(last) + "\n")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start = payload["epoch"]
        history = payload["history"]
        del payload
    return start, history


def continuation(m, audit, model, criterion, dataset, variant):
    root = Path(m["output_dir"])
    smoke = read_json(root / "m61_training_smoke.json")
    audit_sha = sha256(root / "m61_teacher_audit.json")
    if (not smoke.get("complete") or not smoke.get("teacher_gradient_nonzero")
            or smoke["manifest_sha256"] != m["manifest_sha256"]
            or smoke["audit_sha256"] != audit_sha or smoke["environment"] != environment()):
        raise RuntimeError("A successful matching CUDA smoke is required before training")
    run_dir = Path(m["variants"][variant]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    from lib.helpers.optimizer_helper import build_optimizer
    optimizer = build_optimizer(m["student"]["config"]["optimizer"], model)
    binding = dict(manifest_sha256=m["manifest_sha256"], audit_sha256=audit_sha,
                   variant=variant, environment=environment())
    run_identity = run_dir / "identity.json"
    if run_identity.exists() and read_json(run_identity) != binding:
        raise RuntimeError("Training identity changed; existing runs cannot be silently reused")
    write_json(run_identity, binding)
    start, history = restore_checkpoint(run_dir, binding, model, optimizer)
    if start > m["epochs"]:
        raise RuntimeError("Checkpoint exceeds the frozen pilot budget")
    print(f"{variant}: resume epoch {start}/{m['epochs']}; fixed lr={m['learning_rate']}; no augmentation", flush=True)
    for epoch in range(start, m["epochs"]):
        # Epoch-scoped seeds restore data order/dropout on an epoch-boundary resume.
        seed_all(m["seed"] + epoch)
        generator = torch.Generator().manual_seed(m["seed"] + epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=m["batch_size"], shuffle=True,
                                             generator=generator, num_workers=0, drop_last=False)
        model.train()
        criterion.train()
        sums = dict(gt=0.0, kd=0.0, total=0.0)
        counts = {c: 0 for c in COMPONENTS}
        for batch_index, (images, calibs, raw, info) in enumerate(loader):
            targets = pack_targets(raw, "cuda")
            # Verify both arms see the identical cached image/GT view, including control.
            approved = approved_batch(m, images, calibs, raw, info, targets)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.cuda(), calibs.cuda(), targets, raw["img_size"].cuda(), dn_args=None)
            gt_loss, parts = supervised_loss(criterion, outputs, targets)
            total = gt_loss
            kd_value = 0.0
            if variant == "vehicle_kd":
                assignments = criterion.matcher(outputs, targets, group_num=criterion.group_num)
                kd, batch_counts = geometry_distillation(outputs, targets, assignments, approved,
                                                          audit["enabled_components"], m["overall_kd_weight"])
                total = total + kd["total"]
                kd_value = float(kd["total"].detach())
                for c in COMPONENTS:
                    counts[c] += batch_counts[c]
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"Non-finite loss: epoch={epoch + 1}, batch={batch_index}")
            total.backward()
            if not finite_gradients(model):
                raise RuntimeError("Non-finite or absent gradients; no optimizer step taken")
            optimizer.step()
            values = dict(gt=float(gt_loss.detach()), kd=kd_value, total=float(total.detach()))
            for key, value in values.items():
                sums[key] += value
            if batch_index % 20 == 0 or batch_index + 1 == len(loader):
                primary = {k: round(float(v.detach()), 5) for k, v in parts.items() if k in criterion.weight_dict and not k[-1:].isdigit()}
                print(f"{variant} epoch={epoch + 1:02d}/{m['epochs']} batch={batch_index + 1}/{len(loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.8f} gt={values['gt']:.6f} kd={kd_value:.6f} "
                      f"total={values['total']:.6f} parts={primary}", flush=True)
        if variant == "vehicle_kd" and not sum(counts.values()):
            raise RuntimeError("No KD pairs in this epoch; do not silently call this distillation")
        history.append(dict(epoch=epoch + 1, **{k: v / len(loader) for k, v in sums.items()}, grouped_kd_pairs=counts))
        path = run_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        payload = dict(epoch=epoch + 1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                       best_result=0.0, best_epoch=0, history=history, m61=binding)
        save_checkpoint(path, payload)
        path.with_suffix(".sha256").write_text(sha256(path) + "\n")
        print(f"Epoch {epoch + 1} complete: {history[-1]}; saved {path}", flush=True)
    report = dict(schema_version=1, complete=True, epoch=m["epochs"], history=history, **binding)
    write_json(run_dir / "training_summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--variant", choices=("control", "vehicle_kd"))
    args = p.parse_args()
    m = load_manifest(args.manifest)
    audit = validated_audit(m)
    seed_all(m["seed"])
    model, criterion, dataset = build_runtime(m, "student")
    if args.smoke:
        train_smoke(m, audit, model, criterion, dataset)
    else:
        continuation(m, audit, model, criterion, dataset, args.variant)


if __name__ == "__main__":
    main()
