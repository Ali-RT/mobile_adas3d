"""Compact student/A2 pilot: prepare, cache-audit, smoke, train, bundle.

The reviewed epoch120 GT run is read-only. Both continuations start from its
same model weights, fresh AdamW and seed. Offline teacher targets are tied to
TRAIN images/labels/calibration; no validation target can enter KD. No model
architecture or export-time operation is added. CPU teacher/GPU student.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING
from data.compact_kitti import (CompactKittiDataset, CLASSES, project_keypoints,
                               rotation_target, valid_geometry, transform_points)
from data.kitti_parser import KittiObject, parse_kitti_calib_file, parse_kitti_label_file
from data.kitti_r40 import iou_3d
from models.compact_monocular3d import CompactMonocular3D
from models.compact_kitti_loss import (CompactKittiLoss, gather, geometry,
                                     decode_alpha, image_keypoints, solve_center)
from models.compact_distillation import (COMPONENTS, KD_WEIGHTS, box_iou, distillation_loss,
                                        match_teacher, qualification, usable_components, wrapped_error)
from scripts.compact_a2_teacher import A2Teacher, ONNX_SHA, materialize_export, teacher_inputs, decode_teacher
from scripts.train_compact_kitti import (atomic_checkpoint, atomic_json, config, digest_bytes,
                                        environment, evaluate, file_hash, identity, restore_rng,
                                        rng_state, setup_seed, to_device, worker_seed)

REVISION = "COMPACT-A2-SELECTIVE-KD-2026-10-11-r1"
SETTINGS = dict(seed=123, epochs=10, learning_rate=2e-5, batch_size=4,
                workers=2, evaluate_every=5, teacher_score_threshold=.1,
                teacher_match_iou=.5, improvement_ratio=.9,
                minimum_matches_per_class_component=16, freeze_bn_statistics=True,
                gt_weight=1., kd_weights=KD_WEIGHTS, temperature_sweep=False,
                classification_kd=False, feature_kd=False, architecture_changed=False)
CODE = ("scripts/distill_compact_a2.py", "scripts/compact_a2_teacher.py", "models/compact_distillation.py")


def signature(value):
    return digest_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Preserve changed experiment record; choose a new RUN_ID: {path}")
    else:
        atomic_json(path, value)


def code_hashes():
    return {name: file_hash(ROOT/name) for name in CODE}


def load(path):
    m = json.loads(path.read_text())
    saved_identity = m.pop("identity_sha256")
    if signature(m) != saved_identity: raise RuntimeError("Pilot manifest changed")
    m["identity_sha256"] = saved_identity
    if m["revision"] != REVISION or m["code_sha256"] != code_hashes():
        raise RuntimeError("Pilot code changed; preserve output and use a fresh RUN_ID")
    if file_hash(m["checkpoint"]) != m["checkpoint_sha256"]:
        raise RuntimeError("Frozen student checkpoint changed")
    for key in ("baseline_manifest", "baseline_selection"):
        if file_hash(m[key]) != m[key+"_sha256"]: raise RuntimeError(f"Changed {key}")
    original = json.loads(Path(m["baseline_manifest"]).read_text())
    if any(file_hash(ROOT/name) != value for name, value in original["code_sha256"].items()):
        raise RuntimeError("Original compact inference/training source changed")
    return m


def prepare(a):
    baseline = a.baseline.resolve()
    base_manifest = baseline/"training_manifest.json"
    selection_path = baseline/"selection.json"
    original = json.loads(base_manifest.read_text())
    selection = json.loads(selection_path.read_text())
    if not selection.get("complete") or selection.get("selected_epoch") != 120:
        raise RuntimeError("This pilot must start from the reviewed compact epoch120 selection")
    checkpoint = baseline/"checkpoints/epoch_120.pt"
    cfg = config(ROOT/"configs/compact_mnv4_kitti_gt_r1.yaml")
    # Reuse the GT run's data/code validation without mutating its receipts.
    checked, train, val = identity(argparse.Namespace(dataset_root=a.dataset_root, split_dir=a.split_dir,
                                                     output_dir=baseline), cfg)
    if checked["identity_sha256"] != original["identity_sha256"]:
        raise RuntimeError("The completed GT run's identity differs")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved["epoch"] != 120 or saved["identity_sha256"] != original["identity_sha256"]:
        raise RuntimeError("Wrong compact epoch120 checkpoint")
    model = CompactMonocular3D(backbone="mobilenetv4_conv_small")
    model.load_state_dict(saved["model"], strict=True)
    export = materialize_export(a.teacher_export.resolve(), a.output/"teacher_export")
    teacher = A2Teacher(export)
    fixtures = teacher.verify_fixtures(a.dataset_root)
    value = dict(revision=REVISION, complete=True, code_sha256=code_hashes(), settings=SETTINGS,
                 baseline_manifest=str(base_manifest), baseline_manifest_sha256=file_hash(base_manifest),
                 baseline_selection=str(selection_path), baseline_selection_sha256=file_hash(selection_path),
                 checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), checkpoint_epoch=120,
                 baseline_metrics=selection["metrics"], config=cfg, train_ids=train, val_ids=val,
                 dataset_root=str(a.dataset_root.resolve()), split_dir=str(a.split_dir.resolve()),
                 teacher_export=str(export.resolve()), teacher_onnx_sha256=ONNX_SHA,
                 teacher_fixture_parity_passed=True, architecture_changed=False,
                 output=str(a.output.resolve()))
    value["identity_sha256"] = signature(value)
    immutable_json(a.output/"pilot_manifest.json", value)
    atomic_json(a.output/"fixture_checks"/f"{time.time_ns()}.json",
                dict(complete=True, environment=environment(), rows=fixtures))
    print("Prepared: original GT run preserved; A2 exact-input runtime and reconstructed guidance verified; train/val disjoint.", flush=True)


def frozen_student(m, device):
    model = CompactMonocular3D(backbone="mobilenetv4_conv_small").to(device)
    saved = torch.load(m["checkpoint"], weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    return model.eval().requires_grad_(False)


def diagnose(m):
    """Fixed-threshold 2D/3D separation, NOT official 2D AP or a KD gate."""
    baseline = Path(m["baseline_manifest"]).parent
    path = baseline/"evaluations/epoch_120/predictions.json"
    predictions = json.loads(path.read_text())
    if set(predictions) != set(m["val_ids"]): raise RuntimeError("Incomplete original validation predictions")
    totals = {name: dict(gt=0, predictions=0, matched_2d=0, matched_3d=0,
                         depth_errors=[], dimension_errors=[], yaw_errors=[]) for name in CLASSES}
    for sample in m["val_ids"]:
        objects = parse_kitti_label_file(Path(m["dataset_root"])/"training/label_2"/f"{sample}.txt")
        # All valid mapped 3D GT, not KITTI difficulty filtering. Report this
        # explicitly; ignore-region handling differs from the AP evaluator.
        objects = [o for o in objects if KITTI_PRODUCTION_CLASS_MAPPING.get(o.class_name) and valid_geometry(o)]
        rows = [r for r in predictions[sample] if r["score"] >= .1]
        for obj in objects: totals[KITTI_PRODUCTION_CLASS_MAPPING[obj.class_name]]["gt"] += 1
        for row in rows: totals[row["class_name"]]["predictions"] += 1
        # Retain finite invalid boxes for 2D matching, to expose decoder failures.
        matches = match_teacher(objects, [dict(r, invalid_geometry=False) for r in rows], KITTI_PRODUCTION_CLASS_MAPPING)
        for gi, row in matches.items():
            obj = objects[gi]; result = totals[row["class_name"]]
            result["matched_2d"] += 1
            if not row["invalid_geometry"]:
                result["matched_3d"] += int(iou_3d(row, asdict(obj)) >= (.7 if row["class_name"] == "Vehicle" else .5))
            result["depth_errors"].append(abs(row["location_3d"][2]-obj.location_3d[2]))
            result["dimension_errors"].append(float(np.abs(np.asarray(row["dimensions_3d"])-obj.dimensions_3d).mean()))
            result["yaw_errors"].append(math.degrees(wrapped_error(row["rotation_y"], obj.rotation_y)))
    for r in totals.values():
        r["recall_2d"] = r["matched_2d"]/max(1,r["gt"])
        r["geometry_success_given_2d_match"] = r["matched_3d"]/max(1,r["matched_2d"])
        for k in ("depth_errors", "dimension_errors", "yaw_errors"):
            values = r.pop(k)
            r[k] = dict(count=len(values), median=float(np.median(values)) if values else None,
                         p90=float(np.quantile(values,.9)) if values else None)
    report = dict(complete=True, prediction_sha256=file_hash(path), samples=len(predictions), classes=totals,
                  score_threshold=.1, box_iou_threshold=.5, units=dict(depth="m", dimensions="mean H/W/L m", yaw="degrees"),
                  protocol="All valid product GT; class-aware greedy 2D matching; no difficulty/DontCare filtering; NOT AP",
                  validation_used_for_training=False, target_selection_based_on_validation=False)
    immutable_json(Path(m["output"])/"baseline_diagnostic.json", report)
    print(json.dumps(report,indent=2),flush=True)


def source_files(root, sample):
    return {folder: file_hash(root/"training"/folder/(sample+suffix))
            for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt"))}


def source_stats(root, sample):
    result = {}
    for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
        stat = (root/"training"/folder/(sample+suffix)).stat()
        result[folder] = [stat.st_size, stat.st_mtime_ns]
    return result


def unchanged_source(record, root, sample):
    # All three files were SHA256-hashed when first audited. On resume,
    # unchanged size/mtime avoids repeatedly rereading all images from Drive.
    # If metadata differs, verify actual bytes rather than reject a relocation.
    return (source_stats(root, sample) == record["source_stats"]
            or source_files(root, sample) == record["source_sha256"])


@torch.no_grad()
def audit_frame(model, item, objects, matches, device):
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items() if k != "targets"}
    batch["targets"] = {k: v.unsqueeze(0) for k, v in item["targets"].items()}
    batch = to_device(batch, device)
    output = model(batch["image"])
    t = batch["targets"]
    values = {k: gather(output[k], t["indices"]) for k in ("hps", "dim", "rot")}
    locations, _ = geometry(values["hps"], t["indices"], values["dim"], values["rot"],
                            batch["projection"], batch["inverse_affine"], output["hm"].shape[-1])
    alpha = decode_alpha(values["rot"])[0].cpu().numpy()
    prediction = {k: v[0].cpu().numpy() for k, v in values.items()}
    location = locations[0].cpu().numpy()
    rows, oracle = [], []
    target = item["targets"]
    width = output["hm"].shape[-1]
    forward = cv2.invertAffineTransform(item["inverse_affine"].numpy())
    for slot in target["geom_mask"].nonzero().flatten().tolist():
        gt_loc = target["location"][slot].numpy()
        gt_dim = target["dim"][slot].numpy()
        gi = next((i for i, obj in enumerate(objects)
                   if valid_geometry(obj) and np.allclose(obj.location_3d, gt_loc, atol=1e-4)
                   and np.allclose(obj.dimensions_3d, gt_dim, atol=1e-4)), None)
        if gi is None: raise RuntimeError("GT slot/source object correspondence failed")
        obj = objects[gi]
        kp = project_keypoints(obj, item["projection"].numpy())
        expected = gt_loc.copy(); expected[1] -= gt_dim[0]/2
        double = solve_center(torch.tensor(kp, dtype=torch.float64)[None, None],
                              torch.tensor(gt_dim, dtype=torch.float64)[None, None],
                              torch.tensor([[obj.rotation_y]], dtype=torch.float64),
                              item["projection"].double()[None])[0, 0].numpy()
        single = solve_center(torch.tensor(kp)[None, None], torch.tensor(gt_dim)[None, None],
                              torch.tensor([[obj.rotation_y]], dtype=torch.float32),
                              item["projection"][None])[0, 0].numpy()
        oracle.append(dict(double_relative_error=float(np.linalg.norm(double-expected)/max(1., gt_loc[2])),
                           fp32_relative_error=float(np.linalg.norm(single-expected)/max(1., gt_loc[2]))))
        if gi not in matches: continue
        teacher = matches[gi]
        teacher_obj = KittiObject(obj.class_name, 0., 0, teacher["alpha"], teacher["bbox_2d"],
                                 teacher["dimensions_3d"], teacher["location_3d"], teacher["rotation_y"])
        teacher_kp = project_keypoints(teacher_obj, item["projection"].numpy())
        if teacher_kp is None: continue
        index = int(target["indices"][slot]); cx, cy = index % width, index // width
        thps = (transform_points(teacher_kp, forward)-np.asarray([cx, cy])).reshape(18)
        student_errors = dict(hps=float(np.abs(prediction["hps"][slot]-target["hps"][slot].numpy()).mean()),
                              dim=float(np.abs(prediction["dim"][slot]-gt_dim).mean()),
                              rot=wrapped_error(float(alpha[slot]), obj.alpha),
                              depth=float(abs(location[slot, 2]-gt_loc[2])))
        teacher_errors = dict(hps=float(np.abs(thps-target["hps"][slot].numpy()).mean()),
                              dim=float(np.abs(np.asarray(teacher["dimensions_3d"])-gt_dim).mean()),
                              rot=wrapped_error(teacher["alpha"], obj.alpha),
                              depth=float(abs(teacher["location_3d"][2]-gt_loc[2])))
        if not np.isfinite([*student_errors.values(), *teacher_errors.values()]).all():
            raise RuntimeError("Non-finite teacher/student diagnostic")
        rows.append(dict(gt_location=gt_loc.tolist(), gt_dimensions=gt_dim.tolist(), gt_alpha=float(obj.alpha),
                         class_name=teacher["class_name"], teacher=teacher, projected_keypoints=teacher_kp.tolist(),
                         student_errors=student_errors, teacher_errors=teacher_errors))
    return rows, oracle


def cache_audit(a, m):
    root, out = Path(m["dataset_root"]), Path(m["output"])
    teacher = A2Teacher(m["teacher_export"])
    if a.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Select a Colab GPU")
    model = frozen_student(m, a.device)
    dataset = CompactKittiDataset(root, m["train_ids"], training=False)
    allrows, oracles, inventory = [], [], {}
    for index, sample in enumerate(m["train_ids"]):
        path = out/"train_cache"/f"{sample}.json"
        if path.exists():
            record = json.loads(path.read_text())
            checksum = record.pop("record_sha256")
            if (signature(record) != checksum or record["identity_sha256"] != m["identity_sha256"]
                    or record["sample_id"] != sample or not unchanged_source(record, root, sample)):
                raise RuntimeError(f"Changed cache/data: preserve {path} and use a fresh RUN_ID")
            record["record_sha256"] = checksum
        else:
            source = source_files(root, sample)
            objects = parse_kitti_label_file(root/"training/label_2"/f"{sample}.txt")
            p = parse_kitti_calib_file(root/"training/calib"/f"{sample}.txt")["P2"]
            inputs = teacher_inputs(root/"training/image_2"/f"{sample}.png", p)
            predictions = decode_teacher(teacher.raw(inputs), p, inputs["image_size"][0],
                                         SETTINGS["teacher_score_threshold"])
            matches = match_teacher(objects, predictions, KITTI_PRODUCTION_CLASS_MAPPING,
                                    SETTINGS["teacher_match_iou"])
            rows, oracle = audit_frame(model, dataset[index], objects, matches, a.device)
            record = dict(identity_sha256=m["identity_sha256"], sample_id=sample,
                          source_sha256=source, source_stats=source_stats(root, sample), rows=rows, oracle=oracle)
            record["record_sha256"] = signature(record)
            atomic_json(path, record)
        allrows.extend(record["rows"]); oracles.extend(record["oracle"])
        inventory[sample] = record["record_sha256"]
        if (index+1) % 100 == 0: print(f"A2 cache/audit: {index+1}/3712", flush=True)
    approved = qualification(allrows)
    usable = {name: {k: 0 for k in COMPONENTS} for name in CLASSES}
    for row in allrows:
        for k, yes in usable_components(row, approved).items(): usable[row["class_name"]][k] += int(yes)
    # The exact-GT roundtrip distinguishes a decoder bug from imperfect predictions.
    doubles = np.asarray([r["double_relative_error"] for r in oracles])
    singles = np.asarray([r["fp32_relative_error"] for r in oracles])
    oracle_pass = bool(len(doubles) and np.isfinite(doubles).all() and np.isfinite(singles).all()
                       and doubles.max() < .001 and np.quantile(singles, .95) < .005)
    report = dict(complete=True, identity_sha256=m["identity_sha256"], audited_split="train",
                  samples=len(inventory), train_cache_inventory_sha256=signature(inventory),
                  class_component_audit=approved, usable_targets=usable,
                  oracle=dict(objects=len(oracles), passed=oracle_pass,
                              fp64_max_relative_error=float(doubles.max()) if len(doubles) else None,
                              fp32_p95_relative_error=float(np.quantile(singles, .95)) if len(singles) else None),
                  geometry_units=dict(hps="stride4 grid pixels", dim="mean H/W/L metres",
                                      rot="wrapped alpha radians", depth="absolute metres"),
                  pilot_authorized=bool(oracle_pass and any(v for rows in usable.values() for v in rows.values())),
                  validation_targets_used=False, full_training_authorized=False)
    report["audit_sha256"] = signature(report)
    immutable_json(out/"teacher_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pilot_authorized"]:
        raise RuntimeError("Return the audit bundle: decoder roundtrip failed or A2 offers no qualified component. No blind KD.")


def read_audited_cache(m):
    out = Path(m["output"])
    audit = json.loads((out/"teacher_audit.json").read_text())
    digest = audit.pop("audit_sha256")
    if signature(audit) != digest: raise RuntimeError("Teacher audit changed")
    audit["audit_sha256"] = digest
    if not audit.get("pilot_authorized") or audit["identity_sha256"] != m["identity_sha256"]:
        raise RuntimeError("Run the train-only teacher audit first")
    records, inventory = {}, {}
    for sample in m["train_ids"]:
        r = json.loads((out/"train_cache"/f"{sample}.json").read_text())
        checksum = r.pop("record_sha256")
        if signature(r) != checksum or r["identity_sha256"] != m["identity_sha256"] or r["sample_id"] != sample:
            raise RuntimeError(f"Changed training cache {sample}")
        if not unchanged_source(r, Path(m["dataset_root"]), sample):
            raise RuntimeError(f"Training data changed after teacher audit: {sample}")
        records[sample] = r["rows"]; inventory[sample] = checksum
    if signature(inventory) != audit["train_cache_inventory_sha256"]:
        raise RuntimeError("Incomplete or changed teacher target cache")
    return records, audit


class DistillationDataset(CompactKittiDataset):
    def __init__(self, *args, records, approved, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.approved = records, approved

    def __getitem__(self, index):
        item = super().__getitem__(index)
        target = item["targets"]; n = len(target["mask"])
        kd = {k: torch.zeros(n) for k in ("hps_mask", "dim_mask", "rot_mask", "depth_mask", "depth")}
        kd.update(hps=torch.zeros(n, 18), dim=torch.zeros(n, 3),
                  rotbin=torch.zeros(n, 2, dtype=torch.long), rotres=torch.zeros(n, 2))
        forward = cv2.invertAffineTransform(item["inverse_affine"].numpy())
        for slot in target["geom_mask"].nonzero().flatten().tolist():
            row = next((r for r in self.records[item["sample_id"]]
                        if np.allclose(r["gt_location"], target["location"][slot].numpy(), atol=1e-4)
                        and np.allclose(r["gt_dimensions"], target["dim"][slot].numpy(), atol=1e-4)), None)
            if row is None: continue
            teacher = row["teacher"]
            ix = int(target["indices"][slot]); center = np.asarray([ix % 320, ix // 320])
            kd["hps"][slot] = torch.from_numpy((transform_points(row["projected_keypoints"], forward)-center).reshape(18).astype(np.float32))
            kd["dim"][slot] = torch.tensor(teacher["dimensions_3d"])
            kd["depth"][slot] = teacher["location_3d"][2]
            bins, residuals = rotation_target(teacher["alpha"])
            kd["rotbin"][slot], kd["rotres"][slot] = torch.from_numpy(bins), torch.from_numpy(residuals)
            for name, usable in usable_components(row, self.approved).items(): kd[name+"_mask"][slot] = float(usable)
        item["kd"] = kd
        return item


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()


def pilot_update(model, batch, optimizer, kd_scale):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["image"])
    gt, terms = CompactKittiLoss()(output, batch)
    kd, kterms = distillation_loss(output, batch)
    total = gt + kd_scale * kd
    if not bool(torch.isfinite(total)): raise RuntimeError("Non-finite pilot loss")
    total.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("Missing/non-finite pilot gradients; no update applied")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(total.detach()), gt=float(gt.detach()), kd=float(kd.detach()),
                gradient_norm=float(norm), **{"kd_"+k: float(v.detach()) for k, v in kterms.items()})


def train(a, m, smoke=False):
    records, audit = read_audited_cache(m)
    cfg = dict(m["config"], batch_size=SETTINGS["batch_size"], workers=SETTINGS["workers"])
    if a.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Select a Colab GPU")
    role = a.role
    out = Path(m["output"])/role
    if not smoke:
        smoke_path = Path(m["output"])/f"smoke_{role}.json"
        smoke_report = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        if (not smoke_report.get("finite_gradients") or smoke_report.get("optimizer_steps") != 2
                or smoke_report.get("identity_sha256") != m["identity_sha256"]):
            raise RuntimeError(f"Run the two-step {role} gradient smoke first")
    setup_seed(SETTINGS["seed"])
    model = frozen_student(m, a.device).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=1e-4)
    checkpoint = out/"checkpoints/latest.pt"
    history, first = [], 0
    if checkpoint.exists() and not smoke:
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if saved["identity_sha256"] != m["identity_sha256"] or saved["role"] != role:
            raise RuntimeError("Continuation checkpoint identity differs")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        history, first = saved["history"], saved["epoch"]
        restore_rng(saved["rng"])
        print(f"Resume {role} after epoch {first}", flush=True)
    smoke_ids = [s for s in m["train_ids"] if any(any(usable_components(r, audit["class_component_audit"]).values())
                                                for r in records[s])][:16] if smoke else []
    if smoke and not smoke_ids: raise RuntimeError("No usable KD frame for the disposable smoke")
    dataset = DistillationDataset(m["dataset_root"], smoke_ids if smoke else m["train_ids"],
                                  training=True, records=records, approved=audit["class_component_audit"])
    for epoch in range(first, 1 if smoke else SETTINGS["epochs"]):
        model.train(); freeze_bn(model)
        generator = torch.Generator().manual_seed(SETTINGS["seed"]+epoch)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                            worker_init_fn=worker_seed, generator=generator, pin_memory=a.device=="cuda")
        if smoke and len(loader) < 2:
            # A tiny eligible set can still exercise two disposable updates.
            # Neither update is saved as a pilot training checkpoint.
            from itertools import islice, cycle
            loader = islice(cycle(loader), 2)
        updates = []; started = time.perf_counter()
        for index, batch in enumerate(loader):
            batch = to_device(batch, a.device)
            # Identical graph, GT losses and data for both roles. KD ramps over 2 epochs.
            scale = min(1., (epoch+1)/2) if role == "kd" else 0.
            updates.append(pilot_update(model, batch, optimizer, scale))
            if index % 100 == 0: print(f"{role} epoch {epoch+1}/10 batch {index}: {updates[-1]}", flush=True)
            if smoke and index == 1: break
        row = dict(epoch=epoch+1, seconds=time.perf_counter()-started,
                   environment=environment(), loss={k: float(np.mean([u[k] for u in updates])) for k in updates[0]})
        if smoke:
            if role == "kd" and not any(u["kd"] > 0 for u in updates):
                raise RuntimeError("The KD smoke had no active teacher loss; return the audit bundle")
            if len(updates) != 2: raise RuntimeError("Need two KD smoke batches; use at least two eligible frames")
            atomic_json(Path(m["output"])/f"smoke_{role}.json", dict(complete=True, finite_gradients=True,
                        updates=updates, optimizer_steps=len(updates), disposable_weights=True,
                        identity_sha256=m["identity_sha256"], architecture_changed=False))
            return
        if (epoch+1) % SETTINGS["evaluate_every"] == 0:
            eval_args = argparse.Namespace(dataset_root=Path(m["dataset_root"]), split_dir=Path(m["split_dir"]), device=a.device)
            row["validation"] = evaluate(model, eval_args, cfg, m["val_ids"], out/f"evaluations/epoch_{epoch+1:03d}")["metrics"]
        history.append(row); atomic_json(out/"history.json", history)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                       history=history, rng=rng_state(), identity_sha256=m["identity_sha256"], role=role,
                       external_teacher_used=role=="kd", architecture_changed=False)
        atomic_checkpoint(checkpoint, payload)
        if (epoch+1) % 5 == 0: atomic_checkpoint(out/f"checkpoints/epoch_{epoch+1:03d}.pt", payload)
    selected = max((r for r in history if "validation" in r), key=lambda r:r["validation"]["mean_3d_moderate"])
    atomic_json(out/"selection.json", dict(complete=True, identity_sha256=m["identity_sha256"], role=role,
                selected_epoch=selected["epoch"], metrics=selected["validation"],
                external_teacher_used=role=="kd", architecture_changed=False, deployment_qualified=False))
    selections = {r: Path(m["output"])/r/"selection.json" for r in ("control", "kd")}
    if all(p.exists() for p in selections.values()):
        values = {r: json.loads(p.read_text()) for r, p in selections.items()}
        deltas = {k: values["kd"]["metrics"][k]-values["control"]["metrics"][k] for k in m["baseline_metrics"]}
        report = dict(complete=True, baseline=m["baseline_metrics"], selections=values, kd_minus_control=deltas,
                      kd_minus_original={k:values["kd"]["metrics"][k]-v for k,v in m["baseline_metrics"].items()},
                      architecture_changed=False, iphone_retest_required=True, deployment_qualified=False,
                      interpretation="KD must improve over matched control; higher AP than the original alone does not prove KD helped")
        atomic_json(Path(m["output"])/"comparison.json", report)
        print(json.dumps(report, indent=2), flush=True)


def bundle(a):
    out = a.output
    target = out/"compact_a2_kd_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if (p.is_file() and p.suffix in (".json", ".log") and "teacher_export" not in p.parts
                    and "train_cache" not in p.parts and p.name != "predictions.json"):
                z.write(p, p.relative_to(out))
    print(target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "diagnose", "cache-audit", "smoke", "train", "bundle"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--teacher-export", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--role", choices=("control", "kd"), default="kd")
    a = p.parse_args()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        if a.stage == "bundle": bundle(a)
        elif a.stage == "prepare":
            if any(getattr(a, k) is None for k in ("baseline", "teacher_export", "dataset_root", "split_dir")):
                p.error("prepare requires baseline, teacher-export, dataset-root and split-dir")
            prepare(a)
        else:
            m = load(a.output/"pilot_manifest.json")
            if a.stage == "diagnose": diagnose(m)
            elif a.stage == "cache-audit": cache_audit(a, m)
            else: train(a, m, smoke=a.stage=="smoke")
    except Exception as error:
        import traceback
        atomic_json(a.output/"failures"/f"{a.stage}_{time.time_ns()}.json", dict(complete=False,
                    stage=a.stage, error=str(error), traceback=traceback.format_exc(), original_gt_run_preserved=True))
        raise


if __name__ == "__main__":
    main()
