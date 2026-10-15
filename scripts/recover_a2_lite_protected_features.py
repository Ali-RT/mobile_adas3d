"""One bounded protected-feature-KD pilot with a fresh matched GT control.

Original A2-Lite E1/D1,640x192,512 TRAIN frames,five fixed epochs. Reuse
reviewed teacher features, not teacher inference or recovered checkpoints.
No completed experiment source/manifest is edited or re-signed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
import audit_a2_lite_feature_gradients as audit
from models.a2_lite_protected_feature_kd import (TERMS, CAP, COSINE_TOLERANCE, coordinates,
                                               gradient_grams, choose_coefficient)

feature, engine, staging = audit.feature, audit.engine, audit.staging
REVISION = "A2-LITE-PROTECTED-FEATURE-KD-2026-10-15-r1"
AUDIT_PINS = {
    "feature_gradient_manifest.json": "87a12103cf35a5e96a880ebd5cc2cad7e9116037612d27329236d6703b4bb3cf",
    "feature_gradient_comparison.json": "a0d050e109aa82671a5b39a0e129fe6f0ddc7eefdba4a7f80a8b5b49dbbfbf8e",
}
RECIPE = copy.deepcopy(feature.RECIPE)
RECIPE.update(kd_family="same foreground feature loss with per-step scalar safeguards",
    feature_gradient_cap=CAP, conflict_cosine_tolerance=COSINE_TOLERANCE,
    feature_calibration="historical calibrated coefficient is only the scheduled ceiling; actual coefficient checked every TRAIN update",
    guard_reference="GT total and exact Pedestrian/Vehicle GT shares, including class-channel negatives",
    guard_groups=["all_trainable", "visual_encoder", "late_backbone", "other_trainable"],
    conflict_policy="drop the entire combined class-balanced feature term if any checked group conflicts; no redistribution",
    cap_policy="one detached coefficient no larger than scheduled base; cap each group and all trainable coordinates",
    no_GT_gradient_projection=True, adamw_update_safety_guaranteed=False,
    fresh_matched_control=True, teacher_features_reused=True,
    training_images_staged=512, validation_staged_after_training=True)
GUARDRAILS = copy.deepcopy(feature.GUARDRAILS)


def implementation():
    return {**audit.implementation(), "scripts/recover_a2_lite_protected_features.py": engine.sha256(__file__),
            "models/a2_lite_protected_feature_kd.py": engine.sha256(ROOT / "models/a2_lite_protected_feature_kd.py")}


@contextmanager
def recipe_context():
    previous = engine.RECIPE, engine.REVISION
    engine.RECIPE, engine.REVISION = RECIPE, REVISION
    try: yield
    finally: engine.RECIPE, engine.REVISION = previous


def reviewed_audit(source):
    records = {}
    for name, expected in AUDIT_PINS.items():
        raw = engine.read_record(source, name)
        if engine.accuracy.sha_bytes(raw) != expected: raise RuntimeError("Use the exact reviewed endpoint-gradient audit")
        records[name] = audit.signed(json.loads(raw), "signature_sha256" if name.endswith("manifest.json") else "report_sha256")
    m, c = records.values()
    if (m["implementation"] != audit.implementation() or m["historical_run_signature"] != audit.RUN_SIGNATURE
            or c["run_signature"] != m["signature_sha256"] or not c["complete"] or c["optimizer_steps"] != 0
            or c["roles"] != list(audit.ROLES) or c["training_authorized"] or c["validation_inference_performed"]):
        raise RuntimeError("Reviewed gradient audit identity differs")
    return records


def prepare(a):
    parent, records = audit.reviewed(a.feature_run); reviewed_audit(a.gradient_audit)
    protected = [Path(parent["output"]).resolve(), a.initialization.parent.resolve()]
    protected += [p.resolve() for p in (a.feature_run, a.gradient_audit) if p.is_dir()]
    if any(p.is_relative_to(q) for p in (a.output, a.repo, a.dataset) for q in protected):
        raise RuntimeError("Use separate protected-pilot paths; completed runs/checkpoints stay read-only")
    if (a.repo.is_relative_to(Path(parent["source"]).resolve()) or a.dataset.is_relative_to(Path(parent["dataset"]).resolve())
            or any(p.is_relative_to(Path('/content/drive')) or p.is_symlink() for p in (a.repo, a.dataset))):
        raise RuntimeError("Use fresh private local source/data paths; historical inputs are candidates only")
    current = audit.runtime(a.device)
    if not a.initialization.is_file() or engine.sha256(a.initialization) != engine.resolution.CHECKPOINT_SHA:
        raise RuntimeError("Need the exact original untrained pruned initialization, not any recovered endpoint")
    path = a.output / "protected_manifest.json"
    if path.exists():
        old = audit.signed(json.loads(path.read_text()), "signature_sha256")
        if (old["runtime"] != current or old["implementation"] != implementation() or old["recipe"] != RECIPE
                or old["source"] != str(a.repo) or old["dataset"] != str(a.dataset) or old["output"] != str(a.output)):
            raise RuntimeError("Protected run runtime/code/paths changed; preserve it and select a new RUN_ID")
    a.output_verified = True
    engine.base.prepare_repo(a.repo)  # Restore ephemeral source BEFORE full run validation.
    if engine.base.q.source_hash(a.repo) != parent["source_sha256"]: raise RuntimeError("Portable source differs from the reviewed model")
    ids = list(parent["pilot_ids"]); val = list(parent["val_ids"])
    if len(ids) != 512 or len(val) != 3769 or set(ids) & set(val): raise RuntimeError("Expected unchanged512 TRAIN /3769 VAL IDs")
    expected = {s: parent["data_file_sha256"][s] for s in ids + val}
    staging.stage(a.dataset, a.candidate, ids, expected, workers=4)
    for name, values in (("train", ids), ("val", val)):
        engine.accuracy.immutable_bytes(a.dataset / "ImageSets" / f"{name}.txt", ("\n".join(values) + "\n").encode())
    initialization = a.dataset / "weights/original_initialization.pth"
    audit.local_copy(a.initialization, initialization, engine.resolution.CHECKPOINT_SHA)
    student, _ = engine.accuracy.checked_package(a.compact or Path(parent["student_export"]), "compact_640", a.output / "assets/student")
    root = a.dataset / "teacher_features"; coverage = records["feature_cache_complete.json"]
    def copy_feature(sample):
        destination = root / f"{sample}.pt"; digest = coverage["cache_sha256"][sample]
        if not destination.exists():
            candidates = [p / destination.name for p in (*a.feature_candidate, a.features_dir)]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None: raise FileNotFoundError(f"Saved teacher feature missing: {sample}; do not regenerate the reviewed cache")
            audit.local_copy(source, destination, digest)
        feature.read_feature(destination, parent, sample, expected_sha=digest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(copy_feature, ids)):
            if (index + 1) % 64 == 0 or index == len(ids) - 1: print(f"Original teacher features copied/verified {index+1}/512 with4 I/O workers", flush=True)
    for name in audit.REVIEWED:
        engine.accuracy.immutable_bytes(a.output / "reviewed_feature" / name, engine.read_record(a.feature_run, name))
    for name in AUDIT_PINS:
        engine.accuracy.immutable_bytes(a.output / "reviewed_audit" / name, engine.read_record(a.gradient_audit, name))
    m = dict(revision=REVISION, recipe=RECIPE, guardrails=GUARDRAILS, implementation=implementation(), runtime=current,
        output=str(a.output), source=str(a.repo), source_sha256=parent["source_sha256"], dataset=str(a.dataset),
        config=copy.deepcopy(parent["config"]), initialization=str(initialization), initialization_sha256=engine.resolution.CHECKPOINT_SHA,
        student_export=str(student), train_ids=ids, pilot_ids=ids, val_ids=val, data_file_sha256=expected,
        feature_root=str(root), feature_sha256=coverage["cache_sha256"], historical_run_signature=audit.RUN_SIGNATURE,
        scheduled_max_coefficient=records["feature_kd_smoke.json"]["calibration"]["loss_scale"] * feature.RECIPE["kd_max_weight"],
        historical_results_are_context_only=True, architecture_changed=False, full_training_authorized=False, deployment_qualified=False)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(path, m)
    print("Prepared fresh matched512/five-epoch arms. Only TRAIN images staged; VAL staging is deferred.", flush=True)


def load(output, device):
    m = audit.signed(json.loads((output / "protected_manifest.json").read_text()), "signature_sha256")
    if (m["revision"] != REVISION or m["recipe"] != RECIPE or m["guardrails"] != GUARDRAILS
            or m["implementation"] != implementation() or m["runtime"] != audit.runtime(device) or m["output"] != str(output)):
        raise RuntimeError("Protected run identity/runtime changed; preserve run and use a new RUN_ID")
    parent, records = audit.reviewed(output / "reviewed_feature"); reviewed_audit(output / "reviewed_audit")
    if (m["pilot_ids"] != parent["pilot_ids"] or m["train_ids"] != parent["pilot_ids"] or m["val_ids"] != parent["val_ids"]
            or m["config"] != parent["config"] or m["feature_sha256"] != records["feature_cache_complete.json"]["cache_sha256"]
            or m["data_file_sha256"] != {s: parent["data_file_sha256"][s] for s in m["pilot_ids"] + m["val_ids"]}
            or engine.base.q.source_hash(Path(m["source"])) != parent["source_sha256"]
            or engine.sha256(m["initialization"]) != engine.resolution.CHECKPOINT_SHA
            or m["scheduled_max_coefficient"] != records["feature_kd_smoke.json"]["calibration"]["loss_scale"] * feature.RECIPE["kd_max_weight"]):
        raise RuntimeError("Frozen protected-pilot data/model/teacher identity changed")
    return m, parent


def guarded_objective(model, outputs, memory, row, target, base_coefficient, parameters, groups):
    gt, _, _ = engine.gt_loss(outputs, target)
    shares = audit.gt_partition(outputs, target)
    features = feature.object_features(memory, target, feature.STUDENT_SHAPES)
    kd, counts = feature.feature_kd_loss(features, row["features"].to(features.device), target["labels"], row["eligible"].to(features.device))
    values = dict(gt_all=gt, **{k: shares[k] for k in ("gt/Pedestrian", "gt/Vehicle")}, kd=kd)
    gradients = {k: torch.autograd.grad(values[k], parameters, retain_graph=True, allow_unused=True) for k in TERMS}
    decision = choose_coefficient(gradient_grams(groups, gradients), base_coefficient)
    del gradients, shares, values
    total = gt + decision["coefficient"] * kd  # coefficient is detached Python float; no GT gradient edits.
    return total, gt, kd, counts, decision


def read_row(m, parent, sample, target=None):
    return feature.read_feature(Path(m["feature_root"]) / f"{sample}.pt", parent, sample, target, m["feature_sha256"][sample])


def guard_smoke(a, m, parent):
    path = a.output / "protected_smoke.json"
    if path.exists():
        old = engine.checked_report(path, m)
        if not old["passed"]: raise RuntimeError("Preserve failed protected smoke; return evidence")
        return
    engine.tiny_gate(m); engine.setup_device(a.device)
    model = engine.build_model(m, a.device, True); before = engine.resolution.state_signature(model)
    _, parameters, groups = coordinates(model); dataset = engine.native_dataset(m, m["pilot_ids"][:32])
    checks = []; active = Counter()
    with feature.EncoderFeatureTap(model) as tap:
        for index, sample in enumerate(dataset.idx_list):
            engine.verify_files(m, sample); inputs, target, _ = engine.item(dataset, index, a.device)
            row = read_row(m, parent, sample, target); outputs = engine.forward(model, inputs)
            total, gt, kd, counts, decision = guarded_objective(model, outputs, tap.take(), row, target, m["scheduled_max_coefficient"], parameters, groups)
            if not torch.isfinite(total): raise RuntimeError("Nonfinite protected preflight objective")
            if decision["accepted"]: active.update(counts)
            checks.append(dict(sample_id=sample, gt=float(gt.detach()), kd=float(kd.detach()), counts=counts, decision=decision))
            del total, gt, kd, outputs
    unchanged = before == engine.resolution.state_signature(model) and all(p.grad is None for p in model.parameters())
    passed = unchanged and min(active['Pedestrian'], active['Vehicle']) > 0
    engine.save_report(path, dict(complete=True, passed=bool(passed), run_signature=m["signature_sha256"], checks=checks,
        accepted_teacher_object_counts=dict(active), weights_buffers_unchanged=unchanged, optimizer_steps=0,
        safeguard_scope="current raw gradients before clipping/AdamW; no AP guarantee"))
    if not passed: raise RuntimeError("Protected signal preflight failed or rejected all guidance for a class; do not train")


def run_steps(m, parent, model, opt, dataset, total_steps, path, role, device):
    step, history, stats = 0, [], Counter(); before = engine.buffers_signature(model)
    names, parameters, groups = coordinates(model); coordinate_hash = engine.signature(names)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['run_signature'] != m['signature_sha256'] or saved['role'] != role or saved['recipe'] != RECIPE
                or saved['runtime'] != m['runtime'] or saved['coordinate_hash'] != coordinate_hash or not 0 <= saved['step'] <= total_steps):
            raise RuntimeError("Protected resume identity/budget changed; preserve checkpoint")
        model.load_state_dict(saved['model_state'], strict=True); opt.load_state_dict(saved['optimizer_state'])
        step, history, before = saved['step'], saved['history'], saved['buffers_signature']; stats.update(saved['guard_statistics'])
    order = []
    with feature.EncoderFeatureTap(model) as tap:
        for cursor in range(step, total_steps):
            epoch, offset = divmod(cursor, len(dataset))
            if not order or offset == 0:
                order = list(range(len(dataset))); random.Random(RECIPE['seed'] + epoch).shuffle(order)
            index = order[offset]; sample = dataset.idx_list[index]
            engine.verify_files(m, sample); inputs, target, _ = engine.item(dataset, index, device)
            opt.zero_grad(set_to_none=True); outputs = engine.forward(model, inputs); memory = tap.take()
            if role == 'kd':
                base = m['scheduled_max_coefficient'] * min(1., (cursor + 1) / RECIPE['kd_warmup_steps'])
                total, gt, kd, counts, decision = guarded_objective(model, outputs, memory, read_row(m, parent, sample, target), target, base, parameters, groups)
                stats['checked_steps'] += 1
                for key in ('accepted', 'rejected_for_conflict', 'scaled_for_cap', 'no_teacher_gradient'): stats[key] += int(decision[key])
                if decision['accepted']:
                    for cls, count in counts.items(): stats['accepted_' + cls] += count
                stats['cap_violations'] += int(any(v['applied_teacher_to_gt_norm_ratio'] > CAP + 1e-8 for v in decision['checks'].values()))
            else:
                gt, _, _ = engine.gt_loss(outputs, target); kd = memory.sum() * 0; total = gt
                decision = dict(coefficient=0., accepted=False, checks={})
            if not torch.isfinite(total): raise RuntimeError(f"Nonfinite protected {role} loss at step{cursor}")
            total.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip_gradient_norm'])
            if not torch.isfinite(norm): raise RuntimeError("Nonfinite protected training gradient")
            opt.step(); step = cursor + 1
            if step % 32 == 0 or step == total_steps:
                history.append(dict(step=step, epoch=epoch+1, sample_id=sample, gt=float(gt.detach()), kd=float(kd.detach()),
                    coefficient=decision['coefficient'], gradient_norm=float(norm), guard=decision))
                print(f"{role} {step}/{total_steps}: GT={float(gt.detach()):.4f} KD={float(kd.detach()):.4f} coefficient={decision['coefficient']:.5f} accepted={stats['accepted']}", flush=True)
            if step % RECIPE['checkpoint_every_steps'] == 0 or step == total_steps:
                if before != engine.buffers_signature(model): raise RuntimeError("Frozen model buffers changed")
                engine.atomic_checkpoint(path, dict(run_signature=m['signature_sha256'], role=role, step=step, recipe=RECIPE,
                    runtime=m['runtime'], coordinate_hash=coordinate_hash, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                    history=history, buffers_signature=before, guard_statistics=dict(stats)))
    return step, history, dict(stats)


def train(a, m, parent):
    engine.tiny_gate(m); smoke = engine.checked_report(a.output / 'protected_smoke.json', m)
    if not smoke['passed'] or smoke['optimizer_steps'] != 0: raise RuntimeError("Passed zero-update guard smoke required")
    engine.setup_device(a.device); model = engine.build_model(m, a.device, True); dataset = engine.native_dataset(m, m['pilot_ids'])
    path = a.output / a.role / 'last.pth'
    step, history, stats = run_steps(m, parent, model, engine.optimizer(model), dataset, len(dataset)*RECIPE['epochs'], path, a.role, a.device)
    engine.save_report(a.output / a.role / 'training_complete.json', dict(complete=True, run_signature=m['signature_sha256'], role=a.role,
        checkpoint_sha256=engine.sha256(path), optimizer_steps=step, train_images=len(dataset), fixed_endpoint_epoch=5,
        same_initialization_sha256=engine.resolution.CHECKPOINT_SHA, tiny_checkpoint_used=False, architecture_changed=False,
        runtime=m['runtime'], guard_statistics=stats, external_teacher_used=a.role=='kd', history=history, deployment_qualified=False))


def stage_validation(a, m):
    for role in ('control','kd'):
        t = engine.checked_report(a.output / role / 'training_complete.json', m)
        if not t['complete'] or t['optimizer_steps'] != 2560: raise RuntimeError("Finish both bounded training arms before VAL staging")
    staging.stage(Path(m['dataset']), a.candidate, m['val_ids'], m['data_file_sha256'], workers=4)
    engine.save_report(a.output / 'validation_staging.json', dict(complete=True, run_signature=m['signature_sha256'], frames=3769, every_file_sha256_verified=True))


def compare(a, m):
    results = {}; training = {}
    for role in ('control','kd'):
        t = engine.checked_report(a.output / role / 'training_complete.json', m)
        r = json.loads((a.output / 'evaluation' / role / 'accuracy.json').read_text())
        if (not t['complete'] or t['optimizer_steps'] != 2560 or t['train_images'] != 512 or t['fixed_endpoint_epoch'] != 5
                or t['tiny_checkpoint_used'] or t['architecture_changed'] or t['runtime'] != m['runtime']
                or t['same_initialization_sha256'] != engine.resolution.CHECKPOINT_SHA or t['external_teacher_used'] != (role=='kd')
                or t['checkpoint_sha256'] != engine.sha256(a.output / role / 'last.pth')
                or not r['complete'] or r['evaluated_images'] != 3769 or r['run_signature'] != m['signature_sha256']
                or r['checkpoint_sha256'] != t['checkpoint_sha256']): raise RuntimeError("Both fresh matched fixed-epoch endpoints required")
        results[role] = r; training[role] = t
    stats = training['kd']['guard_statistics']
    guard_ok = (stats.get('checked_steps') == 2560 and stats.get('cap_violations',0) == 0
                and min(stats.get('accepted_Pedestrian',0), stats.get('accepted_Vehicle',0)) > 0
                and not any(training['control']['guard_statistics'].values()))
    decision = feature.reference.decision(results['control'],results['kd'])
    passed = guard_ok and decision['kd_benefit_guardrail_passed']
    engine.save_report(a.output / 'protected_comparison.json', dict(complete=True, revision=REVISION, run_signature=m['signature_sha256'],
        recipe=RECIPE, guardrails=GUARDRAILS, results=results, versus_label_only=decision, runtime_guards_passed=guard_ok,
        benefit_guardrail_passed=passed, guard_statistics=stats, fresh_matched_control=True, single_seed=True,
        validation_checkpoint_selection=False, full_training_authorized=False, deployment_qualified=False,
        next_step="Review benefit/repeatability before any longer run" if passed else "Stop this protected recipe; no automatic weight sweep or longer training"))
    print(json.dumps(dict(versus_label_only=decision, runtime_guards_passed=guard_ok, benefit_guardrail_passed=passed, guard_statistics=stats),indent=2),flush=True)


def bundle(a):
    path = a.output / 'a2_lite_protected_feature_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob('*')):
            if source.is_file() and 'assets' not in source.relative_to(a.output).parts and source.suffix in ('.json','.csv','.txt','.log'):
                archive.write(source,source.relative_to(a.output))
    print(path,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','smoke','tiny','guard-smoke','train','stage-validation','evaluate','compare','bundle'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--role',choices=('control','kd'))
    for name in ('feature-run','gradient-audit','features-dir','initialization','repo','dataset','compact'):
        parser.add_argument('--'+name,type=Path)
    for name in ('candidate','feature-candidate'):parser.add_argument('--'+name,type=Path,action='append',default=[])
    a=parser.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    a.candidate=[p.resolve() for p in a.candidate];a.feature_candidate=[p.resolve() for p in a.feature_candidate]
    if a.stage=='prepare' and any(getattr(a,k) is None for k in ('feature_run','gradient_audit','features_dir','initialization','repo','dataset')):parser.error('prepare requires reviewed feature run/audit, features, original initialization and private source/data paths')
    if a.stage in ('train','evaluate') and a.role is None:parser.error('train/evaluate requires --role')
    try:
        with recipe_context():
            if a.stage=='prepare':prepare(a)
            elif a.stage=='bundle':bundle(a)
            else:
                m,parent=load(a.output,a.device)
                if a.stage=='guard-smoke':guard_smoke(a,m,parent)
                elif a.stage=='train':train(a,m,parent)
                elif a.stage=='stage-validation':stage_validation(a,m)
                elif a.stage=='compare':compare(a,m)
                elif a.stage=='evaluate':
                    v=engine.checked_report(a.output/'validation_staging.json',m)
                    if not v['complete']:raise RuntimeError('Stage verified validation inputs first')
                    engine.evaluate(a,m)
                else:getattr(engine,a.stage)(a,m)
    except Exception as error:
        import traceback
        if getattr(a,'output_verified',False) or (a.output/'protected_manifest.json').is_file():
            engine.write_json(a.output/'failures'/f'{a.stage}_{time.time_ns()}.json',dict(complete=False,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
