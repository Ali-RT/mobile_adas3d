"""One new 512-frame geometry-KD arm versus the two reviewed pilot arms.

Same frame IDs/order, pruned initialization, five epochs and GT losses.
Only direct teacher classification/box losses are removed. Completed pilot
and full-data runs remain read-only. No teacher-cache recomputation, longer
training, architecture change or deployment is authorized automatically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import recover_a2_lite_640 as engine
import recover_a2_lite_full as reference
from models.a2_lite_geometry_kd import geometry_kd_loss, ENABLED_COMPONENTS, DISABLED_COMPONENTS

REVISION = "A2-LITE-512-GEOMETRY-KD-2026-10-14-r1"
RECIPE = copy.deepcopy(engine.RECIPE)
RECIPE.update(kd_components=list(ENABLED_COMPONENTS), disabled_kd_components=list(DISABLED_COMPONENTS),
              gt_loss_changed=False, historical_references_retrained=False)
GUARDRAILS = copy.deepcopy(reference.GUARDRAILS)


@contextmanager
def recipe_context():
    old = engine.RECIPE, engine.REVISION, engine.kd_loss
    engine.RECIPE, engine.REVISION, engine.kd_loss = RECIPE, REVISION, geometry_kd_loss
    try:
        yield
    finally:
        engine.RECIPE, engine.REVISION, engine.kd_loss = old


def implementation():
    return {**reference.implementation(), "scripts/recover_a2_lite_geometry.py": engine.sha256(__file__),
            "models/a2_lite_geometry_kd.py": engine.sha256(ROOT / "models/a2_lite_geometry_kd.py")}


def require_reference_environment(previous, current):
    if previous != current:
        changes = {k: dict(saved=previous.get(k), current=current.get(k))
                   for k in set(previous) | set(current) if previous.get(k) != current.get(k)}
        raise RuntimeError("Historical pilot software differs; stop and review whether fresh matched references are needed. "
                           "Do not restore an old CUDA stack or overwrite old runs.\n" + json.dumps(changes, indent=2))


def exact_subset(previous, train, val):
    reference.all_train_ids(train, val)
    selected = previous["pilot_ids"]
    if (train != previous["train_ids"] or val != previous["val_ids"] or len(selected) != 512
            or len(set(selected)) != 512 or not set(selected) <= set(train)):
        raise RuntimeError("Use the exact reviewed512 TRAIN IDs/order and Chen train/val split")
    return list(selected)


def prepare(a):
    previous, comparison, raw_m, raw_c = reference.reviewed_pilot(a.pilot)
    if engine.accuracy.sha_bytes(engine.read_record(a.pilot, "teacher_cache_complete.json")) != reference.PILOT_CACHE_SHA:
        raise RuntimeError("Use the original512 pilot folder/ZIP, not the later full-dataset run")
    for path in (a.pilot, a.compact, a.initialization.parent):
        if path.is_dir() and a.output.is_relative_to(path):
            raise RuntimeError("Use a new geometry output folder; previous runs are read-only")
    require_reference_environment(previous["cpu_environment"], engine.runtime_environment())
    if not a.initialization.is_file() or engine.sha256(a.initialization) != engine.resolution.CHECKPOINT_SHA:
        raise RuntimeError("Use original untrained pruned weights, not a pilot/full-data trained checkpoint")
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != previous["source_sha256"]:
        raise RuntimeError("Fresh portable source differs from the reviewed pilot")
    train = engine.base.q.split_ids(a.split_dir / "train.txt", "train")
    val = engine.base.q.split_ids(a.split_dir / "val.txt", "val")
    selected = exact_subset(previous, train, val)
    for attempt in range(3):
        try:
            files = engine.accuracy.stage_data(a.dataset, a.candidate, train + val)
            break
        except OSError:
            if attempt == 2:
                raise
            print("Transient data read failed; preserving staged files and retrying.", flush=True)
            time.sleep(attempt + 1)
    if files != previous["data_file_sha256"]:
        raise RuntimeError("Dataset bytes differ from the reviewed pilot")
    if engine.choose_ids(train, a.dataset / "training/label_2") != selected:
        raise RuntimeError("Train-only subset selection no longer reproduces the reviewed512 IDs")
    for split, ids in (("train", train), ("val", val)):
        engine.accuracy.immutable_bytes(a.dataset / "ImageSets" / f"{split}.txt", ("\n".join(ids) + "\n").encode())
    student_root, _ = engine.accuracy.checked_package(a.compact, "compact_640", a.output / "assets/student")
    for name, raw in (("recovery_manifest.json", raw_m), ("recovery_comparison.json", raw_c)):
        engine.accuracy.immutable_bytes(a.output / "reviewed_pilot" / name, raw)
    m = dict(revision=REVISION, recipe=RECIPE, guardrails=GUARDRAILS, implementation=implementation(),
             cpu_environment=engine.runtime_environment(), config=copy.deepcopy(previous["config"]),
             source=str(a.repo), source_sha256=previous["source_sha256"],
             initialization=str(a.initialization), initialization_sha256=engine.resolution.CHECKPOINT_SHA,
             teacher_model_sha256=previous["teacher_model_sha256"], student_export=str(student_root),
             dataset=str(a.dataset), data_file_sha256=files, split_dir=str(a.split_dir),
             train_ids=train, val_ids=val, pilot_ids=selected, output=str(a.output),
             reference_results=copy.deepcopy(comparison["results"]),
             reference_run_signature=previous["signature_sha256"],
             same_training_subset=True, same_initialization=True, same_gt_losses=True,
             teacher_cache_reused_only=True, historical_comparison=True,
             architecture_changed=False, longer_training_authorized=False, deployment_qualified=False)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(a.output / "geometry_manifest.json", m)
    print("Prepared one geometry-only arm: exact512 TRAIN frames, five epochs, original pruned initialization.", flush=True)


def load(output):
    m = json.loads((output / "geometry_manifest.json").read_text())
    if (engine.signature({k: v for k, v in m.items() if k != "signature_sha256"}) != m["signature_sha256"]
            or m["revision"] != REVISION or m["recipe"] != RECIPE or m["guardrails"] != GUARDRAILS
            or m["implementation"] != implementation() or m["cpu_environment"] != engine.runtime_environment()):
        raise RuntimeError("Geometry run code/packages/recipe changed; preserve run and use a new RUN_ID")
    previous, comparison, _, _ = reference.reviewed_pilot(output / "reviewed_pilot")
    require_reference_environment(previous["cpu_environment"], m["cpu_environment"])
    if (m["output"] != str(output) or engine.sha256(m["initialization"]) != engine.resolution.CHECKPOINT_SHA
            or engine.base.q.source_hash(Path(m["source"])) != m["source_sha256"]
            or m["reference_results"] != comparison["results"]):
        raise RuntimeError("Frozen weights/source/reference results changed")
    for split, key in (("train", "train_ids"), ("val", "val_ids")):
        if engine.base.q.split_ids(Path(m["split_dir"]) / f"{split}.txt", split) != m[key]:
            raise RuntimeError("Split changed")
    if m["pilot_ids"] != exact_subset(previous, m["train_ids"], m["val_ids"]):
        raise RuntimeError("Geometry training subset changed")
    return m


def verify_reused_values(source, m):
    """Even resumed cache payloads must equal the reviewed teacher values."""
    raw_report = engine.read_record(source, "teacher_cache_complete.json")
    if engine.accuracy.sha_bytes(raw_report) != reference.PILOT_CACHE_SHA:
        raise RuntimeError("Original512 teacher-cache report changed")
    original_report = json.loads(raw_report)
    context = zipfile.ZipFile(source) if source.is_file() else nullcontext(None)
    with context as archive:
        names = archive.namelist() if archive is not None else []
        for sample in m["pilot_ids"]:
            name = f"teacher_cache/{sample}.json"
            if archive is None:
                raw = engine.read_record(source, name)
            else:
                matches = [n for n in names if n == name or n.endswith("/" + name)]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one original teacher record: {sample}")
                raw = archive.read(matches[0])
            if engine.accuracy.sha_bytes(raw) != original_report["cache_sha256"][sample]:
                raise RuntimeError(f"Original teacher record changed: {sample}")
            original = json.loads(raw); original.pop("record_sha256")
            current = engine.cache_record(m, Path(m["output"]) / "teacher_cache", sample)
            if current.pop("reused_from_run_signature", None) != reference.PILOT_SIGNATURE:
                raise RuntimeError("Reused teacher record has wrong historical provenance")
            current["run_signature"] = reference.PILOT_SIGNATURE
            if current != original:
                raise RuntimeError(f"Rebound teacher values differ from original512 pilot: {sample}")


def cache(a, m):
    engine.tiny_gate(m)
    if a.pilot is None:
        raise RuntimeError("cache requires --pilot: reuse the reviewed512 TRAIN teacher records")
    reference.reuse_teacher_cache(a, m)
    verify_reused_values(a.pilot, m)
    root = a.output / "teacher_cache"
    counts = {0: 0, 1: 0}; qualified = {0: 0, 1: 0}
    dataset = engine.native_dataset(m, m["pilot_ids"])
    for index, sample in enumerate(m["pilot_ids"]):
        engine.verify_files(m, sample)
        row = engine.cache_record(m, root, sample)
        _, target, _ = engine.item(dataset, index)
        if row["target_signature"] != engine.target_signature(target):
            raise RuntimeError("Reused teacher target encoding differs from current native GT")
        for label, query in zip(row["target_labels"], row["object_to_teacher_query"]):
            counts[label] += 1; qualified[label] += query >= 0
    report = dict(complete=True, run_signature=m["signature_sha256"], images=len(dataset),
                  encoded_gt_counts=dict(Pedestrian=counts[0], Vehicle=counts[1]),
                  qualified_teacher_matches=dict(Pedestrian=qualified[0], Vehicle=qualified[1]),
                  kd_pilot_authorized=min(qualified.values()) >= RECIPE["minimum_teacher_matches_per_class"],
                  selection_train_only=True, teacher_inference_performed=False,
                  original_cache_sha256=reference.PILOT_CACHE_SHA,
                  cache_sha256={s: engine.sha256(root / f"{s}.json") for s in m["pilot_ids"]})
    engine.save_report(a.output / "teacher_cache_complete.json", report)
    print("Verified/rebound all512 original teacher records and current GT encoding; no teacher inference repeated.", flush=True)


def kd_smoke(a, m):
    destination = a.output / "geometry_kd_smoke.json"
    if destination.exists():
        if not engine.checked_report(destination, m)["passed"]:
            raise RuntimeError("Preserve failed geometry-KD smoke; return evidence")
        print("Unchanged completed geometry-KD smoke reused.", flush=True); return
    coverage = engine.checked_report(a.output / "teacher_cache_complete.json", m)
    if not coverage["kd_pilot_authorized"]:
        raise RuntimeError("Qualified original teacher cache required")
    engine.setup_device(a.device)
    model = engine.build_model(m, a.device, True)
    before = engine.buffers_signature(model)
    dataset = engine.native_dataset(m, m["pilot_ids"][:64])
    totals = {name: {"Pedestrian": 0, "Vehicle": 0} for name in engine.COMPONENTS}
    component_gradients = {name: 0. for name in ENABLED_COMPONENTS}
    parameters = [p for p in model.parameters() if p.requires_grad]
    checks = []
    for index, sample in enumerate(dataset.idx_list):
        engine.verify_files(m, sample)
        inputs, target, _ = engine.item(dataset, index, a.device)
        row = engine.cache_record(m, a.output / "teacher_cache", sample)
        if row["target_signature"] != engine.target_signature(target):
            raise RuntimeError("Geometry smoke GT encoding changed")
        teacher = engine.native_arrays(row["outputs"], a.device)
        mapping = torch.tensor(row["object_to_teacher_query"], device=a.device)
        model.zero_grad(set_to_none=True)
        loss, parts, active = geometry_kd_loss(engine.forward(model, inputs), teacher, target, mapping)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite geometry-KD smoke loss")
        for name in engine.COMPONENTS:
            for label in totals[name]: totals[name][label] += active[name][label]
        if float(loss.detach()) > 0:
            local_gradients = {}
            for name in ENABLED_COMPONENTS:
                if sum(active[name].values()) and float(parts[name].detach()) > 0:
                    values = torch.autograd.grad(parts[name], parameters, retain_graph=True, allow_unused=True)
                    finite = [g for g in values if g is not None]
                    if not all(torch.isfinite(g).all() for g in finite):
                        raise RuntimeError(f"Nonfinite {name} teacher gradients")
                    local_gradients[name] = sum(float(g.detach().abs().sum()) for g in finite)
                    component_gradients[name] = max(component_gradients[name], local_gradients[name])
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            norm = sum(float(g.detach().abs().sum()) for g in gradients)
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or norm <= 0:
                raise RuntimeError("Missing/nonfinite geometry-KD gradients")
            checks.append(dict(sample_id=sample, loss=float(loss.detach()), gradient_abs_sum=norm,
                               component_gradient_abs_sum=local_gradients,
                               parts={k: float(v.detach()) for k, v in parts.items()}))
        if (all(sum(totals[k].values()) > 0 for k in ENABLED_COMPONENTS)
                and all(component_gradients.values())
                and all(sum(totals[k][label] for k in ENABLED_COMPONENTS) > 0 for label in ("Pedestrian", "Vehicle"))):
            break
    passed = bool(checks and all(sum(totals[k].values()) > 0 for k in ENABLED_COMPONENTS)
                  and all(component_gradients.values())
                  and all(sum(totals[k][label] for k in ENABLED_COMPONENTS) > 0 for label in ("Pedestrian", "Vehicle"))
                  and engine.buffers_signature(model) == before
                  and all(sum(totals[k].values()) == 0 for k in DISABLED_COMPONENTS))
    engine.save_report(destination, dict(complete=True, passed=passed, run_signature=m["signature_sha256"],
                       enabled_components=list(ENABLED_COMPONENTS), disabled_components=list(DISABLED_COMPONENTS),
                       active_counts=totals, finite_nonzero_gradients=bool(checks), checks=checks,
                       component_gradient_abs_sum=component_gradients,
                       optimizer_steps=0, buffers_unchanged=engine.buffers_signature(model) == before,
                       gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else None,
                       cuda_build=torch.version.cuda, no_teacher_classification_or_box_loss=True))
    if not passed:
        raise RuntimeError("Geometry-KD smoke did not qualify all three components and both classes; return ZIP")
    print("Geometry KD passed: all three teacher terms active, disabled terms zero, finite gradients, zero updates.", flush=True)


def train(a, m):
    if not engine.checked_report(a.output / "geometry_kd_smoke.json", m)["passed"]:
        raise RuntimeError("Passed geometry-KD smoke required before training")
    engine.train(a, m)


def compare(a, m):
    training = engine.checked_report(a.output / "kd/training_complete.json", m)
    geometry = json.loads((a.output / "evaluation/kd/accuracy.json").read_text())
    if (not training["complete"] or training["train_images"] != 512 or training["optimizer_steps"] != 2560
            or training["fixed_endpoint_epoch"] != 5 or training["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
            or training["tiny_checkpoint_used"] or training["architecture_changed"] or not training["external_teacher_used"]
            or engine.sha256(a.output / "kd/last.pth") != training["checkpoint_sha256"]
            or not geometry["complete"] or geometry["evaluated_images"] != 3769
            or geometry["run_signature"] != m["signature_sha256"]
            or geometry["checkpoint_sha256"] != training["checkpoint_sha256"]
            or any(sum(training["active_kd_counts"][k].values()) for k in DISABLED_COMPONENTS)
            or any(sum(training["active_kd_counts"][k].values()) <= 0 for k in ENABLED_COMPONENTS)):
        raise RuntimeError("Completed geometry-only fixed5-epoch endpoint required")
    control, all_heads = m["reference_results"]["control"], m["reference_results"]["kd"]
    versus_gt = reference.decision(control, geometry)
    versus_all = reference.decision(all_heads, geometry)
    for decision in (versus_gt, versus_all):
        decision["next_review"] = "Review this geometry-only screen; no automatic longer training or deployment"
    result = dict(complete=True, revision=REVISION, run_signature=m["signature_sha256"], recipe=RECIPE,
                  guardrails=GUARDRAILS, results=dict(label_only=control, all_heads=all_heads, geometry_only=geometry),
                  versus_label_only=versus_gt, versus_all_heads=versus_all,
                  geometry_benefit_guardrail_passed=versus_gt["kd_benefit_guardrail_passed"],
                  historical_references=True, historical_reference_signature=m["reference_run_signature"],
                  reference_cpu_environment_matched=True, single_seed=True,
                  comparison_limit="One new arm compared with two reviewed historical arms; not a fresh simultaneous repeat or a per-head causal attribution",
                  initial_weights_and_subset_and_budget_matched=True, validation_selection=False,
                  full_training_authorized=False, deployment_qualified=False, phone_retest_required=True)
    engine.save_report(a.output / "geometry_comparison.json", result)
    print(json.dumps({k: result[k] for k in ("versus_label_only", "versus_all_heads",
                                          "geometry_benefit_guardrail_passed", "comparison_limit")}, indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_geometry_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob("*")):
            if source.is_file() and "assets" not in source.relative_to(a.output).parts and source.suffix in (".json", ".csv", ".txt", ".log"):
                archive.write(source, source.relative_to(a.output))
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "tiny", "cache", "kd-smoke", "train", "evaluate", "compare", "bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name in ("pilot", "initialization", "compact", "repo", "dataset", "split-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args(); a.role = "kd"
    for name in ("output", "pilot", "initialization", "compact", "repo", "dataset", "split_dir"):
        if getattr(a, name) is not None: setattr(a, name, getattr(a, name).resolve())
    a.candidate = [p.resolve() for p in a.candidate]; a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == "prepare" and any(getattr(a, k) is None for k in ("pilot", "initialization", "compact", "repo", "dataset", "split_dir")):
        parser.error("prepare requires the reviewed512 pilot, original initialization, compact export, source and data paths")
    try:
        with recipe_context():
            if a.stage == "prepare": prepare(a)
            elif a.stage == "bundle": bundle(a)
            else:
                m = load(a.output)
                dict(smoke=engine.smoke, tiny=engine.tiny, cache=cache, **{"kd-smoke": kd_smoke},
                     train=train, evaluate=engine.evaluate, compare=compare)[a.stage](a, m)
    except Exception as error:
        import traceback
        engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json",
                          dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__": main()
