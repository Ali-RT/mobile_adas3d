"""Full-train A2-Lite recovery with a matched control and recall guardrail.

The reviewed 512-frame pilot remains immutable. This adapter reuses its
portable model, losses, optimizer, cache and evaluator in a scoped recipe
context, rather than editing frozen pilot code. Only the training set and
fixed update budget change. No automatic third arm or deployment follows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import recover_a2_lite_640 as engine

REVISION = "A2-LITE-FULL-RECOVERY-KD-2026-10-13-r1"
PILOT_MANIFEST_SHA = "4d3a4fde2fc6ad2327c5089bc621508e1d49560cb29d3b2460e2be9072cf2d0f"
PILOT_COMPARISON_SHA = "e0a1ea85f02dffff3c71e64f02fb96120e2adbb48a4502ad670eff0828b5b25b"
PILOT_CACHE_SHA = "dc651b93e7c25435f2b99dadcf2f5f2b82e32d82a8d7707c60095933fea17156"
PILOT_SIGNATURE = "ef9f72b57ae6c6a6d545b4d86378b7cb144f3e59dd836559d53878e4ed305600"
RECIPE = copy.deepcopy(engine.RECIPE)
RECIPE.update(train_images=3712, epochs=10, frame_sampling="all Chen training frames exactly once per epoch",
              training_objective_changed=False, paired_endpoint_selection=False)
for _key in ("pedestrian_containing_frames", "vehicle_only_frames"):
    RECIPE.pop(_key)
GUARDRAILS = dict(maximum_nearby_recall_drop=0.02, minimum_mean_3d_ap_gain=0.1,
                  all_five_ap_metrics_nonregressing=True,
                  nearby_protocol="2D IoU>=0.5, score>=0.001; Vehicle<=40m, Pedestrian<=30m",
                  validation_checkpoint_selection=False, automatic_geometry_only_arm=False)


@contextmanager
def recipe_context():
    """Bind the reusable engine for one stage; never mutate pilot files."""
    old_recipe, old_revision = engine.RECIPE, engine.REVISION
    engine.RECIPE, engine.REVISION = RECIPE, REVISION
    try:
        yield
    finally:
        engine.RECIPE, engine.REVISION = old_recipe, old_revision


def implementation():
    return {**engine.implementation(), "scripts/recover_a2_lite_full.py": engine.sha256(__file__)}


def reviewed_pilot(source):
    raw_manifest = engine.read_record(source, "recovery_manifest.json")
    raw_comparison = engine.read_record(source, "recovery_comparison.json")
    m = engine.signed_record(raw_manifest, PILOT_MANIFEST_SHA)
    if engine.accuracy.sha_bytes(raw_comparison) != PILOT_COMPARISON_SHA:
        raise RuntimeError("Use the exact reviewed recovery comparison; do not overwrite the pilot")
    comparison = json.loads(raw_comparison)
    if (m["signature_sha256"] != PILOT_SIGNATURE or comparison["run_signature"] != PILOT_SIGNATURE
            or not comparison["complete"] or m["recipe"]["train_images"] != 512
            or m["recipe"]["epochs"] != 5 or m["initialization_sha256"] != engine.resolution.CHECKPOINT_SHA):
        raise RuntimeError("Wrong reviewed 512-image pilot identity")
    for role in ("control", "kd"):
        result = comparison["results"][role]
        if (not result["complete"] or result["evaluated_images"] != 3769
                or result["run_signature"] != PILOT_SIGNATURE):
            raise RuntimeError("Both reviewed pilot endpoint evaluations are required")
    return m, comparison, raw_manifest, raw_comparison


def all_train_ids(train, val):
    if (len(train) != 3712 or len(set(train)) != 3712 or len(val) != 3769
            or len(set(val)) != 3769 or set(train) & set(val)):
        raise RuntimeError("Expected disjoint Chen train3712/val3769; never select training frames from validation")
    return list(train)


def prepare(a):
    previous, comparison, raw_m, raw_c = reviewed_pilot(a.pilot)
    for path in (a.pilot, a.compact, a.teacher_export, a.initialization.parent):
        if path.is_dir() and a.output.is_relative_to(path):
            raise RuntimeError("Use a separate full-recovery output; preserve completed experiments")
    if not a.initialization.is_file() or engine.sha256(a.initialization) != engine.resolution.CHECKPOINT_SHA:
        raise RuntimeError("Missing/changed original pruned initialization; do not start from either pilot endpoint")
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != previous["source_sha256"]:
        raise RuntimeError("Fresh portable source differs from the reviewed student")
    train = engine.base.q.split_ids(a.split_dir / "train.txt", "train")
    val = engine.base.q.split_ids(a.split_dir / "val.txt", "val")
    selected = all_train_ids(train, val)
    if train != previous["train_ids"] or val != previous["val_ids"]:
        raise RuntimeError("Chen split bytes/ordering differ from the reviewed pilot")
    for attempt in range(3):
        try:
            files = engine.accuracy.stage_data(a.dataset, a.candidate, train + val)
            break
        except OSError:
            if attempt == 2:
                raise
            print("Transient data read failed; retrying while preserving staged files.", flush=True)
            time.sleep(attempt + 1)
    if files != previous["data_file_sha256"]:
        raise RuntimeError("Train/validation data bytes differ from the reviewed pilot")
    for split, ids in (("train", train), ("val", val)):
        engine.accuracy.immutable_bytes(a.dataset / "ImageSets" / f"{split}.txt", ("\n".join(ids) + "\n").encode())
    student_root, _ = engine.accuracy.checked_package(a.compact, "compact_640", a.output / "assets/student")
    teacher_root = engine.materialize_export(a.teacher_export, a.output / "assets/teacher")
    teacher = engine.A2Teacher(teacher_root)
    if teacher.package["model"]["sha256"] != previous["teacher_model_sha256"]:
        raise RuntimeError("The original A2 teacher changed")
    for fixture in teacher.package["fixtures"]:
        inputs = {n: engine.read_f32(teacher.root, fixture["inputs"][n]) for n in engine.INPUT_NAMES}
        refs = {n: engine.read_f32(teacher.root, fixture["cpu_reference_outputs"][n]) for n in engine.OUTPUT_NAMES}
        if not engine.compare_outputs(teacher.raw(inputs), refs)["passed"]:
            raise RuntimeError("Frozen A2 teacher fixture parity failed")
    config = copy.deepcopy(previous["config"])
    fields = dict(loss_ce="cls_loss_coef", loss_center="3dcenter_loss_coef", loss_bbox="bbox_loss_coef",
                  loss_giou="giou_loss_coef", loss_depth="depth_loss_coef", loss_dim="dim_loss_coef",
                  loss_angle="angle_loss_coef")
    if (config["dataset"]["meanshape"] is not False
            or any(config["model"][fields[k]] != v for k, v in RECIPE["gt_weights"].items())):
        raise RuntimeError("Original A2 target units or GT loss weights changed")
    engine.accuracy.immutable_bytes(a.output / "reviewed_pilot/recovery_manifest.json", raw_m)
    engine.accuracy.immutable_bytes(a.output / "reviewed_pilot/recovery_comparison.json", raw_c)
    m = dict(revision=REVISION, recipe=RECIPE, guardrails=GUARDRAILS, implementation=implementation(),
             cpu_environment=engine.runtime_environment(), config=config, source=str(a.repo),
             source_sha256=previous["source_sha256"], initialization=str(a.initialization),
             initialization_sha256=engine.resolution.CHECKPOINT_SHA, teacher_export=str(teacher_root),
             teacher_model_sha256=previous["teacher_model_sha256"], student_export=str(student_root),
             dataset=str(a.dataset), data_file_sha256=files, split_dir=str(a.split_dir),
             train_ids=train, val_ids=val, pilot_ids=selected, output=str(a.output),
             reviewed_pilot_manifest_sha256=PILOT_MANIFEST_SHA,
             reviewed_pilot_comparison_sha256=PILOT_COMPARISON_SHA,
             previous_pilot_metrics={role: comparison["results"][role]["metrics"] for role in ("control", "kd")},
             target_selection_train_only=True, all_training_frames_used=True,
             frame_groups_balanced=False, object_counts_balanced=False,
             gt_class_weighting_changed=False, kd_class_balanced_means=True,
             bounded_training_authorized=True, longer_training_authorized=False, deployment_qualified=False)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(a.output / "full_recovery_manifest.json", m)
    print("Prepared: same640 student/A2 teacher, all3712 TRAIN frames, two fixed10-epoch endpoints.", flush=True)


def load(output):
    m = json.loads((output / "full_recovery_manifest.json").read_text())
    if (engine.signature({k: v for k, v in m.items() if k != "signature_sha256"}) != m["signature_sha256"]
            or m["revision"] != REVISION or m["recipe"] != RECIPE or m["guardrails"] != GUARDRAILS
            or m["implementation"] != implementation() or m["cpu_environment"] != engine.runtime_environment()):
        raise RuntimeError("Full-recovery code/packages/recipe changed; preserve run and use a new RUN_ID")
    reviewed_pilot(output / "reviewed_pilot")
    if (m["output"] != str(output) or engine.sha256(m["initialization"]) != engine.resolution.CHECKPOINT_SHA
            or engine.base.q.source_hash(Path(m["source"])) != m["source_sha256"]):
        raise RuntimeError("Frozen full-recovery paths, initialization or portable source changed")
    for split, key in (("train", "train_ids"), ("val", "val_ids")):
        if engine.base.q.split_ids(Path(m["split_dir"]) / f"{split}.txt", split) != m[key]:
            raise RuntimeError("Chen split changed")
    if m["pilot_ids"] != all_train_ids(m["train_ids"], m["val_ids"]):
        raise RuntimeError("Full recovery must use all3712 TRAIN frames, not a validation-selected subset")
    return m


def reuse_teacher_cache(a, m):
    """Rebind only hash-verified old TRAIN records with identical inputs/teacher."""
    if a.pilot is None:
        return 0
    old, _, _, _ = reviewed_pilot(a.pilot)
    raw = engine.read_record(a.pilot, "teacher_cache_complete.json")
    if engine.accuracy.sha_bytes(raw) != PILOT_CACHE_SHA:
        raise RuntimeError("Use the exact reviewed completed teacher-cache report")
    report = json.loads(raw)
    if (engine.signature({k: v for k, v in report.items() if k != "report_sha256"}) != report["report_sha256"]
            or report["run_signature"] != old["signature_sha256"] or not report["complete"]
            or set(report["cache_sha256"]) != set(old["pilot_ids"])
            or old["teacher_model_sha256"] != m["teacher_model_sha256"]):
        raise RuntimeError("Unverified previous teacher cache")
    reused = 0
    archive = zipfile.ZipFile(a.pilot) if a.pilot.is_file() else None
    try:
        for sample, expected in report["cache_sha256"].items():
            if sample not in m["pilot_ids"]:
                raise RuntimeError("Previous guidance is not a current TRAIN image")
            name = f"teacher_cache/{sample}.json"
            if archive is None:
                data = engine.read_record(a.pilot, name)
            else:
                matches = [n for n in archive.namelist() if n == name or n.endswith("/" + name)]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one previous teacher record: {sample}")
                data = archive.read(matches[0])
            if engine.accuracy.sha_bytes(data) != expected:
                raise RuntimeError(f"Changed previous teacher cache: {sample}")
            row = json.loads(data)
            if (engine.signature({k: v for k, v in row.items() if k != "record_sha256"}) != row["record_sha256"]
                    or row["run_signature"] != old["signature_sha256"] or row["sample_id"] != sample
                    or row["source_sha256"] != m["data_file_sha256"][sample]):
                raise RuntimeError("Previous cache source identity differs")
            destination = a.output / name
            if destination.exists():
                engine.cache_record(m, a.output / "teacher_cache", sample)
                continue
            row["run_signature"] = m["signature_sha256"]
            row["reused_from_run_signature"] = old["signature_sha256"]
            row.pop("record_sha256")
            row["record_sha256"] = engine.signature(row)
            engine.write_json(destination, row)
            reused += 1
    finally:
        if archive is not None:
            archive.close()
    print(f"Reused {reused} verified TRAIN-only teacher records; no student checkpoint reused.", flush=True)
    return reused


def decision(control, kd):
    for value in (control, kd):
        if (set(value["metrics"]) != set(engine.accuracy.GATES)
                or any(not math.isfinite(v) or not 0 <= v <= 100 for v in value["metrics"].values())
                or any(not math.isfinite(value["nearby"]["classes"][name]["near_recall"])
                       or not 0 <= value["nearby"]["classes"][name]["near_recall"] <= 1
                       for name in ("Vehicle", "Pedestrian"))):
            raise RuntimeError("Invalid AP/nearby-recall comparison values")
    deltas = {k: kd["metrics"][k] - v for k, v in control["metrics"].items()}
    recall = {}
    guards = {}
    for name in ("Vehicle", "Pedestrian"):
        c = control["nearby"]["classes"][name]["near_recall"]
        d = kd["nearby"]["classes"][name]["near_recall"] - c
        recall[name] = d
        guards[name] = d >= -GUARDRAILS["maximum_nearby_recall_drop"] - 1e-12
    ap = all(d >= -1e-12 for d in deltas.values())
    gain = deltas["mean_3d_moderate"] >= GUARDRAILS["minimum_mean_3d_ap_gain"] - 1e-12
    passed = ap and gain and all(guards.values())
    return dict(kd_minus_control=deltas, nearby_recall_delta=recall,
                all_ap_metrics_nonregressing=ap, minimum_mean_3d_gain_passed=gain,
                nearby_recall_guardrails=guards, kd_benefit_guardrail_passed=passed,
                next_review="Review accuracy and export parity before any phone test" if passed else
                "Do not promote KD automatically; review GT control and a separately authorized geometry-only ablation")


def compare(a, m):
    results = {}
    for role in ("control", "kd"):
        training = engine.checked_report(a.output / role / "training_complete.json", m)
        value = json.loads((a.output / "evaluation" / role / "accuracy.json").read_text())
        if (not training["complete"] or training["train_images"] != 3712
                or training["optimizer_steps"] != RECIPE["train_images"] * RECIPE["epochs"]
                or training["fixed_endpoint_epoch"] != RECIPE["epochs"]
                or training["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
                or training["external_teacher_used"] != (role == "kd") or training["tiny_checkpoint_used"]
                or training["architecture_changed"]
                or engine.sha256(a.output / role / "last.pth") != training["checkpoint_sha256"]
                or not value["complete"] or value["evaluated_images"] != 3769
                or value["run_signature"] != m["signature_sha256"]
                or value["checkpoint_sha256"] != training["checkpoint_sha256"]):
            raise RuntimeError("Two completed matched fixed10-epoch evaluations are required")
        results[role] = value
    result = decision(results["control"], results["kd"])
    result.update(complete=True, revision=REVISION, run_signature=m["signature_sha256"], recipe=RECIPE,
                  guardrails=GUARDRAILS, results=results, previous_pilot_metrics=m["previous_pilot_metrics"],
                  same_initialization_sha256=engine.resolution.CHECKPOINT_SHA, tiny_checkpoint_used=False,
                  pilot_endpoint_checkpoint_used=False, validation_selection=False,
                  all_product_accuracy_gates_passed={role: all(v["accuracy_gate_results"].values()) for role, v in results.items()},
                  all_product_nearby_recall_gates_passed={role: v["nearby"]["all_near_recall_gates_passed"] for role, v in results.items()},
                  longer_training_authorized=False, geometry_only_training_authorized=False,
                  phone_retest_required=True, deployment_qualified=False)
    engine.save_report(a.output / "full_recovery_comparison.json", result)
    print(json.dumps({k: result[k] for k in ("kd_minus_control", "nearby_recall_delta", "nearby_recall_guardrails",
                                          "kd_benefit_guardrail_passed", "next_review")}, indent=2), flush=True)


def bundle(a):
    destination = a.output / "a2_lite_full_recovery_results.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob("*")):
            if (source.is_file() and "assets" not in source.relative_to(a.output).parts
                    and source.suffix in (".json", ".csv", ".txt", ".log")):
                archive.write(source, source.relative_to(a.output))
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "tiny", "cache", "train", "evaluate", "compare", "bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--role", choices=("control", "kd"))
    for name in ("pilot", "initialization", "compact", "teacher-export", "repo", "dataset", "split-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args()
    for name in ("output", "pilot", "initialization", "compact", "teacher_export", "repo", "dataset", "split_dir"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    a.candidate = [p.resolve() for p in a.candidate]
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == "prepare" and any(getattr(a, n) is None for n in
                                    ("pilot", "initialization", "compact", "teacher_export", "repo", "dataset", "split_dir")):
        parser.error("prepare requires reviewed pilot, original initialization, exports, source and data paths")
    if a.stage in ("train", "evaluate") and a.role is None:
        parser.error("train/evaluate require --role")
    try:
        with recipe_context():
            if a.stage == "prepare":
                prepare(a)
            elif a.stage == "bundle":
                bundle(a)
            else:
                m = load(a.output)
                if a.stage == "cache":
                    engine.tiny_gate(m)
                    reuse_teacher_cache(a, m)
                dict(smoke=engine.smoke, tiny=engine.tiny, cache=engine.cache, train=engine.train,
                     evaluate=engine.evaluate, compare=compare)[a.stage](a, m)
    except Exception as error:
        import traceback
        engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json",
                          dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
