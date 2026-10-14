"""One reviewed 512-frame/five-epoch orientation-rebalanced A2-Lite arm.

Only the teacher angle coefficient changes from1 to.01. Reuse the frozen
geometry engine in a scoped context; never edit completed scripts, source,
checkpoints or manifests. Full-data training and deployment stay disabled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import audit_a2_lite_gradient_balance as audit
import recover_a2_lite_geometry as geometry
from models.a2_lite_balanced_geometry_kd import balanced_geometry_kd_loss, COMPONENT_WEIGHTS

engine = geometry.engine
reference = geometry.reference
ORIGINAL_RECIPE = copy.deepcopy(geometry.RECIPE)
ORIGINAL_IMPLEMENTATION = geometry.implementation
REVISION = "A2-LITE-512-ANGLE001-KD-2026-10-14-r1"
AUDIT_SIGNATURE = "fa4f61d91be35c7f0144b1e53a7deeeb7abbdd6707785c2ef8cd9e02e0c65d6a"
AUDIT_REPORTS = {
    "gradient_manifest.json": "947f0b2789428328f150dc9d19b6d4e0a64eae673c62a9d8e1c3b6ba29d0ebb6",
    "initial_gradient_report.json": "241c7ea08e360f8af025df4e302c6203dfba5661558eeef8a911f46b8434a627",
    "geometry_epoch5_gradient_report.json": "d2333e4e2d7df4b1e1168f1b75860eceeabf865f30928a117f0fc6346e55a5a9",
}
RECIPE = copy.deepcopy(ORIGINAL_RECIPE)
RECIPE.update(teacher_component_weights=COMPONENT_WEIGHTS,
              only_objective_change="teacher angle multiplier1.0 to0.01; no renormalization",
              reviewed_gradient_signature=AUDIT_SIGNATURE)
GUARDRAILS = copy.deepcopy(geometry.GUARDRAILS)


def implementation():
    return {**ORIGINAL_IMPLEMENTATION(),
            "scripts/recover_a2_lite_balanced_geometry.py": engine.sha256(__file__),
            "models/a2_lite_balanced_geometry_kd.py": engine.sha256(ROOT / "models/a2_lite_balanced_geometry_kd.py"),
            "scripts/audit_a2_lite_gradient_balance.py": engine.sha256(ROOT / "scripts/audit_a2_lite_gradient_balance.py")}


@contextmanager
def recipe_context():
    """Bind the existing adapters without changing their files or identities."""
    old = geometry.RECIPE, geometry.REVISION, geometry.geometry_kd_loss, geometry.implementation
    geometry.RECIPE, geometry.REVISION = RECIPE, REVISION
    geometry.geometry_kd_loss, geometry.implementation = balanced_geometry_kd_loss, implementation
    try:
        with geometry.recipe_context():
            yield
    finally:
        geometry.RECIPE, geometry.REVISION, geometry.geometry_kd_loss, geometry.implementation = old


def reviewed_audit(source):
    """Verify the reviewed byte identities AND all64 frame-report rollups."""
    raw, values = {}, {}
    for name, digest in AUDIT_REPORTS.items():
        raw[name] = engine.read_record(source, name)
        if engine.accuracy.sha_bytes(raw[name]) != digest:
            raise RuntimeError(f"Use the exact reviewed gradient results: {name}")
        values[name] = audit.signed(json.loads(raw[name]), "signature_sha256" if name == "gradient_manifest.json" else "report_sha256")
    m = values["gradient_manifest.json"]
    expected_code = {**ORIGINAL_IMPLEMENTATION(),
                     "scripts/audit_a2_lite_gradient_balance.py": engine.sha256(ROOT / "scripts/audit_a2_lite_gradient_balance.py"),
                     "scripts/stage_a2_lite_data.py": engine.sha256(ROOT / "scripts/stage_a2_lite_data.py")}
    if (m["signature_sha256"] != AUDIT_SIGNATURE or m["revision"] != audit.REVISION
            or m["recipe"] != audit.RECIPE or m["implementation"] != expected_code
            or m["historical_run_signature"] != audit.GEOMETRY_SIGNATURE):
        raise RuntimeError("Reviewed gradient recipe/code/provenance changed")
    historical = {}
    for name, digest in audit.REVIEWED.items():
        key = "reviewed_geometry/" + name
        raw[key] = engine.read_record(source, key)
        if engine.accuracy.sha_bytes(raw[key]) != digest:
            raise RuntimeError("Original completed geometry evidence changed")
        historical[name] = audit.signed(json.loads(raw[key]), "signature_sha256" if name == "geometry_manifest.json" else "report_sha256")
    previous = historical["geometry_manifest.json"]
    if (previous["recipe"] != ORIGINAL_RECIPE or previous["implementation"] != ORIGINAL_IMPLEMENTATION()
            or previous["signature_sha256"] != audit.GEOMETRY_SIGNATURE
            or m["cpu_environment"] != previous["cpu_environment"]
            or m["historical_source_sha256"] != previous["source_sha256"]
            or historical["geometry_comparison.json"]["geometry_benefit_guardrail_passed"] is not False
            or historical["kd/training_complete.json"]["checkpoint_sha256"] != audit.ENDPOINT_SHA):
        raise RuntimeError("Reviewed geometry baseline differs")
    ids = m["view"]["pilot_ids"]
    rows = {}
    for sample in ids:
        name = f"teacher_subset/{sample}.json"
        raw[name] = engine.read_record(source, name)
        digest = engine.accuracy.sha_bytes(raw[name])
        if digest != m["teacher_record_sha256"][sample] or digest != historical["teacher_cache_complete.json"]["cache_sha256"][sample]:
            raise RuntimeError("Gradient audit teacher bytes changed")
        rows[sample] = audit.signed(json.loads(raw[name]), "record_sha256")
        if (rows[sample]["sample_id"] != sample or rows[sample]["run_signature"] != audit.GEOMETRY_SIGNATURE
                or rows[sample]["source_sha256"] != previous["data_file_sha256"][sample]
                or rows[sample]["reused_from_run_signature"] != reference.PILOT_SIGNATURE):
            raise RuntimeError("Gradient audit teacher provenance differs")
    if audit.selected_ids(previous, rows) != ids:
        raise RuntimeError("Gradient audit is not the reviewed train-only subset")
    for role in audit.ROLES:
        report = values[f"{role}_gradient_report.json"]
        if (not report["complete"] or report["run_signature"] != AUDIT_SIGNATURE or report["role"] != role
                or report["frames"] != 32 or report["optimizer_steps"] != 0
                or report["weights_and_buffers_unchanged"] is not True):
            raise RuntimeError("Completed zero-update gradient audit required")
        frames = []
        for sample in ids:
            name = f"frames/{role}/{sample}.json"
            raw[name] = engine.read_record(source, name)
            if engine.accuracy.sha_bytes(raw[name]) != report["frame_sha256"][sample]:
                raise RuntimeError("Gradient frame bytes changed")
            frame = audit.signed(json.loads(raw[name]))
            if (not frame["complete"] or frame["sample_id"] != sample or frame["role"] != role
                    or frame["run_signature"] != AUDIT_SIGNATURE or frame["optimizer_steps"] != 0
                    or frame["model_state_sha256"] != report["state_sha256"]):
                raise RuntimeError("Gradient frame/checkpoint identity differs")
            frames.append(frame)
        if audit.summarize(frames) != report["summary"]:
            raise RuntimeError("Gradient report rollup differs from frame evidence")
    return previous, historical["geometry_comparison.json"], m, raw


def prepare(a):
    old_geometry, _, reviewed, records = reviewed_audit(a.audit)
    pilot, _, _, _ = reference.reviewed_pilot(a.pilot)
    geometry.require_reference_environment(reviewed["cpu_environment"], engine.runtime_environment())
    if (old_geometry["reference_run_signature"] != pilot["signature_sha256"]
            or old_geometry["pilot_ids"] != pilot["pilot_ids"]
            or old_geometry["data_file_sha256"] != pilot["data_file_sha256"]
            or old_geometry["source_sha256"] != pilot["source_sha256"]):
        raise RuntimeError("Use the same original512 data/source as the reviewed geometry audit")
    protected = [Path(value["output"]) for value in (old_geometry, reviewed, pilot)]
    protected += [p for p in (a.audit, a.pilot, a.compact) if p.is_dir()]
    protected.append(a.initialization.parent)
    if any(a.output.is_relative_to(p) for p in protected):
        raise RuntimeError("Historical runs/audit are read-only; use a separate rebalanced output folder")
    source_paths = [Path(old_geometry["source"]), Path(pilot["source"]), Path(reviewed["view"]["source"])]
    dataset_paths = [Path(old_geometry["dataset"]), Path(pilot["dataset"]), Path(reviewed["view"]["dataset"])]
    if any(a.repo.is_relative_to(p) for p in source_paths) or any(a.dataset.is_relative_to(p) for p in dataset_paths):
        raise RuntimeError("Use new local model/data paths; preserve historical checkouts and datasets")
    a.output_verified = True
    a.output.mkdir(parents=True, exist_ok=True)
    for name, data in records.items():
        engine.accuracy.immutable_bytes(a.output / "reviewed_gradient" / name, data)
    with recipe_context():
        geometry.prepare(a)
    print("Prepared ONE new arm: angle teacher multiplier0.01; label orientation unchanged;512 frames/five epochs.", flush=True)


def load(output):
    previous, _, reviewed, _ = reviewed_audit(output / "reviewed_gradient")
    with recipe_context():
        m = geometry.load(output)
    if (m["reference_run_signature"] != previous["reference_run_signature"]
            or m["pilot_ids"] != previous["pilot_ids"] or m["data_file_sha256"] != previous["data_file_sha256"]
            or m["cpu_environment"] != reviewed["cpu_environment"] or m["source_sha256"] != previous["source_sha256"]):
        raise RuntimeError("Rebalanced run differs from the approved comparison setup")
    return m


def compare(a, m):
    _, historical, _, _ = reviewed_audit(a.output / "reviewed_gradient")
    training = engine.checked_report(a.output / "kd/training_complete.json", m)
    result = json.loads((a.output / "evaluation/kd/accuracy.json").read_text())
    if (not training["complete"] or training["train_images"] != 512 or training["optimizer_steps"] != 2560
            or training["fixed_endpoint_epoch"] != 5 or training["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
            or training["tiny_checkpoint_used"] or training["architecture_changed"] or not training["external_teacher_used"]
            or engine.sha256(a.output / "kd/last.pth") != training["checkpoint_sha256"]
            or not result["complete"] or result["evaluated_images"] != 3769 or result["run_signature"] != m["signature_sha256"]
            or result["checkpoint_sha256"] != training["checkpoint_sha256"]
            or any(sum(training["active_kd_counts"][k].values()) for k in geometry.DISABLED_COMPONENTS)
            or any(sum(training["active_kd_counts"][k].values()) <= 0 for k in geometry.ENABLED_COMPONENTS)):
        raise RuntimeError("Completed fixed5-epoch rebalanced geometry endpoint required")
    references = dict(label_only=m["reference_results"]["control"], all_heads=m["reference_results"]["kd"],
                      geometry_angle1=historical["results"]["geometry_only"])
    decisions = {name: reference.decision(value, result) for name, value in references.items()}
    for decision in decisions.values():
        decision["next_review"] = "Review this single coefficient test; no automatic longer training or deployment"
    report = dict(complete=True, revision=REVISION, run_signature=m["signature_sha256"], recipe=RECIPE,
                  teacher_component_weights=COMPONENT_WEIGHTS, guardrails=GUARDRAILS,
                  results={**references, "geometry_angle001": result}, comparisons=decisions,
                  benefit_guardrail_passed=decisions["label_only"]["kd_benefit_guardrail_passed"],
                  historical_reference_signature=reference.PILOT_SIGNATURE,
                  reviewed_gradient_signature=AUDIT_SIGNATURE, reference_cpu_environment_matched=True,
                  single_seed=True, historical_references=True, initial_weights_and_subset_and_budget_matched=True,
                  comparison_limit="One new coefficient arm against reviewed historical endpoints; not a fresh simultaneous repeat or causal proof",
                  validation_checkpoint_selection=False, full_training_authorized=False, deployment_qualified=False,
                  phone_retest_required=True)
    engine.save_report(a.output / "balanced_geometry_comparison.json", report)
    print(json.dumps(dict(comparisons=decisions, benefit_guardrail_passed=report["benefit_guardrail_passed"]), indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_balanced_geometry_results.zip"
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
    for name in ("audit", "pilot", "initialization", "compact", "repo", "dataset", "split-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args(); a.role = "kd"
    for key in ("output", "audit", "pilot", "initialization", "compact", "repo", "dataset", "split_dir"):
        if getattr(a, key) is not None: setattr(a, key, getattr(a, key).resolve())
    a.candidate = [p.resolve() for p in a.candidate]; a.output_verified = False
    if a.stage == "prepare" and any(getattr(a, k) is None for k in ("audit", "pilot", "initialization", "compact", "repo", "dataset", "split_dir")):
        parser.error("prepare requires approved gradient results, original512 pilot, pruned initialization, compact export, source and data paths")
    try:
        if a.stage == "prepare": prepare(a)
        elif a.stage == "bundle":
            saved = a.output / "geometry_manifest.json"
            if saved.exists() and json.loads(saved.read_text()).get("revision") != REVISION:
                raise RuntimeError("Preserve the historical output; bundle only this new rebalanced run")
            a.output.mkdir(parents=True, exist_ok=True)
            bundle(a)
        else:
            saved = a.output / "geometry_manifest.json"
            a.output_verified = saved.is_file() and json.loads(saved.read_text()).get("revision") == REVISION
            m = load(a.output)
            with recipe_context():
                dict(smoke=engine.smoke, tiny=engine.tiny, cache=geometry.cache, **{"kd-smoke": geometry.kd_smoke},
                     train=geometry.train, evaluate=engine.evaluate, compare=compare)[a.stage](a, m)
    except Exception as error:
        import traceback
        if a.output_verified:
            engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json",
                              dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__": main()
