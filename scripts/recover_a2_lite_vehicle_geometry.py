"""One Vehicle-only teacher ablation against four reviewed 512-frame arms.

No Pedestrian teacher contribution and no redistribution to Vehicle.
Both classes still receive all GT losses. Same original initialization,
five epochs, portable model and evaluator; completed runs stay read-only.
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
import torch
import recover_a2_lite_balanced_geometry as balanced
from models.a2_lite_vehicle_geometry_kd import vehicle_geometry_kd_loss, CLASS_WEIGHTS

geometry = balanced.geometry
engine = balanced.engine
reference = balanced.reference
REVISION = "A2-LITE-512-VEHICLE-KD-2026-10-14-r1"
PARENT_SIGNATURE = "1c9fe480b93344bec6a26e053797079c2f670c70e8704afbdd9935fcb736ab36"
PARENT_REPORTS = {
    "geometry_manifest.json": "ab661cb404bcd55e8b46f6448c89546e9435e2e68c5ea11209a76e31140486a1",
    "balanced_geometry_comparison.json": "bc97097f25d4815a130f8229ec8a1fb9a7d9626d4fc378890c9ddd23d7ea803c",
    "kd/training_complete.json": "6e645b4099674286dae057ab952906955145f89dbcbe73cc5498880e30e90636",
    "teacher_cache_complete.json": "2b7a82bdbb3bddfd6330945102b40012dbad390b65f66cf279cb8afcccc32973",
}
RECIPE = copy.deepcopy(balanced.RECIPE)
RECIPE.update(teacher_class_weights=CLASS_WEIGHTS,
              teacher_class_normalization="retain original eligible-class denominator per component; no redistribution",
              only_objective_change="Pedestrian teacher contribution to zero; original Vehicle contribution preserved",
              reviewed_balanced_signature=PARENT_SIGNATURE, gt_class_weighting_changed=False)
GUARDRAILS = copy.deepcopy(balanced.GUARDRAILS)


def implementation():
    return {**balanced.implementation(), "scripts/recover_a2_lite_vehicle_geometry.py": engine.sha256(__file__),
            "models/a2_lite_vehicle_geometry_kd.py": engine.sha256(ROOT / "models/a2_lite_vehicle_geometry_kd.py")}


@contextmanager
def recipe_context():
    old = geometry.RECIPE, geometry.REVISION, geometry.geometry_kd_loss, geometry.implementation
    geometry.RECIPE, geometry.REVISION = RECIPE, REVISION
    geometry.geometry_kd_loss, geometry.implementation = vehicle_geometry_kd_loss, implementation
    try:
        with geometry.recipe_context(): yield
    finally:
        geometry.RECIPE, geometry.REVISION, geometry.geometry_kd_loss, geometry.implementation = old


def parent_record(source, name):
    """Resolve the parent root, not the nested older geometry manifest."""
    if source.is_dir(): return (source / name).read_bytes()
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)): raise RuntimeError("Duplicate parent ZIP entries")
        roots = [n[:-len("balanced_geometry_comparison.json")] for n in names
                 if n == "balanced_geometry_comparison.json" or n.endswith("/balanced_geometry_comparison.json")]
        if len(roots) != 1 or roots[0] + name not in names:
            raise RuntimeError(f"Expected one reviewed balanced parent root: {name}")
        return archive.read(roots[0] + name)


def reviewed_balanced(source):
    raw, records = {}, {}
    for name, digest in PARENT_REPORTS.items():
        raw[name] = parent_record(source, name)
        if engine.accuracy.sha_bytes(raw[name]) != digest:
            raise RuntimeError(f"Use the exact reviewed balanced geometry results: {name}")
        records[name] = balanced.audit.signed(json.loads(raw[name]), "signature_sha256" if name == "geometry_manifest.json" else "report_sha256")
    m = records["geometry_manifest.json"]; c = records["balanced_geometry_comparison.json"]
    t = records["kd/training_complete.json"]; cache = records["teacher_cache_complete.json"]
    if (m["signature_sha256"] != PARENT_SIGNATURE or m["recipe"] != balanced.RECIPE
            or m["implementation"] != balanced.implementation() or m["revision"] != balanced.REVISION
            or any(v["run_signature"] != PARENT_SIGNATURE for k, v in records.items() if k != "geometry_manifest.json")
            or c["benefit_guardrail_passed"] is not False or not c["complete"]
            or not t["complete"] or t["train_images"] != 512 or t["optimizer_steps"] != 2560
            or t["fixed_endpoint_epoch"] != 5 or t["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
            or t["tiny_checkpoint_used"] or not t["external_teacher_used"] or t["architecture_changed"]
            or c["results"]["geometry_angle001"]["checkpoint_sha256"] != t["checkpoint_sha256"]
            or not cache["complete"] or cache["images"] != 512):
        raise RuntimeError("Completed reviewed balanced parent identity differs")
    old, old_comparison, audit, audit_raw = balanced.reviewed_audit(source / "reviewed_gradient" if source.is_dir() else source)
    if (m["pilot_ids"] != old["pilot_ids"] or m["data_file_sha256"] != old["data_file_sha256"]
            or m["source_sha256"] != old["source_sha256"] or m["config"] != old["config"]
            or m["cpu_environment"] != audit["cpu_environment"]
            or c["results"]["geometry_angle1"] != old_comparison["results"]["geometry_only"]):
        raise RuntimeError("Balanced parent differs from its reviewed gradient/source references")
    raw.update({"reviewed_gradient/" + name: value for name, value in audit_raw.items()})
    return m, c, raw


def prepare(a):
    previous, _, raw = reviewed_balanced(a.balanced)
    pilot, _, _, _ = reference.reviewed_pilot(a.pilot)
    geometry.require_reference_environment(previous["cpu_environment"], engine.runtime_environment())
    if (previous["reference_run_signature"] != pilot["signature_sha256"]
            or any(previous[k] != pilot[k] for k in ("pilot_ids", "data_file_sha256", "source_sha256", "config"))):
        raise RuntimeError("Use the same original512 pilot data/source/initialization")
    protected = [Path(previous["output"]), Path(pilot["output"]), a.initialization.parent]
    protected += [p for p in (a.balanced, a.pilot, a.compact) if p.is_dir()]
    for key in ("reviewed_gradient/gradient_manifest.json", "reviewed_gradient/reviewed_geometry/geometry_manifest.json"):
        protected.append(Path(json.loads(raw[key])["output"]))
    if any(a.output.is_relative_to(p) for p in protected):
        raise RuntimeError("Historical runs are read-only; use a new Vehicle-only output")
    if (any(a.repo.is_relative_to(Path(v["source"])) for v in (previous, pilot))
            or any(a.dataset.is_relative_to(Path(v["dataset"])) for v in (previous, pilot))):
        raise RuntimeError("Use new local source/data paths; previous inputs are candidates only")
    a.output_verified = True; a.output.mkdir(parents=True, exist_ok=True)
    for name, value in raw.items(): engine.accuracy.immutable_bytes(a.output / "reviewed_balanced" / name, value)
    with recipe_context(): geometry.prepare(a)
    print("Prepared one Vehicle-only teacher arm; both GT classes unchanged; no weight redistribution.", flush=True)


def load(output):
    previous, _, _ = reviewed_balanced(output / "reviewed_balanced")
    with recipe_context(): m = geometry.load(output)
    if any(m[k] != previous[k] for k in ("pilot_ids", "data_file_sha256", "source_sha256", "config", "cpu_environment", "reference_results")):
        raise RuntimeError("Vehicle-only run differs from the approved matched setup")
    return m


def valid_vehicle_counts(counts):
    return (set(counts) == set(engine.COMPONENTS)
            and all(v.get("Pedestrian") == 0 for v in counts.values())
            and all(counts[k].get("Vehicle", 0) > 0 for k in balanced.COMPONENT_WEIGHTS)
            and all(sum(counts[k].values()) == 0 for k in geometry.DISABLED_COMPONENTS))


def kd_smoke(a, m):
    path = a.output / "vehicle_kd_smoke.json"
    if path.exists():
        old = engine.checked_report(path, m)
        if not old["passed"] or not valid_vehicle_counts(old["active_counts"]):
            raise RuntimeError("Preserve failed Vehicle-only smoke and return evidence")
        print("Unchanged Vehicle-only smoke reused.", flush=True); return
    coverage = engine.checked_report(a.output / "teacher_cache_complete.json", m)
    if not coverage["kd_pilot_authorized"]: raise RuntimeError("Verified original teacher coverage required")
    engine.setup_device(a.device); model = engine.build_model(m, a.device, True)
    before = balanced.audit.model_hash(model); buffers = engine.buffers_signature(model)
    dataset = engine.native_dataset(m, m["pilot_ids"][:64])
    totals = {k: dict(Pedestrian=0, Vehicle=0) for k in engine.COMPONENTS}
    signals = {k: 0. for k in balanced.COMPONENT_WEIGHTS}; checks = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    for index, sample in enumerate(dataset.idx_list):
        engine.verify_files(m, sample); inputs, target, _ = engine.item(dataset, index, a.device)
        row = engine.cache_record(m, a.output / "teacher_cache", sample)
        if row["target_signature"] != engine.target_signature(target): raise RuntimeError("Encoded GT changed")
        teacher = engine.native_arrays(row["outputs"], a.device)
        mapping = torch.tensor(row["object_to_teacher_query"], device=a.device)
        model.zero_grad(set_to_none=True)
        loss, parts, counts = vehicle_geometry_kd_loss(engine.forward(model, inputs), teacher, target, mapping)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite Vehicle-only loss")
        for k in totals:
            for cls in totals[k]: totals[k][cls] += counts[k][cls]
        local = {}
        for name in signals:
            if counts[name]["Vehicle"] and float(parts[name].detach()) > 0:
                gradients = torch.autograd.grad(parts[name], parameters, retain_graph=True, allow_unused=True)
                gradients = [g for g in gradients if g is not None]
                if not all(torch.isfinite(g).all() for g in gradients): raise RuntimeError("Nonfinite teacher gradient")
                local[name] = sum(float(g.detach().abs().sum()) for g in gradients)
                signals[name] = max(signals[name], local[name])
        checks.append(dict(sample_id=sample, weighted_parts={k: float(v.detach()) for k, v in parts.items()}, component_gradient_abs_sum=local))
        if valid_vehicle_counts(totals) and all(signals.values()): break
    unchanged = balanced.audit.model_hash(model) == before and engine.buffers_signature(model) == buffers
    passed = bool(valid_vehicle_counts(totals) and all(signals.values()) and unchanged)
    engine.save_report(path, dict(complete=True, passed=passed, run_signature=m["signature_sha256"], active_counts=totals,
                       component_gradient_abs_sum=signals, checks=checks, optimizer_steps=0,
                       weights_and_buffers_unchanged=unchanged, no_pedestrian_teacher_loss=True,
                       vehicle_weight_not_redistributed=True, teacher_class_weights=CLASS_WEIGHTS,
                       teacher_component_weights=balanced.COMPONENT_WEIGHTS,
                       gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else None, cuda_build=torch.version.cuda))
    if not passed: raise RuntimeError("Vehicle-only smoke failed; return evidence before training")
    print("Vehicle-only teacher gradients passed; Pedestrian teacher counts zero; weights unchanged.", flush=True)


def train(a, m):
    checked = engine.checked_report(a.output / "vehicle_kd_smoke.json", m)
    if not checked["passed"] or not valid_vehicle_counts(checked["active_counts"]):
        raise RuntimeError("Passed Vehicle-only gradient smoke required")
    engine.train(a, m)


def compare(a, m):
    _, parent, _ = reviewed_balanced(a.output / "reviewed_balanced")
    training = engine.checked_report(a.output / "kd/training_complete.json", m)
    current = json.loads((a.output / "evaluation/kd/accuracy.json").read_text())
    if (not training["complete"] or training["train_images"] != 512 or training["optimizer_steps"] != 2560
            or training["fixed_endpoint_epoch"] != 5 or training["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
            or training["tiny_checkpoint_used"] or training["architecture_changed"] or not training["external_teacher_used"]
            or engine.sha256(a.output / "kd/last.pth") != training["checkpoint_sha256"]
            or not valid_vehicle_counts(training["active_kd_counts"])
            or not current["complete"] or current["evaluated_images"] != 3769
            or current["run_signature"] != m["signature_sha256"] or current["checkpoint_sha256"] != training["checkpoint_sha256"]):
        raise RuntimeError("Completed fixed5-epoch Vehicle-only endpoint required")
    references = parent["results"]
    if references["label_only"] != m["reference_results"]["control"] or references["all_heads"] != m["reference_results"]["kd"]:
        raise RuntimeError("Historical reference results changed")
    decisions = {name: reference.decision(value, current) for name, value in references.items()}
    passed = decisions["label_only"]["kd_benefit_guardrail_passed"]
    next_step = "Review relative benefit; no automatic full-data training or deployment" if passed else "Stop this KD recipe; prioritize label-based student recovery rather than adding variants automatically"
    for value in decisions.values(): value["next_review"] = next_step
    report = dict(complete=True, run_signature=m["signature_sha256"], revision=REVISION, recipe=RECIPE,
                  guardrails=GUARDRAILS, results={**references, "vehicle_only_angle001": current}, comparisons=decisions,
                  benefit_guardrail_passed=passed, reviewed_balanced_signature=PARENT_SIGNATURE,
                  teacher_class_weights=CLASS_WEIGHTS, teacher_component_weights=balanced.COMPONENT_WEIGHTS,
                  vehicle_weight_not_redistributed=True, gt_losses_for_both_classes_unchanged=True,
                  historical_references=True, single_seed=True, reference_cpu_environment_matched=True,
                  comparison_limit="One class-contribution ablation against historical endpoints; shared-feature effects and seed variance are not isolated",
                  stop_this_kd_recipe_if_failed=True, next_review=next_step, validation_checkpoint_selection=False,
                  full_training_authorized=False, deployment_qualified=False, phone_retest_required=True)
    engine.save_report(a.output / "vehicle_geometry_comparison.json", report)
    print(json.dumps(dict(comparisons=decisions, benefit_guardrail_passed=passed, next_review=next_step), indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_vehicle_geometry_results.zip"
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
    for name in ("balanced", "pilot", "initialization", "compact", "repo", "dataset", "split-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args(); a.role = "kd"; a.output_verified = False
    for name in ("output", "balanced", "pilot", "initialization", "compact", "repo", "dataset", "split_dir"):
        if getattr(a, name) is not None: setattr(a, name, getattr(a, name).resolve())
    a.candidate = [p.resolve() for p in a.candidate]
    if a.stage == "prepare" and any(getattr(a, k) is None for k in ("balanced", "pilot", "initialization", "compact", "repo", "dataset", "split_dir")):
        parser.error("prepare requires reviewed balanced results, original512 pilot, initialization, compact export, new source and data paths")
    try:
        if a.stage == "prepare": prepare(a)
        elif a.stage == "bundle":
            saved = a.output / "geometry_manifest.json"
            if saved.exists() and json.loads(saved.read_text()).get("revision") != REVISION:
                raise RuntimeError("Bundle only this new Vehicle-only output; preserve historical runs")
            a.output.mkdir(parents=True, exist_ok=True); bundle(a)
        else:
            saved = a.output / "geometry_manifest.json"
            a.output_verified = saved.is_file() and json.loads(saved.read_text()).get("revision") == REVISION
            m = load(a.output)
            with recipe_context():
                dict(smoke=engine.smoke, tiny=engine.tiny, cache=geometry.cache, **{"kd-smoke": kd_smoke},
                     train=train, evaluate=engine.evaluate, compare=compare)[a.stage](a, m)
    except Exception as error:
        import traceback
        if a.output_verified:
            engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json", dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__": main()
