"""Plan a cold Colab session before restoring ephemeral source/data.

No frozen manifest is edited. A changed runtime may get a deterministic new
run only before either main training arm has started. Restore source BEFORE
the frozen launcher attempts its full existing-run validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import recover_a2_lite_features as pilot

REVISION = "A2-LITE-FEATURE-RESTART-2026-10-15-r1"


def current_runtime(device):
    pilot.engine.setup_device(device)
    return dict(cpu_environment=pilot.engine.runtime_environment(),
                execution_identity=pilot.device_identity(device))


def differences(m, current):
    result = {}
    for group, values in current.items():
        before = m[group]
        for key in sorted(set(before) | set(values)):
            if before.get(key) != values.get(key):
                result[f"{group}.{key}"] = dict(saved=before.get(key), current=values.get(key))
    return result


def read_manifest(output, baseline):
    path = output / "feature_manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    if (pilot.engine.signature({k: v for k, v in m.items() if k != "signature_sha256"}) != m["signature_sha256"]
            or m["revision"] != pilot.REVISION or m["recipe"] != pilot.RECIPE
            or m["guardrails"] != pilot.GUARDRAILS or m["implementation"] != pilot.implementation()
            or m["output"] != str(output)):
        raise RuntimeError("Saved feature-run identity/code changed; preserve it rather than bypassing its checks")
    for key in ("config", "source_sha256", "train_ids", "val_ids", "pilot_ids", "data_file_sha256"):
        if m[key] != baseline[key]:
            raise RuntimeError(f"Frozen feature input differs from the reviewed pilot: {key}")
    return m


def main_training_started(output):
    for name in ("control", "kd", "evaluation"):
        path = output / name
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            return True
    return False


def verify_paths(m, output, repo, dataset, initialization, checkpoint):
    expected = dict(output=output, source=repo, dataset=dataset,
                    initialization=initialization, teacher_checkpoint=checkpoint)
    if any(Path(m[k]).resolve() != value.resolve() for k, value in expected.items()):
        raise RuntimeError("Section1 paths differ from the saved run; restore the original paths or choose a new RUN_ID")
    if (m["initialization_sha256"] != pilot.engine.resolution.CHECKPOINT_SHA
            or m["teacher_checkpoint_sha256"] != pilot.engine.base.q.A2_SHA):
        raise RuntimeError("Frozen teacher/student checkpoint identity changed")


def plan(a):
    if (not re.fullmatch(r"[a-z0-9_]+", a.run_id) or a.output.name != a.run_id
            or not a.repo.name.endswith(a.run_id) or not a.dataset.name.endswith(a.run_id)):
        raise RuntimeError("RUN_ID must match the output and disposable source/data path suffixes")
    if any(a.report.is_relative_to(p) for p in (a.output, a.repo, a.dataset)):
        raise RuntimeError("Write the startup plan outside historical output/source/data directories")
    baseline, _, _, _ = pilot.reference.reviewed_pilot(a.pilot)
    for path, expected in ((a.initialization, pilot.engine.resolution.CHECKPOINT_SHA),
                           (a.teacher_checkpoint, pilot.engine.base.q.A2_SHA)):
        if not path.is_file() or pilot.engine.sha256(path) != expected:
            raise RuntimeError(f"Missing/changed pinned checkpoint before staging: {path}")
    current = current_runtime(a.device)
    original = read_manifest(a.output, baseline)
    changes, action = {}, "new_run"
    output, repo, dataset, run_id = a.output, a.repo, a.dataset, a.run_id
    if original is not None:
        verify_paths(original, output, repo, dataset, a.initialization, a.teacher_checkpoint)
        changes = differences(original, current)
        action = "resume_matching_runtime"
        if changes:
            if main_training_started(output):
                raise RuntimeError("Runtime differs after main training started. Do not mix GPUs/software: "
                                   "restore the saved runtime to resume, or explicitly choose a new RUN_ID.\n"
                                   + json.dumps(changes, indent=2))
            if not a.allow_pretraining_restart:
                raise RuntimeError("Runtime changed before main training; enable the explicit pretraining-restart policy or select matching hardware")
            token = pilot.engine.signature(dict(parent=original["signature_sha256"], runtime=current))[:10]
            run_id = a.run_id + "_restart_" + token
            output = a.output.with_name(run_id)
            repo = a.repo.with_name(a.repo.name[:-len(a.run_id)] + run_id)
            dataset = a.dataset.with_name(a.dataset.name[:-len(a.run_id)] + run_id)
            action = "new_runtime_run_before_main_training"
    selected = original if output == a.output else read_manifest(output, baseline)
    if selected is not None:
        verify_paths(selected, output, repo, dataset, a.initialization, a.teacher_checkpoint)
        if differences(selected, current):
            raise RuntimeError("Selected restart run has a different runtime; preserve it and review")
    elif main_training_started(output) or any((output / name).exists() for name in ("smoke.json", "tiny_fit.json", "feature_cache_complete.json")):
        raise RuntimeError("Selected output has orphaned results without a manifest; preserve it and choose a new RUN_ID")
    value = dict(revision=REVISION, action=action, requested_run_id=a.run_id, run_id=run_id,
                 requested_output=str(a.output), requested_repo=str(a.repo), requested_dataset=str(a.dataset),
                 output=str(output), repo=str(repo), dataset=str(dataset),
                 runtime=current, runtime_changes=changes, expected_source_sha256=baseline["source_sha256"],
                 helper_sha256=pilot.engine.sha256(__file__), original_implementation=pilot.implementation(),
                 requested_manifest_sha256=pilot.engine.sha256(a.output / "feature_manifest.json") if original is not None else None,
                 selected_manifest_sha256=pilot.engine.sha256(output / "feature_manifest.json") if selected is not None else None,
                 historical_manifest_rewritten=False, architecture_or_losses_changed=False)
    value["report_sha256"] = pilot.engine.signature(value)
    pilot.engine.write_json(a.report, value)
    print(json.dumps({k: value[k] for k in ("revision", "action", "run_id", "output", "repo", "dataset", "runtime_changes")}, indent=2), flush=True)
    print("Local startup plan: " + str(a.report), flush=True)
    return value


def restore_source(a):
    value = json.loads(a.plan.read_text())
    if (pilot.engine.signature({k: v for k, v in value.items() if k != "report_sha256"}) != value.get("report_sha256")
            or value["revision"] != REVISION or value["helper_sha256"] != pilot.engine.sha256(__file__)
            or value["original_implementation"] != pilot.implementation()
            or value["runtime"] != current_runtime(a.device)):
        raise RuntimeError("Startup plan/runtime changed; rerun section2 before restoring inputs")
    for key, folder in (("requested_manifest_sha256", "requested_output"), ("selected_manifest_sha256", "output")):
        path = Path(value[folder]) / "feature_manifest.json"
        digest = pilot.engine.sha256(path) if path.exists() else None
        if digest != value[key]:
            raise RuntimeError("Saved manifest changed after startup planning; preserve it and rerun section2")
    repo = Path(value["repo"])
    pilot.engine.base.prepare_repo(repo)
    if pilot.engine.base.q.source_hash(repo) != value["expected_source_sha256"]:
        raise RuntimeError("Restored portable source differs from the frozen inference baseline")
    print("Disposable source restored and verified BEFORE full saved-run validation: " + str(repo), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("plan")
    for name in ("pilot", "output", "repo", "dataset", "initialization", "teacher-checkpoint", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--allow-pretraining-restart", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("restore-source")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = parser.parse_args()
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    dict(plan=plan, **{"restore-source": restore_source})[a.stage](a)


if __name__ == "__main__":
    main()
