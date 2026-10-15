"""Resume-safe NumPy scalar compatibility for the frozen feature-KD pilot.

Do not edit/re-sign its manifest or frozen implementation. The only runtime
override is a scoped metadata allowlist while loading the hash-pinned A2
teacher. weights_only=True and strict state loading remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch
import recover_a2_lite_features as pilot

REVISION = "A2-LITE-FEATURE-SAFE-LOAD-2026-10-14-r1"
FROZEN_SCRIPT_SHA = "fc690ba78acfe6487f2bb510f1bbd477c3ad40eb7d4079d95aa751efbc5934fe"
RECEIPT = "compatibility/teacher_numpy_metadata_r1.json"


def numpy_scalar_globals():
    core = np._core if hasattr(np, "_core") else np.core
    return [(core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (core.multiarray.scalar, "numpy._core.multiarray.scalar"),
            np.dtype, type(np.dtype(np.float32)), type(np.dtype(np.float64))]


def verify_teacher(m):
    path = Path(m["teacher_checkpoint"])
    if (m.get("teacher_checkpoint_sha256") != pilot.engine.base.q.A2_SHA
            or pilot.engine.sha256(path) != pilot.engine.base.q.A2_SHA):
        raise RuntimeError("Original A2 teacher SHA256 mismatch before metadata allowlisting")


@contextmanager
def restricted_teacher_loader():
    """No global torch.load replacement or permanent allowlist changes."""
    original = pilot.build_teacher

    def build_teacher(m, device):
        verify_teacher(m)
        # Only floating scalar metadata constructors, never arbitrary globals,
        # modules, ndarray reconstruction, or unsafe pickle fallback.
        existing = set(torch.serialization.get_safe_globals())
        added = [entry for entry in numpy_scalar_globals() if entry not in existing]
        with torch.serialization.safe_globals(added):
            return original(m, device)

    pilot.build_teacher = build_teacher
    try:
        yield
    finally:
        pilot.build_teacher = original


def receipt_value(m):
    return dict(complete=True, run_signature=m["signature_sha256"], revision=REVISION,
                loader_implementation_sha256=pilot.engine.sha256(__file__),
                original_implementation=m["implementation"],
                teacher_checkpoint_sha256=pilot.engine.base.q.A2_SHA,
                weights_only=True, unsafe_pickle_fallback=False, strict_state_loading=True,
                allowlist_scope="original hash-pinned teacher build only; restored on exit",
                numpy_metadata="scalar aliases, dtype, float32 dtype, float64 dtype",
                manifest_rewritten=False, original_source_files_changed=False,
                training_objective_changed=False, optimizer_budget_changed=False,
                completed_smoke_and_tiny_fit_preserved=True)


def ensure_receipt(output):
    if pilot.engine.sha256(pilot.__file__) != FROZEN_SCRIPT_SHA:
        raise RuntimeError("Preserve the original published feature helper; do not bypass frozen implementation checks")
    m = pilot.load(output)  # All original source, weights, data and runtime checks.
    verify_teacher(m)
    path = output / RECEIPT
    value = receipt_value(m)
    if not path.exists():
        # The reported failure occurred before any teacher features/training.
        # Attach the correction before either starts, not retroactively.
        cache = output / "teacher_features"
        started = cache.exists() and any(cache.iterdir())
        started = started or any((output / name).exists() for name in (
            "feature_cache_complete.json", "feature_kd_smoke.json", "control/last.pth", "kd/last.pth",
            "control/training_complete.json", "kd/training_complete.json"))
        if started:
            raise RuntimeError("Cannot attach a new loader receipt after feature caching/training; preserve the run and review")
        pilot.engine.save_report(path, value)
    else:
        saved = pilot.engine.checked_report(path, m)
        if {k: v for k, v in saved.items() if k != "report_sha256"} != value:
            raise RuntimeError("Loader compatibility receipt changed; preserve the run")
    print(REVISION + " — original run signature preserved; restricted teacher metadata loading", flush=True)
    return m


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        pilot.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("stage")
    parser.add_argument("--output", type=Path, required=True)
    a, _ = parser.parse_known_args()
    output = a.output.resolve()
    # Bundling must still work after a preflight/quality failure.
    if a.stage == "bundle":
        pilot.main()
        return
    manifest = output / "feature_manifest.json"
    if a.stage != "prepare" or manifest.exists():
        ensure_receipt(output)
    with restricted_teacher_loader():
        pilot.main()
    if a.stage == "prepare":
        ensure_receipt(output)


if __name__ == "__main__":
    main()
