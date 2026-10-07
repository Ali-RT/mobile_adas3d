"""Run native macOS Core ML parity against an RTM3D export fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


PARITY_TOLERANCE = 2e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("Native Core ML prediction parity must run on macOS")

    import coremltools as ct

    artifact_dir = args.artifact_dir.resolve()
    package_path = artifact_dir / "RTM3D_KM3D_ResNet18.mlpackage"
    fixture_path = artifact_dir / "parity_reference.npz"
    export_report_path = artifact_dir / "rtm3d_coreml_export.json"
    if not package_path.is_dir() or not fixture_path.is_file() or not export_report_path.is_file():
        raise FileNotFoundError("Expected Core ML package, parity fixture, and export report in artifact-dir")

    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    if not export_report.get("complete") or export_report.get("mil_has_custom_op"):
        raise RuntimeError("Export report is incomplete or contains a custom MIL op")
    if sha256_tree(package_path) != export_report["mlpackage_sha256"]:
        raise RuntimeError("ML package hash differs from the frozen export report")
    if sha256_file(fixture_path) != export_report["parity_fixture_sha256"]:
        raise RuntimeError("Parity fixture hash differs from the frozen export report")

    with np.load(fixture_path, allow_pickle=False) as fixture:
        if "image" not in fixture.files:
            raise RuntimeError("Parity fixture is missing its input tensor")
        image = np.asarray(fixture["image"], dtype=np.float32)
        expected = {
            name: np.asarray(fixture[name], dtype=np.float32)
            for name in export_report["outputs"]
        }
    if list(image.shape) != [1, 3, 384, 1280] or not np.isfinite(image).all():
        raise RuntimeError(f"Invalid parity input shape/content: {image.shape}")
    if any(not np.isfinite(value).all() for value in expected.values()):
        raise RuntimeError("Parity fixture contains non-finite reference outputs")

    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    prediction = model.predict({"image": image})
    per_head = {}
    all_finite = True
    for name, reference in expected.items():
        if name not in prediction:
            raise RuntimeError(f"Core ML prediction omitted output {name!r}; got {sorted(prediction)}")
        actual = np.asarray(prediction[name], dtype=np.float32)
        if actual.shape != reference.shape:
            raise RuntimeError(f"Shape mismatch for {name}: Core ML {actual.shape}, reference {reference.shape}")
        finite = bool(np.isfinite(actual).all())
        all_finite &= finite
        delta = float(np.max(np.abs(actual - reference)))
        per_head[name] = {
            "shape": list(actual.shape),
            "max_abs_delta": delta,
            "finite": finite,
            "passed": bool(finite and delta <= PARITY_TOLERANCE),
        }

    complete = all_finite and all(item["passed"] for item in per_head.values())
    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 native macOS Core ML head parity",
        "complete": complete,
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "export_report_sha256": sha256_file(export_report_path),
        "mlpackage_sha256": sha256_tree(package_path),
        "parity_fixture_sha256": sha256_file(fixture_path),
        "fixture_sample_id": export_report["input"]["fixture_sample_id"],
        "compute_units": "CPU_ONLY",
        "parity_tolerance_max_abs": PARITY_TOLERANCE,
        "all_head_outputs_finite": all_finite,
        "per_head": per_head,
        "coreml_prediction_performed": True,
        "geometry_decode_audited": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step_if_passed": "validate calibration-dependent 3D decode parity, then integrate into the iOS app and measure on the target iPhone",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not complete:
        raise RuntimeError(f"Native Core ML raw-head parity failed; see {output}")


if __name__ == "__main__":
    main()
