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
        images = np.asarray(fixture["image"], dtype=np.float32)
        expected = {
            name: np.asarray(fixture[name], dtype=np.float32)
            for name in export_report["outputs"]
        }
    input_report = export_report.get("input", {})
    sample_ids = input_report.get("fixture_sample_ids")
    if sample_ids is None:
        sample_ids = [input_report.get("fixture_sample_id")]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise RuntimeError("Export report has missing or duplicate parity fixture sample IDs")
    if (
        images.ndim != 4
        or list(images.shape[1:]) != [3, 384, 1280]
        or images.shape[0] != len(sample_ids)
        or not np.isfinite(images).all()
    ):
        raise RuntimeError(f"Invalid parity input shape/content: {images.shape}")
    if any(not np.isfinite(value).all() for value in expected.values()):
        raise RuntimeError("Parity fixture contains non-finite reference outputs")
    for name, value in expected.items():
        if value.ndim != 4 or value.shape[0] != len(sample_ids):
            raise RuntimeError(f"Invalid reference batch for {name}: {value.shape}")
        declared_shape = export_report["outputs"][name].get("shape")
        if declared_shape is not None and list(value.shape[1:]) != declared_shape[1:]:
            raise RuntimeError(
                f"Reference shape for {name} differs from export report: {value.shape} vs {declared_shape}"
            )

    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    head_totals = {
        name: {"max_abs_delta": 0.0, "absolute_error_sum": 0.0,
              "element_count": 0, "elements_over_tolerance": 0, "finite": True}
        for name in expected
    }
    per_image = {}
    all_finite = True
    for image_index, sample_id in enumerate(sample_ids):
        prediction = model.predict({"image": images[image_index : image_index + 1]})
        image_heads = {}
        for name, reference_batch in expected.items():
            if name not in prediction:
                raise RuntimeError(f"Core ML prediction omitted output {name!r}; got {sorted(prediction)}")
            reference = reference_batch[image_index : image_index + 1]
            actual = np.asarray(prediction[name], dtype=np.float32)
            if actual.shape != reference.shape:
                raise RuntimeError(
                    f"Shape mismatch for {sample_id}/{name}: Core ML {actual.shape}, reference {reference.shape}"
                )
            finite = bool(np.isfinite(actual).all())
            all_finite &= finite
            totals = head_totals[name]
            totals["finite"] &= finite
            if finite:
                delta = np.abs(actual - reference)
                max_delta = float(np.max(delta))
                totals["max_abs_delta"] = max(totals["max_abs_delta"], max_delta)
                totals["absolute_error_sum"] += float(np.sum(delta, dtype=np.float64))
                totals["element_count"] += int(delta.size)
                totals["elements_over_tolerance"] += int(np.count_nonzero(delta > PARITY_TOLERANCE))
            else:
                max_delta = None
            passed = bool(finite and max_delta <= PARITY_TOLERANCE)
            image_heads[name] = {
                "max_abs_delta": max_delta,
                "finite": finite,
                "passed": passed,
            }
        per_image[sample_id] = image_heads

    per_head = {
        name: {
            "shape_per_image": list(expected[name].shape[1:]),
            "max_abs_delta": totals["max_abs_delta"],
            "mean_abs_delta": (
                totals["absolute_error_sum"] / totals["element_count"]
                if totals["element_count"] else None
            ),
            "elements_over_tolerance": totals["elements_over_tolerance"],
            "finite": bool(totals["finite"]),
            "passed": bool(
                totals["finite"] and totals["max_abs_delta"] <= PARITY_TOLERANCE
            ),
        }
        for name, totals in head_totals.items()
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
        "fixture_sample_id": sample_ids[0],
        "fixture_sample_ids": sample_ids,
        "images_evaluated": len(sample_ids),
        "compute_units": "CPU_ONLY",
        "parity_tolerance_max_abs": PARITY_TOLERANCE,
        "all_head_outputs_finite": all_finite,
        "per_head": per_head,
        "per_image": per_image,
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
