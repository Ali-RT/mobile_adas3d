"""Compare a fixed RTM3D ONNX export with PyTorch references using ONNX Runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PARITY_TOLERANCE = 2e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(artifact_dir: Path, output: Path, provider: str = "CPUExecutionProvider") -> dict:
    """Run all fixed fixtures, recording per-head and per-image raw-output deltas."""
    artifact_dir = artifact_dir.resolve()
    model_path = artifact_dir / "RTM3D_KM3D_ResNet18.onnx"
    fixture_path = artifact_dir / "parity_reference.npz"
    export_report_path = artifact_dir / "rtm3d_onnx_export.json"
    if not model_path.is_file() or not fixture_path.is_file() or not export_report_path.is_file():
        raise FileNotFoundError("Expected ONNX model, parity fixture, and export report in artifact-dir")

    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    if not export_report.get("export_complete") or not export_report.get("onnx_checker_passed"):
        raise RuntimeError("ONNX export report is incomplete or the ONNX checker did not pass")
    if sha256_file(model_path) != export_report["onnx_sha256"]:
        raise RuntimeError("ONNX model hash differs from the frozen export report")
    if sha256_file(fixture_path) != export_report["parity_fixture_sha256"]:
        raise RuntimeError("Parity fixture hash differs from the frozen export report")

    input_report = export_report["input"]
    sample_ids = input_report["fixture_sample_ids"]
    with np.load(fixture_path, allow_pickle=False) as fixture:
        if "image" not in fixture.files:
            raise RuntimeError("Parity fixture is missing its input tensor")
        images = np.asarray(fixture["image"], dtype=np.float32)
        expected = {
            name: np.asarray(fixture[name], dtype=np.float32)
            for name in export_report["outputs"]
        }
    if (
        images.ndim != 4
        or list(images.shape[1:]) != [3, 384, 1280]
        or images.shape[0] != len(sample_ids)
        or not np.isfinite(images).all()
    ):
        raise RuntimeError(f"Invalid parity input shape/content: {images.shape}")
    if any(not np.isfinite(value).all() for value in expected.values()):
        raise RuntimeError("Parity fixture contains non-finite reference outputs")

    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"Requested ONNX Runtime provider {provider!r} is unavailable: {available}")
    providers = [provider]
    if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    if session.get_inputs()[0].name != input_report["name"]:
        raise RuntimeError(f"Unexpected ONNX input name: {session.get_inputs()[0].name}")
    output_names = list(export_report["outputs"])
    actual_names = [item.name for item in session.get_outputs()]
    if actual_names != output_names:
        raise RuntimeError(f"Unexpected ONNX outputs: {actual_names}; expected {output_names}")

    totals = {
        name: {"max_abs_delta": 0.0, "absolute_error_sum": 0.0, "element_count": 0,
              "elements_over_tolerance": 0, "finite": True, "max_image_p99_abs_delta": 0.0}
        for name in expected
    }
    per_image = {}
    all_finite = True
    for image_index, sample_id in enumerate(sample_ids):
        actual_values = session.run(output_names, {input_report["name"]: images[image_index : image_index + 1]})
        image_heads = {}
        for name, actual_value in zip(output_names, actual_values):
            reference = expected[name][image_index : image_index + 1]
            actual = np.asarray(actual_value, dtype=np.float32)
            if actual.shape != reference.shape:
                raise RuntimeError(
                    f"Shape mismatch for {sample_id}/{name}: ONNX Runtime {actual.shape}, reference {reference.shape}"
                )
            finite = bool(np.isfinite(actual).all())
            all_finite &= finite
            aggregate = totals[name]
            aggregate["finite"] &= finite
            if finite:
                delta = np.abs(actual - reference)
                max_delta = float(np.max(delta))
                p99_delta = float(np.quantile(delta, 0.99))
                over_count = int(np.count_nonzero(delta > PARITY_TOLERANCE))
                aggregate["max_abs_delta"] = max(aggregate["max_abs_delta"], max_delta)
                aggregate["max_image_p99_abs_delta"] = max(aggregate["max_image_p99_abs_delta"], p99_delta)
                aggregate["absolute_error_sum"] += float(np.sum(delta, dtype=np.float64))
                aggregate["element_count"] += int(delta.size)
                aggregate["elements_over_tolerance"] += over_count
            else:
                max_delta = p99_delta = None
                over_count = None
            passed = bool(finite and max_delta <= PARITY_TOLERANCE)
            image_heads[name] = {
                "max_abs_delta": max_delta,
                "p99_abs_delta": p99_delta,
                "elements_over_tolerance": over_count,
                "finite": finite,
                "passed": passed,
            }
        per_image[sample_id] = image_heads

    per_head = {
        name: {
            "shape_per_image": list(expected[name].shape[1:]),
            "max_abs_delta": aggregate["max_abs_delta"],
            "mean_abs_delta": (
                aggregate["absolute_error_sum"] / aggregate["element_count"]
                if aggregate["element_count"] else None
            ),
            "max_image_p99_abs_delta": aggregate["max_image_p99_abs_delta"],
            "elements_over_tolerance": aggregate["elements_over_tolerance"],
            "finite": bool(aggregate["finite"]),
            "passed": bool(aggregate["finite"] and aggregate["max_abs_delta"] <= PARITY_TOLERANCE),
        }
        for name, aggregate in totals.items()
    }
    complete = all_finite and all(item["passed"] for item in per_head.values())
    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 ONNX Runtime raw-head parity",
        "complete": complete,
        "onnx_sha256": sha256_file(model_path),
        "parity_fixture_sha256": sha256_file(fixture_path),
        "fixture_sample_ids": sample_ids,
        "images_evaluated": len(sample_ids),
        "onnxruntime_version": ort.__version__,
        "provider_requested": provider,
        "session_providers": session.get_providers(),
        "available_providers": available,
        "parity_tolerance_max_abs": PARITY_TOLERANCE,
        "all_head_outputs_finite": all_finite,
        "per_head": per_head,
        "per_image": per_image,
        "geometry_decode_audited": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()
    report = run_audit(args.artifact_dir, args.output, args.provider)
    if not report["complete"]:
        raise RuntimeError(f"ONNX Runtime raw-head parity failed; see {args.output}")


if __name__ == "__main__":
    main()
