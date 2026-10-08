"""Export pinned RTM3D/KM3D ResNet-18 raw heads to ONNX for edge-runtime testing.

This exports fixed-shape FP32 neural heads only. Calibration-based 3D decode,
training, AP evaluation, and device performance measurements are out of scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_rtm3d_km3d_res18_coreml import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_VAL_SPLIT_SHA256,
    FIXTURE_SAMPLE_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_NAMES,
    RTM3DHeadsWrapper,
    build_pinned_resnet18,
    preprocess_pinned_image,
    select_fixture_ids,
    sha256_file,
    verify_source,
)
from scripts.run_rtm3d_km3d_res18_smoke import (
    HEADS,
    find_sample_paths,
    safe_load_state,
)
from scripts.audit_rtm3d_km3d_res18_onnx import PARITY_TOLERANCE, run_audit


INPUT_NAME = "image"
OPSET_VERSION = 17


def sha256_file_local(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import cv2
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless, onnx, and onnxruntime before export") from exc

    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset_root.resolve()
    split_path = args.split_dir.resolve() / "val.txt"
    output_dir = args.output_dir.resolve()
    if not repo.is_dir() or not checkpoint.is_file() or not split_path.is_file():
        raise FileNotFoundError("Pinned RTM3D source, checkpoint, or Chen val.txt is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty; preserve it and choose a new run directory: {output_dir}")

    source_commit = verify_source(repo)
    checkpoint_sha256 = sha256_file(checkpoint)
    split_sha256 = sha256_file(split_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint hash mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}")
    if split_sha256 != EXPECTED_VAL_SPLIT_SHA256:
        raise RuntimeError(f"Chen validation split hash mismatch: expected {EXPECTED_VAL_SPLIT_SHA256}, got {split_sha256}")
    fixture_sample_ids = select_fixture_ids(split_path, FIXTURE_SAMPLE_COUNT)
    state, checkpoint_epoch = safe_load_state(checkpoint, torch)
    model, head_conv = build_pinned_resnet18(repo, state, torch)
    wrapper = RTM3DHeadsWrapper(model.cpu().eval()).eval()

    first_image_path, _ = find_sample_paths(dataset, fixture_sample_ids[0])
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"OpenCV could not read representative image {first_image_path}")
    first_input = preprocess_pinned_image(first_image, cv2, np, torch, repo)
    if list(first_input.shape) != [1, 3, INPUT_HEIGHT, INPUT_WIDTH] or not bool(torch.isfinite(first_input).all()):
        raise RuntimeError(f"Unexpected or non-finite model input: {list(first_input.shape)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "RTM3D_KM3D_ResNet18.onnx"
    with torch.inference_mode():
        reference_first = wrapper(first_input)
        expected_shapes = {name: [1, channels, 96, 320] for name, channels in HEADS.items()}
        actual_shapes = {name: list(value.shape) for name, value in zip(OUTPUT_NAMES, reference_first)}
        if actual_shapes != expected_shapes:
            raise RuntimeError(f"Unexpected raw head shapes: {actual_shapes}")
        if not all(bool(torch.isfinite(value).all()) for value in reference_first):
            raise RuntimeError("Representative PyTorch heads contain non-finite values")
        torch.onnx.export(
            wrapper,
            first_input,
            str(model_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    onnx_model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)
    custom_domains = sorted({node.domain for node in onnx_model.graph.node if node.domain not in ("", "ai.onnx")})
    if custom_domains:
        raise RuntimeError(f"ONNX graph uses non-standard operator domains: {custom_domains}")
    onnx_input = next((value for value in onnx_model.graph.input if value.name == INPUT_NAME), None)
    if onnx_input is None:
        raise RuntimeError("ONNX graph is missing the named image input")
    input_shape = [dim.dim_value for dim in onnx_input.type.tensor_type.shape.dim]
    if input_shape != [1, 3, INPUT_HEIGHT, INPUT_WIDTH]:
        raise RuntimeError(f"ONNX input is not fixed to [1,3,{INPUT_HEIGHT},{INPUT_WIDTH}]: {input_shape}")

    image_values = np.empty((len(fixture_sample_ids), 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    head_values = {
        name: np.empty((len(fixture_sample_ids), channels, 96, 320), dtype=np.float32)
        for name, channels in HEADS.items()
    }
    fixture_records = []
    with torch.inference_mode():
        for index, sample_id in enumerate(fixture_sample_ids):
            image_path, calibration_path = find_sample_paths(dataset, sample_id)
            image_bgr = first_image if index == 0 else cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"OpenCV could not read parity image {image_path}")
            input_tensor = first_input if index == 0 else preprocess_pinned_image(image_bgr, cv2, np, torch, repo)
            outputs = reference_first if index == 0 else wrapper(input_tensor)
            if not all(bool(torch.isfinite(value).all()) for value in outputs):
                raise RuntimeError(f"Non-finite raw head output for Chen val image {sample_id}")
            image_values[index] = input_tensor[0].detach().cpu().numpy()
            for name, value in zip(OUTPUT_NAMES, outputs):
                if list(value.shape) != expected_shapes[name]:
                    raise RuntimeError(f"Unexpected {name} shape for {sample_id}: {list(value.shape)}")
                head_values[name][index] = value[0].detach().cpu().numpy()
            fixture_records.append({
                "sample_id": sample_id,
                "image_sha256": sha256_file(image_path),
                "calibration_sha256": sha256_file(calibration_path),
            })
            print(f"Prepared ONNX parity fixture {index + 1}/{len(fixture_sample_ids)}: {sample_id}", flush=True)

    fixture_path = output_dir / "parity_reference.npz"
    np.savez_compressed(fixture_path, image=image_values, **head_values)
    output_shapes = {name: list(value.shape) for name, value in zip(OUTPUT_NAMES, reference_first)}
    export_report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 trained-checkpoint ONNX export",
        "export_revision": "2026-10-07-r1",
        "export_complete": True,
        "onnx_checker_passed": True,
        "scope": "fixed-shape FP32 neural heads only; calibration-based 3D decode is excluded",
        "source_url": "https://github.com/Banconxuan/RTM3D.git",
        "source_commit": source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "safe_weights_only_load": True,
        "strict_state_dict_load": True,
        "head_conv": head_conv,
        "input": {
            "name": INPUT_NAME,
            "shape": input_shape,
            "dtype": "float32",
            "preprocessing": "pinned smoke preprocess: OpenCV BGR channel order, affine warp to 1280x384, then (pixel/255 - ImageNet mean)/ImageNet std",
            "fixture_sample_ids": fixture_sample_ids,
            "fixture_samples": fixture_records,
        },
        "outputs": {
            name: {"shape": shape, "dtype": "float32", "meaning": "raw pre-decode network head"}
            for name, shape in output_shapes.items()
        },
        "opset_version": OPSET_VERSION,
        "onnx_ir_version": onnx_model.ir_version,
        "onnx_op_count": len(onnx_model.graph.node),
        "onnx_custom_operator_domains": custom_domains,
        "onnx_file": model_path.name,
        "onnx_sha256": sha256_file_local(model_path),
        "onnx_size_bytes": model_path.stat().st_size,
        "parity_fixture": fixture_path.name,
        "parity_fixture_sha256": sha256_file_local(fixture_path),
        "parity_fixture_real_chen_val_image_count": len(fixture_sample_ids),
        "parity_fixture_selection": "16 evenly spaced unique IDs from the frozen Chen val split",
        "parity_tolerance_max_abs": PARITY_TOLERANCE,
        "training_performed": False,
        "full_val_ap_evaluated": False,
        "geometry_decode_included": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": "compare ONNX Runtime outputs on Mac/target providers; then validate calibration-dependent 3D decode and measure on the target iPhone",
    }
    report_path = output_dir / "rtm3d_onnx_export.json"
    report_path.write_text(json.dumps(export_report, indent=2) + "\n", encoding="utf-8")
    cpu_report_path = output_dir / "onnxruntime_cpu_parity.json"
    cpu_report = run_audit(output_dir, cpu_report_path, "CPUExecutionProvider")
    export_report["onnxruntime_cpu_parity_passed"] = bool(cpu_report["complete"])
    report_path.write_text(json.dumps(export_report, indent=2) + "\n", encoding="utf-8")

    bundle_path = output_dir / "rtm3d_km3d_res18_onnx_export.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (report_path, cpu_report_path, fixture_path, model_path):
            archive.write(path, path.name)
    print(json.dumps(export_report, indent=2), flush=True)
    print(f"ONNX bundle: {bundle_path}", flush=True)
    print(f"ONNX model size: {model_path.stat().st_size} bytes (bundle size {bundle_path.stat().st_size} bytes)", flush=True)
    if not cpu_report["complete"]:
        print("NOTE: ONNX Runtime CPU parity failed its unchanged 2e-4 gate; bundle is still available for review.", flush=True)


if __name__ == "__main__":
    main()
