"""Inspect PyTorch/ONNX raw heads and decoded detections for selected KITTI frames.

This small follow-up to the full-val audit reruns only explicitly selected images
with the same checkpoint, ONNX bundle, preprocessing, decoder, and thresholds.
It writes per-head delta diagnostics, matched decoded detections, and compressed
raw head arrays for local inspection. It is diagnostic-only, not an AP or
deployment gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_rtm3d_km3d_res18_onnx_fullval import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_ONNX_NAME,
    EXPECTED_VAL_SPLIT_SHA256,
    HEADS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    RTM3D_COMMIT,
    RTM3D_URL,
    TOP_K,
    VISIBILITY_THRESHOLD,
    decode_heads,
    find_label_path,
    greedy_match,
    load_calibration,
    load_pinned_package_module,
    preprocess_image,
    prediction_from_row,
    read_full_split,
    sha256_file,
    validate_onnx_head,
)
from scripts.audit_rtm3d_km3d_res18_onnx_decoded import extract_bundle
from scripts.run_rtm3d_km3d_res18_smoke import find_sample_paths, safe_load_state, verify_source


def summarize_head_delta(reference: np.ndarray, candidate: np.ndarray,
                         tolerance: float) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Head shape mismatch: PyTorch {reference.shape}, ONNX {candidate.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise RuntimeError("Cannot summarize a non-finite raw head")
    delta = np.abs(reference - candidate)
    max_index = tuple(int(value) for value in np.unravel_index(np.argmax(delta), delta.shape))
    return {
        "shape": list(reference.shape),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean(dtype=np.float64)),
        "p99_abs_delta": float(np.percentile(delta, 99)),
        "elements_over_tolerance": int(np.count_nonzero(delta > tolerance)),
        "tolerance": float(tolerance),
        "max_delta_index": list(max_index),
        "pytorch_value_at_max": float(reference[max_index]),
        "onnx_value_at_max": float(candidate[max_index]),
    }


def selected_rows_with_indices(rows: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 41)
    finite = np.isfinite(rows).all(axis=1)
    indices = np.flatnonzero(finite & (rows[:, 4] > VISIBILITY_THRESHOLD))
    selected = rows[indices]
    detections = []
    for output_row_index, row in zip(indices, selected):
        record, _line = prediction_from_row(row)
        # Post-processing may reorder candidates, so keep the precise stage clear.
        record["postprocess_row_index"] = int(output_row_index)
        record["decoded_row_41"] = [float(value) for value in row]
        detections.append(record)
    return selected, detections


def wrapped_angle_delta_degrees(first: float, second: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(first - second), math.cos(first - second))))


def compare_selected(reference: np.ndarray, candidate: np.ndarray,
                     reference_records: list[dict], candidate_records: list[dict]) -> dict:
    matches, unmatched_reference, unmatched_candidate = greedy_match(reference, candidate)
    matched = []
    for reference_index, candidate_index, iou in matches:
        ref = reference[reference_index]
        onnx = candidate[candidate_index]
        matched.append({
            "class_name": reference_records[reference_index]["class_name"],
            "pytorch_postprocess_row_index": reference_records[reference_index]["postprocess_row_index"],
            "onnx_postprocess_row_index": candidate_records[candidate_index]["postprocess_row_index"],
            "matched_box_iou": float(iou),
            "bbox_max_abs_px": float(np.max(np.abs(ref[:4] - onnx[:4]))),
            "keypoints_max_abs_px": float(np.max(np.abs(ref[5:23] - onnx[5:23]))),
            "dimensions_max_abs_m": float(np.max(np.abs(ref[32:35] - onnx[32:35]))),
            "location_delta_xyz_m": [float(value) for value in ref[36:39] - onnx[36:39]],
            "location_l2_m": float(np.linalg.norm(ref[36:39] - onnx[36:39])),
            "depth_abs_m": float(abs(ref[38] - onnx[38])),
            "yaw_abs_deg": wrapped_angle_delta_degrees(float(ref[35]), float(onnx[35])),
            "center_score_abs": float(abs(ref[4] - onnx[4])),
            "pytorch_prediction": reference_records[reference_index],
            "onnx_prediction": candidate_records[candidate_index],
        })
    return {
        "pytorch_selected_count": int(len(reference)),
        "onnx_selected_count": int(len(candidate)),
        "matched_count": int(len(matches)),
        "pytorch_unmatched_count": int(len(unmatched_reference)),
        "onnx_unmatched_count": int(len(unmatched_candidate)),
        "matched_predictions": matched,
        "pytorch_unmatched_predictions": [reference_records[index] for index in unmatched_reference],
        "onnx_unmatched_predictions": [candidate_records[index] for index in unmatched_candidate],
    }


def validate_fullval_identity(report_path: Path, bundle: Path, checkpoint: Path,
                              split_path: Path, sample_ids: list[str]) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("complete") is not True or report.get("evaluated_images") != 3769:
        raise RuntimeError("Reference full-val report is not a complete 3,769-image run")
    expected = {
        "source_commit": RTM3D_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "onnx_bundle_sha256": sha256_file(bundle),
        "validation_split_sha256": EXPECTED_VAL_SPLIT_SHA256,
    }
    changed = {
        key: {"expected": value, "saved": report.get(key)}
        for key, value in expected.items() if report.get(key) != value
    }
    if changed:
        raise RuntimeError("Full-val identity differs from the targeted audit inputs: " + json.dumps(changed))
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint hash differs from the reviewed RTM3D checkpoint")
    saved_ids = {sample.get("sample_id") for sample in report.get("per_sample", [])}
    missing = [sample_id for sample_id in sample_ids if sample_id not in saved_ids]
    if missing:
        raise RuntimeError(f"Target samples are missing from the completed full-val report: {missing}")
    return report


def analyze(bundle: Path, repo: Path, checkpoint: Path, dataset: Path,
            split_path: Path, fullval_report: Path, sample_ids: list[str],
            output_dir: Path) -> dict:
    from scripts.export_rtm3d_km3d_res18_coreml import (
        RTM3DHeadsWrapper,
        build_pinned_resnet18,
    )

    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Provide at least one unique sample ID")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; preserve it and choose a fresh one: {output_dir}"
        )
    split_ids = read_full_split(split_path)
    unknown = [sample_id for sample_id in sample_ids if sample_id not in set(split_ids)]
    if unknown:
        raise RuntimeError(f"Target sample IDs are not in the reviewed Chen validation split: {unknown}")
    fullval = validate_fullval_identity(fullval_report, bundle, checkpoint, split_path, sample_ids)
    source_commit = verify_source(repo)
    if source_commit != RTM3D_COMMIT:
        raise RuntimeError(f"Pinned RTM3D source mismatch: {source_commit}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "bundle"
    artifact_dir.mkdir()
    extract_bundle(bundle, artifact_dir)
    export_report = json.loads((artifact_dir / "rtm3d_onnx_export.json").read_text(encoding="utf-8"))
    raw_report = json.loads((artifact_dir / "onnxruntime_cpu_parity.json").read_text(encoding="utf-8"))
    onnx_path = artifact_dir / EXPECTED_ONNX_NAME
    if not export_report.get("export_complete") or not export_report.get("onnx_checker_passed"):
        raise RuntimeError("ONNX export bundle is incomplete or failed ONNX checker")
    if export_report.get("source_commit") != RTM3D_COMMIT:
        raise RuntimeError("ONNX bundle source commit differs from pinned RTM3D")
    if export_report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("ONNX bundle checkpoint identity differs from reviewed checkpoint")
    if raw_report.get("onnx_sha256") != export_report.get("onnx_sha256"):
        raise RuntimeError("Raw parity report and ONNX export identify different graphs")
    if sha256_file(onnx_path) != export_report.get("onnx_sha256"):
        raise RuntimeError("ONNX file hash differs from export manifest")
    if list(export_report.get("outputs", {})) != list(HEADS):
        raise RuntimeError("ONNX output head names/order differ from pinned RTM3D")
    if raw_report.get("all_head_outputs_finite") is not True:
        raise RuntimeError("Frozen ONNX fixture report contains non-finite raw heads")
    if raw_report.get("parity_tolerance_max_abs") is None:
        raise RuntimeError("Frozen ONNX fixture report does not declare its raw-head tolerance")

    import cv2
    import onnxruntime as ort
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned RTM3D geometry decoder requires a Colab GPU runtime")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    models_dir = repo / "src/lib/models"
    utils_dir = repo / "src/lib/utils"
    image_module = load_pinned_package_module("rtm3d_targeted_utils_888c379", utils_dir, "image")
    decoder_module = load_pinned_package_module("rtm3d_targeted_models_888c379", models_dir, "decode")
    postprocess_module = load_pinned_package_module("rtm3d_targeted_utils_888c379", utils_dir, "post_process")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = export_report["input"]["name"]
    output_names = list(export_report["outputs"])
    if [item.name for item in session.get_inputs()] != [input_name]:
        raise RuntimeError("ONNX Runtime input name differs from export manifest")
    if [item.name for item in session.get_outputs()] != output_names:
        raise RuntimeError("ONNX Runtime output names/order differ from export manifest")
    declared_shapes = {item.name: item.shape for item in session.get_outputs()}

    state, checkpoint_epoch = safe_load_state(checkpoint, torch)
    model, head_conv = build_pinned_resnet18(repo, state, torch)
    model = model.cuda().eval()
    wrapper = RTM3DHeadsWrapper(model).eval()
    device = torch.device("cuda")
    const = torch.tensor(
        [[[-1, 0], [0, -1]] * 8], dtype=torch.float32, device=device
    ).view(1, 1, 16, 2)
    expected_shapes = {name: [1, channels, 96, 320] for name, channels in HEADS.items()}
    tolerance = float(raw_report["parity_tolerance_max_abs"])
    arrays: dict[str, np.ndarray] = {}
    samples = []

    for sample_id in sample_ids:
        image_path, calibration_path = find_sample_paths(dataset, sample_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read KITTI image: {image_path}")
        input_cpu, center, scale, inverse = preprocess_image(
            image, cv2, torch, image_module.get_affine_transform
        )
        input_array = input_cpu.numpy()
        calibration = load_calibration(calibration_path, np)
        meta = {
            "trans_output_inv": torch.from_numpy(inverse).unsqueeze(0).to(device),
            "calib": torch.from_numpy(calibration).unsqueeze(0).to(device),
        }

        with torch.inference_mode():
            pytorch_values = wrapper(input_cpu.to(device))
        torch.cuda.synchronize()
        pytorch_heads = {
            name: value.detach().float().cpu().numpy()
            for name, value in zip(output_names, pytorch_values)
        }
        if {name: list(value.shape) for name, value in pytorch_heads.items()} != expected_shapes:
            raise RuntimeError(f"Unexpected PyTorch head shapes at {sample_id}")
        if any(not np.isfinite(value).all() for value in pytorch_heads.values()):
            raise RuntimeError(f"PyTorch reference produced non-finite heads at {sample_id}")

        onnx_values = session.run(output_names, {input_name: input_array})
        onnx_heads = {
            name: validate_onnx_head(
                value, name, sample_id, expected_shapes[name], declared_shapes[name], input_array
            )
            for name, value in zip(output_names, onnx_values)
        }
        raw_deltas = {}
        for name in HEADS:
            raw_deltas[name] = summarize_head_delta(pytorch_heads[name], onnx_heads[name], tolerance)
            arrays[f"{sample_id}__pytorch__{name}"] = pytorch_heads[name]
            arrays[f"{sample_id}__onnx__{name}"] = onnx_heads[name]

        p_heads_gpu = {name: torch.from_numpy(pytorch_heads[name]).to(device) for name in HEADS}
        o_heads_gpu = {name: torch.from_numpy(onnx_heads[name]).to(device) for name in HEADS}
        with torch.inference_mode():
            p_rows = decode_heads(
                p_heads_gpu, meta, const, center, scale, torch, np,
                decoder_module.car_pose_decode, postprocess_module.car_pose_post_process,
            )
            o_rows = decode_heads(
                o_heads_gpu, meta, const, center, scale, torch, np,
                decoder_module.car_pose_decode, postprocess_module.car_pose_post_process,
            )
        p_nonfinite = int((~np.isfinite(p_rows).all(axis=1)).sum())
        o_nonfinite = int((~np.isfinite(o_rows).all(axis=1)).sum())
        p_selected, p_records = selected_rows_with_indices(p_rows)
        o_selected, o_records = selected_rows_with_indices(o_rows)
        samples.append({
            "sample_id": sample_id,
            "image_sha256": sha256_file(image_path),
            "calibration_sha256": sha256_file(calibration_path),
            "image_dimensions_hw": [int(image.shape[0]), int(image.shape[1])],
            "decoded_candidate_counts": {
                "pytorch_total": int(len(p_rows)),
                "onnx_total": int(len(o_rows)),
                "pytorch_nonfinite_discarded": p_nonfinite,
                "onnx_nonfinite_discarded": o_nonfinite,
            },
            "raw_head_deltas": raw_deltas,
            "decoded_comparison": compare_selected(p_selected, o_selected, p_records, o_records),
        })
        print(f"Targeted raw-head and decode audit: {sample_id}", flush=True)

    head_outputs_path = output_dir / "targeted_head_outputs.npz"
    np.savez_compressed(head_outputs_path, **arrays)
    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 targeted full-val outlier audit",
        "complete": len(samples) == len(sample_ids),
        "interpretation_only_no_deployment_gate": True,
        "sample_ids": sample_ids,
        "reference_fullval_report": str(fullval_report.resolve()),
        "reference_fullval_complete": bool(fullval["complete"]),
        "source_url": RTM3D_URL,
        "source_commit": source_commit,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "onnx_bundle_sha256": sha256_file(bundle),
        "onnx_sha256": export_report["onnx_sha256"],
        "validation_split_sha256": sha256_file(split_path),
        "runtime": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "decoder_gpu": torch.cuda.get_device_name(0),
            "onnxruntime_version": ort.__version__,
            "onnxruntime_provider": "CPUExecutionProvider",
            "head_conv": head_conv,
        },
        "selection": {
            "top_k": TOP_K,
            "center_score_strictly_greater_than": VISIBILITY_THRESHOLD,
            "same_class_greedy_2d_iou_match_threshold": 0.5,
            "decoder": "pinned RTM3D car_pose_decode + car_pose_post_process",
            "decoded_row_layout": {
                "0:4": "bbox xyxy pixels",
                "4": "center heatmap score",
                "5:23": "keypoint coordinates",
                "23:32": "keypoint scores",
                "32:35": "dimensions h/w/l meters",
                "35": "yaw camera radians",
                "36:39": "location x/y/z meters",
                "39": "object probability logit",
                "40": "class index: 0 Car, 1 Pedestrian, 2 Cyclist",
            },
        },
        "raw_head_archive": str(head_outputs_path),
        "raw_head_archive_members": sorted(arrays),
        "samples": samples,
        "training_performed": False,
        "full_val_rerun": False,
        "iphone_performance_measured": False,
    }
    report_path = output_dir / "targeted_parity_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    zip_path = output_dir / "targeted_parity_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, arcname=report_path.name)
        archive.write(head_outputs_path, arcname=head_outputs_path.name)
    print(json.dumps({
        "complete": report["complete"],
        "sample_ids": sample_ids,
        "per_sample": [
            {
                "sample_id": sample["sample_id"],
                "raw_hps_max_abs_delta": sample["raw_head_deltas"]["hps"]["max_abs_delta"],
                "decoded": {
                    key: sample["decoded_comparison"][key]
                    for key in (
                        "pytorch_selected_count", "onnx_selected_count", "matched_count",
                        "pytorch_unmatched_count", "onnx_unmatched_count",
                    )
                },
            }
            for sample in samples
        ],
        "report": str(report_path),
        "raw_heads": str(head_outputs_path),
        "zip": str(zip_path),
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--fullval-report", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.bundle, args.checkpoint, args.split_dir / "val.txt", args.fullval_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    analyze(
        args.bundle.resolve(), args.repo.resolve(), args.checkpoint.resolve(),
        args.dataset_root.resolve(), (args.split_dir / "val.txt").resolve(),
        args.fullval_report.resolve(), args.sample_ids, args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
