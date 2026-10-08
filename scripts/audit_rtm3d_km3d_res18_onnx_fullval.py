"""Compare pinned RTM3D PyTorch and ONNX predictions on all Chen val images.

The script runs the same 3,769 images through the checkpoint-matched PyTorch
ResNet-18 and ONNX Runtime CPU heads, decodes both with the pinned RTM3D CUDA
geometry decoder, writes KITTI-format predictions, and computes this project's
AP_R40 implementation for descriptive comparison. It is not a deployment or
official KITTI-leaderboard gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti_parser import parse_kitti_label_file
from data.kitti_r40 import evaluate_kitti_r40
from scripts.audit_rtm3d_km3d_res18_onnx_decoded import (
    MATCH_IOU_THRESHOLD,
    decode_heads,
    distribution,
    extract_bundle,
    greedy_match,
    load_pinned_package_module,
)
from scripts.audit_rtm3d_km3d_res18_onnx import sha256_file
from scripts.run_rtm3d_km3d_res18_smoke import (
    CLASS_NAMES,
    HEADS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    RTM3D_COMMIT,
    RTM3D_URL,
    TOP_K,
    VISIBILITY_THRESHOLD,
    find_sample_paths,
    load_calibration,
    safe_load_state,
    verify_source,
)


EXPECTED_SPLIT_SIZE = 3769
EXPECTED_CHECKPOINT_SHA256 = "5fa355845f79c1afeffab427de32933758e5b4c1e7c9ec19a94a13737691d05b"
EXPECTED_VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
EXPECTED_ONNX_NAME = "RTM3D_KM3D_ResNet18.onnx"
MATCH_KEYS = (
    "matched_box_iou", "bbox_max_abs_px", "keypoints_max_abs_px",
    "dimensions_max_abs_m", "location_l2_m", "depth_abs_m", "yaw_abs_deg",
    "center_score_abs", "export_score_abs",
)


def read_full_split(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != EXPECTED_SPLIT_SIZE or len(set(ids)) != EXPECTED_SPLIT_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_SPLIT_SIZE} unique Chen val IDs, got {len(ids)}")
    if sha256_file(path) != EXPECTED_VAL_SPLIT_SHA256:
        raise RuntimeError("Chen val split hash differs from the reviewed export baseline")
    return ids


def find_label_path(dataset: Path, sample_id: str) -> Path:
    for folder_name in ("label_2", "label_02", "label"):
        candidate = dataset / "training" / folder_name / f"{sample_id}.txt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find KITTI label for {sample_id} under {dataset}/training")


def preprocess_image(image, cv2, torch, affine_transform):
    """Apply the exact BGR affine warp and ImageNet normalization used at export."""
    height, width = image.shape[:2]
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    transform = affine_transform(center, scale, 0, [INPUT_WIDTH, INPUT_HEIGHT])
    warped = cv2.warpAffine(
        image, transform, (INPUT_WIDTH, INPUT_HEIGHT), flags=cv2.INTER_LINEAR
    )
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = ((warped / 255.0 - mean) / std).astype(np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    inverse = affine_transform(
        center, scale, 0, [INPUT_WIDTH // 4, INPUT_HEIGHT // 4], inv=1
    )
    return tensor, center, scale, inverse


def stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def upstream_export_score(row: np.ndarray) -> float:
    """Match RTM3D Debugger.save_kitti_format(is_faster=False)."""
    return (float(row[4]) + stable_sigmoid(float(row[39])) + float(np.mean(row[23:32]))) / 3.0


def prediction_from_row(row: np.ndarray) -> tuple[dict, str]:
    """Return the evaluator record and KITTI text line used by upstream RTM3D."""
    if not np.isfinite(row).all():
        raise RuntimeError("RTM3D selected a non-finite detection; refusing to serialize it")
    class_index = int(round(float(row[40])))
    if class_index < 0 or class_index >= len(CLASS_NAMES):
        raise RuntimeError(f"Decoded candidate has invalid class index {row[40]}")
    box = [float(value) for value in row[:4]]
    h, w, length = [float(value) for value in row[32:35]]
    x, y, z = [float(value) for value in row[36:39]]
    yaw = float(row[35])
    score = upstream_export_score(row)
    if not (box[2] > box[0] and box[3] > box[1]):
        raise RuntimeError(f"RTM3D selected an invalid 2D box: {box}")
    if min(h, w, length, z) <= 0:
        raise RuntimeError(f"RTM3D selected invalid 3D geometry: {[h, w, length, x, y, z]}")

    # Preserve the pinned upstream Debugger.save_kitti_format orientation
    # wrapping and camera-bottom-center conversion exactly.
    two_pi = 2.0 * math.pi
    if yaw > two_pi:
        while yaw > two_pi:
            yaw -= two_pi
    if yaw < -two_pi:
        while yaw < -two_pi:
            yaw += two_pi
    if yaw > math.pi:
        yaw = two_pi - yaw
    if yaw < -math.pi:
        yaw = two_pi + math.pi
    alpha = yaw - math.atan2(x, z)
    bottom_y = y + h / 2.0
    class_name = CLASS_NAMES[class_index]
    record = {
        "class_name": class_name,
        "truncated": -1.0,
        "occluded": -1,
        "alpha": alpha,
        "bbox_2d": box,
        "dimensions_3d_hwl": [h, w, length],
        "location_3d": [x, bottom_y, z],
        "rotation_y": yaw,
        "yaw": yaw,
        "score": score,
    }
    line = (
        f"{class_name} -1.00 -1 {alpha:.7f} "
        f"{box[0]:.7f} {box[1]:.7f} {box[2]:.7f} {box[3]:.7f} "
        f"{h:.7f} {w:.7f} {length:.7f} {x:.7f} {bottom_y:.7f} {z:.7f} "
        f"{yaw:.7f} {score:.7f}\n"
    )
    return record, line


def selected_detections(rows: np.ndarray) -> tuple[np.ndarray, list[dict], list[str]]:
    candidates = rows[np.isfinite(rows[:, 4]) & (rows[:, 4] > VISIBILITY_THRESHOLD)]
    records: list[dict] = []
    lines: list[str] = []
    for row in candidates:
        record, line = prediction_from_row(row)
        records.append(record)
        lines.append(line)
    return candidates, records, lines


def summarize_latency(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50)) if array.size else None,
        "p90_ms": float(np.percentile(array, 90)) if array.size else None,
        "p95_ms": float(np.percentile(array, 95)) if array.size else None,
        "max_ms": float(np.max(array)) if array.size else None,
    }


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rows(reference: np.ndarray, candidate: np.ndarray) -> dict:
    matches, unmatched_reference, unmatched_candidate = greedy_match(
        reference, candidate, MATCH_IOU_THRESHOLD
    )
    per_class = {}
    deltas_by_key = {key: [] for key in MATCH_KEYS}
    for class_index, class_name in enumerate(CLASS_NAMES):
        ref_indices = [i for i, row in enumerate(reference) if int(round(float(row[40]))) == class_index]
        cand_indices = [i for i, row in enumerate(candidate) if int(round(float(row[40]))) == class_index]
        class_matches = [
            (ri, ci, overlap) for ri, ci, overlap in matches
            if int(round(float(reference[ri, 40]))) == class_index
        ]
        per_class[class_name] = {
            "pytorch_selected": len(ref_indices),
            "onnx_selected": len(cand_indices),
            "matched": len(class_matches),
            "pytorch_unmatched": sum(i in unmatched_reference for i in ref_indices),
            "onnx_unmatched": sum(i in unmatched_candidate for i in cand_indices),
        }
        for ri, ci, overlap in class_matches:
            ref, onnx = reference[ri], candidate[ci]
            deltas = {
                "matched_box_iou": float(overlap),
                "bbox_max_abs_px": float(np.max(np.abs(ref[:4] - onnx[:4]))),
                "keypoints_max_abs_px": float(np.max(np.abs(ref[5:23] - onnx[5:23]))),
                "dimensions_max_abs_m": float(np.max(np.abs(ref[32:35] - onnx[32:35]))),
                "location_l2_m": float(np.linalg.norm(ref[36:39] - onnx[36:39])),
                "depth_abs_m": float(abs(ref[38] - onnx[38])),
                "yaw_abs_deg": abs(math.degrees(math.atan2(
                    math.sin(float(ref[35] - onnx[35])),
                    math.cos(float(ref[35] - onnx[35])),
                ))),
                "center_score_abs": float(abs(ref[4] - onnx[4])),
                "export_score_abs": float(abs(upstream_export_score(ref) - upstream_export_score(onnx))),
            }
            for key, value in deltas.items():
                deltas_by_key[key].append(value)
    return {
        "pytorch_selected": int(len(reference)),
        "onnx_selected": int(len(candidate)),
        "matched": int(len(matches)),
        "pytorch_unmatched": int(len(unmatched_reference)),
        "onnx_unmatched": int(len(unmatched_candidate)),
        "per_class": per_class,
        "matched_pair_delta_distributions": {
            key: distribution(values) for key, values in deltas_by_key.items()
        },
    }


def analyze(bundle: Path, repo: Path, checkpoint: Path, dataset: Path,
            split_path: Path, output_dir: Path) -> dict:
    # Keep Torch out of module import time so lightweight serialization tests can
    # run on developer machines without the Colab inference stack installed.
    from scripts.export_rtm3d_km3d_res18_coreml import (
        RTM3DHeadsWrapper,
        build_pinned_resnet18,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; preserve it and choose a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "bundle"
    artifact_dir.mkdir()
    extract_bundle(bundle, artifact_dir)
    export_report = json.loads((artifact_dir / "rtm3d_onnx_export.json").read_text(encoding="utf-8"))
    raw_report = json.loads((artifact_dir / "onnxruntime_cpu_parity.json").read_text(encoding="utf-8"))
    onnx_path = artifact_dir / EXPECTED_ONNX_NAME
    if not export_report.get("export_complete") or not export_report.get("onnx_checker_passed"):
        raise RuntimeError("ONNX bundle does not contain a complete ONNX-checker-validated export")
    if set(export_report.get("outputs", {})) != set(HEADS):
        raise RuntimeError("ONNX export head names differ from pinned RTM3D")
    if set(raw_report.get("per_head", {})) != set(HEADS):
        raise RuntimeError("Raw-parity report is missing one or more expected RTM3D heads")
    if raw_report.get("onnx_sha256") != export_report.get("onnx_sha256"):
        raise RuntimeError("Raw-parity and export reports identify different ONNX models")
    if not raw_report.get("complete"):
        raise RuntimeError("Raw ONNX fixture parity did not complete; resolve that before full validation")
    if raw_report.get("parity_fixture_sha256") != export_report.get("parity_fixture_sha256"):
        raise RuntimeError("Raw-parity and export reports identify different fixtures")
    if sha256_file(onnx_path) != export_report["onnx_sha256"]:
        raise RuntimeError("ONNX model file hash differs from its export manifest")
    fixture_path = artifact_dir / "parity_reference.npz"
    if sha256_file(fixture_path) != export_report.get("parity_fixture_sha256"):
        raise RuntimeError("ONNX reference fixture hash differs from its export manifest")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint hash differs from the reviewed RTM3D ResNet-18 checkpoint")
    if export_report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("ONNX export was not created from the reviewed checkpoint")

    sample_ids = read_full_split(split_path)
    source_commit = verify_source(repo)
    if source_commit != RTM3D_COMMIT or source_commit != export_report.get("source_commit"):
        raise RuntimeError("Pinned RTM3D source commit differs from export metadata")

    sample_paths = []
    label_paths = []
    for sample_id in sample_ids:
        sample_paths.append(find_sample_paths(dataset, sample_id))
        label_paths.append(find_label_path(dataset, sample_id))

    import cv2
    import onnxruntime as ort
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned RTM3D geometry decoder requires a Colab GPU runtime")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    models_dir = repo / "src/lib/models"
    utils_dir = repo / "src/lib/utils"
    image_module = load_pinned_package_module("rtm3d_fullval_utils_888c379", utils_dir, "image")
    decoder_module = load_pinned_package_module("rtm3d_fullval_models_888c379", models_dir, "decode")
    postprocess_module = load_pinned_package_module("rtm3d_fullval_utils_888c379", utils_dir, "post_process")
    decoder = decoder_module.car_pose_decode
    postprocess = postprocess_module.car_pose_post_process

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = export_report["input"]["name"]
    output_names = list(export_report["outputs"])
    if [item.name for item in session.get_inputs()] != [input_name]:
        raise RuntimeError("ONNX Runtime input name differs from export metadata")
    if [item.name for item in session.get_outputs()] != output_names:
        raise RuntimeError("ONNX Runtime output names/order differ from export metadata")

    state, checkpoint_epoch = safe_load_state(checkpoint, torch)
    model, head_conv = build_pinned_resnet18(repo, state, torch)
    model = model.cuda().eval()
    wrapper = RTM3DHeadsWrapper(model).eval()
    device = torch.device("cuda")
    const = torch.tensor(
        [[[-1, 0], [0, -1]] * 8], dtype=torch.float32, device=device
    ).view(1, 1, 16, 2)

    pytorch_dir = output_dir / "pytorch_predictions" / "data"
    onnx_dir = output_dir / "onnx_predictions" / "data"
    pytorch_dir.mkdir(parents=True)
    onnx_dir.mkdir(parents=True)
    gt = {}
    per_sample = []
    pytorch_latency = []
    onnx_latency = []
    raw_head_stats = {
        name: {"max_abs": 0.0, "abs_sum": 0.0, "value_count": 0}
        for name in HEADS
    }
    all_deltas = {key: [] for key in MATCH_KEYS}
    class_counts = {
        role: {name: 0 for name in CLASS_NAMES}
        for role in ("pytorch", "onnx")
    }
    total_detections = {"pytorch": 0, "onnx": 0}
    expected_shapes = {name: [1, channels, 96, 320] for name, channels in HEADS.items()}
    for index, (sample_id, (image_path, calib_path), label_path) in enumerate(
        zip(sample_ids, sample_paths, label_paths)
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        input_cpu, center, scale, inverse = preprocess_image(
            image, cv2, torch, image_module.get_affine_transform
        )
        input_array = input_cpu.numpy()
        projection = load_calibration(calib_path, np)
        meta = {
            "trans_output_inv": torch.from_numpy(inverse).unsqueeze(0).to(device),
            "calib": torch.from_numpy(projection).unsqueeze(0).to(device),
        }

        if index == 0:
            with torch.inference_mode():
                for _ in range(3):
                    wrapper(input_cpu.to(device))
            torch.cuda.synchronize()
            for _ in range(2):
                session.run(output_names, {input_name: input_array})

        start = time.perf_counter()
        with torch.inference_mode():
            pytorch_values = wrapper(input_cpu.to(device))
        torch.cuda.synchronize()
        pytorch_latency.append((time.perf_counter() - start) * 1000.0)
        pytorch_heads = {
            name: value for name, value in zip(output_names, pytorch_values)
        }
        if {name: list(value.shape) for name, value in pytorch_heads.items()} != expected_shapes:
            raise RuntimeError(f"Unexpected PyTorch head shapes at {sample_id}")
        if any(not bool(torch.isfinite(value).all().item()) for value in pytorch_heads.values()):
            raise RuntimeError(f"PyTorch reference produced non-finite heads for {sample_id}")

        start = time.perf_counter()
        onnx_values = session.run(output_names, {input_name: input_array})
        onnx_latency.append((time.perf_counter() - start) * 1000.0)
        pytorch_np = {name: value.detach().float().cpu().numpy() for name, value in pytorch_heads.items()}
        onnx_np = {name: np.asarray(value, dtype=np.float32) for name, value in zip(output_names, onnx_values)}
        for name in HEADS:
            if onnx_np[name].shape != expected_shapes[name] or not np.isfinite(onnx_np[name]).all():
                raise RuntimeError(f"ONNX head {name} has wrong shape or non-finite values at {sample_id}")
            delta = np.abs(pytorch_np[name] - onnx_np[name])
            stat = raw_head_stats[name]
            stat["max_abs"] = max(float(stat["max_abs"]), float(delta.max()))
            stat["abs_sum"] = float(stat["abs_sum"]) + float(delta.sum(dtype=np.float64))
            stat["value_count"] = int(stat["value_count"]) + int(delta.size)

        p_heads_gpu = {name: pytorch_heads[name] for name in HEADS}
        o_heads_gpu = {name: torch.from_numpy(onnx_np[name]).to(device) for name in HEADS}
        with torch.inference_mode():
            p_rows = decode_heads(p_heads_gpu, meta, const, center, scale, torch, np, decoder, postprocess)
            o_rows = decode_heads(o_heads_gpu, meta, const, center, scale, torch, np, decoder, postprocess)
        p_selected, p_records, p_lines = selected_detections(p_rows)
        o_selected, o_records, o_lines = selected_detections(o_rows)
        (pytorch_dir / f"{sample_id}.txt").write_text("".join(p_lines), encoding="utf-8")
        (onnx_dir / f"{sample_id}.txt").write_text("".join(o_lines), encoding="utf-8")
        for role, rows, records in (
            ("pytorch", p_selected, p_records), ("onnx", o_selected, o_records)
        ):
            total_detections[role] += len(rows)
            for record in records:
                class_counts[role][record["class_name"]] += 1
        gt[sample_id] = [
            # Keep all KITTI label classes: the evaluator ignores Van for Car
            # and Person_sitting for Pedestrian when either overlaps a match.
            obj.__dict__ for obj in parse_kitti_label_file(label_path)
        ]
        comparison = compare_rows(p_selected, o_selected)
        # Aggregate paired deltas from the explicit matched pairs so pooled
        # percentiles are computed over detections, not per-image summaries.
        matches, _, _ = greedy_match(p_selected, o_selected, MATCH_IOU_THRESHOLD)
        for ri, oi, _overlap in matches:
            ref, onnx = p_selected[ri], o_selected[oi]
            pair_deltas = {
                "matched_box_iou": float(_overlap),
                "bbox_max_abs_px": float(np.max(np.abs(ref[:4] - onnx[:4]))),
                "keypoints_max_abs_px": float(np.max(np.abs(ref[5:23] - onnx[5:23]))),
                "dimensions_max_abs_m": float(np.max(np.abs(ref[32:35] - onnx[32:35]))),
                "location_l2_m": float(np.linalg.norm(ref[36:39] - onnx[36:39])),
                "depth_abs_m": float(abs(ref[38] - onnx[38])),
                "yaw_abs_deg": abs(math.degrees(math.atan2(
                    math.sin(float(ref[35] - onnx[35])), math.cos(float(ref[35] - onnx[35]))
                ))),
                "center_score_abs": float(abs(ref[4] - onnx[4])),
                "export_score_abs": float(abs(upstream_export_score(ref) - upstream_export_score(onnx))),
            }
            for key, value in pair_deltas.items():
                all_deltas[key].append(value)
        per_sample.append({"sample_id": sample_id, **comparison})
        if (index + 1) % 100 == 0 or index + 1 == len(sample_ids):
            print(f"Full-val inference and decode {index + 1}/{len(sample_ids)}", flush=True)

    if len(list(pytorch_dir.glob("*.txt"))) != EXPECTED_SPLIT_SIZE or len(list(onnx_dir.glob("*.txt"))) != EXPECTED_SPLIT_SIZE:
        raise RuntimeError("Full-val prediction directories are incomplete")
    results = {}
    for role in ("pytorch", "onnx"):
        prediction_dir = pytorch_dir if role == "pytorch" else onnx_dir
        # Parse the emitted KITTI files through the same parser consumed by the
        # project's external-prediction evaluator, ensuring AP uses serialized output.
        from data.kitti_prediction_parser import load_kitti_prediction_directory
        predictions = load_kitti_prediction_directory(
            prediction_dir, sample_ids, allowed_classes=CLASS_NAMES, require_all_files=True
        )
        metrics = [result.to_dict() for result in evaluate_kitti_r40(
            ground_truth=gt, predictions=predictions, classes=CLASS_NAMES
        )]
        results[role] = metrics
        write_metrics_csv(metrics, output_dir / f"{role}_kitti_r40_metrics.csv")

    comparison_rows = []
    for p_result, o_result in zip(results["pytorch"], results["onnx"]):
        if (p_result["class_name"], p_result["difficulty"], p_result["metric"]) != (
            o_result["class_name"], o_result["difficulty"], o_result["metric"]
        ):
            raise RuntimeError("PyTorch and ONNX AP rows are not aligned")
        comparison_rows.append({
            "class_name": p_result["class_name"],
            "difficulty": p_result["difficulty"],
            "metric": p_result["metric"],
            "iou_threshold": p_result["iou_threshold"],
            "pytorch_ap_r40": p_result["ap_r40"],
            "onnx_ap_r40": o_result["ap_r40"],
            "onnx_minus_pytorch_ap_r40": o_result["ap_r40"] - p_result["ap_r40"],
            "pytorch_true_positives": p_result["num_true_positives"],
            "onnx_true_positives": o_result["num_true_positives"],
        })
    write_metrics_csv(comparison_rows, output_dir / "kitti_r40_comparison.csv")
    for stats in raw_head_stats.values():
        stats["mean_abs"] = stats["abs_sum"] / stats["value_count"] if stats["value_count"] else 0.0
        del stats["abs_sum"]
        del stats["value_count"]

    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 full Chen-val ONNX decoded parity diagnostic",
        "revision": "2026-10-08-r1",
        "complete": len(per_sample) == EXPECTED_SPLIT_SIZE,
        "interpretation_only_no_deployment_gate": True,
        "official_kitti_leaderboard_metric": False,
        "split_protocol": "chen_3712_3769",
        "split": "val",
        "evaluated_images": len(per_sample),
        "prediction_files": {
            "pytorch": len(list(pytorch_dir.glob("*.txt"))),
            "onnx": len(list(onnx_dir.glob("*.txt"))),
        },
        "source_commit": source_commit,
        "source_url": RTM3D_URL,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "onnx_bundle": str(bundle.resolve()),
        "onnx_bundle_sha256": sha256_file(bundle),
        "onnx_sha256": export_report["onnx_sha256"],
        "validation_split_sha256": sha256_file(split_path),
        "dataset_root": str(dataset.resolve()),
        "16_image_raw_head_fixture_check": {
            "complete": bool(raw_report.get("complete")),
            "all_heads_within_export_tolerance": all(
                bool(values.get("passed", False))
                for values in raw_report.get("per_head", {}).values()
            ),
            "failed_heads": [
                name for name, values in raw_report.get("per_head", {}).items()
                if not values.get("passed", False)
            ],
            "per_head": raw_report.get("per_head", {}),
        },
        "runtime": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "decoder_gpu": torch.cuda.get_device_name(0),
            "onnxruntime_version": ort.__version__,
            "onnxruntime_provider": "CPUExecutionProvider",
            "opencv_version": cv2.__version__,
            "head_conv": head_conv,
        },
        "selection": {
            "top_k": TOP_K,
            "center_score_strictly_greater_than": VISIBILITY_THRESHOLD,
            "same_class_greedy_2d_iou_match_threshold": MATCH_IOU_THRESHOLD,
            "decoder": "pinned RTM3D car_pose_decode + car_pose_post_process",
            "prediction_score_and_kitti_serialization": "pinned RTM3D Debugger.save_kitti_format semantics",
        },
        "detections": {
            "count": total_detections,
            "class_count": class_counts,
            "mean_per_image": {role: total_detections[role] / len(sample_ids) for role in total_detections},
        },
        "full_val_raw_head_deltas": raw_head_stats,
        "decoded_matched_pair_delta_distributions": {
            key: distribution(values) for key, values in all_deltas.items()
        },
        "prediction_match_counts": {
            role: sum(sample[role + "_selected"] for sample in per_sample)
            for role in ("pytorch", "onnx")
        },
        "decoded_pairing": {
            "total_matched": sum(sample["matched"] for sample in per_sample),
            "pytorch_unmatched": sum(sample["pytorch_unmatched"] for sample in per_sample),
            "onnx_unmatched": sum(sample["onnx_unmatched"] for sample in per_sample),
        },
        "latency_ms_per_image_model_forward_only": {
            "pytorch_cuda": summarize_latency(pytorch_latency),
            "onnxruntime_cpu": summarize_latency(onnx_latency),
            "warning": "Colab GPU/CPU timing only; excludes image loading, preprocessing, decode, and is not an iPhone/device benchmark.",
        },
        "ap_r40": {
            "evaluator": "MobileADAS3D data.kitti_r40 project implementation; standard Car/Pedestrian/Cyclist class thresholds",
            "metrics_by_backend": results,
            "comparison_csv": str(output_dir / "kitti_r40_comparison.csv"),
            "prediction_dirs": {
                "pytorch": str(output_dir / "pytorch_predictions/data"),
                "onnx": str(output_dir / "onnx_predictions/data"),
            },
        },
        "per_sample": per_sample,
        "training_performed": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": "If full-split decoded behavior is stable, measure end-to-end latency on the target iPhone and validate the calibration decoder in the app.",
    }
    report_path = output_dir / "fullval_decoded_parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": report["complete"],
        "evaluated_images": report["evaluated_images"],
        "detections": report["detections"],
        "decoded_pairing": report["decoded_pairing"],
        "ap_comparison_csv": report["ap_r40"]["comparison_csv"],
        "report": str(report_path),
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.bundle, args.checkpoint, args.split_dir / "val.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    analyze(
        args.bundle.resolve(), args.repo.resolve(), args.checkpoint.resolve(),
        args.dataset_root.resolve(), (args.split_dir / "val.txt").resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
