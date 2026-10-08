"""Compare PyTorch-reference and ONNX Runtime RTM3D decoded predictions.

This is a descriptive, inference-only diagnostic on the fixed ONNX parity
fixtures. It uses RTM3D's pinned CUDA decoder and calibration-based postprocess
but intentionally does not define a deployment acceptance threshold or score AP.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_rtm3d_km3d_res18_smoke import (
    CLASS_NAMES,
    HEADS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TOP_K,
    VISIBILITY_THRESHOLD,
    find_sample_paths,
    load_calibration,
    verify_source,
)
from scripts.audit_rtm3d_km3d_res18_onnx import sha256_file


BUNDLE_FILES = {
    "rtm3d_onnx_export.json",
    "onnxruntime_cpu_parity.json",
    "parity_reference.npz",
    "RTM3D_KM3D_ResNet18.onnx",
}
MATCH_IOU_THRESHOLD = 0.5


def extract_bundle(bundle_path: Path, artifact_dir: Path) -> None:
    """Extract only the four expected regular files from the user's bundle."""
    with zipfile.ZipFile(bundle_path) as archive:
        infos = [item for item in archive.infolist() if item.filename in BUNDLE_FILES]
        names = {item.filename for item in infos}
        missing = BUNDLE_FILES - names
        if missing:
            raise RuntimeError(f"ONNX bundle is missing files: {sorted(missing)}")
        if len(infos) != len(BUNDLE_FILES):
            raise RuntimeError("ONNX bundle contains duplicate required filenames")
        total_size = 0
        for item in infos:
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or item.is_dir():
                raise RuntimeError(f"Unsafe or unexpected ONNX bundle entry: {item.filename!r}")
            total_size += item.file_size
        if total_size > 1024**3:
            raise RuntimeError(f"ONNX bundle is unexpectedly large ({total_size} uncompressed bytes)")
        archive.extractall(artifact_dir, members=infos)


def box_iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def load_pinned_package_module(package_name: str, package_dir: Path, module_name: str):
    """Load an RTM3D module under a private package to avoid local-name collisions."""
    package_dir = package_dir.resolve()
    package = sys.modules.get(package_name)
    if package is None:
        package_spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package_spec.submodule_search_locations = [str(package_dir)]
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_dir)]
        package.__spec__ = package_spec
        sys.modules[package_name] = package
    elif list(getattr(package, "__path__", [])) != [str(package_dir)]:
        raise RuntimeError(f"Private RTM3D package already points to another source directory: {package_name}")

    qualified_name = f"{package_name}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    source_path = package_dir / f"{module_name}.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Pinned RTM3D module is missing: {source_path}")
    spec = importlib.util.spec_from_file_location(qualified_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pinned RTM3D module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    setattr(package, module_name, module)
    return module


def greedy_match(reference_rows: np.ndarray, onnx_rows: np.ndarray, threshold: float = MATCH_IOU_THRESHOLD):
    """Greedily match same-class decoded boxes by highest 2D IoU."""
    candidates = []
    for ref_index, ref in enumerate(reference_rows):
        ref_class = int(round(float(ref[40])))
        for onnx_index, pred in enumerate(onnx_rows):
            if int(round(float(pred[40]))) != ref_class:
                continue
            overlap = box_iou_xyxy(ref[:4], pred[:4])
            if overlap >= threshold:
                candidates.append((overlap, ref_index, onnx_index))
    candidates.sort(reverse=True)
    used_reference: set[int] = set()
    used_onnx: set[int] = set()
    matches = []
    for overlap, ref_index, onnx_index in candidates:
        if ref_index in used_reference or onnx_index in used_onnx:
            continue
        used_reference.add(ref_index)
        used_onnx.add(onnx_index)
        matches.append((ref_index, onnx_index, overlap))
    return matches, sorted(set(range(len(reference_rows))) - used_reference), sorted(set(range(len(onnx_rows))) - used_onnx)


def distribution(values: list[float]) -> dict:
    finite_values = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite_values.size == 0:
        return {"count": 0, "p50": None, "p90": None, "max": None}
    return {
        "count": int(finite_values.size),
        "p50": float(np.percentile(finite_values, 50)),
        "p90": float(np.percentile(finite_values, 90)),
        "max": float(np.max(finite_values)),
    }


def selected_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0, 41), dtype=np.float32)
    finite = np.isfinite(rows).all(axis=1)
    return rows[finite & (rows[:, 4] > VISIBILITY_THRESHOLD)]


def decode_heads(heads: dict, meta: dict, const, center, scale, torch, np, decoder, postprocess) -> np.ndarray:
    decoded = decoder(
        heads["hm"].sigmoid(), heads["wh"], heads["hps"], heads["dim"], heads["rot"],
        prob=heads["prob"], reg=heads["reg"], hm_hp=heads["hm_hp"].sigmoid(),
        hp_offset=heads["hp_offset"], K=TOP_K, meta=meta, const=const,
    )
    raw = decoded.detach().float().cpu().numpy().reshape(1, TOP_K, 41)
    rows = postprocess(raw.copy(), [center], [scale], INPUT_HEIGHT // 4, INPUT_WIDTH // 4)[0][1]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 41)


def analyze(artifact_dir: Path, repo: Path, dataset: Path, output: Path) -> dict:
    export_report = json.loads((artifact_dir / "rtm3d_onnx_export.json").read_text(encoding="utf-8"))
    raw_report = json.loads((artifact_dir / "onnxruntime_cpu_parity.json").read_text(encoding="utf-8"))
    if not export_report.get("export_complete") or not export_report.get("onnx_checker_passed"):
        raise RuntimeError("The ONNX export is not complete or did not pass the ONNX checker")
    if set(export_report.get("outputs", {})) != set(HEADS):
        raise RuntimeError("ONNX export does not contain the expected pinned RTM3D output heads")
    if export_report.get("input", {}).get("name") != "image":
        raise RuntimeError("ONNX export input name differs from the pinned RTM3D input")
    if raw_report.get("onnx_sha256") != export_report.get("onnx_sha256"):
        raise RuntimeError("ONNX model identity differs between export and CPU parity reports")
    if raw_report.get("parity_fixture_sha256") != export_report.get("parity_fixture_sha256"):
        raise RuntimeError("Parity fixture identity differs between export and CPU parity reports")
    if raw_report.get("fixture_sample_ids") != export_report.get("input", {}).get("fixture_sample_ids"):
        raise RuntimeError("Fixture sample IDs differ between export and CPU parity reports")
    if sha256_file(artifact_dir / "RTM3D_KM3D_ResNet18.onnx") != export_report["onnx_sha256"]:
        raise RuntimeError("ONNX model hash differs from the export report")
    if sha256_file(artifact_dir / "parity_reference.npz") != export_report["parity_fixture_sha256"]:
        raise RuntimeError("Parity fixture hash differs from the export report")

    repo = repo.resolve()
    source_commit = verify_source(repo)
    if source_commit != export_report["source_commit"]:
        raise RuntimeError("Pinned RTM3D source does not match the ONNX export source commit")

    with np.load(artifact_dir / "parity_reference.npz", allow_pickle=False) as fixture:
        sample_ids = export_report["input"]["fixture_sample_ids"]
        references = {name: np.asarray(fixture[name], dtype=np.float32) for name in export_report["outputs"]}
        images = np.asarray(fixture["image"], dtype=np.float32)
    if len(sample_ids) != 16 or len(set(sample_ids)) != 16:
        raise RuntimeError(f"Expected 16 unique parity images, got {len(sample_ids)}")
    fixture_records = export_report["input"].get("fixture_samples", [])
    if len(fixture_records) != len(sample_ids):
        raise RuntimeError("Export report lacks image/calibration provenance for every fixture sample")
    if images.shape != (len(sample_ids), 3, INPUT_HEIGHT, INPUT_WIDTH):
        raise RuntimeError(f"Unexpected fixture image tensor shape: {images.shape}")
    if any(value.shape[0] != len(sample_ids) for value in references.values()):
        raise RuntimeError("Reference head batches do not match fixture sample IDs")
    if any(not np.isfinite(value).all() for value in references.values()):
        raise RuntimeError("Reference head fixture contains non-finite values")

    import cv2
    import onnxruntime as ort
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("RTM3D's pinned geometry decoder requires a Colab GPU runtime")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is not available")

    pinned_models_dir = repo / "src" / "lib" / "models"
    pinned_utils_dir = repo / "src" / "lib" / "utils"
    image_module = load_pinned_package_module(
        "rtm3d_pinned_utils_888c379", pinned_utils_dir, "image"
    )
    decoder_module = load_pinned_package_module(
        "rtm3d_pinned_models_888c379", pinned_models_dir, "decode"
    )
    postprocess_module = load_pinned_package_module(
        "rtm3d_pinned_utils_888c379", pinned_utils_dir, "post_process"
    )
    get_affine_transform = image_module.get_affine_transform
    car_pose_decode = decoder_module.car_pose_decode
    car_pose_post_process = postprocess_module.car_pose_post_process

    session = ort.InferenceSession(
        str(artifact_dir / "RTM3D_KM3D_ResNet18.onnx"),
        providers=["CPUExecutionProvider"],
    )
    expected_output_names = list(export_report["outputs"])
    if [item.name for item in session.get_inputs()] != [export_report["input"]["name"]]:
        raise RuntimeError("ONNX Runtime input name differs from the frozen export report")
    if [item.name for item in session.get_outputs()] != expected_output_names:
        raise RuntimeError("ONNX Runtime output names/order differ from the frozen export report")
    device = torch.device("cuda")
    const_values = [
        [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
        [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
    ]
    const = torch.tensor(const_values, dtype=torch.float32, device=device).view(1, 1, 16, 2)
    per_sample = []
    summary_values = {
        "matched_box_iou": [], "bbox_max_abs_px": [], "bbox_center_l2_px": [],
        "keypoints_max_abs_px": [], "keypoints_mean_abs_px": [], "keypoints_rmse_px": [],
        "dimensions_max_abs_m": [], "location_l2_m": [], "depth_abs_m": [],
        "yaw_abs_deg": [], "center_score_abs": [],
    }
    class_totals = {
        name: {"reference_selected": 0, "onnx_selected": 0, "matched": 0,
              "reference_unmatched": 0, "onnx_unmatched": 0}
        for name in CLASS_NAMES
    }
    class_metric_values = {
        name: {key: [] for key in summary_values}
        for name in CLASS_NAMES
    }
    for index, sample_id in enumerate(sample_ids):
        image_path, calibration_path = find_sample_paths(dataset, sample_id)
        record = fixture_records[index]
        if record.get("sample_id") != sample_id:
            raise RuntimeError(f"Fixture provenance order does not match sample ID {sample_id}")
        if sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError(f"KITTI image differs from the frozen fixture: {sample_id}")
        if sha256_file(calibration_path) != record["calibration_sha256"]:
            raise RuntimeError(f"KITTI calibration differs from the frozen fixture: {sample_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image dimensions for {sample_id}: {image_path}")
        height, width = image.shape[:2]
        center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
        scale = float(max(height, width))
        inverse = get_affine_transform(center, scale, 0, [INPUT_WIDTH // 4, INPUT_HEIGHT // 4], inv=1)
        projection = load_calibration(calibration_path, np)
        meta = {
            "trans_output_inv": torch.from_numpy(inverse).unsqueeze(0).to(device),
            "calib": torch.from_numpy(projection).unsqueeze(0).to(device),
        }

        onnx_values = session.run(
            expected_output_names,
            {export_report["input"]["name"]: images[index : index + 1]},
        )
        reference_heads = {
            name: torch.from_numpy(value[index : index + 1]).to(device)
            for name, value in references.items()
        }
        onnx_heads = {
            name: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
            for name, value in zip(expected_output_names, onnx_values)
        }
        if any(not bool(torch.isfinite(value).all().item()) for value in onnx_heads.values()):
            raise RuntimeError(f"ONNX Runtime produced non-finite head values for {sample_id}")
        with torch.inference_mode():
            reference_decoded = decode_heads(reference_heads, meta, const, center, scale, torch, np,
                                             car_pose_decode, car_pose_post_process)
            onnx_decoded = decode_heads(onnx_heads, meta, const, center, scale, torch, np,
                                        car_pose_decode, car_pose_post_process)
        reference_selected = selected_rows(reference_decoded)
        onnx_selected = selected_rows(onnx_decoded)
        matches, unmatched_reference, unmatched_onnx = greedy_match(reference_selected, onnx_selected)
        per_class = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            ref_indices = [i for i, row in enumerate(reference_selected) if int(round(float(row[40]))) == class_index]
            onnx_indices = [i for i, row in enumerate(onnx_selected) if int(round(float(row[40]))) == class_index]
            class_matches = [(ri, oi, iou) for ri, oi, iou in matches
                             if int(round(float(reference_selected[ri, 40]))) == class_index]
            totals = class_totals[class_name]
            totals["reference_selected"] += len(ref_indices)
            totals["onnx_selected"] += len(onnx_indices)
            totals["matched"] += len(class_matches)
            totals["reference_unmatched"] += sum(ri in unmatched_reference for ri in ref_indices)
            totals["onnx_unmatched"] += sum(oi in unmatched_onnx for oi in onnx_indices)
            per_class[class_name] = {
                "reference_selected": len(ref_indices),
                "onnx_selected": len(onnx_indices),
                "matched_at_iou_ge_0_5": len(class_matches),
                "reference_unmatched": sum(ri in unmatched_reference for ri in ref_indices),
                "onnx_unmatched": sum(oi in unmatched_onnx for oi in onnx_indices),
            }
            for ref_index, onnx_index, overlap in class_matches:
                reference_row = reference_selected[ref_index]
                onnx_row = onnx_selected[onnx_index]
                delta = {
                    "matched_box_iou": float(overlap),
                    "bbox_max_abs_px": float(np.max(np.abs(reference_row[:4] - onnx_row[:4]))),
                    "bbox_center_l2_px": float(np.linalg.norm(
                        (reference_row[:2] + reference_row[2:4] - onnx_row[:2] - onnx_row[2:4]) / 2.0
                    )),
                    "keypoints_max_abs_px": float(np.max(np.abs(reference_row[5:23] - onnx_row[5:23]))),
                    "keypoints_mean_abs_px": float(np.mean(np.abs(reference_row[5:23] - onnx_row[5:23]))),
                    "keypoints_rmse_px": float(np.sqrt(np.mean((reference_row[5:23] - onnx_row[5:23]) ** 2))),
                    "dimensions_max_abs_m": float(np.max(np.abs(reference_row[32:35] - onnx_row[32:35]))),
                    "location_l2_m": float(np.linalg.norm(reference_row[36:39] - onnx_row[36:39])),
                    "depth_abs_m": float(abs(reference_row[38] - onnx_row[38])),
                    "yaw_abs_deg": float(abs(math.atan2(
                        math.sin(float(reference_row[35] - onnx_row[35])),
                        math.cos(float(reference_row[35] - onnx_row[35])),
                    )) * 180.0 / math.pi),
                    "center_score_abs": float(abs(reference_row[4] - onnx_row[4])),
                }
                for key, value in delta.items():
                    summary_values[key].append(value)
                    class_metric_values[class_name][key].append(value)
        per_sample.append({
            "sample_id": sample_id,
            "reference_selected": len(reference_selected),
            "onnx_selected": len(onnx_selected),
            "matched_at_iou_ge_0_5": len(matches),
            "reference_unmatched": len(unmatched_reference),
            "onnx_unmatched": len(unmatched_onnx),
            "per_class": per_class,
        })
        print(f"Decoded parity {index + 1}/{len(sample_ids)}: {sample_id}", flush=True)

    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 ONNX decoded-prediction diagnostic",
        "diagnostic_revision": "2026-10-08-r1",
        "complete": len(per_sample) == len(sample_ids),
        "interpretation_only_no_deployment_gate": True,
        "source_commit": source_commit,
        "checkpoint_sha256": export_report["checkpoint_sha256"],
        "onnx_sha256": export_report["onnx_sha256"],
        "parity_fixture_sha256": export_report["parity_fixture_sha256"],
        "raw_head_cpu_parity_passed": raw_report.get("complete"),
        "raw_head_failure_summary": {
            name: values for name, values in raw_report.get("per_head", {}).items()
            if not values.get("passed", False)
        },
        "fixture_sample_ids": sample_ids,
        "images_decoded": len(per_sample),
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "onnxruntime_provider": "CPUExecutionProvider",
            "decoder_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
        "selection": {
            "top_k": TOP_K,
            "center_score_strictly_greater_than": VISIBILITY_THRESHOLD,
            "same_class_greedy_2d_iou_match_threshold": MATCH_IOU_THRESHOLD,
            "decoder": "pinned RTM3D car_pose_decode + car_pose_post_process",
        },
        "counts_by_class": class_totals,
        "matched_pair_delta_distributions": {
            key: distribution(values) for key, values in summary_values.items()
        },
        "matched_pair_delta_distributions_by_class": {
            class_name: {key: distribution(values) for key, values in metric_values.items()}
            for class_name, metric_values in class_metric_values.items()
        },
        "per_sample": per_sample,
        "training_performed": False,
        "full_val_ap_evaluated": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": "Review decoded prediction differences; if behavior is stable, proceed to full validation and target-device provider/performance tests",
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing diagnostic report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)
    with tempfile.TemporaryDirectory(prefix="rtm3d_onnx_decode_") as temporary:
        artifact_dir = Path(temporary)
        extract_bundle(args.bundle.resolve(), artifact_dir)
        analyze(artifact_dir, args.repo, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
