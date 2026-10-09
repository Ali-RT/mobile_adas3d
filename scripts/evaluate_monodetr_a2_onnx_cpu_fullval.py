"""Compare original A2 CUDA and ONNX Runtime CPU on all 3769 Chen val images."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_m67_a2_coreml as m67
import m64_teacher_qualification as q
from m68_onnx_common import (
    FULLVAL_AP_DRIFT_LIMIT, NEAR_RECALL_DRIFT_LIMIT, OUTPUT_KEYS, OUTPUT_NAMES,
    REVISION, compare_outputs, sha256, signature, write_json,
)
from export_monodetr_a2_onnx import build_model


def validate_package(output: Path, manifest: dict) -> dict:
    package_root = output / "A2_ONNX"
    record = json.loads((package_root / "manifest.json").read_text())
    saved_signature = record.pop("manifest_sha256", None)
    if signature(record) != saved_signature:
        raise RuntimeError("A2 ONNX phone package manifest signature changed")
    record["manifest_sha256"] = saved_signature
    model_path = package_root / record["model"]["file"]
    if sha256(model_path) != record["model"]["sha256"]:
        raise RuntimeError("A2 ONNX model SHA256 changed")
    if record["checkpoint_sha256"] != q.A2_SHA or record["checkpoint_epoch"] != 130:
        raise RuntimeError("M68 package is not the original A2 epoch-130 model")
    if record["original_m67_manifest_signature"] != manifest["signature_sha256"]:
        raise RuntimeError("M68 package belongs to a different M67/A2 manifest")
    if record["providers"] != ["CPUExecutionProvider"] or record["precision"] != "FP32":
        raise RuntimeError("M68 evaluation requires ONNX Runtime CPU FP32 only")
    return record


def cache_record(path: Path, run_signature: str, sample_id: str, files: dict):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    saved = record.pop("record_sha256", None)
    if signature(record) != saved:
        raise RuntimeError(f"Corrupt/edited full-val cache record: {path}")
    if record.get("run_signature") != run_signature or record.get("sample_id") != sample_id:
        raise RuntimeError(f"Full-val cache identity changed: {path}; preserve it and start a fresh RUN_ID")
    if record.get("input_file_sha256") != files:
        raise RuntimeError(f"KITTI data changed for cached frame {sample_id}; preserve the old run")
    record["record_sha256"] = saved
    return record


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError(f"Preserve conflicting prediction file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def predict_text(out: dict, info: dict, dataset, decode, extract, threshold: float) -> str:
    detections = extract(out, K=50, topk=50).cpu().numpy()
    image_id = int(info["img_id"][0])
    calib = [dataset.get_calib(image_id)]
    decoded = decode(detections.copy(), info, calib, dataset.cls_mean_size, threshold)
    return q.prediction_text(decoded[image_id], dataset.class_name)


def metric_values(summary_path: Path) -> dict:
    report = json.loads(summary_path.read_text())
    if report.get("complete_split") is not True or report.get("evaluated_images") != 3769:
        raise RuntimeError(f"Incomplete Chen val metric summary: {summary_path}")
    values = {}
    for row in report["metrics"]:
        if row["difficulty"] == "moderate":
            values[f"{row['class_name'].lower()}_{row['metric']}_moderate"] = float(row["ap_r40"])
    expected = {"vehicle_bev_moderate", "vehicle_3d_moderate",
                "pedestrian_bev_moderate", "pedestrian_3d_moderate"}
    if set(values) != expected:
        raise RuntimeError(f"Unexpected product metric inventory: {sorted(values)}")
    return values


def nearby_values(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("complete") is not True or report.get("evaluated_images") != 3769:
        raise RuntimeError(f"Incomplete nearby-recall summary: {path}")
    return {name: float(values["near_recall"]) for name, values in report["classes"].items()}


def run_metrics(output: Path, dataset_root: Path, split_dir: Path, checkpoint: Path) -> dict:
    rows = {}
    for role in ("native_cuda", "onnx_cpu"):
        predictions = output / "fullval" / "predictions" / role
        metric_dir = output / "fullval" / "metrics" / role
        geometry_dir = output / "fullval" / "nearby" / role
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/evaluate_kitti_prediction_dir.py"),
            "--config", str(ROOT / "configs/kitti_mobileadas3d_s1.yaml"),
            "--profile", "colab_drive", "--dataset-root", str(dataset_root),
            "--split-dir", str(split_dir), "--prediction-dir", str(predictions),
            "--split", "val", "--classes", "Vehicle", "Pedestrian", "--source-name", f"M68_{role}",
            "--output-dir", str(metric_dir),
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/audit_product_prediction_geometry.py"),
            "--dataset-root", str(dataset_root), "--split-file", str(split_dir / "val.txt"),
            "--prediction-dir", str(predictions), "--output-dir", str(geometry_dir),
            "--checkpoint", str(checkpoint), "--expected-images", "3769",
            "--score-threshold", "0.001", "--match-iou-threshold", "0.5",
        ], cwd=ROOT, check=True)
        rows[role] = {
            "metrics": metric_values(metric_dir / "kitti_r40_summary.json"),
            "near_recall": nearby_values(geometry_dir / "nearby_geometry_summary.json"),
        }
    return rows


def evaluate(args) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime in the isolated M67 environment") from exc
    m67_manifest = m67.load_manifest(args.m67_manifest.resolve())
    output, dataset_root, split_dir = args.output_dir.resolve(), args.dataset_root.resolve(), args.split_dir.resolve()
    export_report = json.loads((output / "m68_onnx_export.json").read_text())
    if export_report.get("revision") != REVISION or not export_report.get("complete"):
        raise RuntimeError("Missing/incomplete M68 ONNX export report")
    package = validate_package(output, m67_manifest)
    ids = q.split_ids(split_dir / "val.txt", "val")
    if len(ids) != 3769:
        raise RuntimeError("Expected the complete Chen 3769-image validation split")
    # Existing completed reports are immutable. Interrupted prediction caches below are resumable.
    report_path = output / "m68_onnx_fullval.json"
    if report_path.exists():
        raise RuntimeError(f"Preserve completed full-val report and use a new RUN_ID: {report_path}")

    model_path = output / "A2_ONNX" / package["model"]["file"]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Full-val must run CPUExecutionProvider only")
    model, dataset = build_model(m67_manifest, dataset_root=dataset_root)
    if len(dataset) != 3769:
        raise RuntimeError(f"Full validation dataset loader has {len(dataset)} rows; expected 3769")
    m67.set_export(model, False)
    from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections

    cache_root = output / "fullval" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "revision": REVISION, "m67_manifest_signature": m67_manifest["signature_sha256"],
        "m68_manifest_signature": package["manifest_sha256"],
        "checkpoint_sha256": q.A2_SHA, "onnx_sha256": package["model"]["sha256"],
        "split_sha256": sha256(split_dir / "val.txt"), "onnxruntime_version": ort.__version__,
        "dataset_root": str(dataset_root), "score_threshold": 0.001, "topk": 50,
    }
    run_signature = signature(run_identity)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    native_texts, cpu_texts, raw_rows = {}, {}, []
    for index, (images, calibrations, _, info_tensors) in enumerate(loader):
        sample_id = f"{int(info_tensors['img_id'][0]):06d}"
        if sample_id != ids[index]:
            raise RuntimeError(f"Dataset order differs from frozen Chen val at {index}: {sample_id}/{ids[index]}")
        source_files = {
            f"image_2/{sample_id}.png": sha256(dataset_root / "training/image_2" / f"{sample_id}.png"),
            f"calib/{sample_id}.txt": sha256(dataset_root / "training/calib" / f"{sample_id}.txt"),
            f"label_2/{sample_id}.txt": sha256(dataset_root / "training/label_2" / f"{sample_id}.txt"),
        }
        record_path = cache_root / f"{sample_id}.json"
        record = cache_record(record_path, run_signature, sample_id, source_files)
        if record is None:
            images = images.float()
            calibrations = calibrations.float()
            sizes = info_tensors["img_size"].float()
            if tuple(images.shape) != (1, 3, 384, 1280):
                raise RuntimeError(f"Unexpected native A2 image shape at {sample_id}: {tuple(images.shape)}")
            with torch.inference_mode():
                native_values = model(images.cuda(), calibrations.cuda(), None, sizes.cuda(), dn_args=0)
            native_outputs = {name: native_values[key].detach().cpu().numpy().astype(np.float32)
                              for name, key in zip(OUTPUT_NAMES, OUTPUT_KEYS)}
            cpu_values = session.run(list(OUTPUT_NAMES), {
                "image": images.numpy().astype(np.float32),
                "calibration": calibrations.numpy().astype(np.float32),
                "image_size": sizes.numpy().astype(np.float32),
            })
            cpu_outputs = {name: np.asarray(value, dtype=np.float32)
                           for name, value in zip(OUTPUT_NAMES, cpu_values)}
            info = {key: value.numpy() for key, value in info_tensors.items()}
            native_out = {key: torch.from_numpy(native_outputs[name])
                          for name, key in zip(OUTPUT_NAMES, OUTPUT_KEYS)}
            cpu_out = {key: torch.from_numpy(cpu_outputs[name])
                       for name, key in zip(OUTPUT_NAMES, OUTPUT_KEYS)}
            text_native = predict_text(native_out, info, dataset, decode_detections,
                                       extract_dets_from_outputs, 0.001)
            text_cpu = predict_text(cpu_out, info, dataset, decode_detections,
                                    extract_dets_from_outputs, 0.001)
            comparison = compare_outputs(cpu_outputs, native_outputs)
            record = {"run_signature": run_signature, "sample_id": sample_id,
                      "input_file_sha256": source_files, "comparison": comparison,
                      "native_text": text_native, "onnx_cpu_text": text_cpu}
            record["record_sha256"] = signature(record)
            write_json(record_path, record)
        native_texts[sample_id] = record["native_text"]
        cpu_texts[sample_id] = record["onnx_cpu_text"]
        raw_rows.append(record["comparison"])
        atomic_text(output / "fullval/predictions/native_cuda" / f"{sample_id}.txt", record["native_text"])
        atomic_text(output / "fullval/predictions/onnx_cpu" / f"{sample_id}.txt", record["onnx_cpu_text"])
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"M68 full Chen val: {index + 1}/{len(ids)}", flush=True)

    if set(native_texts) != set(ids) or set(cpu_texts) != set(ids):
        raise RuntimeError("Incomplete ONNX/native KITTI prediction set")
    metrics = run_metrics(output, dataset_root, split_dir, Path(m67_manifest["checkpoint"]))
    metric_drift = {key: abs(metrics["onnx_cpu"]["metrics"][key] - value)
                    for key, value in metrics["native_cuda"]["metrics"].items()}
    near_drift = {key: abs(metrics["onnx_cpu"]["near_recall"].get(key, float("nan")) - value)
                  for key, value in metrics["native_cuda"]["near_recall"].items()}
    raw_passed = all(row["passed"] for row in raw_rows)
    ap_passed = bool(metric_drift) and all(np.isfinite(value) and value <= FULLVAL_AP_DRIFT_LIMIT
                                           for value in metric_drift.values())
    near_passed = bool(near_drift) and all(np.isfinite(value) and value <= NEAR_RECALL_DRIFT_LIMIT
                                           for value in near_drift.values())
    fixed16_rows = [row["onnx_vs_native_cuda"] for row in export_report.get("comparisons", [])
                    if "onnx_vs_native_cuda" in row]
    if len(fixed16_rows) != 16:
        raise RuntimeError("M68 export report must contain all 16 raw-output comparisons")
    report = {
        "schema_version": 1, "revision": REVISION, "complete": True,
        "experiment": "Original A2 full Chen val: native CUDA versus ONNX Runtime CPU",
        "m67_manifest_signature": m67_manifest["signature_sha256"],
        "m68_manifest_signature": package["manifest_sha256"], "checkpoint_sha256": q.A2_SHA,
        "onnx_sha256": package["model"]["sha256"], "onnxruntime_version": ort.__version__,
        "provider": session.get_providers(), "split_protocol": "chen_3712_3769",
        "evaluated_images": len(ids), "complete_split": len(ids) == 3769,
        "native_cuda_metrics": metrics["native_cuda"]["metrics"],
        "onnx_cpu_metrics": metrics["onnx_cpu"]["metrics"],
        "absolute_ap_drift": metric_drift, "ap_drift_limit": FULLVAL_AP_DRIFT_LIMIT,
        "native_cuda_near_recall": metrics["native_cuda"]["near_recall"],
        "onnx_cpu_near_recall": metrics["onnx_cpu"]["near_recall"],
        "absolute_near_recall_drift": near_drift,
        "near_recall_drift_limit": NEAR_RECALL_DRIFT_LIMIT,
        "fixed16_raw_parity_passed": all(row["passed"] for row in fixed16_rows),
        "fullval_raw_parity_passed": raw_passed,
        "ap_quality_gate_passed": ap_passed, "near_recall_gate_passed": near_passed,
        "quality_gate_passed": ap_passed and near_passed,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": ("Run the existing iOS benchmark app in CPU-only ONNX Runtime mode" if ap_passed and near_passed
                      else "Stop before phone deployment; review CPU output quality/backend drift before any compression or teacher work"),
    }
    write_json(output / "m68_onnx_fullval.json", report)
    if not report["quality_gate_passed"]:
        raise RuntimeError("ONNX Runtime CPU did not preserve A2 full-val AP/near recall; do not treat it as deployable")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m67-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
