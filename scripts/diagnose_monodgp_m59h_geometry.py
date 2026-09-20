"""Measure M59f final-geometry differences on the exact M59g input set.

No new acceptance tolerance: the failed M59g gate remains failed. Preserve
paired raw outputs and compare both continuous geometry and native .2f text.
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_monodgp_m59g_precision import load_inputs, validate_outputs, verify_bundle, selection_summary
from scripts.collect_monodgp_m59g_inputs import ANCHOR_SHA256, TRACE_SHA256
from scripts.export_monodgp_m59f_position_interleave import verify_full_source
from scripts.validate_monodgp_m58_macos_parity import (
    INPUT_SHAPES, OUTPUT_NAMES, decode_candidates, sha256_file, tree_sha256, _sigmoid,
)
from scripts.validate_monodgp_m59f_macos import compare_stage

CONFIG_SHA256 = "4ad6d50241e5a6dd552e5d8b9c043241a11377c26c59b84ab3d2f7f58c7a42af"
UPSTREAM_FILES = {
    "lib/helpers/decode_helper.py": "70ab57ff293f563c14c72121732a327d524b036fc5de6724fd844d61d4c09c7a",
    "lib/datasets/utils.py": "022b1564a6d84d4b700d72ed478fe7b943db7c867a7a42fcbf9100446ed8fbf8",
    "lib/datasets/kitti/kitti_utils.py": "96d7154928e2d11c595c9307e14346c3eadeda596d952ac493c531530723ad64",
    "lib/helpers/tester_helper.py": "c83cec9527237550897ba8e9b89bc45d9865f4d4cec9814fd4b9a21a35b46dab",
    "utils/box_ops.py": "24fbfc914bbbad278b5fdf3078b103d67f771fac24ccec22276fb5392afd9e40",
}
CLASS_NAMES = ("Pedestrian", "Car", "Cyclist")
SCORE_THRESHOLD = 0.001


def upstream_decoder(repo):
    """Load only hash-pinned pure decoding functions, without CUDA/model imports."""
    nodes = []
    hashes = {}
    for relative, expected in UPSTREAM_FILES.items():
        path = repo / relative
        hashes[relative] = sha256_file(path)
        text = path.read_text()
        if hashes[relative] != expected:
            raise RuntimeError(f"Upstream decoding source changed: {relative}")
        if relative.endswith("tester_helper.py"):
            if "f.write(' {:.2f}'.format(results[img_id][i][j]))" not in text:
                raise RuntimeError("Native text serialization is no longer .2f")
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "class2angle", "get_heading_angle", "decode_detections", "extract_dets_from_outputs",
                "box_cxcylrtb_to_xyxy", "box_xyxy_to_cxcywh",
            }:
                nodes.append(node)
            if isinstance(node, ast.ClassDef) and node.name == "Calibration":
                nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in {"img_to_rect", "alpha2ry"})
    if len(nodes) != 8:
        raise RuntimeError("Unexpected upstream decoder interface")
    import torch
    namespace = {"np": np, "torch": torch, "num_heading_bin": 12}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<hash-verified MonoDGP decoder>", "exec"), namespace)
    namespace["box_ops"] = types.SimpleNamespace(**{name: namespace[name] for name in ("box_cxcylrtb_to_xyxy", "box_xyxy_to_cxcywh")})

    def decode(candidates, inputs, threshold=0.0):
        p2 = inputs["calibration"][0]
        calibration = types.SimpleNamespace(cu=p2[0, 2], cv=p2[1, 2], fu=p2[0, 0], fv=p2[1, 1],
                                            tx=p2[0, 3] / -p2[0, 0], ty=p2[1, 3] / -p2[1, 1])
        for name in ("img_to_rect", "alpha2ry"):
            setattr(calibration, name, types.MethodType(namespace[name], calibration))
        values = namespace["decode_detections"](
            candidates.copy(), {"img_id": [0], "img_size": inputs["image_size"].astype(np.int64)},
            [calibration], np.zeros((3, 3), dtype=np.float32), threshold,
        )[0]
        return np.asarray(values, dtype=np.float64).reshape(-1, 14)
    def extract(raw):
        values = {name: torch.from_numpy(np.asarray(raw[name]).copy()) for name in OUTPUT_NAMES}
        with torch.inference_mode():
            candidates = namespace["extract_dets_from_outputs"](values, K=50, topk=50).numpy()
            identities = torch.topk(values["pred_logits"].sigmoid().view(1, -1), 50, dim=1).indices.numpy()[0]
        return candidates, identities
    return decode, extract, hashes


def geometry_rows(candidates, inputs):
    """Native row: class, alpha, bbox[4], hwl[3], bottom_xyz[3], yaw, score.

    Exact frozen policy: meanshape=false; yaw uses the 2D bbox center (not
    the projected 3D center); initial filtering precedes depth confidence.
    Preserve upstream scalar dtypes, including integer original image size.
    """
    width, height = inputs["image_size"][0].astype(np.int64)
    if not np.array_equal(inputs["image_size"][0], [width, height]):
        raise ValueError("Original image size must have integer pixels")
    p2 = inputs["calibration"][0]
    fu, fv, cu, cv = p2[0, 0], p2[1, 1], p2[0, 2], p2[1, 2]
    if fu <= 0 or fv <= 0:
        raise ValueError("Invalid calibration focal lengths")
    tx, ty = p2[0, 3] / -fu, p2[1, 3] / -fv
    rows, bins = [], []
    for detection in candidates[0]:
        cls = int(detection[0])
        if cls not in (0, 1, 2):
            raise ValueError("Unknown native class ID")
        x, y = detection[2] * width, detection[3] * height
        w, h = detection[4] * width, detection[5] * height
        box = [x-w/2, y-h/2, x+w/2, y+h/2]
        dimensions = detection[31:34].copy()
        depth = detection[6]
        x3d, y3d = detection[34] * width, detection[35] * height
        loc_x = ((x3d-cu) * depth) / fu + tx
        loc_y = ((y3d-cv) * depth) / fv + ty
        location = np.array([loc_x, loc_y, depth])
        location[1] += dimensions[0] / 2
        heading_bin = np.argmax(detection[7:19])
        alpha = heading_bin * (2 * np.pi / 12) + detection[19 + heading_bin]
        if alpha > np.pi:
            alpha -= 2 * np.pi
        yaw = alpha + np.arctan2(x-cu, fu)
        if yaw > np.pi:
            yaw -= 2 * np.pi
        if yaw < -np.pi:
            yaw += 2 * np.pi
        score = detection[1] * detection[-1]
        rows.append([cls, alpha, *box, *dimensions.tolist(), *location.tolist(), yaw, score])
        bins.append(int(heading_bin))
    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape != (50, 14) or not np.isfinite(rows).all():
        raise RuntimeError("Invalid final geometry")
    return rows, np.array(bins)


def corners(rows):
    """Float64 derived geometry; KITTI bottom-center, right-handed yaw about Y."""
    result = []
    for row in rows:
        h, w, length = row[6:9]
        yaw = row[12]
        xyz = np.array([[length/2, length/2, -length/2, -length/2] * 2,
                        [0, 0, 0, 0, -h, -h, -h, -h],
                        [w/2, -w/2, -w/2, w/2] * 2])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        result.append((rotation @ xyz).T + row[9:12])
    return np.asarray(result)


def wrapped_degrees(a, b):
    return np.abs(np.rad2deg(np.arctan2(np.sin(a-b), np.cos(a-b))))


def geometry_deltas(reference, actual):
    ref_center, actual_center = reference[:, 9:12].copy(), actual[:, 9:12].copy()
    ref_center[:, 1] -= reference[:, 6] / 2
    actual_center[:, 1] -= actual[:, 6] / 2
    return {
        "bbox_pixel_max_delta": np.abs(reference[:, 2:6] - actual[:, 2:6]).max(axis=1),
        "depth_m_delta": np.abs(reference[:, 11] - actual[:, 11]),
        "dimension_m_max_delta": np.abs(reference[:, 6:9] - actual[:, 6:9]).max(axis=1),
        "bottom_center_m_l2_delta": np.linalg.norm(reference[:, 9:12] - actual[:, 9:12], axis=1),
        "geometric_center_m_l2_delta": np.linalg.norm(ref_center - actual_center, axis=1),
        "radial_distance_m_delta": np.abs(np.linalg.norm(ref_center, axis=1) - np.linalg.norm(actual_center, axis=1)),
        "yaw_degrees_delta": wrapped_degrees(reference[:, 12], actual[:, 12]),
        "alpha_degrees_delta": wrapped_degrees(reference[:, 1], actual[:, 1]),
        "corner_m_max_l2_delta": np.linalg.norm(corners(reference) - corners(actual), axis=2).max(axis=1),
        "final_confidence_delta": np.abs(reference[:, 13] - actual[:, 13]),
    }


def serialize_rows(rows):
    return [f"{CLASS_NAMES[int(row[0])]} 0.0 0" + "".join(f" {x:.2f}" for x in row[1:]) for row in rows]


def rounded_rows(rows):
    return np.array([[row[0], *(float(f"{x:.2f}") for x in row[1:])] for row in rows])


def filter_masks(candidates, rows):
    native = candidates[0, :, 1] >= SCORE_THRESHOLD
    product = np.isin(rows[:, 0], [0, 1])
    return {"native_score": native, "product_export": native & product,
            "nearby_score_before_text": native & product & (rows[:, 13] >= SCORE_THRESHOLD),
            "nearby_score_after_native_text": native & product & (rounded_rows(rows)[:, 13] >= SCORE_THRESHOLD)}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "max": None, "mean": None, "p95": None}
    return {"count": int(values.size), "max": float(values.max()), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95))}


def compare_geometry(candidates_ref, candidates_actual, inputs, expected_ids, actual_ids):
    # Do not compare mismatched ranks or silently discard changed identities.
    if not np.array_equal(expected_ids, actual_ids):
        raise RuntimeError("Top-k identities/order changed; retain raw outputs and investigate before geometry comparison")
    reference, ref_bins = geometry_rows(candidates_ref, inputs)
    actual, actual_bins = geometry_rows(candidates_actual, inputs)
    deltas = geometry_deltas(reference, actual)
    ref_masks, actual_masks = filter_masks(candidates_ref, reference), filter_masks(candidates_actual, actual)
    fields = {name: stats(values) for name, values in deltas.items()}
    decisions = {name: {"reference_count": int(mask.sum()), "actual_count": int(actual_masks[name].sum()),
                        "changed_count": int((mask != actual_masks[name]).sum())} for name, mask in ref_masks.items()}
    text_changes = np.array([a != b for a, b in zip(serialize_rows(reference), serialize_rows(actual))])
    return {"continuous_geometry": fields, "heading_bin_changes": int((ref_bins != actual_bins).sum()),
            "filter_decisions": decisions, "native_text_rows_changed": int(text_changes.sum()),
            "native_export_text_rows_changed": int((text_changes & ref_masks["native_score"] & actual_masks["native_score"]).sum()),
            "product_export_text_rows_changed": int((text_changes & ref_masks["product_export"] & actual_masks["product_export"]).sum()),
            "native_text_geometry": {name: stats(values) for name, values in geometry_deltas(rounded_rows(reference), rounded_rows(actual)).items()},
            "invalid_geometry": {side: int(((rows[:, 6:9] <= 0).any(axis=1) | (rows[:, 11] <= 0)).sum())
                                 for side, rows in (("reference", reference), ("actual", actual))}}, deltas, reference, actual


def topk_ids(raw):
    return np.argsort(-_sigmoid(raw["pred_logits"]).reshape(-1), kind="stable")[:50]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--m58-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--m59g-report", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/monodgp_m56d_runtime_frozen.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "m59h_geometry_diagnostic.json"
    report = {"schema_version": 1, "experiment": "M59h measurement-only final geometry audit", "complete": False,
              "training_performed": False, "weights_changed": False, "thresholds_changed": False,
              "new_acceptance_criteria": None, "ap_evaluation_performed": False, "full_validation_authorized": False,
              "deployment_authorized": False, "all_parity_gates_passed": False, "samples": []}
    try:
        if platform.system() != "Darwin":
            raise RuntimeError("Run on macOS")
        parent = json.loads(args.m59g_report.read_text())
        gate = json.loads((args.artifact_dir / "m59f_export_gate.json").read_text())
        package = args.artifact_dir / "MonoDGP_M59f_fp32.mlpackage"
        source = verify_full_source(args.m58_dir)
        if (not parent.get("complete") or not parent.get("multi_input_complete") or len(parent["samples"]) != 16
                or parent["source_torchscript_sha256"] != TRACE_SHA256
                or source["artifacts"]["torchscript_sha256"] != TRACE_SHA256
                or source["artifacts"]["reference_io_sha256"] != ANCHOR_SHA256
                or sha256_file(args.artifact_dir / "m59f_export_gate.json") != parent["export_gate_sha256"]
                or tree_sha256(package) != parent["mlpackage_tree_sha256"]
                or sha256_file(args.bundle_dir / "m59g_input_manifest.json") != parent["input_manifest_sha256"]
                or sha256_file(args.runtime_config) != CONFIG_SHA256):
            raise RuntimeError("Parent/config/package/input provenance differs from the completed M59g audit")
        with np.load(args.m58_dir / "m58_reference_io.npz", allow_pickle=False) as archive:
            anchor = {name: archive[name].copy() for name in (*INPUT_SHAPES, *OUTPUT_NAMES)}
        inputs_rows = verify_bundle(args.bundle_dir, anchor)
        native_decode, native_extract, native_hashes = upstream_decoder(args.upstream_repo)
        import torch
        import coremltools as ct
        torch.set_num_threads(4)
        original = torch.jit.load(str(args.m58_dir / "MonoDGP_M58_fixed_fp32.pt"), map_location="cpu").eval()
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
        report.update({"m59g_report_sha256": sha256_file(args.m59g_report), "m59g_gate_passed": parent["all_parity_gates_passed"],
                       "input_manifest_sha256": parent["input_manifest_sha256"], "source_torchscript_sha256": TRACE_SHA256,
                       "mlpackage_tree_sha256": parent["mlpackage_tree_sha256"], "runtime_config_sha256": CONFIG_SHA256,
                       "upstream_decoder_hashes": native_hashes, "meanshape": False, "score_threshold": SCORE_THRESHOLD,
                       "topk": 50, "nms_applied": False, "native_class_names": list(CLASS_NAMES),
                       "product_class_mapping": {"Car": "Vehicle", "Pedestrian": "Pedestrian"},
                       "native_serialization": "class 0.0 0 + 13 numeric fields formatted .2f",
                       "software": {"torch": torch.__version__, "numpy": np.__version__, "coremltools": ct.__version__,
                                    "python": platform.python_version(), "platform": platform.platform()},
                       "decoder_implementation_sha256": sha256_file(Path(__file__))})
        all_deltas, all_text_deltas, product_deltas = {}, {}, {}
        for number, item in enumerate(inputs_rows, 1):
            inputs = load_inputs(args.bundle_dir / item["file"])
            tensors = tuple(torch.from_numpy(inputs[name]) for name in INPUT_SHAPES)
            with torch.inference_mode():
                reference = {name: value.numpy().copy() for name, value in zip(OUTPUT_NAMES, original(*tensors))}
            prediction = model.predict(inputs)
            validate_outputs(reference); validate_outputs(prediction)
            pair_path = args.output_dir / f'{item["sample_id"]}_paired_raw.npz'
            np.savez_compressed(pair_path, **inputs, **{"pytorch_"+n: reference[n] for n in OUTPUT_NAMES},
                                **{"coreml_"+n: prediction[n] for n in OUTPUT_NAMES})
            expected, expected_ids = native_extract(reference)
            actual, actual_ids = native_extract(prediction)
            # Existing M59g checks below keep their original NumPy decoding policy.
            # This geometry diagnostic uses the pinned native PyTorch extraction.
            result, deltas, ref_rows, actual_rows = compare_geometry(expected, actual, inputs, expected_ids, actual_ids)
            # Differential checks against exact upstream functions on EVERY input/backend.
            native_errors = [float(np.abs(ours-native_decode(values, inputs)).max())
                             for ours, values in ((ref_rows, expected), (actual_rows, actual))]
            if max(native_errors) != 0.0:
                raise RuntimeError(f"Decoder port is not bit-exact to upstream: {native_errors}")
            for values, ours in ((expected, ref_rows), (actual, actual_rows)):
                if not np.array_equal(ours[filter_masks(values, ours)["native_score"]], native_decode(values, inputs, SCORE_THRESHOLD)):
                    raise RuntimeError("Native threshold behavior differs")
            sample = {"sample_id": item["sample_id"], "paired_raw_file": pair_path.name, "paired_raw_sha256": sha256_file(pair_path),
                      "decoder_port_max_abs_delta": max(native_errors), "selection": selection_summary(reference, prediction),
                      "native_vs_m59g_rank_changes": {"reference": int((expected_ids != topk_ids(reference)).sum()),
                                                      "actual": int((actual_ids != topk_ids(prediction)).sum())},
                      "unchanged_m59g_checks": compare_stage("full", reference, prediction, OUTPUT_NAMES), **result}
            if item["sample_id"] == "000001":
                sample["unchanged_frozen_anchor_checks"] = compare_stage("full", anchor, prediction, OUTPUT_NAMES)
            report["samples"].append(sample)
            product_mask = filter_masks(expected, ref_rows)["product_export"] & filter_masks(actual, actual_rows)["product_export"]
            for name, values in deltas.items():
                all_deltas.setdefault(name, []).extend(values.tolist())
                product_deltas.setdefault(name, []).extend(values[product_mask].tolist())
            for name, values in geometry_deltas(rounded_rows(ref_rows), rounded_rows(actual_rows)).items():
                all_text_deltas.setdefault(name, []).extend(values.tolist())
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f'M59h {number}/16 {item["sample_id"]}: max corner delta {deltas["corner_m_max_l2_delta"].max():.9f} m', flush=True)
        report.update({"complete": True, "samples_evaluated": 16, "candidates_compared": 800,
                       "continuous_geometry": {name: stats(values) for name, values in all_deltas.items()},
                       "product_export_geometry": {name: stats(values) for name, values in product_deltas.items()},
                       "native_text_geometry": {name: stats(values) for name, values in all_text_deltas.items()},
                       "heading_bin_changes": sum(row["heading_bin_changes"] for row in report["samples"]),
                       "native_text_rows_changed": sum(row["native_text_rows_changed"] for row in report["samples"]),
                       "native_export_text_rows_changed": sum(row["native_export_text_rows_changed"] for row in report["samples"]),
                       "product_export_text_rows_changed": sum(row["product_export_text_rows_changed"] for row in report["samples"]),
                       "filter_decision_changes": {name: sum(row["filter_decisions"][name]["changed_count"] for row in report["samples"])
                                                   for name in report["samples"][0]["filter_decisions"]},
                       "interpretation": "Measurement complete, not acceptance. No ground-truth accuracy, AP, runtime, or device claim."})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"M59h report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
