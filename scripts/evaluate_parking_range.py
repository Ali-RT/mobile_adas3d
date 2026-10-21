"""Read-only saved-prediction screen for A2 parking depths, standard library only.

KITTI camera-z error is not ultrasonic surface-range error. No training,
inference, CUDA, image staging, or phone mutation is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING
from data.matching import bbox_iou

CONFIG = ROOT / "configs/parking_range_validation_v0_1.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def immutable(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Preserve changed output and use a new RUN_ID: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Preserve interrupted publication: {temporary}")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_config(path=CONFIG):
    value = json.loads(Path(path).read_text())
    if value["classes"] != ["Vehicle", "Pedestrian"] or value["range_edges_m"] != [2., 3., 4., 5., 6.]:
        raise ValueError("Use the separate frozen Vehicle/Pedestrian 2 to 6 m contract")
    return value


def ids_from_split(path, contract):
    raw = Path(path).read_bytes()
    ids = raw.decode().split()
    if (len(ids) != contract["chen_val"]["images"] or len(set(ids)) != len(ids)
            or any(not re.fullmatch(r"\d{6}", s) for s in ids)
            or digest(raw) != contract["chen_val"]["split_file_sha256"]):
        raise ValueError("Expected the unchanged Chen 3769-frame val.txt; do not create a different subset")
    return ids


def band(depth, edges=(2., 3., 4., 5., 6.)):
    if depth is None or not math.isfinite(depth):
        return None
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        if low <= depth < high or (i == len(edges) - 2 and depth == high):
            return f"{low:g}_{high:g}m"
    return None


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def parse_rows(raw, prediction=False):
    rows = []
    for line_number, line in enumerate(raw.decode().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != (16 if prediction else 15):
            raise ValueError(f"KITTI row {line_number}: expected {'16' if prediction else '15'} fields")
        cls = KITTI_PRODUCTION_CLASS_MAPPING.get(fields[0], fields[0])
        if cls not in ("Vehicle", "Pedestrian"):
            continue
        values = [float(v) for v in fields[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite KITTI values at row {line_number}; repair the source, do not drop it")
        box, dims, location = values[3:7], values[7:10], values[10:13]
        valid_box = box[2] > box[0] and box[3] > box[1]
        # A long, truncated KITTI vehicle can be visible while its annotated
        # box center is behind the camera. Retain it for 2D matching; it is
        # outside this positive camera-center-depth proxy, not corrupt data.
        if not prediction and (not valid_box or min(dims) <= 0):
            raise ValueError(f"Invalid reference geometry at row {line_number}")
        rows.append(dict(class_name=cls, source_class=fields[0], bbox_2d=box,
                         depth_m=location[2], valid_geometry=valid_box and min(dims) > 0 and location[2] > 0,
                         score=values[14] if prediction else 1., truncated=values[0],
                         occluded=int(values[1]), row=line_number))
    return rows


def match(rows, targets, iou):
    """Match all depths first; depth cannot select or rescue a 2D match."""
    remaining = set(range(len(targets)))
    assignments = []
    for row in sorted(rows, key=lambda r: (-r["score"], r["row"])):
        candidates = [(bbox_iou(row["bbox_2d"], targets[j]["bbox_2d"]), -j, j) for j in remaining]
        best = max(candidates) if candidates else (0., 0, None)
        target = best[2] if best[0] >= iou else None
        if target is not None:
            remaining.remove(target)
        assignments.append((row, target, best[0]))
    return assignments


def observations(predictions, ground_truth, settings):
    objects, assigned = [], []
    for sample in sorted(ground_truth):
        for cls in ("Vehicle", "Pedestrian"):
            gt = [r for r in ground_truth[sample] if r["class_name"] == cls]
            pred = [r for r in predictions[sample] if r["class_name"] == cls
                    and r["score"] >= settings["serialized_score_threshold"]]
            pairs = match(pred, gt, settings["match_iou_2d"])
            by_gt = {j: (p, overlap) for p, j, overlap in pairs if j is not None}
            for j, target in enumerate(gt):
                cell = band(target["depth_m"])
                if cell is None:
                    continue
                p, overlap = by_gt.get(j, (None, None))
                valid = p is not None and p["valid_geometry"]
                error = p["depth_m"] - target["depth_m"] if valid else None
                objects.append(dict(sample_id=sample, class_name=cls, band=cell, gt_depth_m=target["depth_m"],
                    gt_row=target["row"], gt_truncated=target["truncated"], gt_occluded=target["occluded"],
                    matched=p is not None, valid_distance=valid, pred_row=p["row"] if p else None,
                    pred_depth_m=p["depth_m"] if p else None, score=p["score"] if p else None,
                    iou_2d=overlap, signed_error_m=error, absolute_error_m=abs(error) if error is not None else None))
            for p, j, overlap in pairs:
                gt_band = band(gt[j]["depth_m"]) if j is not None else None
                pred_band = band(p["depth_m"]) if p["valid_geometry"] else None
                assigned.append(dict(sample_id=sample, class_name=cls, pred_row=p["row"],
                    predicted_band=pred_band, gt_band=gt_band, score=p["score"], pred_depth_m=p["depth_m"],
                    matched=j is not None, correct_near_range_match=j is not None and gt_band is not None,
                    invalid_geometry=not p["valid_geometry"],
                    unassigned_range_false_positive=j is None and not p["valid_geometry"], iou_2d=overlap))
    return objects, assigned


def summarize(objects, assigned, contract):
    settings, thresholds = contract["preliminary_kitti_screen"], contract["thresholds"]
    cells = []
    for cls in contract["classes"]:
        unknown = sum(p["class_name"] == cls and p["unassigned_range_false_positive"] for p in assigned)
        for low, high in zip(contract["range_edges_m"], contract["range_edges_m"][1:]):
            key = f"{low:g}_{high:g}m"
            gt = [r for r in objects if r["class_name"] == cls and r["band"] == key]
            pp = [p for p in assigned if p["class_name"] == cls and p["predicted_band"] == key]
            errors = [r["signed_error_m"] for r in gt if r["valid_distance"]]
            n, frames = len(gt), len({r["sample_id"] for r in gt})
            metrics = dict(recall=sum(r["matched"] for r in gt) / n if n else None,
                precision=sum(p["correct_near_range_match"] for p in pp) / len(pp) if pp else None,
                valid_distance_coverage=len(errors) / n if n else None,
                median_absolute_distance_error_m=quantile([abs(e) for e in errors], .5),
                p95_absolute_distance_error_m=quantile([abs(e) for e in errors], .95),
                absolute_mean_signed_distance_error_m=abs(sum(errors) / len(errors)) if errors else None)
            checks = {name: None if value is None else (value >= thresholds[name] - 1e-12 if name in ("recall", "precision", "valid_distance_coverage")
                      else value <= thresholds[name] + 1e-12) for name, value in metrics.items()}
            enough_gt = n >= settings["minimum_gt_objects_per_cell"] and frames >= settings["minimum_gt_frames_per_cell"]
            enough_distance = len(errors) >= settings["minimum_matched_distances_per_cell"]
            enough_pred = len(pp) >= settings["minimum_predictions_per_precision_cell"]
            supported = {k: enough_gt and (enough_pred if k == "precision" else enough_distance
                         if "error" in k else True) for k in metrics}
            failed = [k for k, passed in checks.items() if passed is False and supported[k]]
            reasons = []
            if not enough_gt: reasons.append("insufficient_ground_truth_objects_or_frames")
            if not enough_distance: reasons.append("insufficient_valid_matched_distances")
            if not enough_pred: reasons.append("insufficient_predictions_for_precision")
            if unknown: reasons.append("unassigned_range_false_positives")
            state = "fail" if failed else "inconclusive" if reasons or any(v is None for v in checks.values()) else "pass"
            cells.append(dict(class_name=cls, band=key, ground_truth=n, ground_truth_frames=frames,
                matched=sum(r["matched"] for r in gt), missed=sum(not r["matched"] for r in gt),
                valid_matched_distances=len(errors), matched_invalid_distances=sum(r["matched"] and not r["valid_distance"] for r in gt),
                predicted_in_band=len(pp), correct_predictions_in_band=sum(p["correct_near_range_match"] for p in pp),
                unknown_range_false_positives=unknown, truncated_gt=sum(r["gt_truncated"] > 0 for r in gt),
                occluded_gt=sum(r["gt_occluded"] > 0 for r in gt), metrics=metrics,
                maximum_absolute_distance_error_m=max(map(abs, errors)) if errors else None,
                observed_threshold_checks=checks, evidence_supported=supported, supported_failures=failed,
                evidence_gaps=reasons, screen_status=state))
    state = "fail" if any(c["screen_status"] == "fail" for c in cells) else "pass" if all(c["screen_status"] == "pass" for c in cells) else "inconclusive"
    return dict(screen_status=state, cells=cells, parking_validation_qualified=False,
                ultrasonic_sensor_qualified=False, qualification_status="inconclusive_requires_independent_mounted_reference")


def named_paths(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name in result or name not in ("A2", "A2_Lite") or not path:
            raise ValueError("Use unique NAME=PATH values, with NAME A2 or A2_Lite")
        result[name] = Path(path).resolve()
    return result


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue().encode()


def provenance(path, name, contract):
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = contract["models"][name]
    if value.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
        raise ValueError(f"Wrong {name} checkpoint in model provenance: {path}")
    model_hash = value.get("onnx_sha256") or value.get("model_sha256")
    if isinstance(value.get("model"), dict):
        model_hash = value["model"].get("sha256")
    if isinstance(model_hash, dict):
        model_hash = model_hash.get("streamed")
    if model_hash not in expected["onnx_sha256"]:
        raise ValueError(f"Wrong {name} ONNX identity in model provenance: {path}")
    return dict(path=str(path), file_sha256=digest(raw), checkpoint_sha256=expected["checkpoint_sha256"],
                onnx_sha256=model_hash, identity_matches=True,
                prediction_files_cryptographically_bound_to_model=False)


def run(args):
    config_path = args.config.resolve()
    contract = load_config(config_path)
    ids = ids_from_split(args.split, contract)
    predictions = named_paths(args.predictions)
    records = named_paths(args.provenance)
    if args.stage == "screen" and not predictions:
        raise ValueError("Supply at least one --predictions A2=DIR or A2_Lite=DIR")
    if set(records) - set(predictions):
        raise ValueError("A provenance record must belong to a supplied prediction source")
    sources = {"labels": args.labels.resolve(), **predictions}
    output = args.output.resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in sources.values()):
        raise ValueError("Use a separate output directory outside every input directory")
    metadata = {k: provenance(v, k, contract) for k, v in records.items()}
    ground_truth, hashes = {}, {name: {} for name in sources}
    parsed = {name: {} for name in predictions}
    for index, sample in enumerate(ids):
        for name, directory in sources.items():
            path = directory / (sample + ".txt")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} frame {sample}: {path}; no result was declared")
            raw = path.read_bytes()
            hashes[name][sample] = digest(raw)
            try:
                rows = parse_rows(raw, name != "labels")
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
            if name == "labels": ground_truth[sample] = rows
            else: parsed[name][sample] = rows
        if (index + 1) % 500 == 0 or index + 1 == len(ids):
            print(f"Read/hashed {index + 1}/{len(ids)} frames; no inference or training", flush=True)
    inventory = []
    for cls in contract["classes"]:
        for key in ("2_3m", "3_4m", "4_5m", "5_6m"):
            selected = [(s, r) for s, rows in ground_truth.items() for r in rows
                        if r["class_name"] == cls and band(r["depth_m"]) == key]
            inventory.append(dict(class_name=cls, band=key, gt_objects=len(selected),
                frames=len({s for s, _ in selected}), truncated_gt=sum(r["truncated"] > 0 for _, r in selected),
                occluded_gt=sum(r["occluded"] > 0 for _, r in selected)))
    identity = dict(contract_sha256=digest(config_path.read_bytes()), split_sha256=digest(args.split.read_bytes()),
                    implementation_sha256=digest(Path(__file__).read_bytes()),
                    parser_dependencies_sha256={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in
                        (ROOT / "data/class_taxonomy.py", ROOT / "data/matching.py")},
                    file_sha256=hashes, sources={k: str(v) for k, v in sources.items()}, provenance=metadata)
    identity["input_signature_sha256"] = digest(encoded(identity))
    results = {}
    files = {"contract.json": config_path.read_bytes(), "input_manifest.json": encoded(identity),
             "inventory.csv": csv_bytes(inventory), "inventory.json": encoded(inventory)}
    for name, pp in parsed.items():
        objects, assigned = observations(pp, ground_truth, contract["preliminary_kitti_screen"])
        results[name] = summarize(objects, assigned, contract)
        flat = [{**{k:v for k,v in c.items() if not isinstance(v, (dict, list))}, **c["metrics"]} for c in results[name]["cells"]]
        files[f"{name}/cells.csv"] = csv_bytes(flat)
        files[f"{name}/objects.csv"] = csv_bytes(objects)
        files[f"{name}/predictions.csv"] = csv_bytes(assigned)
    report = dict(revision=contract["revision"], analysis_complete=True, evaluated_images=len(ids),
        comparison_complete=set(results) == {"A2", "A2_Lite"}, inventory=inventory, models=results,
        input_signature_sha256=identity["input_signature_sha256"],
        distance_quantity=contract["preliminary_kitti_screen"]["distance_quantity"],
        parking_validation_qualified=False, ultrasonic_sensor_qualified=False,
        training_performed=False, inference_performed=False, optimizer_steps=0,
        historical_gates_modified=False, mounted_sensor_data_evaluated=False,
        unknowns=["maximum_vehicle_speed", "required_sampling_rate", "mounted_reference_uncertainty",
                  "camera_to_ultrasonic_transform", "sensor_clock_alignment"],
        reference_objects_nonpositive_camera_z=sum(r["depth_m"] <= 0 for rows in ground_truth.values() for r in rows),
        note="KITTI camera-z screen only. No passing result certifies ultrasonic surface-range accuracy.")
    files["parking_range_screen.json"] = encoded(report)
    for relative, raw in files.items():
        target = output / relative
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError(f"Preserve existing results; new inputs need a new RUN_ID: {target}")
    for relative, raw in files.items():
        immutable(output / relative, raw)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, raw in sorted(files.items()):
            member = zipfile.ZipInfo(relative, date_time=(2026, 10, 21, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, raw)
    immutable(output / "parking_range_results.zip", stream.getvalue())
    print(json.dumps({"models": {k: v["screen_status"] for k, v in results.items()},
                      "comparison_complete": report["comparison_complete"],
                      "parking_validation_qualified": False, "output": str(output)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "screen"))
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--provenance", action="append", default=[], metavar="NAME=JSON")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
