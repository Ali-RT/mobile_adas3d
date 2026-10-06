"""Read-only M65 archive diagnosis. No inference, training, or checkpoint writes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

REVIEWED_ARCHIVE_SHA256 = "4f3504eddfd3c27038237863487674ec36ca390507e31b6c17d35132fcd9dff2"
REVIEWED_MANIFEST = "d52fd78bd039615e932604ea8a51b7a1e9d5c4358b3bfb3c6f31301dd3c10d4b"
FIELDS = ("score", "iou_2d", "iou_bev", "iou_3d", "depth_abs_error_m",
          "depth_relative_error", "dimension_mae_m", "yaw_abs_error_deg",
          "loc_x_abs_error_m", "loc_y_abs_error_m", "center3d_error_m")
GT_FIELDS = ("gt_depth_m", "near_field", "distance_bucket")


def signature(value):
    # Match the historical M64/M65 serializer exactly, including whitespace.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite geometry record")
    return result


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid near_field: {value}")
    return value == "True"


def records(matched, missing):
    """Keep the source's precision; do not invent missing bounding boxes/IDs."""
    result = []
    for matched_flag, source in ((True, matched), (False, missing)):
        for item in source:
            row = dict(item, matched=matched_flag)
            row["gt_depth_m"] = number(row["gt_depth_m"])
            row["near_field"] = boolean(row["near_field"])
            if row["class_name"] not in ("Vehicle", "Pedestrian"):
                raise ValueError("Unexpected class")
            if matched_flag:
                for name in (*FIELDS, "gt_yaw_rad", "pred_yaw_rad"):
                    row[name] = number(row[name])
                # Native score = sigmoid(class_logit) * exp(-depth_log_uncertainty),
                # a ranking value, not a calibrated probability bounded by one.
                if row["score"] < 0 or any(not 0 <= row[name] <= 1 for name in ("iou_2d", "iou_bev", "iou_3d")):
                    raise ValueError("Negative score or IoU outside [0,1]")
            result.append(row)
    return result


def key(row):
    return row["sample_id"], row["class_name"], row["gt_depth_m"]


def pair_records(baseline, control):
    """Pair only keys unique in the complete matched + missed GT universe."""
    a, b = Counter(map(key, baseline)), Counter(map(key, control))
    if a != b:
        raise ValueError("Ground-truth key multisets differ")
    ambiguous = {k for k, count in a.items() if count != 1}
    left = {key(row): row for row in baseline if key(row) not in ambiguous}
    right = {key(row): row for row in control if key(row) not in ambiguous}
    for identity in left:
        names = GT_FIELDS
        if left[identity]["matched"] and right[identity]["matched"]:
            names += ("gt_yaw_rad", "size_bucket")
        if any(left[identity][name] != right[identity][name] for name in names):
            raise ValueError(f"Ground-truth metadata changed for {identity}")
    coverage = dict(total_gt=len(baseline), paired_gt=len(left),
                    ambiguous_keys=len(ambiguous),
                    excluded_gt=sum(a[k] for k in ambiguous),
                    excluded_by_class=dict(Counter(k[1] for k in ambiguous for _ in range(a[k]))),
                    method="Exact image/class/GT-depth keys, unique across matched and missed rows; ambiguous keys excluded")
    return left, right, coverage


def summarize_pairs(left, right, class_name, nearby=False, far=False):
    keys = [k for k, r in left.items() if r["class_name"] == class_name
            and (not nearby or r["near_field"]) and (not far or not r["near_field"])]
    common = [k for k in keys if left[k]["matched"] and right[k]["matched"]]
    lost = [k for k in keys if left[k]["matched"] and not right[k]["matched"]]
    gained = [k for k in keys if not left[k]["matched"] and right[k]["matched"]]
    geometry = {}
    for field in FIELDS:
        a = [left[k][field] for k in common]
        b = [right[k][field] for k in common]
        deltas = [y - x for x, y in zip(a, b)]
        geometry[field] = dict(baseline_mean=statistics.fmean(a) if a else None,
                              control_mean=statistics.fmean(b) if b else None,
                              delta_mean=statistics.fmean(deltas) if deltas else None,
                              delta_median=statistics.median(deltas) if deltas else None)
    threshold = .7 if class_name == "Vehicle" else .5
    crossings = {}
    for field in ("iou_3d", "iou_bev"):
        crossing_lost = [k for k in common if left[k][field] >= threshold > right[k][field]]
        crossing_gained = [k for k in common if left[k][field] < threshold <= right[k][field]]
        crossings[field] = dict(threshold=threshold,
            common_lost=len(crossing_lost), common_gained=len(crossing_gained),
            baseline_success_lost_to_2d_miss=sum(left[k][field] >= threshold for k in lost),
            control_success_gained_from_2d_miss=sum(right[k][field] >= threshold for k in gained),
            common_success_baseline=sum(left[k][field] >= threshold for k in common),
            common_success_control=sum(right[k][field] >= threshold for k in common))
    return dict(gt=len(keys), common_matched=len(common), lost_2d_matches=len(lost),
                gained_2d_matches=len(gained), neither_matched=len(keys)-len(common)-len(lost)-len(gained),
                geometry_on_common_objects=geometry, overlap_threshold_crossings=crossings)


def logged_gradients(log):
    norms = [number(value) for value in re.findall(r"grad_norm=([0-9.eE+-]+)", log)]
    return dict(progress_rows_with_combined_norm=len(norms),
                combined_preclip_norm_min=min(norms) if norms else None,
                combined_preclip_norm_median=statistics.median(norms) if norms else None,
                combined_preclip_norm_max=max(norms) if norms else None,
                component_gradient_norms_recorded=False,
                gt_preservation_gradient_cosine_recorded=False,
                conclusion="Logged combined preclip norms do not establish GT/preservation dominance or conflict")


def read_archive(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REVIEWED_ARCHIVE_SHA256:
        raise ValueError("Use the reviewed, unchanged M65 results archive")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError("Corrupt archive or duplicate members")
        def read(name):
            return json.loads(archive.read(name))
        manifest = read("m65_manifest.json")
        plain = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest["manifest_sha256"] != REVIEWED_MANIFEST or signature(plain) != REVIEWED_MANIFEST:
            raise ValueError("M65 manifest identity differs")
        gate, training = read("m65_control_gate.json"), read("m65_training_summary.json")
        if any(v["manifest_sha256"] != REVIEWED_MANIFEST or not v["complete"] for v in (gate, training)):
            raise ValueError("Run summaries do not belong to the reviewed manifest")
        rows, nearby = {}, {}
        member_hashes = {}
        for role in ("baseline", "control"):
            base = f"evaluation/{role}/metrics/nearby/"
            tables = []
            for name in ("matched_geometry.csv", "false_negatives.csv"):
                contents = archive.read(base + name)
                member_hashes[base + name] = hashlib.sha256(contents).hexdigest()
                tables.append(list(csv.DictReader(io.StringIO(contents.decode()))))
            rows[role] = records(*tables)
            nearby[role] = read(base + "nearby_geometry_summary.json")
            if nearby[role]["evaluated_images"] != 3769 or nearby[role]["score_threshold"] != .001 or nearby[role]["match_2d_iou_threshold"] != .5:
                raise ValueError("Incomplete or different diagnostic matching protocol")
            for name, values in nearby[role]["classes"].items():
                cls = [r for r in rows[role] if r["class_name"] == name]
                actual = dict(gt=len(cls), matched=sum(r["matched"] for r in cls),
                              near_gt=sum(r["near_field"] for r in cls),
                              near_matched=sum(r["matched"] and r["near_field"] for r in cls))
                if any(actual[k] != values[k] for k in actual):
                    raise ValueError("CSV counts do not reconcile to nearby summary")
        left, right, coverage = pair_records(rows["baseline"], rows["control"])
        comparisons = {name: {scope: summarize_pairs(left, right, name, scope == "nearby", scope == "farther")
                              for scope in ("all", "nearby", "farther")} for name in ("Vehicle", "Pedestrian")}
        log = archive.read("colab_logs/m65_train_.log").decode()
    return dict(schema_version=1, complete=True, source_archive_sha256=digest,
        manifest_sha256=REVIEWED_MANIFEST, source_member_sha256=member_hashes,
        pairing_coverage=coverage, product_ap_baseline=gate["baseline"]["product"],
        product_ap_control=gate["control"]["product"],
        product_ap_delta=gate["candidate_minus_baseline"],
        paired_comparisons=comparisons,
        preservation_design=dict(
            protected="Reliability-filtered, independently GT-matched class logits, boxes, point depth, dimensions, and angle outputs; component means balanced across eligible classes",
            not_explicitly_protected=["Depth log uncertainty used in the native ranking score", "Unmatched/background query logits", "Intermediate features", "Objects excluded by reliability masks"],
            ranking_score="sigmoid(class_logit) * exp(-pred_depth[..., 1]); can exceed 1",
            source="Frozen M65 third_party/monodetr/m65_preservation.py and pinned MonoDETR 6994b9f lib/helpers/decode_helper.py",
            causal_attribution_established=False),
        training_evidence=dict(completed_optimizer_steps=training["optimizer_steps"],
            loss_means=training["loss_means"], preservation_pairs=training["preservation_pairs"],
            batchnorm_buffers_unchanged=training["running_buffers_unchanged"],
            frozen_anchor_unchanged=training["anchor_unchanged"], external_teacher_used=False,
            gradients=logged_gradients(log)),
        scope=dict(new_optimizer_steps=0, inference_performed=False, checkpoint_selected=False,
                   training_authorized=False, kd_authorized=False),
        limitations=["Geometry is conditional on score>=0.001 and greedy class-specific 2D IoU>=0.5 matches, not AP matching.",
            "Exported scores are native uncertainty-weighted ranking values, not probabilities; they can exceed one. Raw class logits and depth uncertainty are absent from the CSVs.",
            "No explicit label IDs, boxes, occlusion or truncation in this export. Ambiguous GT-depth keys are excluded; no difficulty-specific pairing is possible.",
            "Paired threshold crossings are diagnostic counts, not an AP decomposition or proof of which loss caused regression.",
            "The archive contains no checkpoint tensors or raw prediction files; state differences and independent AP recomputation were not performed.",
            "Separate GT and preservation gradients were not recorded. Scalar loss magnitude is not gradient magnitude.",
            "A zero-update gradient probe at the completed checkpoint would describe that endpoint, not reconstruct all 928 historical updates."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-zip", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = read_archive(args.results_zip)
    report["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report.exists() and args.report.read_text() != content:
        raise RuntimeError("Preserve the previous diagnostic and choose another report path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(content)
    print(json.dumps(dict(complete=True, report=str(args.report),
                         coverage=report["pairing_coverage"],
                         optimizer_steps_taken=0), indent=2))


if __name__ == "__main__":
    main()
