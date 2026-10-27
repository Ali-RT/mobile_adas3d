"""Raw metric-depth screen with unchanged Urban + two-depth phone graphs.

Sparse nuScenes mini LiDAR is an engineering reference, not dense camera GT.
Urban overlays and point coverage are diagnostics, not object recall or 3D AP.
No model training, scale alignment, metric cuboid reconstruction or app edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts import compare_parking_depth_models as comparison
from scripts import prepare_parking_lidar_depth as ground_truth
from scripts.prepare_parking_candidates_phone import SPECS, letterbox, restored_predictions
from scripts.prepare_parking_depth_phone import GRAPH_SHA, preprocess, restore_depth
from scripts.prepare_parking_two_depth_phone import check_depth
from scripts.screen_parking_edge_candidates import roi_depth

REVISION = "PARKING-URBAN-LIDAR-DEPTH-2026-10-27-r1"
DAV2_SHA = "6206c5ac06f385f2af7585c5353de8de1b25ec4a65e84d28b91e306f76561ccc"
MODELS = ["yolo26n_depth", "depth_anything_v2_metric_outdoor_small"]
SCOPES = ["all_projected", "conservative", "conservative_static_objects"]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def read_manifest(folder):
    """Validate the frozen inputs without rebasing or regenerating any GT."""
    manifest = json.loads((folder / "input_manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != common.identity(body) or not manifest.get("complete"):
        raise ValueError("Prepared ground-truth manifest changed")
    identity = json.loads((folder / "prepare_identity.json").read_text())
    if (common.identity(identity) != manifest["prepare_identity_sha256"]
            or identity["mapping"] != manifest["mapping"]
            or json.loads(ground_truth.CONFIG.read_text()) != manifest["config"]
            or common.file_sha(ground_truth.CONFIG) != identity["config_sha256"]):
        raise ValueError("Prepared configuration/identity changed")
    root = Path(manifest["dataset_root"])
    for table, digest in identity["table_sha256"].items():
        if common.file_sha(ground_truth.safe_file(root, f"v1.0-mini/{table}.json")) != digest:
            raise ValueError("Source nuScenes metadata changed")
    samples = manifest["samples"]
    if (len(samples) != manifest["config"]["maximum_images"]
            or len({s["sample_token"] for s in samples}) != len(samples)
            or len({s["camera_token"] for s in samples}) != len(samples)):
        raise ValueError("Frozen keyframe/image inventory changed")
    for sample in samples:
        for key in ["image", "cloud", "labels"]:
            if common.file_sha(ground_truth.safe_file(root, sample[f"{key}_file"])) != sample[f"{key}_sha256"]:
                raise ValueError(f"Source {key} bytes changed")
        if common.file_sha(ground_truth.safe_file(folder, sample["ground_truth_file"])) != sample["ground_truth_sha256"]:
            raise ValueError("Projected GT cache changed")
    return manifest


def load_points(folder, sample):
    import numpy as np
    with np.load(ground_truth.safe_file(folder, sample["ground_truth_file"]), allow_pickle=False) as archive:
        if set(archive.files) != {"xy", "z", "semantic", "static", "conservative"}:
            raise ValueError("Unexpected GT fields")
        points = {k: archive[k] for k in archive.files}
    n = len(points["z"])
    if (points["xy"].shape != (n, 2) or points["xy"].dtype != np.int32
            or any(points[k].shape != (n,) for k in ["z", "semantic", "static", "conservative"])
            or points["static"].dtype != bool or points["conservative"].dtype != bool
            or not np.isfinite(points["z"]).all() or not (points["z"] > 0).all()
            or (points["xy"] < 0).any() or (points["xy"][:, 0] >= sample["width"]).any()
            or (points["xy"][:, 1] >= sample["height"]).any()):
        raise ValueError("Malformed projected GT arrays")
    return points


def metrics(reference, prediction):
    """Score raw metres; never normalize or fit either input."""
    import numpy as np
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (reference.ndim != 1 or prediction.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(prediction).all()
            or not (reference > 0).all() or not (prediction > 0).all()):
        raise ValueError("Metric arrays must be matched, positive and finite")
    if not len(reference):
        return None
    error = prediction - reference
    absolute = np.abs(error)
    ratio = np.maximum(prediction / reference, reference / prediction)
    return dict(mean_absolute_error_m=float(absolute.mean()), median_absolute_error_m=float(np.median(absolute)),
        p95_absolute_error_m=float(np.quantile(absolute, .95)), mean_signed_error_m=float(error.mean()),
        fraction_within_0_10m=float(np.mean(absolute <= .1)), fraction_within_0_20m=float(np.mean(absolute <= .2)),
        rmse_m=float(np.sqrt(np.mean(error ** 2))), abs_rel=float(np.mean(absolute / reference)),
        delta1=float(np.mean(ratio < 1.25)))


def mask(points, mapping, scope, cls, low, high, inclusive_high=False):
    import numpy as np
    result = (points["z"] >= low) & ((points["z"] <= high) if inclusive_high else (points["z"] < high))
    if cls != "All":
        result &= np.isin(points["semantic"], mapping[cls])
    if scope != "all_projected":
        result &= points["conservative"]
    if scope == "conservative_static_objects":
        result &= points["static"]
    if scope not in SCOPES or cls not in ["All", "Vehicle", "Pedestrian"]:
        raise ValueError("Unknown metric cell")
    return result


def urban_coverage(points, rows):
    """Point coverage is deliberately NOT instance detection recall."""
    import numpy as np
    result = {name: np.zeros(len(points["z"]), dtype=bool) for name in ["Vehicle", "Pedestrian"]}
    x, y = points["xy"].T
    for row in rows:
        cls = "Pedestrian" if row["coco_class_id"] == 0 else "Vehicle"
        x1, y1, x2, y2 = row["bbox"]
        result[cls] |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return result


def summarize(manifest, data):
    import numpy as np
    config = manifest["config"]
    bands = [(f"{lo}-{hi}", lo, hi, hi == 80) for lo, hi in config["range_bands_m"]]
    bands += [("2-6", 2, 6, True)]
    result = {}
    for scope in SCOPES:
        result[scope] = {}
        for cls in ["All", "Vehicle", "Pedestrian"]:
            result[scope][cls] = {}
            for name, lo, hi, inc in bands:
                cells = [(sample, p, pred, mask(p, manifest["mapping"], scope, cls, lo, hi, inc))
                    for sample, p, pred in data]
                cells = [(s, p, pred, m) for s, p, pred, m in cells if m.any()]
                count = sum(int(m.sum()) for _, _, _, m in cells)
                measured = {}
                for model in MODELS:
                    gt = np.concatenate([p["z"][m] for _, p, _, m in cells]) if cells else np.empty(0)
                    values = np.concatenate([pred[model][m] for _, _, pred, m in cells]) if cells else np.empty(0)
                    measured[model] = metrics(gt, values)
                    if measured[model] is not None:
                        measured[model]["macro_image_mean_absolute_error_m"] = float(np.mean([
                            np.abs(pred[model][m].astype(np.float64)-p["z"][m]).mean() for _, p, pred, m in cells]))
                result[scope][cls][name] = dict(points=count, images=len(cells),
                    scenes=dict(Counter(s["scene"] for s, _, _, _ in cells)),
                    descriptive_coverage_adequate=count >= config["minimum_points_for_descriptive_cell"]
                        and len(cells) >= config["minimum_images_for_descriptive_cell"], models=measured)
    return result


def preview(rgb, depths, rows, points, token):
    """Identical Urban overlays; sparse LiDAR dots are measured z, not box centres."""
    import numpy as np
    from PIL import Image, ImageDraw
    height, width = rgb.shape[:2]
    scale = 800 / width
    w, h = 800, round(height * scale)
    out = Image.new("RGB", (w*2, h+95), "#101820")
    for col, model in enumerate(MODELS):
        image = Image.fromarray(rgb).resize((w, h))
        draw = ImageDraw.Draw(image)
        for row in rows:
            colour = "#ffcc33" if row["coco_class_id"] == 0 else "#22ff88"
            box = [v*scale for v in row["bbox"]]
            draw.rectangle(box, outline=colour, width=2)
            corners = row["projected_corners_xy_confidence"]
            for a, b in EDGES:
                if corners[a][2] >= .25 and corners[b][2] >= .25:
                    draw.line([corners[a][0]*scale,corners[a][1]*scale,corners[b][0]*scale,corners[b][1]*scale], fill="#55ddff", width=1)
            name = "Person" if row["coco_class_id"] == 0 else "Vehicle"
            distance = roi_depth(depths[model], row["bbox"])["distance_m"]
            label = f"{name} {row['score']:.2f} | ROI {distance:.2f}m" if distance is not None else f"{name} no range"
            draw.text((box[0], max(0, box[1]-12)), label, fill=colour, stroke_width=1, stroke_fill="black")
        near = points["conservative"] & (points["z"] >= 2) & (points["z"] <= 6)
        for xy in points["xy"][near][::3]:
            x, y = xy*scale
            draw.ellipse([x-1,y-1,x+1,y+1], fill="#ff44dd")
        out.paste(image, (col*w, 52))
        heading = ImageDraw.Draw(out)
        heading.text((col*w+8, 7), "Urban + " + model, fill="white")
        heading.text((col*w+8, 25), "Cyan: projected cuboid | Magenta: sparse LiDAR 2-6m", fill="white")
    ImageDraw.Draw(out).text((8, h+62), f"{token} | Displayed ROI depth is not nearest surface or ultrasonic range", fill="white")
    ImageDraw.Draw(out).text((8, h+79), "Qualitative overlay: NOT 3D AP, detection recall, or a 10cm qualification", fill="#ffcc33")
    buffer = io.BytesIO();out.save(buffer, format="PNG")
    return buffer.getvalue()


def evaluate(a):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if a.output.is_symlink():
        raise ValueError("Preserve symlinked output")
    manifest = read_manifest(a.prepared)
    root = Path(manifest["dataset_root"])
    environment = comparison.runtime()
    weights = comparison.verify_weights(a.weights)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    graphs = {
        "urban_nano": (a.assets/"PARKING_CANDIDATES_ONNX/models/urban_nano.onnx", SPECS["urban_nano"]["graph"], [1,3,640,640], [1,300,30]),
        MODELS[0]: (a.assets/"PARKING_TWO_DEPTH_ONNX/models/yolo26n-depth.onnx", GRAPH_SHA, [1,3,768,768], [1,1,768,768]),
        MODELS[1]: (a.assets/"PARKING_TWO_DEPTH_ONNX/models/depth_anything_v2_metric_outdoor_small.onnx", DAV2_SHA, [1,3,518,924], [1,1,518,924]),
    }
    for path, digest, _, _ in graphs.values():
        if common.file_sha(path) != digest:
            raise ValueError("Frozen phone graph changed")
    helpers = [Path(__file__), Path(comparison.__file__), Path(ground_truth.__file__),
        common.ROOT/"scripts/prepare_parking_depth_phone.py", common.ROOT/"scripts/prepare_parking_candidates_phone.py",
        common.ROOT/"scripts/prepare_parking_two_depth_phone.py", common.ROOT/"scripts/screen_parking_edge_candidates.py"]
    identity = dict(revision=REVISION, input_manifest_sha256=common.file_sha(a.prepared/"input_manifest.json"),
        environment=environment, graph_sha256={k:v[1] for k,v in graphs.items()},
        helper_sha256={p.name:common.file_sha(p) for p in helpers}, hf_files=weights,
        policy=dict(no_scale_alignment=True, no_training=True, paired_identical_GT_points=True,
            urban_confidence=.25, urban_max_detections=100, corner_display_confidence=.25,
            metric_3d_box_evaluated=False, detection_AP_or_recall_evaluated=False,
            same_Urban_outputs_for_both_depth_models=True, native_DAV2_parity_first_images=3,
            preview_selection="First two images with >=5 conservative near points for each class, GT only"))
    common.immutable(a.output/"evaluation_identity.json", common.encoded(identity))
    identity_sha = common.identity(identity)
    sessions = {k:comparison.ort_session(str(p), shape, out) for k,(p,_,shape,out) in graphs.items()}
    processor = AutoImageProcessor.from_pretrained(a.weights, local_files_only=True, trust_remote_code=False, use_fast=False)
    native = AutoModelForDepthEstimation.from_pretrained(a.weights, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager").float().cpu().eval()
    previews = set()
    for cls in ["Vehicle", "Pedestrian"]:
        previews.update(s["camera_token"] for s in [x for x in manifest["samples"] if x["counts"][f"{cls}:2-6"] >= 5][:2])
    data = [];records = []
    for index, sample in enumerate(manifest["samples"]):
        token = sample["camera_token"]
        points = load_points(a.prepared, sample)
        record_path = a.output/f"records/{token}.json"
        prediction_file = f"predictions/{token}.npz"
        path = a.output/prediction_file
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record["identity_sha256"] != identity_sha or record["sample"] != sample
                    or common.file_sha(path) != record["prediction_sha256"]):
                raise ValueError("Preserve mismatched cached prediction")
            if token in previews and common.file_sha(a.output/record["preview_file"]) != record["preview_sha256"]:
                raise ValueError("Saved combined preview changed")
            with np.load(path, allow_pickle=False) as saved:
                pred = {k:saved[k] for k in MODELS}
        else:
            rgb = np.asarray(Image.open(ground_truth.safe_file(root, sample["image_file"])).convert("RGB"))
            if rgb.shape != (sample["height"], sample["width"], 3):
                raise ValueError("Original camera image shape changed")
            t0 = time.perf_counter()
            urban_raw = sessions["urban_nano"].run(["output0"], {"images":letterbox(rgb)})[0]
            rows = restored_predictions(urban_raw, 30, sample["width"], sample["height"], "urban_nano")
            yolo_raw = sessions[MODELS[0]].run(["output0"], {"images":preprocess(rgb)})[0]
            hf = processor(images=rgb, return_tensors="pt")["pixel_values"]
            if list(hf.shape) != [1,3,518,924]:
                raise ValueError("Official HF processor shape is incompatible with this frozen phone graph")
            dav2_raw = sessions[MODELS[1]].run(["output0"], {"images":hf.numpy()})[0]
            parity = None
            if index < 3:
                with torch.inference_mode():
                    reference = native(pixel_values=hf).predicted_depth.unsqueeze(1).numpy()
                parity = check_depth(reference, dav2_raw)
            h,w = sample["height"], sample["width"]
            depths = {MODELS[0]:restore_depth(yolo_raw, h, w),
                MODELS[1]:comparison.restore_anything(torch.from_numpy(dav2_raw[:,0]), h, w)}
            x,y = points["xy"].T
            pred = {k:depths[k][y,x].astype(np.float32) for k in MODELS}
            buffer = io.BytesIO();np.savez_compressed(buffer, **pred)
            common.immutable(path, buffer.getvalue())
            coverage = urban_coverage(points, rows)
            diagnostic = {}
            for cls in ["Vehicle", "Pedestrian"]:
                m = mask(points, manifest["mapping"], "conservative", cls, 2, 6, True)
                diagnostic[cls] = dict(near_GT_points=int(m.sum()),
                    points_inside_same_class_predicted_box=int((m & coverage[cls]).sum()))
            for row in rows:
                row["nominal_depth_ROIs"] = {k:roi_depth(depths[k], row["bbox"]) for k in MODELS}
                corners = row["projected_corners_xy_confidence"]
                row["drawable_edges_of_12"] = sum(corners[u][2] >= .25 and corners[v][2] >= .25 for u,v in EDGES)
            record = dict(identity_sha256=identity_sha, sample=sample, prediction_file=prediction_file,
                prediction_sha256=common.file_sha(path), urban_predictions=rows,
                point_coverage_diagnostic_NOT_object_recall=diagnostic,
                native_DAV2_to_ONNX_parity=parity, host_seconds_including_preprocessing=time.perf_counter()-t0)
            if token in previews:
                name = f"previews/{token}.png"
                image = preview(rgb, depths, rows, points, token)
                common.immutable(a.output/name, image)
                record.update(preview_file=name, preview_sha256=common.sha(image))
            common.immutable(record_path, common.encoded(record))
        if any(pred[k].shape != points["z"].shape for k in MODELS):
            raise ValueError("Prediction/GT point identities differ")
        for k in MODELS:
            metrics(points["z"], pred[k])
        records.append(record);data.append((sample, points, pred))
        print(f"Verified/scored {index+1}/{len(manifest['samples'])}: {sample['scene']} {sample['channel']} {token}", flush=True)
    coverage = {}
    for cls in ["Vehicle", "Pedestrian"]:
        cells = [r["point_coverage_diagnostic_NOT_object_recall"][cls] for r in records]
        n = sum(c["near_GT_points"] for c in cells)
        hit = sum(c["points_inside_same_class_predicted_box"] for c in cells)
        coverage[cls] = dict(near_GT_points=n, points_inside_same_class_predicted_box=hit,
            fraction=hit/n if n else None, interpretation="Point-weighted box coverage, NOT instance detection recall")
    all_rows = [r for record in records for r in record["urban_predictions"]]
    summary = dict(complete=True, revision=REVISION, identity_sha256=identity_sha,
        images=len(records), scenes=manifest["inventory"]["selected_scenes"],
        metrics=summarize(manifest,data), inventory=manifest["inventory"],
        urban_diagnostics=dict(detections=len(all_rows), near_same_class_point_coverage=coverage,
            projected_cuboids_with_all_12_drawable_edges=sum(r["drawable_edges_of_12"] == 12 for r in all_rows),
            interpretation="Corner confidence/drawability is not cuboid accuracy; no metric XYZ, dimensions or orientation decoder evaluated"),
        native_DAV2_parity=[r["native_DAV2_to_ONNX_parity"] for r in records[:3]],
        previews=[r["preview_file"] for r in records if "preview_file" in r],
        limitations=manifest["limitations"] + [
            "Sparse projected LiDAR is not dense visible-surface truth; cross-sensor occlusions and moving objects can remain",
            "Pixel errors are correlated; macro per-image MAE also reported. This is not an official nuScenes or KITTI score",
            "No Pedestrian 2-4m points are available in these validation scenes; do not claim complete 2-6m pedestrian coverage",
            "Urban point coverage is not detection recall; projected cuboid display is not metric 3D-box accuracy",
            "No 10cm qualification or iPhone accuracy/latency qualification follows from this small screen"],
        scale_fitting_performed=False, training_performed=False, model_weights_changed=False,
        phone_app_changed=False, metric_3d_box_evaluated=False, detection_AP_or_recall_evaluated=False,
        deployment_qualified=False)
    common.immutable(a.output/"depth_comparison.json", common.encoded(summary))
    print(json.dumps({cls:summary["metrics"]["conservative"][cls]["2-6"] for cls in ["Vehicle","Pedestrian"]}, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--assets",type=Path,default=common.ROOT/"ios/M60Benchmark")
    p.add_argument("--weights",type=Path,default=common.ROOT/"outputs/parking_depth_comparison_20261026/checkpoints/depth_anything_v2_metric_outdoor_small")
    evaluate(p.parse_args())


if __name__ == "__main__":main()
