"""Inference-only YOLO26 detection/depth and UrbanOmniDetect-2 ONNX screens.

No training, depth-scale fitting, KITTI 3D AP claim, or phone qualification.
Each candidate runs in its own pinned Ultralytics environment. Downloads are
hash-checked before the trusted upstream checkpoint loader is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parking_edge_common as common


def flags():
    return dict(training_performed=False, optimizer_steps=0, depth_calibration_performed=False,
                full_val_ap_evaluated=False, independent_surface_distance_evaluated=False,
                iphone_performance_measured=False, deployment_qualified=False,
                parking_validation_qualified=False, ultrasonic_sensor_qualified=False)


def runtime():
    return dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                packages={k: importlib.metadata.version(k) for k in
                          ("torch", "torchvision", "ultralytics", "numpy", "onnx", "onnxruntime", "pillow")})


def download(spec, folder):
    path = folder / spec["file"]
    if not path.exists():
        print(f"Downloading pinned checkpoint: {spec['file']}", flush=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "MobileADAS3D-parking-screen/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        if common.sha(raw) != spec["sha256"]:
            raise RuntimeError(f"Upstream checkpoint hash changed: {spec['file']}; do not load it")
        common.immutable(path, raw)
    if path.is_symlink() or common.file_sha(path) != spec["sha256"]:
        raise RuntimeError(f"Checkpoint identity mismatch: {path}")
    return path


def urban_source(path, settings):
    if not path.exists():
        subprocess.run(["git", "clone", "--no-checkout", settings["source_url"], str(path)], check=True)
        subprocess.run(["git", "checkout", "--detach", settings["source_commit"]], cwd=path, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=path, text=True).strip()
    if (head != settings["source_commit"] or origin != settings["source_url"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=path).strip()):
        raise RuntimeError("Preserve changed Urban checkout; choose a fresh source directory")
    sys.path.insert(0, str(path))
    module = importlib.import_module("v2.hybrid_v2")
    if not Path(module.__file__).resolve().is_relative_to(path.resolve()):
        raise RuntimeError("Imported the wrong v2 package")
    return head


def iou(a, b):
    area = lambda v: max(0., v[2] - v[0]) * max(0., v[3] - v[1])
    intersection = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.


def detections(result, c, corners=False):
    import numpy as np
    if result.boxes is None:
        raise RuntimeError("Detection result has no boxes field")
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    keypoints = None
    if corners:
        if result.keypoints is None:
            raise RuntimeError("Urban output lost its cuboid head; load ONNX with task='pose'")
        keypoints = result.keypoints.data.cpu().numpy()
        if keypoints.shape != (len(xyxy), 8, 3):
            raise RuntimeError(f"Expected eight (x,y,confidence) projected corners; got {keypoints.shape}")
    rows = []
    for index, (box, score, cls) in enumerate(zip(xyxy, scores, classes)):
        if not np.isfinite(box).all() or not math.isfinite(float(score)) or not math.isfinite(float(cls)):
            raise RuntimeError("Nonfinite detection output")
        if int(cls) != cls or not 0 <= score <= 1 or box[2] < box[0] or box[3] < box[1]:
            raise RuntimeError("Invalid detection output")
        product = c["coco_mapping"].get(str(int(cls)))
        if product is None:
            continue
        row = dict(class_name=product, coco_class_id=int(cls), score=float(score), bbox=box.astype(float).tolist())
        if corners:
            if not np.isfinite(keypoints[index]).all():
                raise RuntimeError("Nonfinite projected corners")
            row["projected_corners_xy_confidence"] = keypoints[index].astype(float).tolist()
            row["distance_m"] = None
        rows.append(row)
    return rows


def depth_array(result, shape):
    import numpy as np
    if getattr(result, "depth", None) is None:
        raise RuntimeError("No numeric result.depth; this package/checkpoint does not support the required depth API")
    a = result.depth.data.cpu().numpy()
    if a.shape != tuple(shape) or a.ndim != 2 or not np.isfinite(a).all() or not (a > 0).all():
        raise RuntimeError(f"Expected a finite positive HxW metric depth map at original image size, got {a.shape}")
    return a.astype(np.float32)


def roi_depth(depth, box, minimum_fraction=.9):
    """A fixed diagnostic ROI, not segmentation, nearest surface or sensor range."""
    import numpy as np
    h, w = depth.shape
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = max(0., x2 - x1) / 4, max(0., y2 - y1) / 4
    left, right = max(0, int(math.floor(cx - dx))), min(w, int(math.ceil(cx + dx)))
    top, bottom = max(0, int(math.floor(cy - dy))), min(h, int(math.ceil(cy + dy)))
    if right <= left or bottom <= top:
        return dict(distance_m=None, valid_fraction=0., roi_xyxy=[left, top, right, bottom])
    values = depth[top:bottom, left:right]
    good = values[np.isfinite(values) & (values > 0)]
    fraction = float(good.size / values.size)
    valid = good.size > 0 and fraction >= minimum_fraction
    return dict(distance_m=float(np.median(good)) if valid else None, valid_fraction=fraction,
                roi_xyxy=[left, top, right, bottom],
                p10_m=float(np.quantile(good, .1)) if valid else None,
                p90_m=float(np.quantile(good, .9)) if valid else None)


def detection_parity(a, b, limits):
    """Match exact COCO class first; product merging cannot hide a class flip."""
    remaining = set(range(len(b)))
    deltas = []; scores = []; corner_deltas = []
    passed = len(a) == len(b)
    for row in sorted(a, key=lambda r: -r["score"]):
        options = [(iou(row["bbox"], b[j]["bbox"]), -j, j) for j in remaining
                   if b[j]["coco_class_id"] == row["coco_class_id"]]
        best = max(options) if options else (0., 0, None)
        if best[2] is None or best[0] < .99:
            passed = False
            continue
        other = b[best[2]]; remaining.remove(best[2])
        deltas.append(max(abs(x-y) for x, y in zip(row["bbox"], other["bbox"])))
        scores.append(abs(row["score"] - other["score"]))
        if "projected_corners_xy_confidence" in row:
            corner_deltas.append(max(abs(row["projected_corners_xy_confidence"][i][j] -
                                        other["projected_corners_xy_confidence"][i][j])
                                     for i in range(8) for j in range(2)))
    box, score, corner = max(deltas, default=0.), max(scores, default=0.), max(corner_deltas, default=0.)
    passed = passed and not remaining and box <= limits["box_max_abs_px"] and score <= limits["score_max_abs"] and corner <= limits["corner_max_abs_px"]
    return dict(passed=bool(passed), native_count=len(a), onnx_count=len(b),
                max_box_delta_px=box, max_score_delta=score, max_corner_delta_px=corner)


def depth_parity(a, b, limits):
    import numpy as np
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False, reason="shape_or_nonfinite")
    delta = np.abs(a - b)
    return dict(passed=bool(np.allclose(a, b, atol=limits["depth_atol_m"], rtol=limits["depth_rtol"])),
                max_abs_delta_m=float(delta.max()), mean_abs_delta_m=float(delta.mean()))


def recall_cells(rows, labels):
    gt = common.targets(labels)
    remaining = set(range(len(gt))); matched = set()
    for p in sorted(rows, key=lambda r: -r["score"]):
        candidates = [(iou(p["bbox"], gt[j]["bbox"]), -j, j) for j in remaining if p["class_name"] == gt[j]["class_name"]]
        best = max(candidates) if candidates else (0., 0, None)
        if best[0] >= .5:
            matched.add(best[2]); remaining.remove(best[2])
    return [dict(class_name=g["class_name"], source_class=g["source_class"],
                 band=common.band(g["camera_center_z_m"]), matched=j in matched,
                 truncated=g["truncated"], occluded=g["occluded"])
            for j, g in enumerate(gt) if common.band(g["camera_center_z_m"]) is not None]


def audit_graph(path, threads):
    import onnx
    import onnxruntime as ort
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    custom = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    if custom or any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise RuntimeError(f"Use a self-contained standard-operator graph, not custom/external operators: {custom}")
    options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"] or len(session.get_inputs()) != 1:
        raise RuntimeError("Expected a single-image-input ONNX CPU graph")
    inp = session.get_inputs()[0]
    if inp.type != "tensor(float)" or any(not isinstance(v, int) or v <= 0 for v in inp.shape):
        raise RuntimeError(f"Expected fixed FP32 input shape, got {inp.type} {inp.shape}")
    return session, dict(sha256=common.file_sha(path), bytes=path.stat().st_size,
                         input_name=inp.name, input_shape=inp.shape, output_names=[v.name for v in session.get_outputs()],
                         custom_operator_domains=custom, providers=session.get_providers(), cpu_threads=threads)


def export_component(native, path, spec, c, run_id):
    receipt = path.with_suffix(".export.json")
    if path.exists():
        if not receipt.is_file():
            raise RuntimeError(f"Unreceipted export preserved: {path}; use a new RUN_ID")
        value = json.loads(receipt.read_text())
        if value["run_identity"] != run_id or value["onnx_sha256"] != common.file_sha(path):
            raise RuntimeError("Export source/runtime/graph changed; preserve the run")
    else:
        # Ultralytics exports next to the private per-run checkpoint copy.
        produced = Path(native.export(format="onnx", imgsz=spec["imgsz"], batch=1, dynamic=False,
                                      simplify=False, opset=c["onnx_opset"], half=False, device="cpu", nms=False))
        if produced.resolve() != path.resolve():
            raise RuntimeError(f"Unexpected export path {produced}; preserve it")
        common.immutable(receipt, common.encoded(dict(run_identity=run_id, onnx_sha256=common.file_sha(path))))
    return audit_graph(path, c["cpu_threads"])


def overlay(path, image, rows, title, c):
    from PIL import ImageDraw
    canvas = image.copy(); draw = ImageDraw.Draw(canvas)
    for p in rows:
        color = "#00ff88" if p["class_name"] == "Vehicle" else "#ffcc00"
        draw.rectangle(p["bbox"], outline=color, width=3)
        label = f"{p['class_name']} {p['score']:.2f}"
        if p.get("distance_m") is not None:
            label += f"  ROI z~{p['distance_m']:.2f}m (unvalidated)"
        draw.text((p["bbox"][0], max(16, p["bbox"][1]) - 14), label, fill=color, stroke_width=1, stroke_fill="black")
        if "projected_corners_xy_confidence" in p:
            corners = p["projected_corners_xy_confidence"]
            for a, b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
                if min(corners[a][2], corners[b][2]) >= .5:
                    draw.line([tuple(corners[a][:2]), tuple(corners[b][:2])], fill=color, width=2)
    draw.rectangle((0, 0, canvas.width, 16), fill="black")
    draw.text((4, 2), title, fill="white")
    data = io.BytesIO(); canvas.save(data, format="PNG")
    common.immutable(path, data.getvalue())


def percentiles_ms(values):
    import numpy as np
    return dict(samples=len(values), median_ms=float(np.median(values) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000))


def screen(args):
    # Keep per-version Ultralytics settings out of the user's global profile.
    os.environ["YOLO_CONFIG_DIR"] = str(args.local_data / "runtime_settings" / args.candidate)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    protected = [args.local_data, args.weights]
    if args.source is not None:
        protected.append(args.source)
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in protected):
        raise RuntimeError("Keep experiment output separate from data, shared weights and upstream source")
    import numpy as np
    import torch
    from PIL import Image
    c = common.load_config(args.config); setting = c["candidates"][args.candidate]
    if importlib.metadata.version("ultralytics") != setting["ultralytics"]:
        raise RuntimeError(f"Use ultralytics=={setting['ultralytics']} in this candidate's separate runtime")
    torch.set_num_threads(c["cpu_threads"])
    m = common.verify_manifest(json.loads((args.local_data / "input_manifest.json").read_text()), c)
    for name, expected in m["files"].items():
        if common.file_sha(args.local_data / name) != expected:
            raise RuntimeError(f"Staged input changed: {name}")
    source = urban_source(args.source, setting) if args.candidate == "urban_nano" else None
    from ultralytics import YOLO
    run = dict(revision=c["revision"], candidate=args.candidate, config_sha256=common.identity(c),
               input_manifest_sha256=m["manifest_sha256"], runtime=runtime(), source_commit=source,
               implementation_sha256=common.file_sha(Path(__file__)), **flags())
    run_id = common.identity(run)
    common.immutable(args.output / "run_identity.json", common.encoded(dict(run, run_identity=run_id)))
    native = {}; exported = {}; graphs = {}; sessions = {}; specs = setting["components"]
    for name, spec in specs.items():
        original = download(spec, args.weights)
        local = args.output / "models" / spec["file"]
        common.immutable(local, original.read_bytes())
        native[name] = YOLO(str(local), task=spec["task"])
        graph = local.with_suffix(".onnx")
        sessions[name], graphs[name] = export_component(native[name], graph, spec, c, run_id)
        graphs[name]["preprocessing"] = dict(input="RGB_FP32_BCHW_divide_255",
                                            resize="stretch_to_square" if name == "depth" else "centered_letterbox_square_fill_114",
                                            resolution=spec["imgsz"],
                                            output_restore="bilinear_align_corners_true_to_original_image" if name == "depth" else "upstream_scale_boxes_and_keypoints_to_original_image")
        exported[name] = YOLO(str(graph), task=spec["task"])
    common.immutable(args.output / "graphs.json", common.encoded(graphs))
    def predict(model, name, image):
        kwargs = dict(source=np.asarray(image), imgsz=specs[name]["imgsz"], device="cpu", verbose=False, save=False)
        # PIL is supplied below so the common RGB->BGR conversion is performed
        # by the upstream predictor, not an accidental NumPy RGB/BGR swap.
        kwargs["source"] = image
        if name == "detector":
            kwargs.update(conf=c["confidence"], iou=c["nms_iou"], max_det=c["max_detections"], rect=False, nms=False)
        return model.predict(**kwargs)[0]
    # Warm both high-level paths before collecting per-frame timings.
    first = Image.open(args.local_data / "training/image_2" / (m["ids"][0] + ".png")).convert("RGB")
    for _ in range(2):
        for name in specs:
            predict(native[name], name, first); predict(exported[name], name, first)
    records = []
    for n, sample in enumerate(m["ids"]):
        destination = args.output / "records" / (sample + ".json")
        if destination.exists():
            record = json.loads(destination.read_text())
            if record.get("run_identity") != run_id or record.get("sample_id") != sample:
                raise RuntimeError("Cached per-frame identity changed; preserve the run")
            records.append(record)
            print(f"Verified saved frame {n+1}/{len(m['ids'])}: {sample}", flush=True)
            continue
        image_path = args.local_data / "training/image_2" / (sample + ".png")
        image = Image.open(image_path).convert("RGB")
        outputs = {}; times = {}; depth_maps = {}
        for backend, models in (("pytorch", native), ("onnx_cpu", exported)):
            started = time.perf_counter()
            det = predict(models["detector"], "detector", image)
            rows = detections(det, c, corners=args.candidate == "urban_nano")
            if "depth" in models:
                depth_maps[backend] = depth_array(predict(models["depth"], "depth", image), (image.height, image.width))
                for row in rows:
                    row.update(roi_depth(depth_maps[backend], row["bbox"], c["roi"]["valid_fraction_min"]))
            outputs[backend] = rows; times[backend] = time.perf_counter() - started
        parity = detection_parity(outputs["pytorch"], outputs["onnx_cpu"], c["parity"])
        depth_check = depth_parity(depth_maps["pytorch"], depth_maps["onnx_cpu"], c["parity"]) if depth_maps else None
        if n < 16:
            title = "ONNX CPU - ROI surface z is unvalidated" if depth_maps else "ONNX CPU - projected cuboids only, NO metric range"
            overlay(args.output / "overlays" / (sample + ".png"), image, outputs["onnx_cpu"], title, c)
        record = dict(sample_id=sample, run_identity=run_id, image_sha256=m["files"][f"training/image_2/{sample}.png"],
                      predictions=outputs, detection_parity=parity, depth_parity=depth_check,
                      pipeline_seconds_excluding_image_read=times,
                      near_detection_observations=recall_cells(outputs["onnx_cpu"], (args.local_data / "training/label_2" / (sample + ".txt")).read_bytes()))
        common.immutable(destination, common.encoded(record)); records.append(record)
        print(f"Completed {n+1}/{len(m['ids'])}: {sample}; export decision parity={parity['passed']}", flush=True)
    # Raw graph timings use explicit ORT CPU threads. These are one-image,
    # warm-cache laptop/Colab measurements, NOT camera or iPhone timings.
    profile_path = args.output / "raw_onnx_profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if profile["run_identity"] != run_id:
            raise RuntimeError("Profiling identity differs")
    else:
        import cv2
        image_bgr = cv2.imread(str(args.local_data / "training/image_2" / (m["ids"][0] + ".png")))
        feeds = {}
        for name in specs:
            tensor = native[name].predictor.preprocess([image_bgr]).cpu().numpy().astype(np.float32)
            if list(tensor.shape) != graphs[name]["input_shape"]:
                raise RuntimeError("Predictor tensor shape differs from fixed ONNX input")
            feeds[name] = {graphs[name]["input_name"]: tensor}
            values = sessions[name].run(None, feeds[name])
            if any(not np.isfinite(v).all() for v in values):
                raise RuntimeError("Nonfinite raw ONNX fixture")
            fixture = io.BytesIO()
            np.savez_compressed(fixture, image=tensor, **{f"output_{i}": v for i, v in enumerate(values)})
            common.immutable(args.output / "fixtures" / (name + ".npz"), fixture.getvalue())
        durations = {name: [] for name in specs}; total = []
        for iteration in range(25):
            begin = time.perf_counter()
            for name in specs:
                start = time.perf_counter(); sessions[name].run(None, feeds[name]); elapsed = time.perf_counter() - start
                if iteration >= 5:
                    durations[name].append(elapsed)
            if iteration >= 5:
                total.append(time.perf_counter() - begin)
        profile = dict(run_identity=run_id, cpu_threads=c["cpu_threads"], provider="CPUExecutionProvider",
                       components={name: percentiles_ms(v) for name, v in durations.items()},
                       sequential_pipeline_neural_only=percentiles_ms(total), iphone_measurement=False)
        common.immutable(profile_path, common.encoded(profile))
    cells = []
    for cls in c["classes"]:
        for b in range(2, 6):
            values = [v for r in records for v in r["near_detection_observations"] if v["class_name"] == cls and v["band"] == b]
            cells.append(dict(class_name=cls, gt_camera_center_band_m=[b, b+1], objects=len(values),
                              matched=sum(v["matched"] for v in values),
                              recall=sum(v["matched"] for v in values)/len(values) if values else None,
                              truncated=sum(v["truncated"] > 0 for v in values),
                              not_a_full_val_or_surface_distance_gate=True))
    passed = all(r["detection_parity"]["passed"] and (r["depth_parity"] is None or r["depth_parity"]["passed"]) for r in records)
    report = dict(complete=True, revision=c["revision"], candidate=args.candidate, run_identity=run_id,
                  frames=len(records), technical_export_screen_passed=passed,
                  raw_onnx_profile=profile, graphs=graphs, diagnostic_near_detection_cells=cells,
                  roi=c["roi"] if "depth" in specs else None, metric_range_available=setting["metric_range_available"],
                  distance_quantity=setting["distance_quantity"], independent_distance_accuracy_status="not_evaluated",
                  checkpoint_bytes_unchanged=all(common.file_sha(args.output/"models"/s["file"]) == s["sha256"] for s in specs.values()),
                  pipeline_timings={b: percentiles_ms([r["pipeline_seconds_excluding_image_read"][b] for r in records]) for b in ("pytorch", "onnx_cpu")},
                  notes=["64 near-range-enriched frames are not representative full-VAL AP.",
                         "COCO bus is Vehicle; Tram and Person_sitting have no dedicated COCO output class.",
                         "No per-image median alignment, GT scale fitting, offline smoothing or future frames.",
                         "ROI median can contain background and is not nearest surface or ultrasonic range.",
                         "No comparison of surface depth with annotated KITTI box-center depth."], **flags())
    common.immutable(args.output / "candidate_screen.json", common.encoded(report))
    common.immutable(args.output / "input_manifest.json", common.encoded(m))
    common.immutable(args.output / "screen_config.json", common.encoded(c))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Export parity failed; bundle the completed diagnostic, do not deploy")


def bundle(args):
    # Include failures and overlays for review, but never the source dataset or
    # pickle checkpoints. The two ONNX files and fixed tensor fixtures are the
    # portable runtime screen, not a ready-to-install iPhone package.
    names = [p for p in args.output.rglob("*") if p.is_file() and p.suffix in (".json", ".onnx", ".npz", ".png", ".log")]
    if not names:
        raise RuntimeError("No diagnostics to bundle")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(names):
            archive.write(path, path.relative_to(args.output))
        for path in (Path(__file__), Path(common.__file__)):
            archive.writestr("implementation/" + path.name, path.read_bytes())
    destination = args.output / (args.candidate + "_parking_edge_results.zip")
    # A bundle can include a later retry's logs; preserve each snapshot.
    if destination.exists() and destination.read_bytes() != data.getvalue():
        destination = args.output / (args.candidate + "_parking_edge_results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".zip")
    common.immutable(destination, data.getvalue())
    print("RESULT BUNDLE:", destination, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("screen", "bundle"))
    p.add_argument("--candidate", choices=("yolo_depth", "urban_nano"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--local-data", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--config", type=Path, default=common.CONFIG)
    a = p.parse_args()
    if a.stage == "bundle":
        bundle(a); return
    if a.local_data is None or a.weights is None or (a.candidate == "urban_nano" and a.source is None):
        p.error("screen needs --local-data, --weights and (for Urban) --source")
    try:
        screen(a)
    except Exception as e:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        common.immutable(a.output / ("failure_" + stamp + ".json"), common.encoded(dict(stage="screen", error_type=type(e).__name__, error=str(e), **flags())))
        raise


if __name__ == "__main__":
    main()
