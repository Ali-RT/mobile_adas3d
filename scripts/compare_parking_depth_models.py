"""Matched, inference-only parking depth screen with a fixed YOLO26n detector.

KITTI box-centre labels are NOT surface-distance references. This screen does
not fit scale, choose a winning ROI using tape error, or qualify phone latency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts.prepare_parking_candidates_phone import (
    INPUT_SHA, MANIFEST_SHA, SPECS, checked_archive, letterbox, restored_predictions,
)
from scripts.prepare_parking_depth_phone import GRAPH_SHA, parity, preprocess, restore_depth
from scripts.screen_parking_edge_candidates import roi_depth

REVISION = "PARKING-TWO-DEPTH-SCREEN-2026-10-26-r1"
REPO = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
HF_REVISION = "fd2c22027eaf20374204f14099b8341e1925ad39"
FILES = {
    "config.json": (1063, "git-sha1", "d3569a37f098a26fb76fd33ae6ba23e33bea1a13"),
    "preprocessor_config.json": (437, "git-sha1", "4b48811870a9d567ee23549c08249fe943e57916"),
    "model.safetensors": (99173660, "sha256", "ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124"),
    "README.md": (4969, "git-sha1", "32917ca7070a1edfaada195c0de4dab873c1fc16"),
}
PACKAGES = {"transformers": "4.57.6", "huggingface-hub": "0.36.0", "tokenizers": "0.22.1", "onnxruntime": "1.30.0"}
CLASS_NAMES = {0: "Pedestrian", 2: "Vehicle", 5: "Vehicle", 7: "Vehicle"}


def verify_hf_file(name, raw):
    size, algorithm, expected = FILES[name]
    actual = (hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
              if algorithm == "git-sha1" else common.sha(raw))
    if len(raw) != size or actual != expected:
        raise ValueError(f"Pinned official depth checkpoint file changed: {name}")


def verify_weights(folder):
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError("Use a private non-symlink checkpoint folder")
    result = {}
    for name in FILES:
        path = folder / name
        if path.is_symlink():
            raise ValueError("Symlinked checkpoint file")
        raw = path.read_bytes(); verify_hf_file(name, raw)
        result[name] = common.sha(raw)
    config = json.loads((folder / "config.json").read_text())
    if config["depth_estimation_type"] != "metric" or config["max_depth"] != 80:
        raise ValueError("Use the metric OUTDOOR model, not relative or indoor depth")
    return result


def fetch(args):
    folder = args.weights
    if folder.is_symlink():
        raise ValueError("Use a private checkpoint folder")
    for name, (size, _, _) in FILES.items():
        path = folder / name
        if path.exists():
            if path.is_symlink():
                raise ValueError("Symlinked checkpoint file")
            verify_hf_file(name, path.read_bytes()); continue
        url = f"https://huggingface.co/{REPO}/resolve/{HF_REVISION}/{name}"
        print(f"Downloading pinned official {name} ({size:,} bytes)", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": REVISION})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(size + 1)
        verify_hf_file(name, raw); common.immutable(path, raw)
    print(json.dumps(dict(repo=REPO, revision=HF_REVISION, files=verify_weights(folder)), indent=2))


def runtime():
    import onnxruntime as ort
    actual = {name: importlib.metadata.version(name) for name in PACKAGES if name != "onnxruntime"}
    # Some locally prepared binary targets omit wheel metadata. Check the
    # imported runtime itself, as the reviewed phone-preparation helpers do.
    actual["onnxruntime"] = ort.__version__
    if actual != PACKAGES:
        raise RuntimeError(f"Use the isolated comparison dependencies: {PACKAGES}; got {actual}")
    actual.update({name: importlib.metadata.version(name) for name in ["torch", "numpy", "pillow"]})
    import cv2
    actual["opencv_import"] = cv2.__version__
    return dict(packages=actual, python=platform.python_version(), system=platform.system(), machine=platform.machine())


def ort_session(graph, shape, output_shape):
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    for name in ["intra_op", "inter_op"]:
        options.add_session_config_entry(f"session.{name}.allow_spinning", "0")
    session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
    if (session.get_providers() != ["CPUExecutionProvider"]
            or [(x.name, x.shape, x.type) for x in session.get_inputs()] != [("images", shape, "tensor(float)")]
            or [(x.name, x.shape, x.type) for x in session.get_outputs()] != [("output0", output_shape, "tensor(float)")]):
        raise ValueError("Reviewed ONNX graph interface changed")
    return session


def restore_anything(prediction, height, width):
    """Official HF model-card restoration; do not borrow YOLO interpolation."""
    import numpy as np
    import torch
    result = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=(height, width),
                                              mode="bicubic", align_corners=False)[0, 0].numpy()
    if result.shape != (height, width) or not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError("Depth Anything produced invalid restored metric depth")
    return result.astype(np.float32)


def fixed_rows(raw, width, height):
    rows = restored_predictions(raw, 6, width, height, "yolo26n")
    for row in rows:
        row["class_name"] = CLASS_NAMES[row["coco_class_id"]]
        b = row["bbox"]
        if b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError("Degenerate detector box; do not silently change the shared ROI")
    return rows


def compare_rows(rows, yolo, anything):
    """Same source boxes and same fixed central-half ROI for both depth maps."""
    if yolo.shape != anything.shape:
        raise ValueError("Compare depth maps at the same original image size")
    result = []
    for row in rows:
        a = roi_depth(yolo, row["bbox"]); b = roi_depth(anything, row["bbox"])
        if a["roi_xyxy"] != b["roi_xyxy"] or a["distance_m"] is None or b["distance_m"] is None:
            raise ValueError("Shared fixed ROI did not yield two valid depth values")
        result.append(dict(**row, yolo26n_depth=a, depth_anything_v2_metric_outdoor_small=b,
                           anything_minus_yolo_m=b["distance_m"]-a["distance_m"],
                           difference_is_not_error=True))
    return result


def preview(rgb, yolo, anything, rows, sample_id):
    """Two native RGB overlays, then two maps with a shared 0–10 m colour range."""
    import numpy as np
    from PIL import Image, ImageDraw
    height, width = rgb.shape[:2]
    scale = min(1., 640 / width); w, h = round(width * scale), round(height * scale)
    canvas = Image.new("RGB", (w * 2, (h + 38) * 2 + 24), "#111111")
    for col, (model, depth, key) in enumerate([
        ("YOLO26n-depth", yolo, "yolo26n_depth"),
        ("Depth Anything V2 Metric Outdoor Small", anything, "depth_anything_v2_metric_outdoor_small")]):
        image = Image.fromarray(rgb).resize((w, h)); draw = ImageDraw.Draw(image)
        for n, row in enumerate(rows):
            box = [x * scale for x in row["bbox"]]
            colour = "#00ff88" if row["class_name"] == "Vehicle" else "#ffcc00"
            draw.rectangle(box, outline=colour, width=2)
            label = f"{n} {row['class_name']} {row[key]['distance_m']:.2f}m"
            draw.text((box[0], max(0, box[1]-12)), label, fill=colour, stroke_width=1, stroke_fill="black")
        t = np.clip(depth / 10, 0, 1)
        colours = np.stack([np.clip(2*t-0.5, 0, 1), 1-np.abs(2*t-1), np.clip(1.5-2*t, 0, 1)], axis=2)
        heat = Image.fromarray((colours * 255).astype(np.uint8)).resize((w, h))
        canvas.paste(image, (col*w, 38)); canvas.paste(heat, (col*w, h+76))
        draw = ImageDraw.Draw(canvas)
        draw.text((col*w+5, 5), model, fill="white")
        draw.text((col*w+5, 20), "Nominal ROI depth; NOT measured accuracy", fill="#ffcc00")
        draw.text((col*w+5, h+45), "Same colour scale: blue near / red >=10m", fill="white")
    ImageDraw.Draw(canvas).text((5, canvas.height-18), f"{sample_id} | same YOLO26n boxes | no scale fitting | no 3D cuboids", fill="white")
    buffer = io.BytesIO(); canvas.save(buffer, format="PNG"); return buffer.getvalue()


def input_items(args, archive):
    """Private inputs are read-only; reject synthetic phone maps as field data."""
    import numpy as np
    from PIL import Image
    if args.inputs is not None:
        config = common.load_config()
        m = common.verify_manifest(json.loads(archive.read("input_manifest.json")), config)
        if m["manifest_sha256"] != MANIFEST_SHA:
            raise ValueError("Reviewed fixed64 input manifest differs")
        with checked_archive(args.inputs, INPUT_SHA) as inputs:
            if set(inputs.namelist()) != set(m["files"]):
                raise ValueError("Unexpected fixed64 input inventory")
            for name, expected in m["files"].items():
                if common.sha(inputs.read(name)) != expected:
                    raise ValueError("Fixed64 input hash mismatch")
            for sample in m["ids"][:args.limit]:
                raw = inputs.read(f"training/image_2/{sample}.png")
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                yield dict(id=sample, rgb=rgb, source_sha256=common.sha(raw),
                           dataset="KITTI_fixed64_engineering_screen", independent_surface_reference=None)
    else:
        from scripts.audit_parking_distance_samples import analyze
        for run in args.clean_run:
            audit = analyze(run)
            if audit["synthetic"]:
                raise ValueError("Synthetic preflight data cannot substitute for real parking captures")
            manifest = json.loads((run / "distance_diagnostics/manifest.json").read_text())
            for sample in manifest["samples"]:
                raw = (run / sample["png"]["file"]).read_bytes()
                yield dict(id=run.name + f"_{sample['frame_id']:06d}",
                           rgb=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")),
                           source_sha256=common.sha(raw), dataset="real_clean_phone_capture",
                           independent_surface_reference=sample["reference"],
                           reference_is_user_entered_unverified=True,
                           automatic_reference_to_detection_matching=False)


def summarize(records):
    import numpy as np
    result = {}
    for cls in ["Vehicle", "Pedestrian"]:
        values = [r["anything_minus_yolo_m"] for item in records for r in item["detections"] if r["class_name"] == cls]
        result[cls] = dict(detected_boxes=len(values),
            median_anything_minus_yolo_m=float(np.median(values)) if values else None,
            median_absolute_model_disagreement_m=float(np.median(np.abs(values))) if values else None,
            interpretation="Model disagreement, NOT distance error, accuracy, or evidence that either model is better")
    return result


def screen(args):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if not 1 <= args.limit <= 64 or args.output.is_symlink():
        raise ValueError("Use 1–64 frames and a private output folder")
    weights = verify_weights(args.weights); environment = runtime()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with checked_archive(args.yolo_bundle, SPECS["yolo26n"]["bundle"]) as archive:
        detector_graph = archive.read(SPECS["yolo26n"]["member"])
        depth_graph = archive.read("models/yolo26n-depth.onnx")
        if common.sha(detector_graph) != SPECS["yolo26n"]["graph"] or common.sha(depth_graph) != GRAPH_SHA:
            raise ValueError("Reviewed YOLO detector/depth graph hash changed")
        if not json.loads(archive.read("candidate_screen.json"))["technical_export_screen_passed"]:
            raise ValueError("Source YOLO screen did not pass")
        items = list(input_items(args, archive))[:args.limit]
        if not items:
            raise ValueError("No real input images were supplied")
        sources = [{k:v for k,v in item.items() if k != "rgb"} for item in items]
        identity = dict(revision=REVISION, implementation_sha256=common.file_sha(Path(__file__)),
            runtime=environment, detector_sha256=common.sha(detector_graph), yolo_depth_sha256=GRAPH_SHA,
            depth_anything_repo=REPO, depth_anything_revision=HF_REVISION, depth_anything_files_sha256=weights,
            sources=sources, threads=2, processor="official pinned DPTImageProcessor use_fast=False",
            hf_attention="eager", restoration="HF bicubic align_corners=False; YOLO bilinear align_corners=True",
            shared_roi="central half of the same YOLO26n bbox; median", scale_alignment=False)
        run_sha = common.identity(identity)
        common.immutable(args.output / "run_identity.json", common.encoded(dict(identity, identity_sha256=run_sha)))
        detector = ort_session(detector_graph, [1,3,640,640], [1,300,6])
        yolo = ort_session(depth_graph, [1,3,768,768], [1,1,768,768])
        with np.load(io.BytesIO(archive.read("fixtures/depth.npz")), allow_pickle=False) as fixture:
            checked = parity(fixture["output_0"], yolo.run(["output0"], {"images":fixture["image"]})[0])
            if not checked["passed"]:
                raise ValueError("YOLO raw export fixture parity failed")
        processor = AutoImageProcessor.from_pretrained(args.weights, local_files_only=True,
                                                       trust_remote_code=False, use_fast=False)
        model = AutoModelForDepthEstimation.from_pretrained(args.weights, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, attn_implementation="eager").float().cpu().eval()
        if model.config.depth_estimation_type != "metric" or model.config.max_depth != 80:
            raise ValueError("Wrong Depth Anything depth head")
        records = []
        for index, item in enumerate(items):
            path = args.output / "records" / (item["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if record["run_identity_sha256"] != run_sha or not record["complete"]:
                    raise ValueError("Preserve conflicting per-frame output; choose a new run")
                for name, digest in record["artifacts"].items():
                    if common.file_sha(args.output / name) != digest:
                        raise ValueError("Completed per-frame artifact changed")
                records.append(record); print(f"Verified {index+1}/{len(items)} {item['id']}", flush=True); continue
            rgb = item["rgb"]; height, width = rgb.shape[:2]
            detector_input = letterbox(rgb); yolo_input = preprocess(rgb)
            t = time.perf_counter(); hf_input = processor(images=rgb, return_tensors="pt")["pixel_values"]
            hf_pre_ms = (time.perf_counter()-t)*1000
            # Warm every backend on the first image; exclude warmups from records.
            if index == 0:
                detector.run(["output0"], {"images":detector_input}); yolo.run(["output0"], {"images":yolo_input})
                with torch.inference_mode(): model(pixel_values=hf_input)
            t = time.perf_counter(); raw_boxes = detector.run(["output0"], {"images":detector_input})[0]
            detector_ms = (time.perf_counter()-t)*1000; rows = fixed_rows(raw_boxes, width, height)
            t = time.perf_counter(); raw_depth = yolo.run(["output0"], {"images":yolo_input})[0]
            yolo_ms = (time.perf_counter()-t)*1000; yolo_map = restore_depth(raw_depth, height, width)
            t = time.perf_counter()
            with torch.inference_mode(): hf_depth = model(pixel_values=hf_input).predicted_depth
            hf_ms = (time.perf_counter()-t)*1000; anything_map = restore_anything(hf_depth, height, width)
            paired = compare_rows(rows, yolo_map, anything_map)
            buffer = io.BytesIO(); np.savez_compressed(buffer, yolo26n_depth=yolo_map,
                                                      depth_anything_v2_metric_outdoor_small=anything_map)
            files = {f"maps/{item['id']}.npz":buffer.getvalue(),
                     f"previews/{item['id']}.png":preview(rgb,yolo_map,anything_map,paired,item['id'])}
            for name, raw in files.items(): common.immutable(args.output / name, raw)
            record = dict(complete=True, run_identity_sha256=run_sha, sample=sources[index], detections=paired,
                source_size=[width,height], depth_anything_input_shape=list(hf_input.shape),
                models_unchanged=True, independent_surface_accuracy_evaluated=False,
                cpu_timing_ms=dict(detector_ort_forward=detector_ms,yolo_depth_ort_forward=yolo_ms,
                                   depth_anything_torch_forward=hf_ms,depth_anything_preprocess=hf_pre_ms),
                timing_caveat="Host screen only; different native runtimes; no iPhone speed or end-to-end claim",
                artifacts={name:common.sha(raw) for name,raw in files.items()})
            common.immutable(path, common.encoded(record)); records.append(record)
            print(f"Screened {index+1}/{len(items)} {item['id']}: {len(paired)} boxes; YOLO depth {yolo_ms:.1f}ms / DAV2 {hf_ms:.1f}ms (HOST ONLY)", flush=True)
        timing = {key:dict(median_ms=float(np.median([r["cpu_timing_ms"][key] for r in records])),
                          p95_ms=float(np.quantile([r["cpu_timing_ms"][key] for r in records],.95)))
                  for key in records[0]["cpu_timing_ms"]}
        result = dict(revision=REVISION, complete=True, frames=len(records), run_identity_sha256=run_sha,
            detector="same YOLO26n for both depth candidates", shared_roi="central-half bbox median",
            depth_anything_parameter_count=sum(p.numel() for p in model.parameters()),
            yolo_raw_fixture_parity=checked, model_disagreement=summarize(records), host_cpu_timing=timing,
            accuracy_evaluated=False, independent_surface_distance_evaluated=False,
            box_centre_depth_used_as_surface_gt=False, scale_fitting_performed=False,
            training_performed=False, optimizer_steps=0, model_weights_changed=False,
            iphone_performance_measured=False, deployment_qualified=False,
            next_step="Collect real clean tagged 2/4/6m captures; independently match the measured target before scoring surface distance; then export/benchmark the depth challenger on the phone",
            limitations=["Enriched KITTI screen is not representative full validation",
                         "Different depth values do not establish which model is more accurate",
                         "No nearest-surface or ultrasonic sensor-frame range claim",
                         "Host ORT versus Torch latency is not a fair phone-runtime comparison"])
        common.immutable(args.output / "depth_comparison.json", common.encoded(result))
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    p = commands.add_parser("fetch"); p.add_argument("--weights",type=Path,required=True)
    p = commands.add_parser("screen")
    p.add_argument("--weights",type=Path,required=True); p.add_argument("--yolo-bundle",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int,default=64)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs",type=Path); source.add_argument("--clean-run",type=Path,action="append")
    args = parser.parse_args(); dict(fetch=fetch,screen=screen)[args.stage](args)


if __name__ == "__main__":
    main()
