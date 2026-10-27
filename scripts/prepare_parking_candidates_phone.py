"""Freeze the reviewed Urban/YOLO detection graphs and shared fixed16 phone inputs.

Offline, inference-only. No checkpoint loading, model rewriting, training, or
metric-distance claims. Refuse conflicting output instead of replacing it.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts.screen_parking_edge_candidates import detection_parity

REVISION = "PARKING-CANDIDATES-CPU-ABBA-2026-10-21-r1"
INPUT_SHA = "977ad14618c491dc2d9de0aaf007ece30fce47c4b85d6b06d8df3ab929dc4dc5"
MANIFEST_SHA = "291ad4d43c9c796bde5278e9ae65053ec314112a0689a2986d3ed19a89b4c6c1"
SPECS = {
    "urban_nano": dict(bundle="099d99628fa9e18ef9823f3d585d3185fa70d53339b98d4a71bbf3efd4e37b69",
                       graph="51e7ee3fff8ae6f0e1809e9eca8ab6b90f1adc7a4b11f73e5a50887d5f259afc",
                       member="models/urbanomnidetect_v2_n_640.onnx", columns=30),
    "yolo26n": dict(bundle="141ee9b523054ea1c04c9aefeff54ffa5c9f485eb66b6f9e0405191915d9566f",
                    graph="41316c0e8f8beb640a036dd05f52e35103b7621e96900fbff56220480b8dde1f",
                    member="models/yolo26n.onnx", columns=6),
}


def checked_archive(path, expected):
    if common.file_sha(path) != expected:
        raise ValueError(f"Uploaded archive identity changed: {path}")
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    if len(names) != len(set(names)) or any(
        n.startswith("/") or ".." in Path(n).parts or "\\" in n for n in names
    ) or archive.testzip() is not None:
        archive.close()
        raise ValueError("Unsafe, duplicate, or damaged ZIP entries")
    return archive


def letterbox(rgb):
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    gain = min(640 / h, 640 / w)
    nw, nh = round(w * gain), round(h * gain)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (640 - nw) / 2, (640 - nh) / 2
    padded = cv2.copyMakeBorder(resized, round(dy - .1), round(dy + .1),
                               round(dx - .1), round(dx + .1), cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255


def eligible(output, columns):
    """YOLO26 end-to-end output: no second NMS, original top-100 rule."""
    import numpy as np
    if output.shape != (1, 300, columns) or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError("Invalid fixed FP32 end-to-end output")
    rows = output[0]
    if ((rows[:, 4] < 0) | (rows[:, 4] > 1)).any() or (rows[:, 5] != np.floor(rows[:, 5])).any():
        raise ValueError("Invalid output scores/classes")
    selected = rows[rows[:, 4] > .25][:100]
    return selected[np.isin(selected[:, 5], [0, 2, 5, 7])].copy()


def restored_predictions(output, columns, width, height, candidate):
    rows = eligible(output, columns)
    gain = min(640 / height, 640 / width)
    nh, nw = round(height * gain), round(width * gain)
    px, py = round((640 - nw) / 2 - .1), round((640 - nh) / 2 - .1)
    # Preserve each reviewed Ultralytics version's restoration convention.
    gx, gy = (nw / width, nh / height) if candidate == "yolo26n" else (gain, gain)
    result = []
    for row in rows:
        box = [float(max(0, min(limit, (row[i] - pad) / scale)))
               for i, pad, scale, limit in [(0, px, gx, width), (1, py, gy, height),
                                             (2, px, gx, width), (3, py, gy, height)]]
        item = dict(coco_class_id=int(row[5]), score=float(row[4]), bbox=box)
        if columns == 30:
            # Pose 8.4.61 scale_coords uses half-padding after rounding the
            # resized image; unlike boxes it does not round that half-padding.
            cx, cy = (640 - nw) / 2, (640 - nh) / 2
            item["projected_corners_xy_confidence"] = [
                [float(max(0, min(width, (row[6 + j*3] - cx) / gain))),
                 float(max(0, min(height, (row[7 + j*3] - cy) / gain))), float(row[8 + j*3])]
                for j in range(8)]
        result.append(item)
    return result


def prepare(args):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
    if ort.__version__ != "1.30.0":
        raise RuntimeError("Use the reviewed ONNX Runtime 1.30.0")
    output = args.output.resolve()
    if args.output.is_symlink():
        raise ValueError("Use a private, non-symlink output folder")
    config = common.load_config(common.CONFIG)
    with checked_archive(args.inputs, INPUT_SHA) as inputs:
        packages = {key: checked_archive(getattr(args, key), spec["bundle"]) for key, spec in SPECS.items()}
        try:
            manifests = [json.loads(z.read("input_manifest.json")) for z in packages.values()]
            if manifests[0] != manifests[1] or manifests[0]["manifest_sha256"] != MANIFEST_SHA:
                raise RuntimeError("Candidates do not use the identical reviewed inputs")
            source = common.verify_manifest(manifests[0], config)
            if set(inputs.namelist()) != set(source["files"]):
                raise RuntimeError("Input archive inventory changed")
            for name, digest in source["files"].items():
                if common.sha(inputs.read(name)) != digest:
                    raise RuntimeError(f"Input bytes changed: {name}")
            ids = source["ids"][:16]
            image_tensors = []
            fixtures = []
            def tensor_file(name, array):
                raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
                common.immutable(output / name, raw)
                return dict(file=name, sha256=common.sha(raw), bytes=len(raw), shape=list(array.shape))
            for sample in ids:
                rgb = np.asarray(Image.open(io.BytesIO(inputs.read(f"training/image_2/{sample}.png"))).convert("RGB"))
                tensor = letterbox(rgb)
                image_tensors.append((tensor, rgb.shape))
                fixtures.append(dict(sample_id=sample, input=tensor_file(f"fixtures/{sample}/image.bin", tensor)))
            models = []
            for candidate, spec in SPECS.items():
                archive = packages[candidate]
                report = json.loads(archive.read("candidate_screen.json"))
                if not report["complete"] or not report["technical_export_screen_passed"] or report["training_performed"]:
                    raise RuntimeError("Candidate technical screen did not pass")
                graph = archive.read(spec["member"])
                if common.sha(graph) != spec["graph"]:
                    raise RuntimeError("Reviewed graph changed")
                model = onnx.load_model_from_string(graph)
                onnx.checker.check_model(model)
                if any(n.domain not in ("", "ai.onnx") for n in model.graph.node) or any(
                    t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer
                ):
                    raise RuntimeError("Phone graph must be standard and self-contained")
                path = f"models/{candidate}.onnx"
                common.immutable(output / path, graph)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
                if ([(v.name, v.shape, v.type) for v in session.get_inputs()] != [("images", [1,3,640,640], "tensor(float)")]
                        or [(v.name, v.shape) for v in session.get_outputs()] != [("output0", [1,300,spec["columns"]])]):
                    raise RuntimeError("Graph tensor interface changed")
                saved = np.load(io.BytesIO(archive.read("fixtures/detector.npz")), allow_pickle=False)
                if not np.array_equal(saved["image"], image_tensors[0][0]):
                    raise RuntimeError("Phone preprocessing differs from the original export fixture")
                refs = []
                for index, (sample, (tensor, shape)) in enumerate(zip(ids, image_tensors)):
                    raw = session.run(["output0"], {"images": tensor})[0]
                    eligible(raw, spec["columns"])
                    restored = restored_predictions(raw, spec["columns"], shape[1], shape[0], candidate)
                    record = json.loads(archive.read(f"records/{sample}.json"))
                    parity = detection_parity(record["predictions"]["onnx_cpu"], restored, config["parity"])
                    if not parity["passed"]:
                        raise RuntimeError(f"Host-vs-reviewed original-image decisions differ: {candidate}/{sample}: {parity}")
                    refs.append(tensor_file(f"fixtures/{sample}/{candidate}.bin", raw))
                    print(f"{candidate}: verified phone fixture {index+1}/16 {sample}", flush=True)
                saved.close()
                models.append(dict(id=candidate, file=path, sha256=spec["graph"], bytes=len(graph),
                                   columns=spec["columns"], references=refs, source_bundle_sha256=spec["bundle"]))
            manifest = dict(revision=REVISION, complete=True, input_manifest_sha256=MANIFEST_SHA,
                            input_archive_sha256=INPUT_SHA, fixtures=fixtures, models=models,
                            input_name="images", output_name="output0", onnxruntime_version="1.30.0",
                            provider="CPUExecutionProvider", intra_op_threads=2, inter_op_threads=1,
                            order=["urban_nano", "yolo26n", "yolo26n", "urban_nano"],
                            confidence=.25, max_detections=100, warmups=5, timed_predictions=100, sustained_seconds=60,
                            measurement_scope="model only; fixed preprocessed inputs; no camera, image preprocessing, depth network, tracking or rendering",
                            distance_measured=False, training_performed=False, deployment_qualified=False,
                            reference_environment=dict(onnxruntime=ort.__version__, numpy=np.__version__, opencv=cv2.__version__))
            data = common.encoded(manifest)
            common.immutable(output / "manifest.json", data)
            print(f"Package manifest SHA256: {common.sha(data)}", flush=True)
        finally:
            for archive in packages.values():
                archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "urban-nano", "yolo26n", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
