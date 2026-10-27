"""Prepare unchanged YOLO26n-depth and fixed16 references for iPhone CPU tests.

No downloads, checkpoint deserialization, training, scale fitting or accuracy
qualification. Original-image ROI checks preserve archived behavior only.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts.prepare_parking_candidates_phone import INPUT_SHA, MANIFEST_SHA, SPECS, checked_archive
from scripts.screen_parking_edge_candidates import roi_depth

REVISION = "PARKING-YOLO26N-DEPTH-CPU-2026-10-21-r1"
GRAPH_SHA = "3da972d8da3621fbcb54ccb2e90efdbb8a8b78307be4307c34110e76beadffe9"
INPUT_SHAPE = [1, 3, 768, 768]
OUTPUT_SHAPE = [1, 1, 768, 768]
ATOL = .001
RTOL = .001


def preprocess(rgb):
    import cv2
    import numpy as np
    # DepthPredictor.scale_fill=True: stretch, not detector letterboxing.
    resized = cv2.resize(rgb, (768, 768), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255


def validate_depth(value):
    import numpy as np
    if (value.dtype != np.float32 or list(value.shape) != OUTPUT_SHAPE
            or not np.isfinite(value).all() or not (value > 0).all()):
        raise ValueError("Depth must be fixed FP32, finite and positive at every pixel")


def parity(reference, actual):
    import numpy as np
    validate_depth(reference)
    validate_depth(actual)
    delta = np.abs(actual - reference)
    return dict(passed=bool(np.all(delta <= ATOL + RTOL * np.abs(reference))),
                max_abs_delta_m=float(delta.max()), mean_abs_delta_m=float(delta.mean()))


def restore_depth(value, height, width):
    import torch
    validate_depth(value)
    # Identical causal, per-image interpolation to the reviewed DepthPredictor.
    return torch.nn.functional.interpolate(torch.from_numpy(value), size=(height, width),
                                            mode="bilinear", align_corners=True)[0, 0].numpy()


def prepare(args):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
    if ort.__version__ != "1.30.0":
        raise RuntimeError("Use reviewed ONNX Runtime 1.30.0")
    if args.output.is_symlink():
        raise ValueError("Use private non-symlink output")
    output = args.output.resolve()
    with checked_archive(args.inputs, INPUT_SHA) as inputs, checked_archive(args.bundle, SPECS["yolo26n"]["bundle"]) as archive:
        config = common.load_config(common.CONFIG)
        source = common.verify_manifest(json.loads(archive.read("input_manifest.json")), config)
        if source["manifest_sha256"] != MANIFEST_SHA or set(inputs.namelist()) != set(source["files"]):
            raise RuntimeError("Reviewed input inventory changed")
        for name, expected in source["files"].items():
            if common.sha(inputs.read(name)) != expected:
                raise RuntimeError(f"Input bytes changed: {name}")
        screen = json.loads(archive.read("candidate_screen.json"))
        if not screen["complete"] or not screen["technical_export_screen_passed"] or screen["training_performed"]:
            raise RuntimeError("Original screen did not pass")
        graph = archive.read("models/yolo26n-depth.onnx")
        if common.sha(graph) != GRAPH_SHA:
            raise RuntimeError("Depth graph identity changed")
        model = onnx.load_model_from_string(graph)
        onnx.checker.check_model(model)
        if any(n.domain not in ("", "ai.onnx") for n in model.graph.node) or any(
            t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer
        ):
            raise RuntimeError("Use the self-contained, standard-operator graph")
        common.immutable(output / "yolo26n-depth.onnx", graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
        if (session.get_providers() != ["CPUExecutionProvider"] or
                [(v.name, v.shape, v.type) for v in session.get_inputs()] != [("images", INPUT_SHAPE, "tensor(float)")] or
                [(v.name, v.shape, v.type) for v in session.get_outputs()] != [("output0", OUTPUT_SHAPE, "tensor(float)")]):
            raise RuntimeError("Depth graph interface changed")
        fixtures = []
        with np.load(io.BytesIO(archive.read("fixtures/depth.npz")), allow_pickle=False) as saved:
            for index, sample in enumerate(source["ids"][:16]):
                rgb = np.asarray(Image.open(io.BytesIO(inputs.read(f"training/image_2/{sample}.png"))).convert("RGB"))
                tensor = preprocess(rgb)
                if index == 0 and not np.array_equal(tensor, saved["image"]):
                    raise RuntimeError("Depth preprocessing differs from the original export fixture")
                actual = session.run(["output0"], {"images": tensor})[0]
                validate_depth(actual)
                raw_check = parity(saved["output_0"], actual) if index == 0 else None
                if raw_check is not None and not raw_check["passed"]:
                    raise RuntimeError(f"Original raw depth fixture parity failed: {raw_check}")
                restored = restore_depth(actual, *rgb.shape[:2])
                record = json.loads(archive.read(f"records/{sample}.json"))
                if not record["depth_parity"]["passed"]:
                    raise RuntimeError("Original per-image depth preservation did not pass")
                max_roi_delta = 0.
                for row in record["predictions"]["onnx_cpu"]:
                    original = row["distance_m"]
                    recovered = roi_depth(restored, row["bbox"])["distance_m"]
                    if (original is None) != (recovered is None):
                        raise RuntimeError("Archived ROI validity changed")
                    if original is not None:
                        delta = abs(original - recovered)
                        if delta > ATOL + RTOL * abs(original):
                            raise RuntimeError(f"Archived nominal ROI output changed: {sample}")
                        max_roi_delta = max(max_roi_delta, delta)
                def write_tensor(name, value):
                    raw = np.ascontiguousarray(value, dtype="<f4").tobytes()
                    common.immutable(output / name, raw)
                    return dict(file=name, sha256=common.sha(raw), bytes=len(raw), shape=list(value.shape))
                fixtures.append(dict(sample_id=sample,
                    input=write_tensor(f"fixtures/{sample}/image.bin", tensor),
                    reference=write_tensor(f"fixtures/{sample}/depth.bin", actual),
                    original_raw_fixture_parity=raw_check, restored_roi_max_abs_delta_m=max_roi_delta))
                print(f"Depth phone fixture verified {index+1}/16: {sample}", flush=True)
        manifest = dict(revision=REVISION, complete=True, model_file="yolo26n-depth.onnx",
            model_sha256=GRAPH_SHA, model_bytes=len(graph), source_bundle_sha256=SPECS["yolo26n"]["bundle"],
            input_manifest_sha256=MANIFEST_SHA, input_archive_sha256=INPUT_SHA,
            input_name="images", input_shape=INPUT_SHAPE, output_name="output0", output_shape=OUTPUT_SHAPE,
            fixtures=fixtures, onnxruntime_version=ort.__version__, provider="CPUExecutionProvider", intra_op_threads=2,
            spinning="explicitly_disabled", parity_atol_m=ATOL, parity_rtol=RTOL,
            repetitions=2, warmups=5, timed_predictions=100, sustained_seconds=60,
            preprocessing="RGB FP32 BCHW /255; bilinear stretch to 768x768; no letterbox",
            output_quantity="released nominal per-pixel depth; no rescaling or GT alignment; accuracy unvalidated",
            measurement_scope="session.run only; no camera, preprocessing, original-image interpolation, ROI extraction, detector or rendering",
            training_performed=False, scale_fitting_performed=False, independent_distance_accuracy_evaluated=False,
            deployment_qualified=False, reference_environment=dict(onnxruntime=ort.__version__, numpy=np.__version__, opencv=cv2.__version__))
        raw = common.encoded(manifest)
        common.immutable(output / "manifest.json", raw)
        print(f"Depth package manifest SHA256: {common.sha(raw)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "bundle", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
