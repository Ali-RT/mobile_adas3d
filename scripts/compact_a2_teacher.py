"""Pinned, previously full-val-audited A2 ONNX teacher, CPU only.

No MonoDETR source import, namespace patch, native attention extension,
Core ML, or historical CUDA runtime is required. Preprocessing and decoding
follow ZrrSkywalker/MonoDETR commit 6994b9f512400b258c6edb75f77423beb9c126f2.
The original M68 fixed16 input/output fixtures are verified before caching.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from scripts.train_compact_kitti import file_hash
from scripts.m68_onnx_common import (INPUT_NAMES, OUTPUT_NAMES, OUTPUT_SHAPES,
                                    compare_outputs, read_f32, signature)

A2_SHA = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"
ONNX_SHA = "82344b2bc7a96ff3bb24cfc6d9b3aeea61c138b1e3bcbae07b58e2bed562c0c1"


def preprocessing_comparison(inputs, saved):
    """Allow only sparse single-8-bit-level PIL affine rounding differences.

    Linux/macOS OpenCV affine coefficients can differ near rounding ties.
    Model/runtime parity is checked separately on the exact signed inputs;
    this is not a relaxation of the M68 raw-output tolerances.
    """
    result = {}
    for name in INPUT_NAMES:
        actual, reference = inputs[name], saved[name]
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise RuntimeError(f"Teacher preprocessing has invalid {name}")
        delta = np.abs(actual-reference)
        row = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
        if name == "image":
            byte_delta = delta * np.asarray([.229, .224, .225], np.float32)[None, :, None, None] * 255
            row.update(changed_fraction=float((delta > 1e-6).mean()),
                       max_rgb_level_delta=float(byte_delta.max()))
            passed = (byte_delta.max() <= 1.0001 and delta.mean() <= 1e-6
                      and row["changed_fraction"] <= 1e-4)
        else:
            passed = delta.max() <= 1e-6
        row["passed"] = bool(passed); result[name] = row
    if not all(r["passed"] for r in result.values()):
        raise RuntimeError(f"Teacher preprocessing changed beyond sparse RGB rounding: {result}")
    return result


def decoded_fixture_comparison(actual, reference, projection, image_size):
    """Check guidance targets after reconstructing images, not inactive queries."""
    old = {(r["query_index"], r["class_name"]): r for r in decode_teacher(reference, projection, image_size)}
    new = {(r["query_index"], r["class_name"]): r for r in decode_teacher(actual, projection, image_size)}
    if old.keys() != new.keys(): raise RuntimeError("Teacher fixture selected detections changed")
    limits = dict(bbox_2d=.05, dimensions_3d=.01, location_3d=.01,
                  alpha=.001, rotation_y=.001, class_confidence=.001, score_relative=.01)
    maximum = {k: 0. for k in limits}
    for key, a in old.items():
        b = new[key]
        if a["invalid_geometry"] != b["invalid_geometry"]:
            raise RuntimeError("Teacher fixture geometry validity changed")
        for name in maximum:
            if name == "score_relative":
                difference = abs(a["score"]-b["score"])/max(1., abs(a["score"]))
            elif name in ("alpha", "rotation_y"):
                d = a[name]-b[name]; difference = abs(math.atan2(math.sin(d), math.cos(d)))
            else:
                difference = float(np.abs(np.asarray(a[name])-np.asarray(b[name])).max())
            maximum[name] = max(maximum[name], difference)
    passed = all(maximum[k] <= limits[k] for k in maximum)
    result = dict(detections=len(old), max_deltas=maximum, limits=limits, passed=passed)
    if not passed: raise RuntimeError(f"Reconstructed teacher guidance changed: {result}")
    return result


def teacher_inputs(image_path, projection):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        # Exact zero-rotation upstream get_affine_transform, including float32
        # control points and PIL's resampling (NOT the compact cv2 warp).
        src = np.asarray([[width/2, height/2], [width/2, height/2-width/2],
                          [0, height/2-width/2]], np.float32)
        dst = np.asarray([[640, 192], [640, -448], [0, -448]], np.float32)
        inverse = cv2.getAffineTransform(dst, src)
        image = image.transform((1280, 384), Image.Transform.AFFINE,
                                tuple(inverse.reshape(-1)), Image.Resampling.BILINEAR)
        pixels = np.asarray(image).astype(np.float32)/255
    pixels = (pixels-np.asarray([.485, .456, .406], np.float32))/np.asarray([.229, .224, .225], np.float32)
    return dict(image=np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]),
                calibration=np.asarray(projection, np.float32)[None],
                image_size=np.asarray([[width, height]], np.float32))


def decode_teacher(values, projection, image_size, threshold=.1):
    """Upstream M68 decode: top50 class/query pairs; meanshape=False.

    Keep native alpha and native yaw conventions explicit. Derived targets
    are projected from the decoded metric box, not aligned query positions.
    """
    for name in OUTPUT_NAMES:
        a = np.asarray(values[name])
        if a.shape != OUTPUT_SHAPES[name] or not np.isfinite(a).all():
            raise ValueError(f"Invalid teacher {name} tensor")
    logits = values["logits"][0]
    probs = 1/(1+np.exp(-np.clip(logits, -80, 80)))
    order = np.argsort(-probs.reshape(-1), kind="stable")[:50]
    width, height = image_size
    p = projection
    rows = []
    for flat in order:
        query, cls = divmod(int(flat), 3)
        base_score = float(probs[query, cls])
        if cls == 2 or base_score < threshold:
            continue
        xc, yc, left, right, top, bottom = values["boxes"][0, query]
        box = np.asarray([(xc-left)*width, (yc-top)*height,
                          (xc+right)*width, (yc+bottom)*height])
        dim = values["dimensions"][0, query].copy()
        depth, uncertainty = values["depth"][0, query]
        location = np.asarray([((xc*width-p[0, 2])*depth-p[0, 3])/p[0, 0],
                               ((yc*height-p[1, 2])*depth-p[1, 3])/p[1, 1]+dim[0]/2, depth])
        angle = values["angle"][0, query]
        bin_id = int(angle[:12].argmax())
        alpha = bin_id * (2*math.pi/12) + float(angle[12+bin_id])
        if alpha > math.pi: alpha -= 2*math.pi
        yaw = alpha + math.atan2(float((box[0]+box[2])/2-p[0, 2]), float(p[0, 0]))
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        score = base_score * float(np.exp(np.clip(-uncertainty, -80, 80)))
        invalid = bool(not np.isfinite([*box, *dim, *location, yaw, score]).all()
                       or min(dim) <= 0 or depth <= .1 or (box[2:] <= box[:2]).any())
        rows.append(dict(class_name="Pedestrian" if cls == 0 else "Vehicle", score=score,
                         query_index=query, class_confidence=base_score,
                         bbox_2d=box.tolist(), dimensions_3d=dim.tolist(), location_3d=location.tolist(),
                         rotation_y=yaw, alpha=float(alpha), invalid_geometry=invalid))
    return rows


def materialize_export(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_dir(): return source
    # Extract only the validated teacher export; prevent absolute/traversal paths.
    with zipfile.ZipFile(source) as z:
        manifests = [n for n in z.namelist() if n.endswith("A2_ONNX/manifest.json")]
        if len(manifests) != 1: raise ValueError("Expected one M68 A2_ONNX package")
        prefix = manifests[0][:-len("A2_ONNX/manifest.json")]
        for name in z.namelist():
            if not name.startswith(prefix) or name.endswith("/"): continue
            relative = Path(name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts: raise ValueError("Unsafe ZIP path")
            if relative.parts[0] != "A2_ONNX" and str(relative) != "m68_onnx_fullval.json": continue
            target = destination / relative
            content = z.read(name)
            if target.exists():
                if target.read_bytes() != content: raise RuntimeError(f"Preserve changed teacher file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    return destination


class A2Teacher:
    def __init__(self, export):
        import onnxruntime as ort
        export = Path(export)
        package = json.loads((export/"A2_ONNX/manifest.json").read_text())
        saved = package.pop("manifest_sha256")
        if signature(package) != saved: raise RuntimeError("Teacher package signature differs")
        self.package = package
        model = export/"A2_ONNX"/package["model"]["file"]
        if (not model.resolve().is_relative_to((export/"A2_ONNX").resolve())
                or file_hash(model) != ONNX_SHA or package["checkpoint_sha256"] != A2_SHA
                or package["checkpoint_epoch"] != 130):
            raise RuntimeError("Teacher is not the reviewed A2 epoch130 ONNX model")
        fullval = json.loads((export/"m68_onnx_fullval.json").read_text())
        if (fullval.get("complete") is not True or fullval.get("quality_gate_passed") is not True
                or fullval.get("evaluated_images") != 3769 or fullval.get("onnx_sha256") != ONNX_SHA):
            raise RuntimeError("Missing passed M68 decoded full-validation audit")
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]: raise RuntimeError("Teacher must be CPU-only")
        if [v.name for v in self.session.get_inputs()] != list(INPUT_NAMES): raise RuntimeError("Teacher input contract differs")
        self.root = export/"A2_ONNX"

    def raw(self, inputs):
        return dict(zip(OUTPUT_NAMES, self.session.run(list(OUTPUT_NAMES), inputs)))

    def verify_fixtures(self, dataset):
        from data.kitti_parser import parse_kitti_calib_file
        rows = []
        for fixture in self.package["fixtures"]:
            sample = fixture["sample_id"]
            p = parse_kitti_calib_file(Path(dataset)/"training/calib"/f"{sample}.txt")["P2"]
            inputs = teacher_inputs(Path(dataset)/"training/image_2"/f"{sample}.png", p)
            saved_inputs = {n: read_f32(self.root, fixture["inputs"][n]) for n in INPUT_NAMES}
            preprocessing = preprocessing_comparison(inputs, saved_inputs)
            reference = {n: read_f32(self.root, fixture["cpu_reference_outputs"][n]) for n in OUTPUT_NAMES}
            parity = compare_outputs(self.raw(saved_inputs), reference)
            if not parity["passed"]: raise RuntimeError(f"Teacher CPU fixture changed: {sample}")
            decoded = decoded_fixture_comparison(self.raw(inputs), reference, p, inputs["image_size"][0])
            rows.append(dict(sample_id=sample, preprocessing=preprocessing,
                             exact_input_output_comparison=parity, decoded_guidance_comparison=decoded))
        return rows
