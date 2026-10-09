"""Export only verified camera calibration and original image sizes (no images).

Standalone/standard-library only: upload this file to Colab and point it at the
existing RTM3D ONNX bundle and KITTI dataset. Existing outputs are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile


MODEL_SHA = "cd0647611e2cd498136d7153b1032a08f991280d1306e88b4f5e100ea1e31017"
BUNDLE_SHA = "c8bd09938cdd3d0ec4b501dde2595c9fd82a3fdc8d45b4f3d64e49d71ce42ad8"
CHECKPOINT_SHA = "5fa355845f79c1afeffab427de32933758e5b4c1e7c9ec19a94a13737691d05b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Not a PNG with an IHDR header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if not 0 < width <= 10000 or not 0 < height <= 10000:
        raise RuntimeError(f"Invalid original image size: {path}")
    return width, height


def prepare(bundle: Path, dataset: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"Preserve existing metadata and choose a new output: {output}")
    if sha256(bundle) != BUNDLE_SHA:
        raise RuntimeError("ONNX bundle differs from the reviewed phone model")
    with zipfile.ZipFile(bundle) as archive:
        export = json.loads(archive.read("rtm3d_onnx_export.json"))
    if (export.get("onnx_sha256") != MODEL_SHA
            or export.get("checkpoint_sha256") != CHECKPOINT_SHA
            or not export.get("export_complete")):
        raise RuntimeError("Unexpected model identity in ONNX export")
    ids = export["input"]["fixture_sample_ids"]
    originals = export["input"]["fixture_samples"]
    if len(ids) != 16 or len(set(ids)) != 16 or [x["sample_id"] for x in originals] != ids:
        raise RuntimeError("Expected sixteen frozen fixture records in the exact order")
    records = []
    calibrations = {}
    for item in originals:
        sample_id = item["sample_id"]
        image = next((dataset/"training"/folder/f"{sample_id}.png"
                      for folder in ("image_2", "image_02")
                      if (dataset/"training"/folder/f"{sample_id}.png").is_file()), None)
        calibration = dataset/"training/calib"/f"{sample_id}.txt"
        if image is None or not calibration.is_file():
            raise FileNotFoundError(f"Missing original image/calibration: {sample_id}")
        if sha256(image) != item["image_sha256"] or sha256(calibration) != item["calibration_sha256"]:
            raise RuntimeError(f"KITTI metadata source differs from export fixture: {sample_id}")
        width, height = png_size(image)
        name = f"calib/{sample_id}.txt"
        calibrations[name] = calibration.read_bytes()
        records.append(dict(sample_id=sample_id, width=width, height=height,
                            image_sha256=item["image_sha256"],
                            calibration_sha256=item["calibration_sha256"], calibration_file=name))
    manifest = dict(revision="RTM3D-PHONE-DECODE-METADATA-2026-10-09-r1", complete=True,
                    onnx_sha256=MODEL_SHA, onnx_bundle_sha256=BUNDLE_SHA,
                    checkpoint_sha256=CHECKPOINT_SHA, sample_ids=ids, records=records,
                    original_images_included=False, calibration_and_sizes_only=True,
                    training_performed=False, deployment_qualified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(manifest, indent=2)+"\n")
        for name, data in calibrations.items():
            archive.writestr(name, data)
    print(f"Verified metadata ready: {output}; {len(records)} frames; {output.stat().st_size} bytes")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-bundle", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.onnx_bundle, a.dataset_root, a.output)


if __name__ == "__main__":
    main()
