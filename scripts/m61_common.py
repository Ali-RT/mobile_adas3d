"""Frozen, train-only MonoDGP -> A2 pilot utilities (no upstream imports here)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "m61-20260922-v1"
STUDENT_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
TEACHER_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
STUDENT_SHA = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"
TEACHER_SHA = "8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847"
SPLITS = {
    "train": (3712, "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c"),
    "val": (3769, "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"),
}
CLASS_MAPPING = dict(Car="Car", Van="Car", Truck="Car", Tram="Car",
                     Pedestrian="Pedestrian", Person_sitting="Pedestrian")
# Native index order is NOT the product display order!
NATIVE_CLASSES = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}
TARGET_KEYS = ("labels", "boxes", "calibs", "depth", "size_3d",
               "heading_bin", "heading_res", "boxes_3d")
OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle")
COMPONENTS = ("depth", "dimensions", "center", "angle")
AP_GATES = dict(vehicle_3d_moderate=15.8713, pedestrian_3d_moderate=5.1493,
                mean_3d_moderate=10.5103, vehicle_bev_moderate=21.3134,
                pedestrian_bev_moderate=5.9365)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def array_hash(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        a = np.ascontiguousarray(value)
        h.update(json.dumps([key, a.dtype.str, a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def npz_write(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def npz_read(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def check_split(path, split):
    ids = Path(path).read_text().splitlines()
    count, expected = SPLITS[split]
    if (len(ids) != count or len(set(ids)) != count or sha256(path) != expected
            or any(len(i) != 6 or not i.isdigit() for i in ids)):
        raise RuntimeError(f"Not the frozen Chen {split} split: {path}")
    return ids


def source_hash(repo):
    """Bind patched Python/CUDA source; configs are bound separately in the manifest."""
    repo = Path(repo)
    files = sorted(p for folder in ("lib", "utils", "tools")
                   for p in (repo / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".cuh")
                   and "build" not in p.relative_to(repo).parts)
    return json_hash({str(p.relative_to(repo)): sha256(p) for p in files})


def code_hash():
    files = sorted((ROOT / "scripts").glob("*m61*.py"))
    files += [ROOT / "third_party/monodetr/m61_distillation_loss.py"]
    return json_hash({str(p.relative_to(ROOT)): sha256(p) for p in files})


def environment():
    import torch
    import timm
    import torchvision
    import PIL
    import cv2
    import scipy
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "numpy": np.__version__, "timm": timm.__version__,
            "torchvision": str(torchvision.__version__), "pillow": PIL.__version__,
            "opencv": cv2.__version__, "scipy": scipy.__version__,
            "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_manifest(path, verify_sources=True):
    m = read_json(path)
    signature = m.pop("manifest_sha256")
    if json_hash(m) != signature or m["revision"] != REVISION or m["code_sha256"] != code_hash():
        raise RuntimeError("M61 manifest/code changed; use the frozen code or a new output directory")
    m["manifest_sha256"] = signature
    for role, expected in (("student", STUDENT_SHA), ("teacher", TEACHER_SHA)):
        if m[role]["checkpoint_sha256"] != expected or sha256(m[role]["checkpoint"]) != expected:
            raise RuntimeError(f"Changed {role} checkpoint")
        if verify_sources and source_hash(m[role]["repo"]) != m[role]["source_sha256"]:
            raise RuntimeError(f"Changed patched {role} source; rerun setup with the frozen code")
    for split in SPLITS:
        check_split(Path(m["dataset_root"]) / "ImageSets" / f"{split}.txt", split)
    return m


def activate_repo(m, role):
    repo = Path(m[role]["repo"])
    if "lib" in sys.modules:
        raise RuntimeError("Teacher/student must run in separate processes (upstream lib namespace)")
    model = "monodgp" if role == "teacher" else "monodetr"
    sys.path[:0] = [str(repo), str(repo / f"lib/models/{model}/ops")]
    return repo


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_runtime(m, role):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("M61 inference/training smoke requires a CUDA GPU")
    if environment()["timm"] != "1.0.20":
        raise RuntimeError("M61 requires the A2 backbone library timm==1.0.20")
    activate_repo(m, role)
    from lib.helpers.model_helper import build_model
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    cfg = m[role]["config"]
    dataset = KITTI_Dataset("train", cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != NATIVE_CLASSES or np.any(dataset.cls_mean_size):
        raise RuntimeError("Changed native class order or dimension encoding")
    model, criterion = build_model(cfg["model"])
    # Only load the exact, user-provided checkpoint after its frozen SHA check.
    payload = torch.load(m[role]["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    model = model.cuda()
    criterion = criterion.cuda()
    return model, criterion, dataset


def pack_targets(raw, device):
    return [{k: raw[k][i][raw["mask_2d"][i]].to(device) for k in TARGET_KEYS}
            for i in range(len(raw["labels"]))]


def input_fingerprint(image, calibration, image_size):
    return array_hash(dict(image=image, calibration=calibration, image_size=image_size))


def validate_cache(directory, m, role):
    directory = Path(directory)
    report = read_json(directory / "cache_manifest.json")
    if (not report["complete"] or report["manifest_sha256"] != m["manifest_sha256"]
            or report["role"] != role or report["environment"] != environment()):
        raise RuntimeError(f"Changed or incomplete {role} cache/environment")
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    if set(report["files"]) != set(ids) or {p.stem for p in directory.glob("*.npz")} != set(ids):
        raise RuntimeError("Cache does not contain exactly the training IDs")
    for image_id in ids:
        if sha256(directory / f"{image_id}.npz") != report["files"][image_id]:
            raise RuntimeError(f"Changed {role} cache sample {image_id}")
    return report
