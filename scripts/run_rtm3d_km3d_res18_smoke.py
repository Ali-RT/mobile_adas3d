"""Run a fixed-16 inference-only smoke for the official RTM3D/KM3D ResNet-18 model.

This intentionally builds the plain ResNet-18 path directly.  Upstream's generic
model factory imports optional DLA/DCNv2 and training/evaluation CUDA modules;
those are not part of this candidate's ResNet-18 forward path.  The smoke does
not modify the upstream checkout, train, calculate AP, or claim Core ML/iPhone
support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


RTM3D_URL = "https://github.com/Banconxuan/RTM3D.git"
RTM3D_COMMIT = "888c379e79d8a6d134f06a9b7d669118679e06dc"
CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
HEADS = {
    "hm": 3,
    "wh": 2,
    "hps": 18,
    "rot": 8,
    "dim": 3,
    "prob": 1,
    "reg": 2,
    "hm_hp": 9,
    "hp_offset": 2,
}
INPUT_HEIGHT = 384
INPUT_WIDTH = 1280
TOP_K = 100
SAMPLES = 16
SCORE_THRESHOLD = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != 3769 or len(set(ids)) != 3769:
        raise RuntimeError(f"Expected the complete Chen val split (3769 unique IDs); got {len(ids)}")
    return ids[:SAMPLES]


def verify_source(repo: Path) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != RTM3D_COMMIT:
        raise RuntimeError(f"RTM3D source commit mismatch: expected {RTM3D_COMMIT}, got {head}")
    remote = subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"], text=True
    ).strip()
    if remote.rstrip("/").removesuffix(".git") != RTM3D_URL.removesuffix(".git"):
        raise RuntimeError(f"Unexpected RTM3D origin: {remote}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            "RTM3D checkout has modified or untracked files; preserve it and use a fresh checkout:\n"
            + dirty
        )
    return head


def find_sample_paths(dataset: Path, sample_id: str) -> tuple[Path, Path]:
    image_dirs = [dataset / "training" / "image_2", dataset / "training" / "image_02"]
    calib_dirs = [dataset / "training" / "calib"]
    image = next((folder / f"{sample_id}.png" for folder in image_dirs
                  if (folder / f"{sample_id}.png").is_file()), None)
    calib = next((folder / f"{sample_id}.txt" for folder in calib_dirs
                  if (folder / f"{sample_id}.txt").is_file()), None)
    if image is None or calib is None:
        raise FileNotFoundError(f"Missing KITTI image/calibration for sample {sample_id} under {dataset}")
    return image, calib


def load_calibration(path: Path, np: Any):
    projection = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") in {"P2", "P_rect_02"}:
            values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            if values.size == 12:
                projection = values.reshape(3, 4)
                break
    if projection is None:
        raise RuntimeError(f"Could not find a 3x4 P2/P_rect_02 calibration matrix in {path}")
    return projection


def safe_load_state(checkpoint_path: Path, torch: Any) -> tuple[dict[str, Any], int | None]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("This smoke requires a PyTorch version with weights_only=True support") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a safe tensor dictionary")
    raw_state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError("Checkpoint does not contain a non-empty state_dict/model mapping")
    state = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("Checkpoint state contains non-tensor entries; refusing unsafe/unexpected format")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in state:
            raise RuntimeError(f"Duplicate parameter after removing DataParallel prefix: {normalized}")
        state[normalized] = value
    epoch = checkpoint.get("epoch")
    return state, int(epoch) if isinstance(epoch, (int, float)) else None


def build_strict_resnet18(repo: Path, state: dict[str, Any], torch: Any):
    sys.path.insert(0, str(repo / "src" / "lib"))
    from models.networks.msra_resnet import PoseResNet, resnet_spec

    block, layers = resnet_spec[18]
    failures = []
    for head_conv in (64, 256, 128, 0):
        model = PoseResNet(block, layers, HEADS, head_conv=head_conv)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            failures.append(f"head_conv={head_conv}: {exc}")
            del model
            continue
        return model, head_conv
    raise RuntimeError("No strict ResNet-18/head configuration matched the checkpoint:\n" + "\n".join(failures))


def preprocess(image: Any, cv2: Any, np: Any, torch: Any, transform_fn: Any, device: Any):
    height, width = image.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    transform = transform_fn(center, scale, 0, [INPUT_WIDTH, INPUT_HEIGHT])
    resized = cv2.resize(image, (width, height))
    warped = cv2.warpAffine(resized, transform, (INPUT_WIDTH, INPUT_HEIGHT), flags=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = ((warped / 255.0 - mean) / std).astype(np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    inverse = transform_fn(
        center, scale, 0, [INPUT_WIDTH // 4, INPUT_HEIGHT // 4], inv=1
    )
    meta = {
        "trans_output_inv": torch.from_numpy(inverse).unsqueeze(0).to(device),
        "calib": None,
    }
    return tensor, meta, center, scale


def probability(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def rounded_finite(value: float, digits: int) -> float | None:
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset = args.dataset_root.resolve()
    split_path = args.split_dir.resolve() / "val.txt"
    if not repo.is_dir() or not checkpoint_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("RTM3D repo, checkpoint, or Chen val.txt is missing")

    source_commit = verify_source(repo)
    sample_ids = read_split(split_path)
    sample_paths = [find_sample_paths(dataset, sample_id) for sample_id in sample_ids]

    sys.path.insert(0, str(repo / "src" / "lib"))
    import cv2
    import numpy as np
    import torch
    from utils.image import get_affine_transform
    from models.decode import car_pose_decode
    from utils.post_process import car_pose_post_process

    if not torch.cuda.is_available():
        raise RuntimeError("RTM3D's upstream geometry decoder hard-codes CUDA; select a Colab GPU runtime")
    device = torch.device("cuda")
    state, checkpoint_epoch = safe_load_state(checkpoint_path, torch)
    model, head_conv = build_strict_resnet18(repo, state, torch)
    model = model.to(device).eval()

    const_values = [
        [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
        [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
    ]
    const = torch.tensor(const_values, dtype=torch.float32, device=device).view(1, 1, 16, 2)
    output_shapes: dict[str, list[int]] = {}
    per_sample = []
    forward_ms = []
    decode_ms = []
    all_outputs_finite = True
    selected_geometry_finite = True
    total_selected = 0
    valid_2d = 0
    valid_3d = 0
    class_counts = {name: 0 for name in CLASS_NAMES}

    for index, (sample_id, (image_path, calib_path)) in enumerate(zip(sample_ids, sample_paths)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        tensor, meta, center, scale = preprocess(image, cv2, np, torch, get_affine_transform, device)
        projection = load_calibration(calib_path, np)
        meta["calib"] = torch.from_numpy(projection).unsqueeze(0).to(device)

        if index == 0:
            for _ in range(3):
                with torch.inference_mode():
                    model(tensor)
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensor)[-1]
        torch.cuda.synchronize()
        forward_ms.append((time.perf_counter() - start) * 1000.0)
        if set(outputs) != set(HEADS):
            raise RuntimeError(f"Unexpected model head names: {sorted(outputs)}")
        output_shapes = {key: list(value.shape) for key, value in outputs.items()}
        all_outputs_finite &= all(bool(torch.isfinite(value).all().item()) for value in outputs.values())

        start = time.perf_counter()
        decoded = car_pose_decode(
            outputs["hm"].sigmoid(), outputs["wh"], outputs["hps"], outputs["dim"], outputs["rot"],
            prob=outputs["prob"], reg=outputs["reg"], hm_hp=outputs["hm_hp"].sigmoid(),
            hp_offset=outputs["hp_offset"], K=TOP_K, meta=meta, const=const,
        )
        torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - start) * 1000.0)
        raw = decoded.detach().float().cpu().numpy().reshape(1, TOP_K, 41)
        rows = car_pose_post_process(
            raw.copy(), [center], [scale], INPUT_HEIGHT // 4, INPUT_WIDTH // 4
        )[0][1]
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 41)

        selected_rows = []
        sample_selected = 0
        for row in rows:
            combined_score = (float(row[4]) + probability(float(row[39])) + float(np.mean(row[23:32]))) / 3.0
            if not math.isfinite(combined_score):
                selected_geometry_finite = False
                continue
            if combined_score < SCORE_THRESHOLD:
                continue
            sample_selected += 1
            total_selected += 1
            finite_box = bool(np.isfinite(row[:4]).all()) and row[2] > row[0] and row[3] > row[1]
            finite_geometry = (
                bool(np.isfinite(row[32:39]).all())
                and bool((row[32:35] > 0).all())
                and float(row[38]) > 0
            )
            valid_2d += int(finite_box)
            valid_3d += int(finite_geometry)
            selected_geometry_finite &= finite_box and finite_geometry
            class_index = int(row[40]) if math.isfinite(float(row[40])) else -1
            class_name = CLASS_NAMES[class_index] if 0 <= class_index < len(CLASS_NAMES) else "unknown"
            if class_name != "unknown":
                class_counts[class_name] += 1
            if len(selected_rows) < 10:
                selected_rows.append({
                    "class_name": class_name,
                    "score": rounded_finite(combined_score, 6),
                    "bbox_xyxy_px": [rounded_finite(value, 3) for value in row[:4]],
                    "dimensions_hwl_m": [rounded_finite(value, 4) for value in row[32:35]],
                    "yaw_camera_rad": rounded_finite(row[35], 6),
                    "location_xyz_m": [rounded_finite(value, 4) for value in row[36:39]],
                })
        per_sample.append({"sample_id": sample_id, "detections_score_ge_0_1": sample_selected,
                           "top_detections": selected_rows})

    def percentile(values: list[float], fraction: float) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), fraction)) if values else 0.0

    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D official ResNet-18 fixed16 inference screen",
        "stage": "colab_inference_smoke",
        "complete": len(per_sample) == SAMPLES,
        "smoke_passed": bool(all_outputs_finite and selected_geometry_finite and total_selected > 0),
        "source_url": RTM3D_URL,
        "source_commit": source_commit,
        "source_clean_tracked_files": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "safe_weights_only_load": True,
        "strict_state_dict_load": True,
        "architecture": {"backbone": "ResNet-18", "head_conv": head_conv, "input_nchw": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
                         "heads": HEADS, "classes": list(CLASS_NAMES)},
        "dataset": {"split_protocol": "chen_3712_3769", "split": "val", "split_sha256": sha256_file(split_path),
                    "sample_ids": sample_ids, "sample_count": len(sample_ids)},
        "output_shapes": output_shapes,
        "raw_outputs_finite": all_outputs_finite,
        "decoded_2d_and_3d_selected_outputs_finite_and_valid": selected_geometry_finite,
        "selected_detections_score_ge_0_1": total_selected,
        "valid_2d_box_count": valid_2d,
        "valid_3d_box_count": valid_3d,
        "selected_class_counts": class_counts,
        "runtime": {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_forward_p50_ms": percentile(forward_ms, 50), "gpu_forward_p95_ms": percentile(forward_ms, 95),
                    "gpu_decode_p50_ms": percentile(decode_ms, 50), "gpu_decode_p95_ms": percentile(decode_ms, 95)},
        "per_sample": per_sample,
        "training_performed": False,
        "optimizer_steps": 0,
        "full_val_ap_evaluated": False,
        "coreml_conversion_performed": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step_if_passed": "export neural network heads to Core ML; implement/validate calibration-based geometry decode separately",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["smoke_passed"]:
        raise RuntimeError(f"RTM3D/KM3D ResNet-18 fixed16 inference smoke failed; see {args.output}")


if __name__ == "__main__":
    main()
