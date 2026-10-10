"""Train the phone-screened compact CNN from KITTI labels, then report AP_R40.

Stages: preflight, smoke (two gradient steps), tiny (16 train-only frames),
train (Chen 3712), evaluate (all Chen 3769), bundle (small result archive).
No external teacher, native extension build, or earlier M62/M67 runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING, map_objects
from data.compact_kitti import CLASSES, INPUT_HW, PREPROCESSING, CompactKittiDataset, read_ids, valid_geometry
from data.kitti_parser import parse_kitti_label_file
from data.kitti_r40 import PRODUCT_IOU_THRESHOLDS, PRODUCT_NEIGHBOR_CLASSES, evaluate_kitti_r40
from models.compact_monocular3d import CompactMonocular3D, MOBILENETV4_ARCHITECTURE_REVISION
from models.compact_kitti_loss import CompactKittiLoss, LOSS_WEIGHTS, decode

REVISION = "COMPACT-MNV4-KITTI-GT-2026-10-09-r1"
GATES = {"vehicle_3d_moderate": 15.8713, "pedestrian_3d_moderate": 5.1493,
         "mean_3d_moderate": 10.5103, "vehicle_bev_moderate": 21.3134,
         "pedestrian_bev_moderate": 5.9365}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def environment():
    import cv2
    import timm
    import torchvision
    # torch.__version__ is a TorchVersion subclass. Store plain strings so
    # training history remains loadable with weights_only=True.
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "timm": str(timm.__version__),
            "numpy": str(np.__version__), "opencv": str(cv2.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get("revision") != REVISION or cfg.get("backbone") != "mobilenetv4_conv_small"
            or cfg.get("timm_version") != "1.0.27" or cfg.get("input_hw") != list(INPUT_HW)
            or cfg.get("classes") != list(CLASSES) or cfg.get("preprocessing") != PREPROCESSING
            or cfg.get("external_teacher") is not None or cfg.get("amp") is not False):
        raise ValueError("This baseline requires its reviewed architecture, RGB contract, FP32 and no teacher")
    if any(cfg[name] <= 0 for name in ("epochs", "batch_size", "learning_rate", "max_objects",
                                      "evaluate_every", "save_every", "warmup_epochs", "geometry_warmup_epochs")):
        raise ValueError("Invalid training schedule")
    if cfg["workers"] < 0 or cfg["evaluate_every"] % cfg["save_every"]:
        raise ValueError("workers must be nonnegative; evaluated epochs must have numbered checkpoints")
    return cfg


def identity(args, cfg):
    train, val = read_ids(args.split_dir / "train.txt"), read_ids(args.split_dir / "val.txt")
    if len(train) != 3712 or len(val) != 3769 or set(train) & set(val):
        raise ValueError("Expected disjoint Chen 3712/3769 splits; no validation data may train")
    data_hash = hashlib.sha256()
    counts = {split: {"source": Counter(), "mapped": Counter(), "valid_3d": Counter()} for split in ("train", "val")}
    for split, ids in (("train", train), ("val", val)):
        for sample_id in ids:
            for directory, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
                path = args.dataset_root / "training" / directory / (sample_id + suffix)
                if not path.is_file() or path.stat().st_size == 0 and directory != "label_2":
                    raise FileNotFoundError(path)
                if directory != "image_2":
                    data_hash.update(f"{split}/{directory}/{sample_id}\0".encode())
                    data_hash.update(path.read_bytes())
            objects = parse_kitti_label_file(args.dataset_root / "training/label_2" / (sample_id + ".txt"))
            mapped_count = 0
            for obj in objects:
                counts[split]["source"][obj.class_name] += 1
                mapped = KITTI_PRODUCTION_CLASS_MAPPING.get(obj.class_name)
                if mapped:
                    mapped_count += 1
                    counts[split]["mapped"][mapped] += 1
                    if valid_geometry(obj):
                        counts[split]["valid_3d"][mapped] += 1
            if mapped_count > cfg["max_objects"]:
                raise ValueError(f"{sample_id}: increase max_objects instead of truncating labels")
    code_files = ["models/compact_monocular3d.py", "models/compact_kitti_loss.py",
                  "data/compact_kitti.py", "data/kitti_parser.py", "data/class_taxonomy.py",
                  "data/kitti_r40.py", "scripts/train_compact_kitti.py"]
    code = {name: file_hash(ROOT / name) for name in code_files}
    immutable = {"config": cfg, "architecture": MOBILENETV4_ARCHITECTURE_REVISION,
                 "code_sha256": code, "label_calibration_sha256": data_hash.hexdigest(),
                 "train_split_sha256": file_hash(args.split_dir / "train.txt"),
                 "val_split_sha256": file_hash(args.split_dir / "val.txt"),
                 "taxonomy": KITTI_PRODUCTION_CLASS_MAPPING, "loss_weights": LOSS_WEIGHTS}
    value = {"schema_version": 1, "revision": REVISION, "complete": True,
             "identity_sha256": digest_bytes(json.dumps(immutable, sort_keys=True).encode()),
             **immutable, "counts": counts, "environment": environment(),
             "dataset_root": str(args.dataset_root.resolve()), "split_dir": str(args.split_dir.resolve()),
             "external_teacher_used": False, "image_pixels_hashed": False,
             "pretrained_backbone": bool(cfg["pretrained_backbone"]),
             "initialization": "ImageNet backbone + new neck/heads" if cfg["pretrained_backbone"] else "random",
             "accuracy_qualified": False, "full_phone_pipeline_qualified": False}
    path = args.output_dir / "training_manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity_sha256"] != value["identity_sha256"]:
            raise RuntimeError("This run's code/config/data differs. Preserve it and choose a new RUN_ID; no files overwritten")
    else:
        atomic_json(path, value)
    return value, train, val


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True) if isinstance(batch, torch.Tensor) else batch


def loader(args, cfg, ids, augment, epoch=0):
    dataset = CompactKittiDataset(args.dataset_root, ids, training=augment, max_objects=cfg["max_objects"])
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=augment,
                      num_workers=cfg["workers"], pin_memory=args.device == "cuda",
                      drop_last=False, worker_init_fn=worker_seed, generator=generator)


def finite_update(model, criterion, optimizer, batch, clip, geometry_scale):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["image"])
    if not all(torch.isfinite(v).all() for v in output.values()):
        raise RuntimeError("Non-finite raw heads; last checkpoint preserved")
    total, terms = criterion(output, batch, geometry_scale=geometry_scale)
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite loss; last checkpoint preserved")
    total.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Missing/non-finite gradients; no optimizer step taken")
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return {"total": float(total.detach()), "gradient_norm": float(gradient),
            **{k: float(v.detach()) for k, v in terms.items()}}


def rng_state():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, np.uint32), pos, gaussian, cached))


@torch.no_grad()
def evaluate(model, args, cfg, ids, directory):
    model.eval()
    predictions, ground_truth = {}, {}
    invalid = 0
    for batch in loader(args, cfg, ids, False):
        batch = to_device(batch, args.device)
        decoded = decode(model(batch["image"]), batch["projection"], batch["inverse_affine"],
                         topk=cfg["topk"], threshold=cfg["score_threshold"])
        for sample, rows in zip(batch["sample_id"], decoded):
            predictions[sample] = rows
            invalid += sum(row["invalid_geometry"] for row in rows)
            objects = [asdict(obj) for obj in parse_kitti_label_file(args.dataset_root / "training/label_2" / f"{sample}.txt")]
            # Match A2's existing product-taxonomy evaluator, not a different
            # official leaderboard protocol or undocumented ignore policy.
            ground_truth[sample] = map_objects(objects, KITTI_PRODUCTION_CLASS_MAPPING)
        if len(predictions) % 100 < cfg["batch_size"]:
            print(f"val: {len(predictions)}/{len(ids)}", flush=True)
    atomic_json(directory / "predictions.json", predictions)
    scoring = {sample: [dict(row, dimensions_3d=[0., 0., 0.]) if row["invalid_geometry"] else row for row in rows]
               for sample, rows in predictions.items()}
    results = evaluate_kitti_r40(ground_truth, scoring, list(CLASSES),
                                iou_thresholds=PRODUCT_IOU_THRESHOLDS, neighbor_classes=PRODUCT_NEIGHBOR_CLASSES)
    metrics = {f"{r.class_name.lower()}_{r.metric}_moderate": r.ap_r40
               for r in results if r.difficulty == "moderate"}
    metrics["mean_3d_moderate"] = (metrics["vehicle_3d_moderate"] + metrics["pedestrian_3d_moderate"]) / 2
    report = {"schema_version": 1, "complete": len(predictions) == len(ids), "samples": len(ids),
              "protocol": "Chen product-taxonomy KITTI-difficulty AP_R40; not official leaderboard",
              "metrics": metrics, "all_metrics": [r.to_dict() for r in results],
              "comparison_gates": GATES, "gate_results": {k: metrics[k] >= v for k, v in GATES.items()},
              "all_accuracy_gates_passed": all(metrics[k] >= v for k, v in GATES.items()),
              "invalid_geometry_predictions": invalid,
              "invalid_geometry_policy": "preserved in raw report; zero-volume boxes for 3D/BEV AP",
              "nearby_recall_evaluated": False, "iphone_performance_measured": False,
              "deployment_qualified": False}
    atomic_json(directory / "accuracy.json", report)
    model.train()
    return report


def run(args):
    cfg = config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "bundle":
        target = args.output_dir / "compact_mnv4_gt_results.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output_dir.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".yaml"):
                    archive.write(path, path.relative_to(args.output_dir))
        print(target)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before proceeding; do not replace PyTorch or install drivers")
    # Limit CPU test oversubscription; CUDA run still uses CPU quality targets.
    torch.set_num_threads(2)
    setup_seed(cfg["seed"])
    manifest, train, val = identity(args, cfg)
    atomic_json(args.output_dir / "current_environment.json", environment())
    if args.stage == "preflight":
        batch = next(iter(loader(args, cfg, train[:16], False)))
        atomic_json(args.output_dir / "preflight.json", {"complete": True, "counts": manifest["counts"],
                    "image_shape": list(batch["image"].shape), "target_heads": list(batch["targets"]),
                    "validation_used_for_training": False, "external_teacher_used": False})
        print(json.dumps(manifest["counts"], indent=2))
        return
    checkpoint_path = args.checkpoint or args.output_dir / "checkpoints/latest.pt"
    resume = args.stage in ("train", "evaluate") and checkpoint_path.exists()
    if args.stage == "evaluate" and not resume:
        raise FileNotFoundError(checkpoint_path)
    if args.stage == "train" and resume and not args.resume:
        raise RuntimeError("Checkpoint already exists. Use --resume or a new RUN_ID; never overwrite an existing run")
    initial_path = args.output_dir / "checkpoints/initial.pt"
    use_initial = args.stage == "train" and initial_path.exists() and not resume
    model = CompactMonocular3D(backbone=cfg["backbone"],
                               pretrained_backbone=cfg["pretrained_backbone"] and not resume and not use_initial).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = CompactKittiLoss().to(args.device)
    first_epoch, history = 0, []
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["identity_sha256"] != manifest["identity_sha256"]:
            raise RuntimeError("Checkpoint belongs to a different code/config/data identity")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch, history = saved["epoch"], saved["history"]
        if args.stage == "train":
            restore_rng(saved["rng"])
        print(f"Loaded epoch {first_epoch} from {checkpoint_path}", flush=True)
    elif use_initial:
        saved = torch.load(initial_path, map_location="cpu", weights_only=True)
        if saved["identity_sha256"] != manifest["identity_sha256"]:
            raise RuntimeError("Initial checkpoint identity differs; preserve it and choose a new RUN_ID")
        model.load_state_dict(saved["model"], strict=True)
        print("Reusing the full run's original initialization (not smoke/tiny weights)", flush=True)
    if args.stage == "evaluate":
        report = evaluate(model, args, cfg, val, args.output_dir / f"evaluations/epoch_{first_epoch:03d}")
        print(json.dumps(report["metrics"], indent=2))
        return
    model.train()
    if args.stage in ("smoke", "tiny"):
        # The smoke and tiny run are disposable; their weights NEVER seed full training.
        batch = to_device(next(iter(loader(args, cfg, train[:16], False))), args.device)
        with torch.no_grad():
            initial, _ = criterion(model(batch["image"]), batch, geometry_scale=1.)
        initial = float(initial)
        steps = 2 if args.stage == "smoke" else args.steps
        updates = []
        batches = iter(loader(args, cfg, train[:16], False))
        start = time.perf_counter()
        for step in range(steps):
            # Cycle all 16 train-only images for tiny; fixed minibatch for smoke.
            if args.stage == "tiny":
                try:
                    batch = to_device(next(batches), args.device)
                except StopIteration:
                    batches = iter(loader(args, cfg, train[:16], False, epoch=step))
                    batch = to_device(next(batches), args.device)
            update = finite_update(model, criterion, optimizer, batch, cfg["grad_clip"], min(1., (step+1)/50))
            updates.append(update)
            if step % 25 == 0:
                print(f"{args.stage}: {step+1}/{steps} loss={update['total']:.4f}", flush=True)
        # Compare the same initial minibatch, same train mode, full geometry weight.
        compare = to_device(next(iter(loader(args, cfg, train[:16], False))), args.device)
        with torch.no_grad():
            final, _ = criterion(model(compare["image"]), compare, geometry_scale=1.)
        final = float(final)
        passed = bool(math.isfinite(final) and (args.stage == "smoke" or final < 0.8 * initial))
        report = {"complete": True, "passed": passed, "stage": args.stage, "steps": steps,
                  "identity_sha256": manifest["identity_sha256"],
                  "initial_same_batch_loss": initial, "final_same_batch_loss": final,
                  "loss_ratio": final/initial, "finite_gradients_all_parameters": True,
                  "elapsed_seconds": time.perf_counter()-start, "updates": updates,
                  "device": args.device, "external_teacher_used": False,
                  "accuracy_qualified": False, "full_training_initialized_from_this_run": False}
        atomic_json(args.output_dir / f"{args.stage}.json", report)
        if not passed:
            raise RuntimeError("Tiny learnability check failed: stop before full training and return the result bundle")
        return
    tiny_path = args.output_dir / "tiny.json"
    tiny = json.loads(tiny_path.read_text()) if tiny_path.exists() else {}
    if not tiny.get("passed") or tiny.get("identity_sha256") != manifest["identity_sha256"]:
        raise RuntimeError("Run and pass the train-only tiny learnability check before full training")
    if not resume and not use_initial:
        atomic_checkpoint(initial_path, {"model": model.state_dict(), "epoch": 0,
                          "identity_sha256": manifest["identity_sha256"], "preprocessing": PREPROCESSING,
                          "external_teacher_used": False})
    atomic_json(args.output_dir / "initialization.json", {"complete": True,
                "initial_checkpoint_sha256": file_hash(initial_path), "pretrained_backbone": cfg["pretrained_backbone"],
                "seed": cfg["seed"], "tiny_weights_used": False, "external_teacher_used": False})
    for epoch in range(first_epoch, cfg["epochs"]):
        schedule = min(1., (epoch + 1)/cfg["warmup_epochs"]) * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / cfg["epochs"])) / 2)
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * schedule
        records = []
        started = time.perf_counter()
        geometry_scale = min(1., epoch / cfg["geometry_warmup_epochs"])
        for index, batch in enumerate(loader(args, cfg, train, True, epoch)):
            batch = to_device(batch, args.device)
            records.append(finite_update(model, criterion, optimizer, batch, cfg["grad_clip"], geometry_scale))
            if index % 100 == 0:
                print(f"epoch {epoch+1}/{cfg['epochs']} batch {index} loss {records[-1]['total']:.4f}", flush=True)
        summary = {"epoch": epoch + 1, "seconds": time.perf_counter()-started,
                   "lr": optimizer.param_groups[0]["lr"],
                   "environment": environment(),
                   "loss": {key: float(np.mean([row[key] for row in records])) for key in records[0]}}
        if (epoch + 1) % cfg["evaluate_every"] == 0 or epoch + 1 == cfg["epochs"]:
            summary["validation"] = evaluate(model, args, cfg, val, args.output_dir / f"evaluations/epoch_{epoch+1:03d}")["metrics"]
        history.append(summary)
        atomic_json(args.output_dir / "history.json", history)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1,
                   "identity_sha256": manifest["identity_sha256"], "history": history, "rng": rng_state(),
                   "architecture_revision": model.architecture_revision, "preprocessing": PREPROCESSING,
                   "external_teacher_used": False}
        atomic_checkpoint(args.output_dir / "checkpoints/latest.pt", payload)
        if (epoch+1) % cfg["save_every"] == 0 or epoch+1 == cfg["epochs"]:
            atomic_checkpoint(args.output_dir / f"checkpoints/epoch_{epoch+1:03d}.pt", payload)
        print(json.dumps(summary), flush=True)
    candidates = [row for row in history if "validation" in row]
    selected = max(candidates, key=lambda row: row["validation"]["mean_3d_moderate"])
    atomic_json(args.output_dir / "selection.json", {"complete": True, "selected_epoch": selected["epoch"],
                "criterion": "maximum product mean moderate 3D AP_R40 among scheduled val evaluations",
                "metrics": selected["validation"], "checkpoint": str(args.output_dir / f"checkpoints/epoch_{selected['epoch']:03d}.pt"),
                "gate_results": {k: selected["validation"][k] >= v for k, v in GATES.items()},
                "all_accuracy_gates_passed": all(selected["validation"][k] >= v for k, v in GATES.items()),
                "external_teacher_used": False, "iphone_performance_measured": False,
                "nearby_recall_review_required": True, "deployment_qualified": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "tiny", "train", "evaluate", "bundle"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/compact_mnv4_kitti_gt_r1.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    try:
        run(args)
    except Exception as error:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{time.time_ns()}"
        atomic_json(args.output_dir / "failures" / f"{args.stage}_{stamp}.json",
                    {"complete": False, "stage": args.stage, "type": type(error).__name__,
                     "error": str(error), "traceback": traceback.format_exc(),
                     "existing_checkpoints_preserved": True})
        raise


if __name__ == "__main__":
    main()
