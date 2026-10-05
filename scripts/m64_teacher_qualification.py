"""M64: A2 baseline and one candidate teacher; no optimization or KD authorization."""
from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M64-TEACHER-QUALIFICATION-2026-10-04-r1"
A2_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
TEACHER_COMMIT = "884b7d8562e528031616c4773170bfc4fe211bf0"
A2_SHA = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"
SPLITS = {
    "train": (3712, "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c"),
    "val": (3769, "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"),
}
ASSETS = {
    "checkpoint": ("1UENFz-poULnTnfwKJoNal6sIgpDFXVfO", "monoprio_seed444_validation.pth"),
    "prior": ("1T0KXFiafpbZPPjXp0arM9BumrDE4z6X4", "monoprio_validation_unified.npz"),
    "log": ("1jMVHLFyUKMYfSEsJZS-43cYGChWDfmiG", "monoprio_seed444_published.log"),
}
# Bytes independently inspected locally on 2026-10-04, not publisher-signed checksums.
ASSET_SHA256 = {
    "checkpoint": "6ac885b5902b98110e42da0217a01232e61cf8534884f188bd794ff97e50c098",
    "prior": "e2f519187b72f1f4c62d44cbef63f0d6d05f87c7c47fff160049b34d230b464c",
    "log": "8dd018b12a36adc29842a54a36addc890ade4f34f5efb737b47e2d321aa71850",
}
PUBLISHED = {"Car": [30.4130, 21.9274, 18.7024], "Pedestrian": [12.4075, 9.6523, 7.4288]}
BASELINE = dict(vehicle_3d_moderate=15.457290707384603,
                pedestrian_3d_moderate=7.532768186994984,
                vehicle_bev_moderate=21.37498561663574,
                pedestrian_bev_moderate=8.489171811764757)
BASELINE_NEARBY = {"Vehicle": 0.88293448, "Pedestrian": 0.69223986}
TARGET_KEYS = ("labels", "boxes", "calibs", "depth", "size_3d", "heading_bin", "heading_res", "boxes_3d")
OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle")
CLASS_MAPPING = dict(Car="Car", Van="Car", Truck="Car", Tram="Car",
                     Pedestrian="Pedestrian", Person_sitting="Pedestrian")


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def split_ids(path: Path, split: str) -> list[str]:
    ids = path.read_text().splitlines()
    count, digest = SPLITS[split]
    normalized = "\n".join(ids) + "\n"
    if (len(ids) != count or len(set(ids)) != count
            or any(len(i) != 6 or not i.isdigit() for i in ids)
            or hashlib.sha256(normalized.encode()).hexdigest() != digest):
        raise RuntimeError(f"Not the exact Chen {split} IDs/order: {path}")
    return ids


def source_hash(repo: Path) -> str:
    inventory = {str(p.relative_to(repo)): sha(p)
                 for folder in ("lib", "utils", "tools")
                 for p in sorted((repo / folder).rglob("*"))
                 if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}}
    if not inventory:
        raise RuntimeError(f"Missing model source: {repo}")
    return signature(inventory)


def implementation_hash() -> str:
    # Bind the CPU evaluators/configs too, without importing frozen M62 loaders.
    paths = [Path(__file__), ROOT / "scripts/patch_m64_inference_sources.py",
             ROOT / "scripts/setup_m64_runtime.py", ROOT / "scripts/restore_m63_data.py",
             ROOT / "scripts/build_m64_attention.py",
             ROOT / "scripts/evaluate_kitti_prediction_dir.py",
             ROOT / "scripts/audit_product_prediction_geometry.py"]
    paths += sorted((ROOT / "data").glob("*.py"))
    paths += sorted((ROOT / "tools").glob("*.py"))
    paths += sorted((ROOT / "configs").glob("*.yaml"))
    return signature({str(p.relative_to(ROOT)): sha(p) for p in paths})


def package_environment() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "pillow", "opencv-python-headless",
                 "scikit-image", "numba", "timm", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    return dict(python=sys.version, packages=packages)


def environment() -> dict:
    import torch
    return dict(**package_environment(), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda_home=os.environ.get("CUDA_HOME"))


def download_assets(output: Path) -> None:
    import gdown
    directory = output / "assets"
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "asset_receipt.json"
    previous = read_json(receipt_path) if receipt_path.exists() else {}
    receipt = {}
    for role, (file_id, filename) in ASSETS.items():
        path = directory / filename
        if path.exists() and role in previous:
            if previous[role]["sha256"] != sha(path) or previous[role]["drive_file_id"] != file_id:
                raise RuntimeError(f"Existing {role} asset changed; preserve it, use a new RUN_ID")
        elif not path.exists():
            temporary = directory / (filename + ".download")
            print(f"Downloading the official validation {role}: {file_id}", flush=True)
            if not gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True):
                raise RuntimeError(f"Download unavailable: https://drive.google.com/file/d/{file_id}/view")
            temporary.replace(path)
        digest = sha(path)
        if digest != ASSET_SHA256[role]:
            raise RuntimeError(f"Official {role} bytes differ from the locally inspected release: {digest}")
        receipt[role] = dict(path=str(path.resolve()), drive_file_id=file_id, sha256=sha(path),
                             bytes=path.stat().st_size, source="pinned official README validation link",
                             locally_inspected_release_bytes_verified=True,
                             independently_published_checksum_available=False)
        write_json(receipt_path, receipt)
    print("Locally inspected release hashes verified. No publisher-signed checksum is claimed.", flush=True)


def model_config(repo: Path, role: str, dataset: Path, prior: Path) -> dict:
    import yaml
    name = "monodetr" if role == "a2" else "monoprio"
    cfg = yaml.safe_load((repo / f"configs/{name}.yaml").read_text())
    cfg["random_seed"] = 444
    cfg["dataset"].update(root_dir=str(dataset), train_split="train", test_split="val",
                          batch_size=1, class_merging=False, use_dontcare=False, meanshape=False,
                          aug_pd=False, aug_crop=False, random_flip=0.0,
                          random_crop=0.0, random_mixup3d=0.0)
    if role == "a2":
        cfg["dataset"].update(writelist=["Car", "Pedestrian"], class_mapping=CLASS_MAPPING)
        cfg["model"].update(backbone_source="timm", backbone="mobilenetv4_conv_medium.e500_r256_in1k",
                            backbone_out_indices=[2, 3, 4], backbone_pretrained=False)
    else:
        cfg["dataset"].pop("class_mapping", None)
        cfg["dataset"]["writelist"] = ["Car", "Pedestrian", "Cyclist"]
        cfg["model"]["prior_path"] = str(prior)
        if cfg["model"].get("use_monoprio") is not True:
            raise RuntimeError("Candidate prior conditioning unexpectedly disabled")
    return cfg


def prepare(args) -> None:
    output, dataset = args.output.resolve(), args.dataset_root.resolve()
    receipt = read_json(output / "assets/asset_receipt.json")
    if set(receipt) != set(ASSETS) or any(
            entry["drive_file_id"] != ASSETS[role][0] or sha(entry["path"]) != ASSET_SHA256[role]
            for role, entry in receipt.items()):
        raise RuntimeError("Teacher assets differ from the locally inspected release")
    selection = read_json(args.a2_selection)
    checkpoint = Path(selection["selected_checkpoint"])
    if (selection.get("complete") is not True or selection.get("selected_epoch") != 130
            or sha(checkpoint) != A2_SHA):
        raise RuntimeError("A2 selection must identify the original epoch130 checkpoint/hash")
    ids = {s: split_ids(dataset / "ImageSets" / (s + ".txt"), s) for s in SPLITS}
    if set(ids["train"]) & set(ids["val"]):
        raise RuntimeError("Train/validation overlap")
    data_identity = {}
    for s in ids:
        rows = {}
        for i in ids[s]:
            for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
                path = dataset / "training" / folder / (i + suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if folder != "image_2":
                    rows[f"{folder}/{i}{suffix}"] = sha(path)
        data_identity[s] = signature(rows)
    models = {}
    for role, repo, commit in (("a2", args.a2_repo.resolve(), A2_COMMIT),
                              ("teacher", args.teacher_repo.resolve(), TEACHER_COMMIT)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Wrong {role} source commit: {actual}")
        if role == "teacher":
            # Validate original release IDs, tolerating only line-ending differences.
            for s in SPLITS:
                split_ids(repo / "data/kitti/ImageSets" / (s + ".txt"), s)
        cfg = model_config(repo, role, dataset, Path(receipt["prior"]["path"]))
        model_checkpoint = checkpoint if role == "a2" else Path(receipt["checkpoint"]["path"])
        name = "monodetr" if role == "a2" else "monoprio"
        build_path = repo / f"lib/models/{name}/ops/m64_build_receipt.json"
        build = read_json(build_path)
        if not build.get("import_verified") or sha(build["binary"]) != build["binary_sha256"]:
            raise RuntimeError("Missing or changed locally built attention extension")
        models[role] = dict(repo=str(repo), upstream_commit=commit,
                            patched_source_sha256=source_hash(repo), config=cfg,
                            attention_build_receipt=str(build_path), attention_build_sha256=sha(build_path),
                            attention_binary=build["binary"], attention_binary_sha256=build["binary_sha256"],
                            checkpoint=str(model_checkpoint), checkpoint_sha256=sha(model_checkpoint))
    manifest = dict(schema_version=1, revision=REVISION, output=str(output), dataset_root=str(dataset),
                    implementation_sha256=implementation_hash(), environment=environment(), models=models,
                    asset_receipt=receipt, data_identity=data_identity,
                    published_seed444_3d_r40=PUBLISHED, native_reproduction_tolerance_ap=0.5,
                    product_score_threshold=0.001, published_score_threshold=0.2, topk=50,
                    baseline_ap_drift_tolerance=0.15, original_a2_product_metrics=BASELINE,
                    baseline_nearby_drift_tolerance=0.01, original_a2_nearby=BASELINE_NEARBY,
                    optimizer_steps=0, kd_authorized=False, full_training_authorized=False,
                    deployment_authorized=False, prior_training_ids_independently_verified=False,
                    historical_manifests_rewritten=False)
    if manifest["environment"]["gpu"] is None:
        raise RuntimeError("Use a CUDA GPU runtime")
    manifest["manifest_sha256"] = signature(manifest)
    path = output / "m64_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise RuntimeError("M64 identity differs. Preserve this output; change RUN_ID for a new environment")
    write_json(path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def load(path: Path, native_evaluator=False) -> dict:
    m = read_json(path)
    digest = m.pop("manifest_sha256")
    if m.get("revision") != REVISION or signature(m) != digest:
        raise RuntimeError("M64 manifest changed")
    m["manifest_sha256"] = digest
    if m["implementation_sha256"] != implementation_hash():
        raise RuntimeError("M64 implementation changed; preserve results and use a new RUN_ID")
    # The isolated AP process must not initialize a Torch CUDA context first.
    current = package_environment() if native_evaluator else environment()
    saved = {k: m["environment"][k] for k in current}
    if saved != current:
        raise RuntimeError("Prospective environment differs; choose a new RUN_ID, not a historical rollback")
    for entry in m["models"].values():
        if sha(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise RuntimeError("Frozen model weights changed")
        if source_hash(Path(entry["repo"])) != entry["patched_source_sha256"]:
            raise RuntimeError("Prospective model source changed")
        if (sha(entry["attention_build_receipt"]) != entry["attention_build_sha256"]
                or sha(entry["attention_binary"]) != entry["attention_binary_sha256"]):
            raise RuntimeError("Bound attention extension changed")
    for entry in m["asset_receipt"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Teacher asset changed")
    dataset = Path(m["dataset_root"])
    for s in SPLITS:
        ids = split_ids(dataset / "ImageSets" / (s + ".txt"), s)
        rows = {f"{folder}/{i}.txt": sha(dataset / "training" / folder / (i + ".txt"))
                for folder in ("label_2", "calib") for i in ids}
        if signature(rows) != m["data_identity"][s]:
            raise RuntimeError(f"Original {s} labels/calibration changed")
    return m


def safe_payload(path: Path):
    import numpy as np
    import torch
    core = np._core if hasattr(np, "_core") else np.core
    allowed = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"),
               (core.multiarray.scalar, "numpy._core.multiarray.scalar"), np.dtype, codecs.encode,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise RuntimeError("Expected a restricted tensor-state checkpoint")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
           for k, v in payload["model_state"].items()):
        raise RuntimeError("Non-tensor model-state entry")
    return payload


def seed() -> None:
    import numpy as np
    import torch
    random.seed(444)
    np.random.seed(444)
    torch.manual_seed(444)
    torch.cuda.manual_seed_all(444)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_hash(values) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, arr.dtype.str, arr.shape]).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def runtime(m: dict, role: str, split="val"):
    import torch
    if "lib" in sys.modules:
        raise RuntimeError("Each model must run in a fresh subprocess")
    seed()
    entry = m["models"][role]
    repo = Path(entry["repo"])
    name = "monodetr" if role == "a2" else "monoprio"
    ops = repo / f"lib/models/{name}/ops"
    sys.path[:0] = [str(repo), str(ops)]
    import MultiScaleDeformableAttention as extension
    if not Path(extension.__file__).resolve().is_relative_to(ops.resolve()):
        raise RuntimeError("Loaded another experiment's attention extension")
    from lib.helpers.model_helper import build_model
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    if any("kitti_eval_python.eval" in key or "rotate_iou" in key for key in sys.modules):
        raise RuntimeError("Dataset import unexpectedly initialized AP evaluation")
    cfg = copy.deepcopy(entry["config"])
    dataset = KITTI_Dataset(split, cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != {"Pedestrian": 0, "Car": 1, "Cyclist": 2}:
        raise RuntimeError("Native class IDs changed")
    print(f"{role}: source imports passed; AP evaluator has not been imported", flush=True)
    model, criterion = build_model(cfg["model"])
    payload = safe_payload(Path(entry["checkpoint"]))
    if role == "teacher":
        prior_buffers = {k: v for k, v in model.state_dict().items()
                         if k.startswith("router.bank_") or k == "router.class_offsets"}
        if not prior_buffers or any(k not in payload["model_state"] or not torch.equal(v, payload["model_state"][k])
                                    for k, v in prior_buffers.items()):
            raise RuntimeError("Released prior bank does not match checkpoint's fixed router buffers")
    model.load_state_dict(payload["model_state"], strict=True)
    checkpoint_epoch = int(payload.get("epoch", -1))
    del payload
    model = model.cuda().eval()
    if role == "teacher":
        model.requires_grad_(False)
    return model, criterion.cuda(), dataset, checkpoint_epoch


def smoke(m: dict, role: str) -> None:
    import torch
    from torch.utils.data import DataLoader
    print(f"{role}: entering forward/backward preflight", flush=True)
    model, criterion, dataset, epoch = runtime(m, role, "train" if role == "a2" else "val")
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    images, calibs, raw, info = next(iter(loader))
    images, calibs, sizes = images.cuda(), calibs.cuda(), info["img_size"].cuda()
    with torch.no_grad():
        out = model(images, calibs, None, sizes, dn_args=0)
    if any(not bool(torch.isfinite(out[k]).all()) for k in OUTPUT_KEYS):
        raise RuntimeError("Non-finite native predictions")
    shapes = {k: list(out[k].shape) for k in OUTPUT_KEYS}
    del out
    report = dict(schema_version=1, complete=True, role=role, manifest_sha256=m["manifest_sha256"],
                  checkpoint_epoch=epoch, finite_outputs=True, output_shapes=shapes,
                  dataset_import_without_cuda_ap=True, safe_weights_only_load=True,
                  optimizer_steps=0, training_authorized=False)
    if role == "a2":
        # Real backward, without creating an optimizer or changing any parameter.
        before = tensor_hash(dict(model.named_parameters()))
        buffers_before = tensor_hash(dict(model.named_buffers()))
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        targets = [{k: raw[k][i][raw["mask_2d"][i]].cuda() for k in TARGET_KEYS}
                   for i in range(len(raw["labels"]))]
        out = model(images, calibs, targets, sizes, dn_args=None)
        losses = criterion(out, targets, None)
        total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Non-finite A2 GT loss")
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("Missing or non-finite native backward gradients")
        nonzero = any(bool(torch.any(g != 0)) for g in gradients)
        if not nonzero or before != tensor_hash(dict(model.named_parameters())):
            raise RuntimeError("Backward failed or changed source parameters")
        if buffers_before != tensor_hash(dict(model.named_buffers())):
            raise RuntimeError("Frozen running buffers changed during smoke")
        report.update(gt_total_loss=float(total.detach()), finite_nonzero_gradients=True,
                      parameters_unchanged=True, running_buffers_unchanged=True)
    else:
        report.update(prior_buffers_match_checkpoint=True,
                      prior_training_ids_independently_verified=False)
    torch.cuda.synchronize()
    write_json(Path(m["output"]) / f"m64_{role}_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


def prediction_text(rows, class_names) -> str:
    return "".join(class_names[int(row[0])] + " 0.0 0" +
                   "".join(f" {float(value):.2f}" for value in row[1:]) + "\n" for row in rows)


def cached_prediction(path: Path, identity: dict):
    if not path.exists():
        return None
    record = read_json(path)
    digest = record.pop("record_sha256")
    if signature(record) != digest or record.get("identity") != identity:
        raise RuntimeError(f"Prediction cache changed or belongs to another input: {path}")
    return record


def infer(m: dict, role: str) -> None:
    import torch
    from torch.utils.data import DataLoader
    model, _, dataset, _ = runtime(m, role)
    from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
    root = Path(m["output"]) / "predictions" / role
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ids = split_ids(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    files = {"product": {}}
    if role == "teacher":
        files["published"] = {}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for index, (images, calibs, _, info) in enumerate(loader):
        image_id = f"{int(info['img_id'][0]):06d}"
        if image_id != ids[index]:
            raise RuntimeError("Validation order changed")
        identity = dict(manifest_sha256=m["manifest_sha256"], role=role, image_id=image_id,
                        input_sha256=tensor_hash(dict(image=images, calib=calibs, size=info["img_size"])))
        path = cache / (image_id + ".json")
        record = cached_prediction(path, identity)
        if record is None:
            with torch.no_grad():
                out = model(images.cuda(), calibs.cuda(), None, info["img_size"].cuda(), dn_args=0)
                if any(not bool(torch.isfinite(out[k]).all()) for k in OUTPUT_KEYS):
                    raise RuntimeError(f"Non-finite {role} output at {image_id}")
                detections = extract_dets_from_outputs(out, K=50, topk=m["topk"]).cpu().numpy()
            calibration = [dataset.get_calib(int(info["img_id"][0]))]
            native_info = {k: v.numpy() for k, v in info.items()}
            texts = {}
            for protocol in files:
                threshold = m["published_score_threshold"] if protocol == "published" else m["product_score_threshold"]
                decoded = decode_detections(detections.copy(), native_info, calibration,
                                            dataset.cls_mean_size, threshold)
                texts[protocol] = prediction_text(decoded[int(info["img_id"][0])], dataset.class_name)
            record = dict(identity=identity, texts=texts)
            record["record_sha256"] = signature(record)
            write_json(path, record)
        # Recover a text export interrupted after its single atomic cache record.
        for protocol, text in record["texts"].items():
            dest = root / protocol / (image_id + ".txt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_text() != text:
                raise RuntimeError(f"Existing exported prediction changed: {dest}")
            if not dest.exists():
                tmp = dest.with_suffix(".txt.tmp")
                tmp.write_text(text)
                tmp.replace(dest)
            files[protocol][image_id] = sha(dest)
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"{role}: inferred/verified {index + 1}/{len(ids)}", flush=True)
    if {p.stem for p in cache.glob("*.json")} != set(ids):
        raise RuntimeError("Unexpected prediction cache IDs")
    for protocol in files:
        if {p.stem for p in (root / protocol).glob("*.txt")} != set(ids):
            raise RuntimeError("Incomplete or extra validation predictions")
    write_json(root / "complete.json", dict(complete=True, role=role,
               manifest_sha256=m["manifest_sha256"], prediction_files=files))


def validate_predictions(m: dict, role: str) -> Path:
    root = Path(m["output"]) / "predictions" / role
    report = read_json(root / "complete.json")
    ids = split_ids(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    if not report.get("complete") or report.get("manifest_sha256") != m["manifest_sha256"]:
        raise RuntimeError(f"Missing or stale complete {role} predictions")
    expected = {"product", "published"} if role == "teacher" else {"product"}
    if set(report["prediction_files"]) != expected:
        raise RuntimeError("Missing or extra prediction protocols")
    for protocol, files in report["prediction_files"].items():
        if set(files) != set(ids) or {p.stem for p in (root / protocol).glob("*.txt")} != set(ids):
            raise RuntimeError("Prediction IDs differ from the full validation split")
        if any(sha(root / protocol / (i + ".txt")) != digest for i, digest in files.items()):
            raise RuntimeError("Prediction content changed")
    return root


def run(command) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def metrics(m: dict, role: str) -> None:
    import yaml
    directory = Path(m["output"]) / "metrics" / role
    predictions = validate_predictions(m, role) / "product"
    shared = ["--profile", "colab_drive", "--dataset-root", m["dataset_root"],
              "--split-dir", Path(m["dataset_root"]) / "ImageSets",
              "--prediction-dir", predictions, "--split", "val", "--source-name", f"M64_{role}"]
    run([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
         "--config", ROOT / "configs/kitti_mobileadas3d_s1.yaml", *shared,
         "--classes", "Vehicle", "Pedestrian", "--output-dir", directory / "product"])
    native_config = dict(base_config=str(ROOT / "configs/kitti_mobileadas3d_s1.yaml"),
                         dataset=dict(classes=["Car", "Pedestrian"], class_mapping=None,
                                      require_taxonomy_manifest=False))
    config_path = directory / "native_classes.yaml"
    config_path.write_text(yaml.safe_dump(native_config, sort_keys=False))
    run([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
         "--config", config_path, *shared, "--classes", "Car", "Pedestrian",
         "--output-dir", directory / "native_independent"])
    run([sys.executable, "-u", ROOT / "scripts/audit_product_prediction_geometry.py",
         "--dataset-root", m["dataset_root"], "--split-file", Path(m["dataset_root"]) / "ImageSets/val.txt",
         "--prediction-dir", predictions, "--output-dir", directory / "nearby",
         "--checkpoint", m["models"][role]["checkpoint"], "--expected-images", "3769",
         "--score-threshold", "0.001", "--match-iou-threshold", "0.5"])


def native_reference(m: dict) -> None:
    # This is deliberately a separate process. A native evaluator failure must
    # not corrupt or invalidate the already-complete CUDA prediction cache.
    root = validate_predictions(m, "teacher")
    repo = Path(m["models"]["teacher"]["repo"])
    sys.path.insert(0, str(repo))
    print("Separate published-protocol AP evaluator: importing Numba CUDA now", flush=True)
    from lib.datasets.kitti.kitti_eval_python import kitti_common
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
    ids = split_ids(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    gt = kitti_common.get_label_annos(str(Path(m["dataset_root"]) / "training/label_2"), [int(i) for i in ids])
    dt = kitti_common.get_label_annos(str(root / "published"), [int(i) for i in ids])
    values = {}
    for cls, native_id in (("Car", 0), ("Pedestrian", 1)):
        text, result, _ = get_official_eval_result(gt, dt, native_id)
        print(text, flush=True)
        values[cls] = [float(result[f"{cls}_3d_{d}_R40"]) for d in ("easy", "moderate", "hard")]
    checks = {cls: all(abs(a - b) <= m["native_reproduction_tolerance_ap"]
                       for a, b in zip(values[cls], PUBLISHED[cls])) for cls in PUBLISHED}
    write_json(Path(m["output"]) / "m64_teacher_native_reference.json",
               dict(complete=True, manifest_sha256=m["manifest_sha256"], reproduced=values,
                    published_seed444=PUBLISHED, checks=checks,
                    reference_reproduced=all(checks.values()), tolerance_ap=0.5,
                    kd_authorized=False))


def metric_rows(summary: dict) -> dict:
    if not summary.get("complete_split") or summary.get("evaluated_images") != 3769:
        raise RuntimeError("Incomplete validation metrics")
    values = {f"{r['class_name'].lower()}_{r['metric']}_moderate": float(r["ap_r40"])
              for r in summary["metrics"] if r["difficulty"] == "moderate"}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError("Non-finite AP values")
    return values


def review(m: dict) -> None:
    output = Path(m["output"])
    rows = {}
    for role in ("a2", "teacher"):
        validate_predictions(m, role)
        smoke_report = read_json(output / f"m64_{role}_smoke.json")
        if not smoke_report.get("complete") or smoke_report["manifest_sha256"] != m["manifest_sha256"]:
            raise RuntimeError("Missing or stale model smoke report")
        rows[role] = dict(product=metric_rows(read_json(output / f"metrics/{role}/product/kitti_r40_summary.json")),
                          native_classes=metric_rows(read_json(output / f"metrics/{role}/native_independent/kitti_r40_summary.json")),
                          nearby=read_json(output / f"metrics/{role}/nearby/nearby_geometry_summary.json")["classes"])
    drift = {k: abs(rows["a2"]["product"][k] - value) <= m["baseline_ap_drift_tolerance"]
             for k, value in BASELINE.items()}
    drift.update({f"{k.lower()}_near_recall": abs(rows["a2"]["nearby"][k]["near_recall"] - value)
                  <= m["baseline_nearby_drift_tolerance"] for k, value in BASELINE_NEARBY.items()})
    native_path = output / "m64_teacher_native_reference.json"
    native = read_json(native_path) if native_path.exists() else None
    if native and native["manifest_sha256"] != m["manifest_sha256"]:
        raise RuntimeError("Native reference belongs to a different manifest")
    report = dict(schema_version=1, complete=True, revision=REVISION,
                  manifest_sha256=m["manifest_sha256"], rows=rows,
                  a2_baseline_checks=drift, a2_baseline_reproduced=all(drift.values()),
                  published_native_reference=native,
                  teacher_native_reproduced=bool(native and native.get("reference_reproduced")),
                  teacher_minus_a2_product={k: rows["teacher"]["product"][k] - v
                                            for k, v in rows["a2"]["product"].items()},
                  teacher_product_taxonomy_adapted=False,
                  prior_training_ids_independently_verified=False,
                  checkpoint_and_release_split_verified=True,
                  teacher_selected=False, kd_authorized=False, full_training_authorized=False,
                  phone_deployment_authorized=False, optimizer_steps=0,
                  next_step="Review prior provenance, class/range strengths and baseline stability before freezing a preservation/KD pilot")
    write_json(output / "m64_teacher_qualification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("STOP FOR REVIEW. This notebook never authorizes KD or trains the teacher.", flush=True)


def bundle(output: Path) -> None:
    path = output / "m64_results.zip"
    files = [p for p in output.rglob("*") if p.is_file() and p != path
             and p.suffix in {".json", ".yaml", ".log", ".csv", ".txt"}
             and "predictions" not in p.relative_to(output).parts
             and "assets" not in p.relative_to(output).parts]
    # Include the official log/receipt, but not hundreds of MB of model weights.
    files += [p for p in (output / "assets").glob("*") if p.suffix in {".log", ".json"}]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            archive.write(file, file.relative_to(output))
    print(f"Return this file: {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("assets", "prepare", "smoke", "infer", "metrics", "native", "review", "bundle"))
    p.add_argument("--output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--role", choices=("a2", "teacher"))
    for key in ("a2-repo", "teacher-repo", "dataset-root", "a2-selection"):
        p.add_argument("--" + key, type=Path)
    args = p.parse_args()
    if args.stage in {"assets", "bundle", "prepare"} and args.output is None:
        p.error("--output is required for this stage")
    if args.stage == "assets":
        download_assets(args.output.resolve())
    elif args.stage == "bundle":
        bundle(args.output.resolve())
    elif args.stage == "prepare":
        if any(getattr(args, k) is None for k in ("a2_repo", "teacher_repo", "dataset_root", "a2_selection")):
            p.error("prepare requires both repositories, dataset root and A2 selection")
        prepare(args)
    else:
        if args.manifest is None:
            p.error("--manifest is required")
        if args.stage in {"smoke", "infer", "metrics"} and args.role is None:
            p.error("--role is required")
        m = load(args.manifest, native_evaluator=args.stage == "native")
        functions = dict(smoke=smoke, infer=infer, metrics=metrics, native=native_reference, review=review)
        if args.role and args.stage in {"smoke", "infer", "metrics"}:
            functions[args.stage](m, args.role)
        else:
            functions[args.stage](m)


if __name__ == "__main__":
    main()
