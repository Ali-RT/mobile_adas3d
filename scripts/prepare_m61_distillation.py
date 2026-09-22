"""Freeze M54/A2 provenance and the two-run M61 experiment, without training."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import yaml

from m61_common import (AP_GATES, CLASS_MAPPING, REVISION, SPLITS, STUDENT_COMMIT,
                        STUDENT_SHA, TEACHER_COMMIT, TEACHER_SHA, check_split,
                        code_hash, json_hash, read_json, sha256, source_hash, write_json)


def make_config(base, dataset, role, output, repo):
    cfg = copy.deepcopy(base)
    cfg["random_seed"] = 20268
    cfg["dataset"].update(root_dir=str(dataset), train_split="train", test_split="val",
                          writelist=["Car", "Pedestrian"], class_mapping=CLASS_MAPPING,
                          class_merging=False, use_dontcare=False, meanshape=False,
                          batch_size=4, aug_pd=False, aug_crop=False, aug_calib=False,
                          random_flip=0.0, random_crop=0.0, random_mixup3d=0.0)
    if role == "student":
        cfg["model"].update(backbone_source="timm",
                            backbone="mobilenetv4_conv_medium.e500_r256_in1k",
                            backbone_out_indices=[2, 3, 4], backbone_pretrained=False)
    cfg["optimizer"].update(type="adamw", lr=1e-5, weight_decay=1e-4)
    cfg["lr_scheduler"].update(type="step", warmup=False, decay_list=[], decay_rate=1.0)
    cfg["trainer"].update(max_epoch=10, save_all=True, save_frequency=1, gpu_ids="0",
                          save_path=os.path.relpath(output, repo),
                          pretrain_model=None, resume_model=False)
    cfg["tester"].update(mode="single", checkpoint=10, threshold=0.001, topk=50)
    cfg.pop("distillation", None)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("student-repo", "teacher-repo", "a2-selection", "m54-selection",
                 "dataset-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    output = args.output_dir.resolve()
    dataset = args.dataset_root.resolve()
    ids = {split: check_split(dataset / "ImageSets" / f"{split}.txt", split) for split in SPLITS}
    if set(ids["train"]) & set(ids["val"]):
        raise RuntimeError("Train/validation leakage")
    # Check completeness here; exact training inputs/GT are hashed during caching.
    for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
        for image_id in ids["train"] + ids["val"]:
            if not (dataset / "training" / folder / (image_id + suffix)).is_file():
                raise FileNotFoundError(f"KITTI {folder}/{image_id}{suffix}")
    m = dict(schema_version=1, revision=REVISION, complete=True, code_sha256=code_hash(),
             dataset_root=str(dataset), output_dir=str(output), seed=20268, epochs=10,
             batch_size=4, learning_rate=1e-5, augmentation=False,
             vehicle_native_class_id=1, temperature=None, overall_kd_weight=0.25,
             audit_policy=dict(min_score=0.3, min_iou=0.5, max_depth_exclusive=60.0,
                               min_component_pairs=100, individual_error_ratio_max=0.95),
             ap_gates=AP_GATES, evaluation_epochs=[5, 10], decision_epoch=10,
             full_run_authorized=False, phone_deployment_authorized=False)
    for role, repo, selection_path, commit, expected, epoch, name in (
        ("student", args.student_repo, args.a2_selection, STUDENT_COMMIT, STUDENT_SHA, 130, "monodetr"),
        ("teacher", args.teacher_repo, args.m54_selection, TEACHER_COMMIT, TEACHER_SHA, 100, "monodgp"),
    ):
        repo = repo.resolve()
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Wrong {role} upstream commit: {actual}")
        selection = read_json(selection_path)
        checkpoint = Path(selection["selected_checkpoint"]).resolve()
        if not selection.get("complete") or selection["selected_epoch"] != epoch or sha256(checkpoint) != expected:
            raise RuntimeError(f"Wrong/incomplete frozen {role} checkpoint selection")
        base_path = repo / f"configs/{name}.yaml"
        cfg = make_config(yaml.safe_load(base_path.read_text()), dataset, role, output, repo)
        m[role] = dict(repo=str(repo), upstream_commit=commit, checkpoint=str(checkpoint),
                       checkpoint_sha256=expected, epoch=epoch, config=cfg,
                       selection_sha256=sha256(selection_path), source_sha256=source_hash(repo))
    # This workflow is independent of missing transient A2/M54 runtime YAML files.
    m["variants"] = {}
    for variant in ("control", "vehicle_kd"):
        cfg = copy.deepcopy(m["student"]["config"])
        cfg["model_name"] = f"m61_{variant}"
        path = output / "configs" / f"{variant}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != encoded:
            raise RuntimeError(f"Refusing changed existing config: {path}")
        path.write_text(encoded)
        m["variants"][variant] = dict(config=str(path), config_sha256=sha256(path),
                                     run_dir=str(output / cfg["model_name"]))
    m["manifest_sha256"] = json_hash(m)
    path = output / "m61_manifest.json"
    if path.exists() and read_json(path) != m:
        raise RuntimeError("Existing M61 manifest differs; do not mix experiments in one output directory")
    write_json(path, m)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
