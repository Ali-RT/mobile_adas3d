"""M63c: one full-val inference with source A2 BatchNorm statistics; no training."""
from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from m63_common import ROOT, check_split, load_manifest, read_json, sha256, write_json
from diagnose_m63_continuation import load_bound_checkpoint
from m63c_bn_intervention import restore_running_statistics, state_fingerprints

REVISION = "M63c-2026-09-29-r1"
REVIEWED_M63B = "0c4f2b86fb469fda0452dd0fcdc3a543b415852476019d2410d4556ac2086463"
VARIANT = "control_a2_bn"

def checkpoint_for(m, variant, epoch):
    if variant != VARIANT or epoch != 1:
        raise ValueError("Only the GT-only epoch1 normalization intervention is authorized")
    return Path(m["variants"]["control"]["run_dir"]) / "checkpoint_epoch_1.pth"

def inference(m, variant, epoch, directory):
    import torch
    from m63_common import build_runtime, seed_all
    seed_all(m["seed"])
    candidate_path = checkpoint_for(m, variant, epoch)
    candidate_sha = sha256(candidate_path)
    source_sha = sha256(m["student"]["checkpoint"])
    # build_runtime first loads the hash-verified source A2 checkpoint.
    model, _, _ = build_runtime(m, "student")
    source_statistics = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                         if k.endswith(("running_mean","running_var"))}
    payload = load_bound_checkpoint(m, "control", 1)
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    intervention = restore_running_statistics(model, source_statistics)
    if intervention["normalization_modules"] != 76:
        raise RuntimeError("Expected the 76 normalization modules observed in M63b")
    model.eval()
    before_inference = state_fingerprints(model)
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.helpers.tester_helper import Tester
    cfg = copy.deepcopy(m["student"]["config"])
    dataset = KITTI_Dataset("val", cfg["dataset"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=m["batch_size"], shuffle=False, num_workers=0)
    repo = Path(m["student"]["repo"])
    cfg["trainer"]["save_path"] = os.path.relpath(directory,repo)
    os.chdir(repo)
    tester = Tester(cfg["tester"],model,loader,logging.getLogger("M63c"),
                    train_cfg=cfg["trainer"],model_name="native")
    with torch.no_grad():
        tester.inference()
    if state_fingerprints(model) != before_inference:
        raise RuntimeError("Inference changed model state")
    if sha256(candidate_path) != candidate_sha or sha256(m["student"]["checkpoint"]) != source_sha:
        raise RuntimeError("On-disk checkpoint changed during diagnostic")
    write_json(directory / "intervention_audit.json",dict(complete=True, revision=REVISION,
        manifest_sha256=m["manifest_sha256"], candidate_checkpoint_sha256=candidate_sha,
        source_checkpoint_sha256=source_sha, optimizer_steps=0, checkpoint_written=False,
        post_inference_state_unchanged=True, **intervention))

def run(command, cwd=ROOT):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def evaluate_one(m, manifest_path, variant, epoch):
    output = Path(m["output_dir"])
    directory = output / "diagnostics_m63c" / "evaluation" / f"{variant}_epoch{epoch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(m, variant, epoch)
    prediction_dir = directory / "native/outputs/data"
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    identity = dict(manifest_sha256=m["manifest_sha256"], variant=variant, epoch=epoch,
                    checkpoint_sha256=sha256(checkpoint), intervention="source_a2_bn_running_mean_var",
                    script_sha256=sha256(Path(__file__)), helper_sha256=sha256(ROOT / "scripts/m63c_bn_intervention.py"))
    done = directory / "inference_manifest.json"
    if done.exists():
        old = read_json(done)
        if any(old.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Cached evaluation belongs to different weights/provenance")
        if (set(old["prediction_files"]) != set(ids)
                or {p.stem for p in prediction_dir.glob("*.txt")} != set(ids)):
            raise RuntimeError("Cached validation predictions are incomplete")
        for image_id, digest in old["prediction_files"].items():
            if sha256(prediction_dir / f"{image_id}.txt") != digest:
                raise RuntimeError("Changed cached prediction file")
    else:
        run([sys.executable, "-u", Path(__file__), "--manifest", manifest_path,
             "--infer", variant, "--epoch", epoch])
        if {p.stem for p in prediction_dir.glob("*.txt")} != set(ids):
            raise RuntimeError("Expected exactly 3,769 validation prediction IDs")
        identity["prediction_files"] = {i: sha256(prediction_dir / f"{i}.txt") for i in ids}
        write_json(done, identity)
    audit = read_json(directory / "intervention_audit.json")
    if (not audit["complete"] or not audit["post_inference_state_unchanged"]
            or audit["candidate_checkpoint_sha256"] != sha256(checkpoint)):
        raise RuntimeError("Missing or invalid intervention audit")
    split_dir = Path(m["dataset_root"]) / "ImageSets"
    run([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
         "--config", ROOT / "configs/kitti_mobileadas3d_s1.yaml", "--profile", "colab_drive",
         "--dataset-root", m["dataset_root"], "--split-dir", split_dir,
         "--prediction-dir", prediction_dir, "--split", "val", "--classes", "Vehicle", "Pedestrian",
         "--source-name", f"M63_{variant}_{epoch}", "--output-dir", directory / "product"])
    run([sys.executable, "-u", ROOT / "scripts/audit_product_prediction_geometry.py",
         "--dataset-root", m["dataset_root"], "--split-file", split_dir / "val.txt",
         "--prediction-dir", prediction_dir, "--output-dir", directory / "nearby",
         "--checkpoint", checkpoint, "--expected-images", "3769",
         "--score-threshold", "0.001", "--match-iou-threshold", "0.5"])
    summary = read_json(directory / "product/kitti_r40_summary.json")
    if not summary.get("complete_split") or summary["evaluated_images"] != 3769:
        raise RuntimeError("Incomplete product evaluation")
    from sweep_monodetr_r0_product_checkpoints import metric_value
    metrics = {f"{cls.lower()}_{metric}_moderate": metric_value(summary, metric, cls)
               for cls in ("Vehicle", "Pedestrian") for metric in ("3d", "bev")}
    metrics["mean_3d_moderate"] = (metrics["vehicle_3d_moderate"] + metrics["pedestrian_3d_moderate"]) / 2
    nearby = read_json(directory / "nearby/nearby_geometry_summary.json")
    for cls in ("Vehicle", "Pedestrian"):
        metrics[f"{cls.lower()}_near_recall"] = nearby["classes"][cls]["near_recall"]
    if any(not __import__("math").isfinite(v) for v in metrics.values()):
        raise RuntimeError("Non-finite evaluation metric")
    return dict(variant=variant, epoch=epoch, checkpoint_sha256=sha256(checkpoint), manifest_sha256=m["manifest_sha256"], **metrics)

def compare(baseline, control, restored):
    keys = ("vehicle_3d_moderate","pedestrian_3d_moderate","mean_3d_moderate",
            "vehicle_bev_moderate","pedestrian_bev_moderate",
            "vehicle_near_recall","pedestrian_near_recall")
    result = {}
    for k in keys:
        loss = baseline[k] - control[k]
        result[k] = dict(delta_vs_original_control=restored[k]-control[k],
                         delta_vs_source_a2=restored[k]-baseline[k],
                         fraction_of_control_regression_recovered=(restored[k]-control[k])/loss
                         if loss > 0 else None)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--infer",choices=(VARIANT,))
    p.add_argument("--epoch",type=int,choices=(1,))
    args=p.parse_args()
    m=load_manifest(args.manifest)
    root=Path(m["output_dir"])
    previous=root / "diagnostics_m63b/m63b_continuation_diagnostic.json"
    if sha256(previous) != REVIEWED_M63B:
        raise RuntimeError("M63b report differs from the reviewed diagnostic")
    old=read_json(previous)
    baseline=next(r for r in old["rows"] if r["variant"]=="baseline")
    control=next(r for r in old["rows"] if r["variant"]=="control" and r["epoch"]==1)
    if sha256(checkpoint_for(m,VARIANT,1)) != control["checkpoint_sha256"]:
        raise RuntimeError("Not the reviewed GT-only epoch1 checkpoint")
    directory=root / "diagnostics_m63c"
    identity=dict(revision=REVISION,manifest_sha256=m["manifest_sha256"],
        m63b_report_sha256=REVIEWED_M63B,script_sha256=sha256(Path(__file__)),
        helper_sha256=sha256(ROOT / "scripts/m63c_bn_intervention.py"),
        candidate_checkpoint_sha256=control["checkpoint_sha256"],
        source_checkpoint_sha256=m["student"]["checkpoint_sha256"])
    path=directory / "identity.json"
    if path.exists() and read_json(path)!=identity:
        raise RuntimeError("Existing diagnostic identity differs; preserve it")
    write_json(path,identity)
    if args.infer:
        if args.epoch != 1: p.error("--infer requires --epoch 1")
        inference(m,args.infer,1,directory / "evaluation" / f"{VARIANT}_epoch001")
        return
    restored=evaluate_one(m,args.manifest,VARIANT,1)
    report=dict(complete=True,**identity,rows=[baseline,control,restored],
        comparison=compare(baseline,control,restored),
        optimizer_steps=0,training_authorized=False,checkpoint_promotion_authorized=False,
        original_m63_pilot_passed=False,
        interpretation="Isolates inference-time contribution of BN statistics for one checkpoint. Does not prove frozen-BN training will succeed; no significance or deployment claim.")
    write_json(directory / "m63c_bn_statistics_comparison.json",report)
    print(json.dumps(report,indent=2))
    print("STOP for review. No training, saved-weight changes, or checkpoint promotion.")

if __name__=="__main__": main()
