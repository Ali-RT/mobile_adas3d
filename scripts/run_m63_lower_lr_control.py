"""M63e: one GT-only epoch from original A2; frozen BN, learning rate reduced to 1e-6."""
from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import torch
from m63_common import (ROOT, build_runtime, check_split, environment, json_hash,
    load_manifest, pack_targets, read_json, require_baseline, seed_all, sha256, write_json)
from train_m63_student import (approved_batch, finite_gradients, save_checkpoint,
    supervised_loss, validated_audit)
from m63c_bn_intervention import state_fingerprints
from m63d_frozen_bn import freeze_bn_statistics, bn_fingerprints, assert_bn_frozen

REVISION="M63e-2026-09-30-r1"
REVIEWED_M63D="b881bfa08900b0ac82fdaa4d742ce17b3dc2e8e97d3f2f4fbfc0506b7dbdf1e8"
VARIANT="control_frozen_bn_lr1e6"

def output_root(m):
    return Path(m["output_dir"]) / "m63e_frozen_bn_lr"

def checkpoint_for(m,variant,epoch):
    if variant!=VARIANT or epoch!=1:
        raise ValueError("Only one GT-only frozen-BN epoch is authorized")
    return output_root(m) / "run/checkpoint_epoch_1.pth"

def bind(m):
    previous=Path(m["output_dir"]) / "m63d_frozen_bn/m63d_comparison.json"
    if sha256(previous)!=REVIEWED_M63D:
        raise RuntimeError("Not the reviewed M63d report")
    b=dict(revision=REVISION,manifest_sha256=m["manifest_sha256"],
        m63d_report_sha256=REVIEWED_M63D,script_sha256=sha256(Path(__file__)),
        helper_sha256=sha256(ROOT/"scripts/m63d_frozen_bn.py"),
        state_helper_sha256=sha256(ROOT/"scripts/m63c_bn_intervention.py"),
        audit_sha256=sha256(Path(m["output_dir"])/"m63_teacher_audit.json"),
        baseline_sha256=require_baseline(m),environment=environment(),
        source_checkpoint_sha256=m["student"]["checkpoint_sha256"],
        variant=VARIANT,epochs=1,batch_size=4,seed=20268,learning_rate=1e-6,
        optimizer="fresh native AdamW",augmentation=False,distillation=False,
        bn_mode="eval during training; affine trainability unchanged")
    if (m["seed"]!=20268 or m["batch_size"]!=4 or m["learning_rate"]!=1e-5
            or m["augmentation"] is not False):
        raise RuntimeError("M63 comparison settings changed")
    path=output_root(m)/"identity.json"
    if path.exists() and read_json(path)!=b:
        raise RuntimeError("Existing M63e identity differs; preserve it")
    write_json(path,b)
    return b

def load_completed(m,binding):
    path=checkpoint_for(m,VARIANT,1)
    digest=path.with_suffix(".sha256")
    if digest.exists() and digest.read_text().strip()!=sha256(path):
        raise RuntimeError("Checkpoint checksum mismatch; inspect before resuming")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["m63e"]!=binding or payload["epoch"]!=1:
        raise RuntimeError("M63e checkpoint lineage mismatch")
    if not digest.exists(): digest.write_text(sha256(path)+"\n")
    return payload

def training_mode(model):
    model.train()
    names=freeze_bn_statistics(model)
    if len(names)!=76: raise RuntimeError("Expected 76 A2 BatchNorm modules")
    assert_bn_frozen(model)
    return names

def loader_for(dataset,m):
    generator=torch.Generator().manual_seed(m["seed"])
    return torch.utils.data.DataLoader(dataset,batch_size=m["batch_size"],shuffle=True,
        generator=generator,num_workers=0,drop_last=False)

def smoke(m,binding):
    validated_audit(m)
    seed_all(m["seed"])
    model,criterion,dataset=build_runtime(m,"student")
    before=state_fingerprints(model)
    expected=bn_fingerprints(model)
    seed_all(m["seed"])
    loader=loader_for(dataset,m)
    training_mode(model)
    criterion.train()
    images,calibs,raw,info=next(iter(loader))
    targets=pack_targets(raw,"cuda")
    approved_batch(m,images,calibs,raw,info,targets)
    outputs=model(images.cuda(),calibs.cuda(),targets,raw["img_size"].cuda(),dn_args=None)
    loss,_=supervised_loss(criterion,outputs,targets)
    if not torch.isfinite(loss): raise RuntimeError("Non-finite smoke GT loss")
    loss.backward()
    if not finite_gradients(model): raise RuntimeError("Invalid smoke gradients")
    assert_bn_frozen(model,expected)
    if state_fingerprints(model)!=before:
        raise RuntimeError("Smoke unexpectedly changed model state")
    report=dict(complete=True,binding_sha256=json_hash(binding),optimizer_steps=0,
        finite_gradients=True,bn_state_unchanged=True,all_model_state_unchanged=True,
        gt_loss=float(loss.detach()),batch_size=len(images),normalization_modules=76)
    write_json(output_root(m)/"m63e_smoke.json",report)
    print(json.dumps(report,indent=2))

def lower_lr_optimizer_config(m):
    """Copy only optimizer LR; never mutate the reviewed manifest."""
    cfg=copy.deepcopy(m["student"]["config"]["optimizer"])
    if cfg["lr"] != 1e-5 or cfg["type"].lower() != "adamw":
        raise RuntimeError("Expected original M63 optimizer settings")
    cfg["lr"]=1e-6
    return cfg

def train(m,binding):
    audit=validated_audit(m)
    report=read_json(output_root(m)/"m63e_smoke.json")
    if not report["complete"] or report["binding_sha256"]!=json_hash(binding):
        raise RuntimeError("Matching M63e CUDA smoke required")
    seed_all(m["seed"])
    model,criterion,dataset=build_runtime(m,"student")
    expected=bn_fingerprints(model)
    checkpoint=checkpoint_for(m,VARIANT,1)
    if checkpoint.exists():
        payload=load_completed(m,binding)
        model.load_state_dict(payload["model_state"],strict=True)
        training_mode(model)
        assert_bn_frozen(model,expected)
        write_json(output_root(m)/"m63e_training_summary.json",payload["summary"])
        print("One epoch already complete; no additional training performed.")
        return
    from lib.helpers.optimizer_helper import build_optimizer
    optimizer_config=lower_lr_optimizer_config(m)
    optimizer=build_optimizer(optimizer_config,model)
    if any(group["lr"] != 1e-6 for group in optimizer.param_groups):
        raise RuntimeError("All optimizer groups must use LR 1e-6")
    seed_all(m["seed"])
    loader=loader_for(dataset,m)
    training_mode(model)
    criterion.train()
    loss_sum=0.
    for index,(images,calibs,raw,info) in enumerate(loader):
        assert_bn_frozen(model)
        targets=pack_targets(raw,"cuda")
        approved_batch(m,images,calibs,raw,info,targets)
        optimizer.zero_grad(set_to_none=True)
        outputs=model(images.cuda(),calibs.cuda(),targets,raw["img_size"].cuda(),dn_args=None)
        loss,parts=supervised_loss(criterion,outputs,targets)
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at batch {index+1}")
        loss.backward()
        if not finite_gradients(model): raise RuntimeError("Invalid gradients; no update performed")
        optimizer.step()
        loss_sum+=float(loss.detach())
        if index%20==0 or index+1==len(loader):
            assert_bn_frozen(model,expected)
            print(f"M63e epoch=1/1 batch={index+1}/{len(loader)} "
                  f"lr={optimizer.param_groups[0]['lr']:.8f} gt={float(loss.detach()):.6f} "
                  "kd=0 BN=frozen",flush=True)
    assert_bn_frozen(model,expected)
    if len(loader)!=928: raise RuntimeError("Expected exactly 928 optimizer steps")
    summary=dict(complete=True,epoch=1,optimizer_steps=len(loader),gt_loss=loss_sum/len(loader),
        learning_rate=1e-6,kd_loss=0.,binding_sha256=json_hash(binding),bn_state_unchanged=True,
        source_bn_sha256=expected,final_bn_sha256=bn_fingerprints(model),
        full_run_authorized=False,checkpoint_promotion_authorized=False)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(checkpoint,dict(epoch=1,model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),m63e=binding,summary=summary,
        best_result=0.,best_epoch=0))
    checkpoint.with_suffix(".sha256").write_text(sha256(checkpoint)+"\n")
    write_json(output_root(m)/"m63e_training_summary.json",summary)
    print(f"Saved {checkpoint}. STOP training; evaluate the fixed epoch1.")

def inference(m,variant,epoch,directory):
    checkpoint=checkpoint_for(m,variant,epoch)
    binding=bind(m)
    seed_all(m["seed"])
    model,_,_=build_runtime(m,"student")
    expected=bn_fingerprints(model)
    payload=load_completed(m,binding)
    model.load_state_dict(payload["model_state"],strict=True)
    model.eval()
    assert_bn_frozen(model,expected)
    before=state_fingerprints(model)
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.helpers.tester_helper import Tester
    cfg=copy.deepcopy(m["student"]["config"])
    dataset=KITTI_Dataset("val",cfg["dataset"])
    loader=torch.utils.data.DataLoader(dataset,batch_size=m["batch_size"],shuffle=False,num_workers=0)
    repo=Path(m["student"]["repo"])
    cfg["trainer"]["save_path"]=os.path.relpath(directory,repo)
    os.chdir(repo)
    tester=Tester(cfg["tester"],model,loader,logging.getLogger("M63e"),
                  train_cfg=cfg["trainer"],model_name="native")
    with torch.no_grad(): tester.inference()
    if state_fingerprints(model)!=before: raise RuntimeError("Inference changed model state")
    write_json(directory/"frozen_bn_eval_audit.json",dict(complete=True,
        candidate_checkpoint_sha256=sha256(checkpoint),post_inference_state_unchanged=True,
        bn_state_matches_source=True,binding_sha256=json_hash(binding)))

def run(command, cwd=ROOT):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def evaluate_one(m, manifest_path, variant, epoch):
    output = Path(m["output_dir"])
    directory = output / "m63e_frozen_bn_lr" / "evaluation" / f"{variant}_epoch{epoch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(m, variant, epoch)
    prediction_dir = directory / "native/outputs/data"
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    identity = dict(manifest_sha256=m["manifest_sha256"], variant=variant, epoch=epoch,
                    checkpoint_sha256=sha256(checkpoint), intervention="gt_only_one_epoch_bn_frozen_lr1e6",
                    script_sha256=sha256(Path(__file__)), helper_sha256=sha256(ROOT / "scripts/m63d_frozen_bn.py"))
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
    audit = read_json(directory / "frozen_bn_eval_audit.json")
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke",action="store_true")
    action.add_argument("--train",action="store_true")
    action.add_argument("--evaluate",action="store_true")
    action.add_argument("--infer",choices=(VARIANT,))
    p.add_argument("--epoch",type=int,choices=(1,))
    args=p.parse_args()
    m=load_manifest(args.manifest)
    binding=bind(m)
    if args.smoke: smoke(m,binding); return
    if args.train: train(m,binding); return
    if args.infer:
        if args.epoch!=1: p.error("--infer requires --epoch 1")
        inference(m,args.infer,1,output_root(m)/"evaluation"/f"{VARIANT}_epoch001")
        return
    new=evaluate_one(m,args.manifest,VARIANT,1)
    old=read_json(Path(m["output_dir"])/"m63d_frozen_bn/m63d_comparison.json")
    keys=("vehicle_3d_moderate","pedestrian_3d_moderate","mean_3d_moderate",
          "vehicle_bev_moderate","pedestrian_bev_moderate","vehicle_near_recall","pedestrian_near_recall")
    deltas={row["variant"]:{k:new[k]-row[k] for k in keys} for row in old["rows"]}
    report=dict(complete=True,binding_sha256=json_hash(binding),rows=old["rows"]+[new],
        deltas_vs_comparators=deltas,decision_epoch=1,
        original_m63_pilot_passed=False,full_run_authorized=False,
        checkpoint_promotion_authorized=False,distillation_authorized=False,
        interpretation="Single-seed one-factor continuation diagnostic; review before any KD or longer run.")
    write_json(output_root(m)/"m63e_comparison.json",report)
    print(json.dumps(report,indent=2))
    print("STOP for review. No further epochs or distillation authorized.")

if __name__=="__main__": main()
