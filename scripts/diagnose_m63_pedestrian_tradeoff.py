"""M63g: read-only pedestrian paired-error and control-state audit; no inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from m61_common import sha256, json_hash, read_json, write_json, check_split
from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING, map_objects
from data.kitti_parser import parse_kitti_label_file
from data.kitti_prediction_parser import parse_kitti_prediction_file
from data.matching import bbox_iou
from audit_product_prediction_geometry import match_class, geometry_row, resolve_label_dir
from diagnose_a2_pedestrian_false_negatives import kitti_difficulty

REVIEWED="e5addd27afed6faa19dd271a8b99b1494a3c173bdd51fe4e7d84932de8d68eb2"
FIELDS=("score","iou_2d","iou_3d","iou_bev","depth_abs_error_m",
        "dimension_mae_m","yaw_abs_error_deg","loc_x_abs_error_m",
        "loc_y_abs_error_m","loc_z_abs_error_m","center3d_error_m")

def sample_rows(sample_id, targets, predictions):
    """Stable IDs use mapped label order, not prediction order."""
    gt=[dict(t,gt_id=f"{sample_id}:{i}") for i,t in enumerate(targets)
        if t["class_name"]=="Pedestrian"]
    preds=[p for p in predictions if p["class_name"]=="Pedestrian"]
    matches,fp,_=match_class(preds,gt,0.001,0.5)
    matched={t["gt_id"]:(p,t,iou) for p,t,iou in matches}
    rows={}
    for t in gt:
        key=t["gt_id"]
        row=dict(gt_id=key,sample_id=sample_id,gt_depth_m=float(t["location_3d"][2]),
                 near_field=float(t["location_3d"][2])<30,
                 difficulty=kitti_difficulty(t),matched=key in matched)
        if key in matched:
            p,_,overlap=matched[key]
            row.update(geometry_row(sample_id,"Pedestrian",p,t,overlap))
            row["status"]="matched"
        else:
            overlaps=[(bbox_iou(p["bbox_2d"],t["bbox_2d"]),p) for p in preds]
            good=[p for ov,p in overlaps if ov>=0.5]
            if any(float(p["score"])>=0.001 for p in good): reason="one_to_one_competition"
            elif good: reason="below_score_floor_in_saved_predictions"
            elif any(p["class_name"]!="Pedestrian" and float(p["score"])>=0.001
                     and bbox_iou(p["bbox_2d"],t["bbox_2d"])>=0.5 for p in predictions):
                reason="other_class_overlap_candidate"
            else: reason="no_overlapping_pedestrian_in_saved_predictions"
            row["status"]=reason
        rows[key]=row
    thresholds={}
    for threshold in (0.001,0.1,0.3,0.5):
        pairs,nfp,_=match_class(preds,gt,threshold,0.5)
        thresholds[str(threshold)]=dict(matched=len(pairs),fp=nfp,gt=len(gt),
            near_matched=sum(float(t["location_3d"][2])<30 for _,t,_ in pairs),
            near_gt=sum(float(t["location_3d"][2])<30 for t in gt))
    return rows,thresholds

def averages(rows):
    return {k:float(np.mean([r[k] for r in rows])) if rows else None for k in FIELDS}

def compare(source,candidate):
    if set(source)!=set(candidate): raise RuntimeError("GT identity sets differ")
    result={}
    for scope in ("all","moderate_including_easy","near"):
        keys=[k for k,r in source.items() if scope=="all"
              or (scope=="moderate_including_easy" and r["difficulty"] in ("easy","moderate"))
              or (scope=="near" and r["near_field"])]
        common=[k for k in keys if source[k]["matched"] and candidate[k]["matched"]]
        a=averages([source[k] for k in common]);b=averages([candidate[k] for k in common])
        result[scope]=dict(gt=len(keys),common_matched=len(common),
            lost_detections=sum(source[k]["matched"] and not candidate[k]["matched"] for k in keys),
            gained_detections=sum(not source[k]["matched"] and candidate[k]["matched"] for k in keys),
            source_common=a,candidate_common=b,
            delta_on_common={k:b[k]-a[k] if a[k] is not None else None for k in FIELDS},
            lost_3d_iou_0_5=sum(source[k]["iou_3d"]>=.5 and candidate[k]["iou_3d"]<.5 for k in common),
            gained_3d_iou_0_5=sum(source[k]["iou_3d"]<.5 and candidate[k]["iou_3d"]>=.5 for k in common))
    return result

def state_difference(a,b):
    if set(a)!=set(b): raise RuntimeError("Checkpoint state keys differ")
    groups=defaultdict(lambda:dict(tensors=0,changed=0,numel=0,sum_abs=0.,max_abs=0.))
    for key,x in a.items():
        y=b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype: raise RuntimeError(key)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise RuntimeError("Nonfinite "+key)
        group="bn_buffers" if key.endswith(("running_mean","running_var","num_batches_tracked")) else key.split(".")[0]
        g=groups[group];delta=(x.double()-y.double()).abs()
        g["tensors"]+=1;g["changed"]+=int(not torch.equal(x,y));g["numel"]+=x.numel()
        g["sum_abs"]+=float(delta.sum());g["max_abs"]=max(g["max_abs"],float(delta.max()) if delta.numel() else 0.)
    return {k:dict(v,mean_abs=v["sum_abs"]/max(v["numel"],1)) for k,v in groups.items()}

def validate_predictions(directory, ids, checkpoint_sha, manifest_sha):
    info=read_json(directory/"inference_manifest.json")
    if info["checkpoint_sha256"]!=checkpoint_sha or info["manifest_sha256"]!=manifest_sha:
        raise RuntimeError("Prediction lineage mismatch")
    data=directory/"native/outputs/data"
    if set(info["prediction_files"])!=set(ids) or {p.stem for p in data.glob("*.txt")}!=set(ids):
        raise RuntimeError("Incomplete prediction IDs")
    for i in ids:
        if sha256(data/f"{i}.txt")!=info["prediction_files"][i]:
            raise RuntimeError(f"Changed prediction {i}")
    return data,info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    args=p.parse_args()
    m=read_json(args.manifest)
    copied=dict(m);saved=copied.pop("manifest_sha256")
    if json_hash(copied)!=saved: raise RuntimeError("Manifest self-hash mismatch")
    root=Path(m["output_dir"]);report_path=root/"m63f_frozen_bn_kd/m63f_comparison.json"
    if sha256(report_path)!=REVIEWED: raise RuntimeError("Not reviewed M63f")
    prior=read_json(report_path)
    if any(r["manifest_sha256"]!=saved for r in prior["rows"]):
        raise RuntimeError("Reviewed report belongs to another manifest")
    rows={r["variant"]:r for r in prior["rows"]}
    old={r["variant"]:r for r in prior["historical_rows"]}
    specs={
        "a2":(root/"evaluation/baseline_epoch000",Path(m["student"]["checkpoint"]),rows["baseline"]),
        "m63d":(root/"m63d_frozen_bn/evaluation/control_frozen_bn_epoch001",
                root/"m63d_frozen_bn/run/checkpoint_epoch_1.pth",old["control_frozen_bn"]),
        "m63f_control":(root/"m63f_frozen_bn_kd/evaluation/control_epoch001",
                       root/"m63f_frozen_bn_kd/runs/control/checkpoint_epoch_1.pth",rows["control"]),
        "m63f_kd":(root/"m63f_frozen_bn_kd/evaluation/vehicle_kd_epoch001",
                  root/"m63f_frozen_bn_kd/runs/vehicle_kd/checkpoint_epoch_1.pth",rows["vehicle_kd"])}
    ids=check_split(Path(m["dataset_root"])/"ImageSets/val.txt","val")
    label_dir=resolve_label_dir(Path(m["dataset_root"]))
    ground_truth={i:map_objects([asdict(t) for t in parse_kitti_label_file(label_dir/f"{i}.txt")],
                               KITTI_PRODUCTION_CLASS_MAPPING) for i in ids}
    all_rows={};summaries={};prediction_hashes={};states={}
    for name,(directory,checkpoint,row) in specs.items():
        print(f"{name}: checking saved predictions and checkpoint",flush=True)
        if sha256(checkpoint)!=row["checkpoint_sha256"]: raise RuntimeError("Changed "+name+" checkpoint")
        data,info=validate_predictions(directory,ids,row["checkpoint_sha256"],saved)
        prediction_hashes[name]=json_hash(info["prediction_files"])
        model_rows={};counts=defaultdict(Counter)
        for index,i in enumerate(ids):
            preds=map_objects(parse_kitti_prediction_file(data/f"{i}.txt"),dict(KITTI_PRODUCTION_CLASS_MAPPING,Vehicle="Vehicle"))
            r,c=sample_rows(i,ground_truth[i],preds);model_rows.update(r)
            for threshold,v in c.items(): counts[threshold].update(v)
            if (index+1)%500==0: print(f"{name}: {index+1}/{len(ids)}",flush=True)
        all_rows[name]=model_rows
        summaries[name]=dict(status_counts=dict(Counter(r["status"] for r in model_rows.values())),
            score_threshold_counts={k:dict(v) for k,v in counts.items()},
            all_matched_geometry=averages([r for r in model_rows.values() if r["matched"]]))
        states[name]=torch.load(checkpoint,map_location="cpu",weights_only=True)["model_state"]
    pairs=(("a2","m63d"),("a2","m63f_control"),("a2","m63f_kd"),
           ("m63d","m63f_control"),("m63f_control","m63f_kd"))
    changes={a+"_to_"+b:compare(all_rows[a],all_rows[b]) for a,b in pairs}
    state_changes={a+"_to_"+b:state_difference(states[a],states[b]) for a,b in pairs}
    settings={}
    for name,path in (("m63d",root/"m63d_frozen_bn/identity.json"),
                      ("m63f",root/"m63f_frozen_bn_kd/identity.json")):
        settings[name]=read_json(path)
    if json_hash(settings["m63f"])!=prior["binding_sha256"]:
        raise RuntimeError("M63f settings identity mismatch")
    d_report=root/"m63d_frozen_bn/m63d_comparison.json"
    if sha256(d_report)!="b881bfa08900b0ac82fdaa4d742ce17b3dc2e8e97d3f2f4fbfc0506b7dbdf1e8":
        raise RuntimeError("M63d report changed")
    if json_hash(settings["m63d"])!=read_json(d_report)["binding_sha256"]:
        raise RuntimeError("M63d settings identity mismatch")
    compared_keys=("environment","source_checkpoint_sha256","seed","batch_size",
                   "learning_rate","optimizer","augmentation","bn_mode")
    settings_equal={k:settings["m63d"].get(k)==settings["m63f"].get(k) for k in compared_keys}
    output=root/"diagnostics_m63g"
    identity=dict(revision="M63g-2026-10-01-r1",reviewed_report_sha256=REVIEWED,
        manifest_sha256=saved,script_sha256=sha256(Path(__file__)),
        dependency_sha256={str(p):sha256(ROOT/p) for p in (
            "scripts/audit_product_prediction_geometry.py","scripts/diagnose_a2_pedestrian_false_negatives.py",
            "data/kitti_prediction_parser.py","data/kitti_parser.py","data/kitti_r40.py","data/matching.py","data/class_taxonomy.py")},
        label_tree_sha256=json_hash({i:sha256(label_dir/f"{i}.txt") for i in ids}),
        prediction_tree_sha256=prediction_hashes,
        checkpoint_sha256={n:r["checkpoint_sha256"] for n,(_,_,r) in specs.items()})
    if (output/"identity.json").exists() and read_json(output/"identity.json")!=identity:
        raise RuntimeError("Diagnostic identity changed; preserve previous outputs")
    report=dict(complete=True,identity_sha256=json_hash(identity),samples=len(ids),
        summaries=summaries,paired_comparisons=changes,checkpoint_state_differences=state_changes,
        recorded_control_settings_equal=settings_equal,
        training_performed=False,inference_performed=False,checkpoint_promotion_authorized=False,
        caveats=["Diagnostic 2D matching, not official AP attribution; DontCare exclusions not applied.",
                 "Threshold views describe saved/top-k-limited predictions; no threshold is selected.",
                 "Geometry deltas use common GT matches; localization uses KITTI bottom-center coordinates.",
                 "Different weights prove training endpoints differ, not the cause; RNG/kernel histories were not recorded.",
                 "No repeat training or repeat inference performed; variation cannot be assigned to nondeterminism alone."])
    write_json(output/"identity.json",identity)
    write_json(output/"m63g_diagnostic.json",report)
    write_json(output/"m63g_pedestrian_objects.json",all_rows)
    print(json.dumps(dict(complete=True,output=str(output),recorded_control_settings_equal=settings_equal),indent=2))
    print("STOP for review. No training, inference, or promotion.")

if __name__=="__main__": main()
