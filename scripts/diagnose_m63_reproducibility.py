"""M63h: two fresh processes, three GT-only updates each; no candidate saved."""
from __future__ import annotations
import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import numpy as np
import torch
from m63_common import ROOT,load_manifest,seed_all,build_runtime,pack_targets,sha256,read_json,write_json,json_hash,environment,input_fingerprint,array_hash
from train_m63_student import validated_audit,approved_batch,supervised_loss,finite_gradients
from run_m63_frozen_bn_control import training_mode,loader_for
from m63d_frozen_bn import bn_fingerprints,assert_bn_frozen
from m63c_bn_intervention import state_fingerprints

STEPS=3
REVIEWED="75d88dc42a7b99b16498b820528c79155d51a8b2f3d0f1a3c275f850d09fc66d"

def tensor_hash(t):
    t=t.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(t.dtype),list(t.shape))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()

def rng_snapshot():
    n=np.random.get_state()
    return dict(python=json_hash(random.getstate()),
                numpy=json_hash([n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])]),
                torch_cpu=tensor_hash(torch.get_rng_state()),
                torch_cuda=[tensor_hash(t) for t in torch.cuda.get_rng_state_all()])

def backend_settings():
    return dict(environment=environment(),cudnn_version=torch.backends.cudnn.version(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"))

def compare_tensors(a,b):
    if set(a)!=set(b): raise RuntimeError("Tensor keys differ between replicas")
    out={}
    for key,x in a.items():
        y=b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype: raise RuntimeError("Tensor layout differs: "+key)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise RuntimeError("Nonfinite "+key)
        delta=(x.double()-y.double()).abs()
        out[key]=dict(exact=torch.equal(x,y),max_abs=float(delta.max()) if delta.numel() else 0.,
                      mean_abs=float(delta.mean()) if delta.numel() else 0.)
    return dict(all_exact=all(v["exact"] for v in out.values()),
                changed_tensors=sum(not v["exact"] for v in out.values()),tensors=out)

def worker(m,destination,replica):
    seed_all(m["seed"])
    model,criterion,dataset=build_runtime(m,"student")
    initial=state_fingerprints(model);bn=bn_fingerprints(model)
    from lib.helpers.optimizer_helper import build_optimizer
    optimizer=build_optimizer(m["student"]["config"]["optimizer"],model)
    if any(g["lr"]!=1e-5 for g in optimizer.param_groups): raise RuntimeError("Expected LR1e-5")
    seed_all(m["seed"])
    loader=loader_for(dataset,m)
    training_mode(model);criterion.train()
    report=dict(replica=replica,initial_state_sha256=json_hash(initial),
                backend=backend_settings(),steps=[],optimizer_steps=0,complete=False)
    destination.mkdir(parents=True,exist_ok=True)
    for index,(images,calibs,raw,info) in enumerate(islice(loader,STEPS)):
        targets=pack_targets(raw,"cuda")
        approved_batch(m,images,calibs,raw,info,targets)
        optimizer.zero_grad(set_to_none=True)
        before=rng_snapshot()
        outputs=model(images.cuda(),calibs.cuda(),targets,raw["img_size"].cuda(),dn_args=None)
        loss,parts=supervised_loss(criterion,outputs,targets)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
        output_tensors={k:v.detach().cpu().clone() for k,v in outputs.items() if isinstance(v,torch.Tensor)}
        loss.backward()
        if not finite_gradients(model): raise RuntimeError("Invalid gradients")
        gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
        after_backward=rng_snapshot()
        optimizer.step()
        assert_bn_frozen(model,bn)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        step=dict(step=index+1,sample_ids=[int(i) for i in info["img_id"]],
            input_sha256=[input_fingerprint(images[i],calibs[i],raw["img_size"][i]) for i in range(len(images))],
            target_sha256=[array_hash(t) for t in targets],rng_before_forward=before,
            rng_after_backward=after_backward,gt_loss=float(loss.detach()),
            losses={k:float(v.detach()) for k,v in parts.items()},
            outputs_sha256=json_hash({k:tensor_hash(v) for k,v in output_tensors.items()}),
            gradients_sha256=json_hash({k:tensor_hash(v) for k,v in gradients.items()}),
            updated_state_sha256=json_hash({k:tensor_hash(v) for k,v in state.items()}))
        # Temporary CPU tensors only; parent removes them after comparison.
        torch.save(dict(outputs=output_tensors,gradients=gradients,state=state),destination/f"step{index+1}.pt")
        report["steps"].append(step);report["optimizer_steps"]+=1
        print(f"M63h replica={replica} step={index+1}/{STEPS} GT={step['gt_loss']:.6f}",flush=True)
    if report["optimizer_steps"]!=STEPS: raise RuntimeError("Incorrect diagnostic budget")
    report.update(complete=True,bn_state_unchanged=True)
    write_json(destination/"trace.json",report)

def compare_replicas(a,b,folder_a,folder_b):
    if len(a["steps"])!=STEPS or len(b["steps"])!=STEPS:
        raise RuntimeError("Incomplete replica traces")
    comparisons=[]
    for index,(x,y) in enumerate(zip(a["steps"],b["steps"]),1):
        ax=torch.load(folder_a/f"step{index}.pt",map_location="cpu",weights_only=True)
        bx=torch.load(folder_b/f"step{index}.pt",map_location="cpu",weights_only=True)
        comparisons.append(dict(step=index,sample_ids_equal=x["sample_ids"]==y["sample_ids"],
            inputs_equal=x["input_sha256"]==y["input_sha256"],targets_equal=x["target_sha256"]==y["target_sha256"],
            rng_before_equal=x["rng_before_forward"]==y["rng_before_forward"],
            rng_after_backward_equal=x["rng_after_backward"]==y["rng_after_backward"],
            gt_loss_abs_delta=abs(x["gt_loss"]-y["gt_loss"]),
            numerical={k:compare_tensors(ax[k],bx[k]) for k in ("outputs","gradients","state")}))
    return comparisons

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--worker",choices=("a","b"))
    p.add_argument("--temporary-output",type=Path)
    args=p.parse_args()
    m=load_manifest(args.manifest)
    if args.worker:
        if args.temporary_output is None: p.error("--worker requires --temporary-output")
        worker(m,args.temporary_output,args.worker);return
    reviewed=Path(m["output_dir"])/"diagnostics_m63g/m63g_diagnostic.json"
    if sha256(reviewed)!=REVIEWED: raise RuntimeError("Not reviewed M63g evidence")
    validated_audit(m)
    if m["seed"]!=20268 or m["batch_size"]!=4 or m["augmentation"] is not False:
        raise RuntimeError("Diagnostic settings changed")
    output=Path(m["output_dir"])/"diagnostics_m63h"
    identity=dict(revision="M63h-2026-10-02-r1",manifest_sha256=m["manifest_sha256"],
        reviewed_m63g_sha256=REVIEWED,steps_per_replica=STEPS,replicas=2,
        script_sha256=sha256(Path(__file__)),environment=environment(),
        dependencies={p:sha256(ROOT/p) for p in ("scripts/run_m63_frozen_bn_control.py",
                      "scripts/m63d_frozen_bn.py","scripts/m63c_bn_intervention.py")},
        checkpoint_promotion_authorized=False,full_run_authorized=False)
    identity_path=output/"identity.json"
    if identity_path.exists() and read_json(identity_path)!=identity:
        raise RuntimeError("Existing M63h identity changed")
    write_json(identity_path,identity)
    completed=output/"m63h_reproducibility.json"
    if completed.exists():
        old=read_json(completed)
        if old.get("complete") and old.get("identity_sha256")==json_hash(identity):
            print("M63h already complete; no additional optimizer steps.");return
        raise RuntimeError("Unexpected existing report; preserve for review")
    with tempfile.TemporaryDirectory(prefix="m63h_") as temporary:
        root=Path(temporary)
        traces={}
        for replica in ("a","b"):
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--manifest",str(args.manifest),
                "--worker",replica,"--temporary-output",str(root/replica)],check=True,cwd=ROOT)
            traces[replica]=read_json(root/replica/"trace.json")
            write_json(output/f"trace_{replica}.json",traces[replica])
        comparisons=compare_replicas(traces["a"],traces["b"],root/"a",root/"b")
        report=dict(complete=True,identity_sha256=json_hash(identity),optimizer_steps_total=2*STEPS,
            initial_states_equal=traces["a"]["initial_state_sha256"]==traces["b"]["initial_state_sha256"],
            backend_settings_equal=traces["a"]["backend"]==traces["b"]["backend"],
            comparisons=comparisons,checkpoint_saved_for_promotion=False,
            full_run_authorized=False,distillation_authorized=False,
            caveat="Three instrumented GT-only batches cannot prove full-run determinism or identify a specific faulty kernel.")
        write_json(completed,report)
    print(f"Return {completed}. STOP; temporary tensors deleted, no candidate promoted.")

if __name__=="__main__": main()
