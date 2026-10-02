import ast
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_m63_frozen_bn_distillation as workflow
from run_m63_frozen_bn_distillation import paired_loss, checkpoint_for, load_completed
from train_m63_student import save_checkpoint
from evaluate_m63_pilot import decide

class FrozenBNKDTests(unittest.TestCase):
    def test_control_is_exact_same_tensor(self):
        gt=torch.tensor(2.,requires_grad=True)
        total,kd,count=paired_loss("control",gt,None,None,None,None)
        self.assertIs(total,gt)
        self.assertEqual((kd,count),(0.,0))

    def test_only_vehicle_depth_gets_kd_gradient(self):
        depth=torch.tensor([[[10.,0.],[8.,0.]]],requires_grad=True)
        criterion=SimpleNamespace(group_num=1,matcher=lambda *a,**k: [(torch.tensor([0,1]),torch.tensor([0,1]))])
        targets=[{"labels":torch.tensor([1,0])}]
        approved=[{"masks":np.array([[True,False,False,False],[False,False,False,False]]),
                   "pred_depth":np.array([[9.,0.],[2.,0.]])}]
        gt=depth.sum()*0
        total,kd,count=paired_loss("vehicle_kd",gt,criterion,{"pred_depth":depth},targets,approved)
        total.backward()
        self.assertEqual(count,1)
        self.assertAlmostEqual(kd,.25)
        self.assertEqual(float(depth.grad[0,1].abs().sum()),0.)
        self.assertAlmostEqual(float(depth.grad[0,0,0]),.25)
        with self.assertRaises(ValueError):
            paired_loss("other",gt,None,None,None,None)

    def test_checkpoint_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            m={"output_dir":temp}; binding={"revision":"test"}
            for variant in ("control","vehicle_kd"):
                path=checkpoint_for(m,variant,1); path.parent.mkdir(parents=True)
                save_checkpoint(path,dict(epoch=1,variant=variant,m63f=binding))
                self.assertEqual(load_completed(m,binding,variant)["variant"],variant)
            self.assertNotEqual(checkpoint_for(m,"control",1),checkpoint_for(m,"vehicle_kd",1))
            with self.assertRaises(ValueError): checkpoint_for(m,"control",2)
            with self.assertRaises(RuntimeError): load_completed(m,{},"control")

    def test_weak_control_cannot_justify_promotion(self):
        keys=("vehicle_3d_moderate","pedestrian_3d_moderate","mean_3d_moderate",
              "vehicle_bev_moderate","pedestrian_bev_moderate","vehicle_near_recall","pedestrian_near_recall")
        source=dict.fromkeys(keys,1.)
        control=dict.fromkeys(keys,.8)
        student=dict.fromkeys(keys,.9); student["vehicle_3d_moderate"]=1.2
        self.assertFalse(all(decide(source,control,student).values()))
        student=dict(source); student["vehicle_3d_moderate"]=1.2
        self.assertTrue(all(decide(source,control,student).values()))

    def test_notebook_fixed_actions(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"][-6:-2]
        for code in cells: ast.parse(code)
        for code,action in zip(cells,("--smoke","'control'","'vehicle_kd'","--evaluate")):
            self.assertIn(action,code)
            self.assertIn("run_m63_frozen_bn_distillation.py",code)
        self.assertIn("m63f_results.zip",cells[-1])
        script=(ROOT/"scripts/run_m63_frozen_bn_distillation.py").read_text()
        tree=ast.parse(script)
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=="VARIANT" for n in ast.walk(tree)))
        self.assertIn("seed=20268,learning_rate=1e-5",script)
        self.assertIn("kd_weight=0.25",script)


    def test_smoke_executes_without_updates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.depth=torch.nn.Parameter(torch.tensor([[[10.,0.]]]))
            def forward(self,*args,**kwargs):
                return {"pred_depth":self.depth}
        model=Model()
        criterion=SimpleNamespace(train=lambda:None,group_num=1,
            matcher=lambda *a,**k:[(torch.tensor([0]),torch.tensor([0]))])
        targets=[{"labels":torch.tensor([1])}]
        approved=[{"masks":np.array([[True,False,False,False]]),"pred_depth":np.array([[9.,0.]])}]
        batch=(torch.zeros(1),torch.zeros(1),{"img_size":torch.zeros(1)},{})
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            replacements={
                "validated_audit":lambda m:{},
                "build_runtime":lambda *a:(model,criterion,[]),
                "loader_for":lambda *a:[batch],
                "training_mode":lambda m:m.train(),
                "bn_fingerprints":lambda m:{},
                "assert_bn_frozen":lambda *a:None,
                "pack_targets":lambda *a:targets,
                "approved_batch":lambda *a:approved,
                "supervised_loss":lambda *a:(model.depth.sum()*0+1,{})}
            for name,value in replacements.items():
                stack.enter_context(patch.object(workflow,name,value))
            stack.enter_context(patch.object(torch.Tensor,"cuda",lambda self:self))
            workflow.smoke({"seed":20268,"output_dir":temp},{})
            report=json.loads((Path(temp)/"m63f_frozen_bn_kd/m63f_smoke.json").read_text())
            self.assertTrue(report["teacher_gradient_nonzero"])
            self.assertEqual(report["optimizer_steps"],0)
            self.assertEqual(float(model.depth[0,0,0]),10.)

    def test_evaluation_writes_gate_decision(self):
        keys=("vehicle_3d_moderate","pedestrian_3d_moderate","mean_3d_moderate",
              "vehicle_bev_moderate","pedestrian_bev_moderate","vehicle_near_recall","pedestrian_near_recall")
        baseline=dict.fromkeys(keys,1.); baseline["variant"]="baseline"
        control=dict(baseline,variant="control")
        kd=dict(baseline,variant="vehicle_kd",vehicle_3d_moderate=1.2)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            m={"output_dir":temp}
            stack.enter_context(patch.object(sys,"argv",["test","--manifest","unused","--evaluate"]))
            stack.enter_context(patch.object(workflow,"load_manifest",lambda p:m))
            stack.enter_context(patch.object(workflow,"bind",lambda m:{}))
            stack.enter_context(patch.object(workflow,"read_json",lambda p:{"rows":[baseline]}))
            stack.enter_context(patch.object(workflow,"evaluate_one",lambda m,p,v,e:control if v=="control" else kd))
            workflow.main()
            report=json.loads((Path(temp)/"m63f_frozen_bn_kd/m63f_comparison.json").read_text())
            self.assertTrue(report["pilot_passed"])
            self.assertFalse(report["checkpoint_promotion_authorized"])

if __name__=="__main__": unittest.main()
