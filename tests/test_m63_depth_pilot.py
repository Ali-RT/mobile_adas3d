import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m63_common import BASELINE, baseline_checks, EVIDENCE
from evaluate_m63_pilot import decide, checkpoint_for
from train_m63_student import restore_checkpoint, save_checkpoint
from third_party.monodetr.m63_depth_loss import geometry_distillation

class M63Tests(unittest.TestCase):
    def inputs(self):
        out = {"pred_depth": torch.tensor([[[12., 3.], [30., 4.], [11., 5.]]], requires_grad=True)}
        targets = [{"labels": torch.tensor([1, 0])}]
        a = {"masks": np.array([[1,0,0,0],[0,0,0,0]], bool),
             "pred_depth": torch.tensor([[10., 8.], [20., 9.]], requires_grad=True)}
        assignment = [(torch.tensor([0,1,2]), torch.tensor([0,1,0]))]
        return out, targets, assignment, [a]

    def test_metre_loss_group_matching_and_gradient_isolation(self):
        o,t,i,a = self.inputs()
        loss,count = geometry_distillation(o,t,i,a,["depth"])
        self.assertEqual(float(loss["depth"].detach()), 1.5)
        self.assertEqual(float(loss["total"].detach()), .375)
        self.assertEqual(count["depth"], 2)
        loss["total"].backward()
        self.assertEqual(float(o["pred_depth"].grad[0,1].abs().sum()), 0)
        self.assertEqual(float(o["pred_depth"].grad[...,1].abs().sum()), 0)
        self.assertIsNone(a[0]["pred_depth"].grad)
        self.assertTrue(torch.isfinite(o["pred_depth"].grad).all())

    def test_wrong_masks_rejected(self):
        for row,col in [(1,0),(0,1),(0,2),(0,3)]:
            o,t,i,a = self.inputs()
            a[0]["masks"][row,col] = True
            with self.assertRaises(ValueError):
                geometry_distillation(o,t,i,a,["depth"])

    def test_fixed_weight_and_component(self):
        o,t,i,a = self.inputs()
        for enabled, weight in [(["depth"], .5), (["angle"], .25)]:
            with self.assertRaises(ValueError):
                geometry_distillation(o,t,i,a,enabled,weight)

    def test_empty_mask_has_zero_gradient(self):
        o,t,i,a = self.inputs()
        a[0]["masks"][:] = False
        loss,count = geometry_distillation(o,t,i,a,["depth"])
        loss["total"].backward()
        self.assertEqual(count["depth"],0)
        self.assertEqual(float(o["pred_depth"].grad.abs().sum()),0)

    def test_invalid_teacher_depth_rejected(self):
        for value in (float("nan"), -1.):
            o,t,i,a = self.inputs()
            a[0]["pred_depth"] = np.array([[value,0],[20,0]])
            with self.assertRaises(ValueError):
                geometry_distillation(o,t,i,a,["depth"])

    def test_baseline_drift(self):
        self.assertTrue(all(baseline_checks(BASELINE).values()))
        changed = dict(BASELINE, vehicle_3d_moderate=BASELINE["vehicle_3d_moderate"]+.16)
        self.assertFalse(all(baseline_checks(changed).values()))

    def test_incremental_gate_is_not_historical_product_gate(self):
        improved = dict(BASELINE)
        improved["vehicle_3d_moderate"] += .11
        improved["mean_3d_moderate"] += .055
        self.assertTrue(all(decide(BASELINE,BASELINE,improved).values()))
        self.assertLess(improved["vehicle_3d_moderate"],15.8713)

    def test_vehicle_gain_must_beat_both(self):
        control = dict(BASELINE, vehicle_3d_moderate=14.)
        self.assertFalse(all(decide(BASELINE,control,BASELINE).values()))

    def test_pedestrian_regression_rejected(self):
        improved = dict(BASELINE,vehicle_3d_moderate=16.,pedestrian_near_recall=.68)
        self.assertFalse(all(decide(BASELINE,BASELINE,improved).values()))

    def test_resume_and_sidecar_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            model=torch.nn.Linear(2,1)
            opt=torch.optim.AdamW(model.parameters())
            model(torch.ones(1,2)).sum().backward()
            opt.step()
            saved=copy.deepcopy(model.state_dict())
            binding={"variant":"control","baseline_sha256":"baseline"}
            p=d/"checkpoint_epoch_2.pth"
            save_checkpoint(p,dict(epoch=2,model_state=saved,optimizer_state=opt.state_dict(),
                                  history=[{"epoch":2}],m63=binding))
            with torch.no_grad():
                model.weight.zero_()
            epoch,history=restore_checkpoint(d,binding,model,opt)
            self.assertEqual(epoch,2)
            self.assertTrue(torch.equal(model.weight,saved["weight"]))
            self.assertTrue(p.with_suffix(".sha256").is_file())
            with self.assertRaises(RuntimeError):
                restore_checkpoint(d,{"variant":"other"},model,opt)

    def test_fixed_epoch_and_notebook(self):
        m={"student":{"checkpoint":"source"},"variants":{"control":{"run_dir":"run"}},
           "evaluation_epochs":[10]}
        with self.assertRaises(ValueError):
            checkpoint_for(m,"control",5)
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
        self.assertEqual(len(cells),10)
        for code in cells: ast.parse(code)
        self.assertIn("def pilot",cells[0])
        self.assertIn("M63-NOTEBOOK-2026-09-28-r2",cells[0])
        joined="\n".join(cells)
        self.assertNotIn("MonoDGP.git",joined)
        self.assertNotIn("reset",joined)
        self.assertIn("--baseline-only",cells[4])
        self.assertIn("--smoke",cells[5])


    def test_setup_cell_executes_fresh_and_existing_checkout(self):
        # Syntax parsing alone cannot catch a bare-name expression such as mobile_.
        from types import SimpleNamespace
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        setup=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"][1]
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                mobile, repo=root/"mobile", root/"monodetr"
                if existing:
                    mobile.mkdir()
                    repo.mkdir()
                calls=[]
                def run(command, name, cwd=None):
                    calls.append((list(map(str, command)), name))
                def check_output(command, **kwargs):
                    return "main" if "--show-current" in command else "frozen"
                env={k:"test-version" for k in ("timm","numpy","scipy","pillow")}
                scope=dict(MOBILE_REPO=mobile, REPO=repo, COMMIT="frozen",
                           M62_OUTPUT=root/"m62", M62={"environment":env}, run=run,
                           sys=sys, os=SimpleNamespace(environ={}),
                           subprocess=SimpleNamespace(check_output=check_output))
                exec(compile(setup, "<M63 setup cell>", "exec"), scope)
                names=[name for _,name in calls]
                self.assertLess(names.index("update_mobile"), names.index("dependencies"))
                self.assertLess(names.index("dependencies"), names.index("verify_m62_environment"))
                self.assertLess(names.index("verify_m62_environment"), names.index("cuda_build"))
                self.assertEqual(scope["mobile_url"], "https://github.com/Ali-RT/mobile_adas3d.git")
                self.assertEqual(scope["url"], "https://github.com/ZrrSkywalker/MonoDETR.git")
                clones=[cmd for cmd,name in calls if name.startswith("clone_")]
                self.assertEqual(len(clones), 0 if existing else 2)
                if not existing:
                    self.assertIn(scope["mobile_url"], clones[0])
                    self.assertIn(scope["url"], clones[1])

if __name__ == "__main__":
    unittest.main()
