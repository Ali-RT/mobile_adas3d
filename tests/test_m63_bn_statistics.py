import ast
import copy
import json
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from m63c_bn_intervention import restore_running_statistics, state_fingerprints
from diagnose_m63_bn_statistics import checkpoint_for, compare, VARIANT

class BNInterventionTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
        with torch.no_grad():
            model[1].running_mean.copy_(torch.tensor([3.,4.]))
            model[1].running_var.copy_(torch.tensor([5.,6.]))
            model[1].num_batches_tracked.fill_(10)
            model[1].weight.fill_(2.)
        return model

    def test_only_mean_variance_replaced(self):
        model=self.model()
        before=copy.deepcopy(model.state_dict())
        source=copy.deepcopy(before)
        source["1.running_mean"].zero_()
        source["1.running_var"].fill_(1.)
        source["1.weight"].fill_(99.)
        source["1.num_batches_tracked"].fill_(999)
        report=restore_running_statistics(model,source)
        self.assertEqual(report["replaced_buffers"],2)
        for key,value in model.state_dict().items():
            expected=source[key] if key in report["replaced_buffer_names"] else before[key]
            self.assertTrue(torch.equal(value,expected),key)
        self.assertEqual(int(model[1].num_batches_tracked),10)
        self.assertTrue(torch.equal(source["1.weight"],torch.full((2,),99.)))

    def test_validation_happens_before_mutation(self):
        for mode in ("missing","shape","dtype","nan","negative_variance"):
            model=self.model()
            original=state_fingerprints(model)
            source=copy.deepcopy(model.state_dict())
            if mode=="missing": del source["1.running_var"]
            if mode=="shape": source["1.running_var"]=torch.ones(3)
            if mode=="dtype": source["1.running_var"]=torch.ones(2,dtype=torch.float64)
            if mode=="nan": source["1.running_var"][0]=float("nan")
            if mode=="negative_variance": source["1.running_var"][0]=-1
            with self.assertRaises(ValueError): restore_running_statistics(model,source)
            self.assertEqual(original,state_fingerprints(model))

    def test_no_batchnorm_rejected(self):
        with self.assertRaises(ValueError):
            restore_running_statistics(torch.nn.Linear(2,2),{})

    def test_eval_forward_does_not_change_state(self):
        model=self.model().eval()
        source=copy.deepcopy(model.state_dict())
        source["1.running_var"].fill_(1)
        restore_running_statistics(model,source)
        before=state_fingerprints(model)
        with torch.no_grad(): model(torch.ones(4,2))
        self.assertEqual(before,state_fingerprints(model))

    def test_only_control_epoch_one(self):
        m={"variants":{"control":{"run_dir":"run"}}}
        self.assertEqual(checkpoint_for(m,VARIANT,1),Path("run/checkpoint_epoch_1.pth"))
        for variant,epoch in (("vehicle_kd",1),(VARIANT,10),("control",1)):
            with self.assertRaises(ValueError): checkpoint_for(m,variant,epoch)

    def test_recovery_is_unclamped_and_recall_no_loss_is_none(self):
        from m63_common import BASELINE
        baseline=dict(BASELINE)
        control=dict(BASELINE,vehicle_3d_moderate=BASELINE["vehicle_3d_moderate"]-1)
        restored=dict(BASELINE,vehicle_3d_moderate=BASELINE["vehicle_3d_moderate"]-.25)
        result=compare(baseline,control,restored)
        self.assertEqual(result["vehicle_3d_moderate"]["fraction_of_control_regression_recovered"],.75)
        self.assertIsNone(result["vehicle_near_recall"]["fraction_of_control_regression_recovered"])

    def test_notebook_latest_section_is_no_training(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        code="".join(nb["cells"][-1]["source"])
        ast.parse(code)
        self.assertIn("diagnose_m63_bn_statistics.py",code)
        self.assertNotIn("train_m63_student",code)
        script=(ROOT/"scripts/diagnose_m63_bn_statistics.py").read_text()
        self.assertNotIn("torch.save(",script)
        self.assertNotIn(".backward(",script)
        self.assertNotIn("optimizer.step(",script)
        self.assertIn("checkpoint_written=False",script)

if __name__=="__main__": unittest.main()
