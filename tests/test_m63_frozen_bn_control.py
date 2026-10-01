import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from m63d_frozen_bn import freeze_bn_statistics, bn_fingerprints, assert_bn_frozen
from run_m63_frozen_bn_control import checkpoint_for, VARIANT, load_completed
from train_m63_student import save_checkpoint

class FrozenBNControlTests(unittest.TestCase):
    def model(self):
        return torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3),torch.nn.Dropout(.2),torch.nn.Linear(3,1))

    def test_statistics_fixed_affine_and_other_weights_train(self):
        torch.manual_seed(42)
        model=self.model()
        requires=[p.requires_grad for p in model.parameters()]
        expected=bn_fingerprints(model)
        before=copy.deepcopy(model.state_dict())
        model.train()
        freeze_bn_statistics(model)
        self.assertTrue(model[2].training)
        self.assertFalse(model[1].training)
        self.assertEqual(requires,[p.requires_grad for p in model.parameters()])
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(4,3)).square().mean().backward()
            opt.step()
            assert_bn_frozen(model,expected)
        self.assertFalse(torch.equal(before["1.weight"],model[1].weight))
        self.assertFalse(torch.equal(before["0.weight"],model[0].weight))

    def test_model_train_requires_reapplying_freeze(self):
        model=self.model()
        freeze_bn_statistics(model)
        model.train()
        with self.assertRaises(RuntimeError): assert_bn_frozen(model)
        freeze_bn_statistics(model)
        assert_bn_frozen(model)

    def test_buffer_change_detected(self):
        model=self.model()
        freeze_bn_statistics(model)
        expected=bn_fingerprints(model)
        model[1].num_batches_tracked.add_(1)
        with self.assertRaises(RuntimeError): assert_bn_frozen(model,expected)

    def test_originally_frozen_affine_stays_frozen(self):
        model=self.model()
        model[1].weight.requires_grad_(False)
        freeze_bn_statistics(model)
        self.assertFalse(model[1].weight.requires_grad)

    def test_missing_or_untracked_bn_rejected(self):
        for model in (torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3,track_running_stats=False)):
            with self.assertRaises(ValueError): freeze_bn_statistics(model)

    def test_one_epoch_only_and_completion_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            m={"output_dir":temp}
            path=checkpoint_for(m,VARIANT,1)
            path.parent.mkdir(parents=True)
            binding={"variant":VARIANT}
            save_checkpoint(path,dict(epoch=1,m63d=binding,model_state={},summary={}))
            self.assertEqual(load_completed(m,binding)["epoch"],1)
            self.assertTrue(path.with_suffix(".sha256").exists())
            with self.assertRaises(RuntimeError): load_completed(m,{"variant":"other"})
            with self.assertRaises(ValueError): checkpoint_for(m,VARIANT,2)
            with self.assertRaises(ValueError): checkpoint_for(m,"vehicle_kd",1)

    def test_notebook_three_actions_and_no_distillation(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"][-11:-8]
        for code in cells: ast.parse(code)
        for code,action in zip(cells,("--smoke","--train","--evaluate")):
            self.assertIn("run_m63_frozen_bn_control.py",code)
            self.assertIn(action,code)
        script=(ROOT/"scripts/run_m63_frozen_bn_control.py").read_text()
        self.assertNotIn("geometry_distillation(",script)
        self.assertIn('if len(loader)!=928',script)
        self.assertIn('bn_state_unchanged=True',script)

if __name__=="__main__": unittest.main()
