import ast
import json
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from diagnose_m63_continuation import EPOCHS, checkpoint_for, buffer_drift, summarize

class ContinuationDiagnosticTests(unittest.TestCase):
    def test_bounded_epochs(self):
        self.assertEqual(EPOCHS,(1,3,5))
        m={"variants":{"control":{"run_dir":"run"}}}
        self.assertEqual(checkpoint_for(m,"control",3),Path("run/checkpoint_epoch_3.pth"))
        for epoch in (2,6,11):
            with self.assertRaises(ValueError): checkpoint_for(m,"control",epoch)

    def test_buffer_drift_is_read_only(self):
        source={"bn.running_mean":torch.tensor([1.,2.]),"layer.weight":torch.ones(2),
                "bn.num_batches_tracked":torch.tensor(10)}
        other={"bn.running_mean":torch.tensor([2.,0.]),"layer.weight":torch.zeros(2),
               "bn.num_batches_tracked":torch.tensor(15)}
        rows=buffer_drift(source,other)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["mean_absolute_change"],1.5)
        self.assertEqual(rows[1]["max_absolute_change"],5)
        self.assertTrue(torch.equal(source["bn.running_mean"],torch.tensor([1.,2.])))
        with self.assertRaises(RuntimeError): buffer_drift(source,{})

    def test_nonfinite_buffers_rejected(self):
        with self.assertRaises(RuntimeError):
            buffer_drift({"bn.running_var":torch.ones(1)},{"bn.running_var":torch.tensor([float("nan")])})

    def test_diagnostic_deltas_do_not_select_checkpoint(self):
        from m63_common import BASELINE
        b=dict(BASELINE,variant="baseline",epoch=0)
        t=dict(BASELINE,variant="vehicle_kd",epoch=1)
        t["vehicle_3d_moderate"]-=1
        rows=summarize([b,t])
        self.assertAlmostEqual(rows[0]["delta_vs_baseline"]["vehicle_3d_moderate"],-1)
        self.assertNotIn("selected_checkpoint",rows[0])

    def test_notebook_diagnostic_has_no_training_calls(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        code="".join(nb["cells"][-1]["source"])
        ast.parse(code)
        self.assertIn("diagnose_m63_continuation.py",code)
        self.assertNotIn("train_m63_student",code)
        self.assertIn("sections **1–3**","".join(nb["cells"][-2]["source"]))
        script=(ROOT/"scripts/diagnose_m63_continuation.py").read_text()
        self.assertNotIn(".backward(",script)
        self.assertNotIn("optimizer.step(",script)
        self.assertIn("selected_checkpoint=None",script)
        self.assertIn('output / "diagnostics_m63b" / "evaluation"',script)

if __name__=="__main__": unittest.main()
