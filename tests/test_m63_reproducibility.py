import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from diagnose_m63_reproducibility import compare_tensors,compare_replicas,tensor_hash,STEPS

class ReproducibilityTests(unittest.TestCase):
    def test_equal_and_changed_tensors(self):
        a={"w":torch.tensor([1.,2.])}
        self.assertTrue(compare_tensors(a,a)["all_exact"])
        b={"w":torch.tensor([1.,3.])}
        result=compare_tensors(a,b)
        self.assertFalse(result["all_exact"])
        self.assertEqual(result["tensors"]["w"]["max_abs"],1.)
        self.assertEqual(result["tensors"]["w"]["mean_abs"],.5)

    def test_invalid_layout_and_nonfinite_rejected(self):
        a={"w":torch.ones(1)}
        for b in ({},{"w":torch.ones(2)},{"w":torch.tensor([float("nan")])}):
            with self.assertRaises(RuntimeError):compare_tensors(a,b)

    def test_hash_covers_dtype_shape_value(self):
        a=torch.tensor([1.,2.])
        self.assertEqual(tensor_hash(a),tensor_hash(a.clone()))
        self.assertNotEqual(tensor_hash(a),tensor_hash(a.double()))
        self.assertNotEqual(tensor_hash(a),tensor_hash(a.reshape(1,2)))

    def test_trace_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/"a";b=root/"b";a.mkdir();b.mkdir()
            trace={"steps":[]}
            for i in range(1,STEPS+1):
                value={k:{"w":torch.ones(1)} for k in ("outputs","gradients","state")}
                torch.save(value,a/f"step{i}.pt");torch.save(value,b/f"step{i}.pt")
                trace["steps"].append(dict(sample_ids=[1],input_sha256=["i"],target_sha256=["t"],
                    rng_before_forward={},rng_after_backward={},gt_loss=1.))
            results=compare_replicas(trace,trace,a,b)
            self.assertEqual(len(results),3)
            self.assertTrue(all(r["numerical"]["gradients"]["all_exact"] for r in results))

    def test_notebook_and_budget(self):
        self.assertEqual(STEPS,3)
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        code="".join(nb["cells"][-1]["source"]);ast.parse(code)
        self.assertIn("diagnose_m63_reproducibility.py",code)
        self.assertIn("m63h_results.zip",code)
        script=(ROOT/"scripts/diagnose_m63_reproducibility.py").read_text()
        self.assertNotIn("geometry_distillation(",script)
        self.assertIn("TemporaryDirectory",script)
        self.assertIn("completed.exists()",script)

if __name__=="__main__":unittest.main()
