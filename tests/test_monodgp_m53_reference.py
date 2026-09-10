from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from scripts.evaluate_monodgp_m53_reference import (
    CHECKPOINT_SHA256,
    MAX_ABS_AP_DIFFERENCE,
    PINNED_COMMIT,
    PUBLISHED,
    VAL_SPLIT_SHA256,
)
from scripts.prepare_monodgp_m53_reference import (
    CHECKPOINT_FILE_ID,
    TRAIN_SPLIT_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M53_Official_Reference_Colab.ipynb"


class MonoDGPM53ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_official_provenance_is_frozen(self):
        self.assertEqual(PINNED_COMMIT, "aa059a18214aebf644510e7f0793971b403f9d14")
        self.assertEqual(
            CHECKPOINT_SHA256,
            "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d",
        )
        self.assertEqual(CHECKPOINT_FILE_ID, "1nfCiFIxCIm0WG--cbllkqzgeuVRPpNI5")
        self.assertEqual(
            TRAIN_SPLIT_SHA256,
            "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c",
        )
        self.assertEqual(
            VAL_SPLIT_SHA256,
            "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8",
        )
        self.assertEqual(PUBLISHED, {"easy": 30.1314, "moderate": 22.7109, "hard": 19.3978})
        self.assertEqual(MAX_ABS_AP_DIFFERENCE, 0.5)

    def test_checkpoint_loading_remains_restricted(self):
        source = (ROOT / "scripts/patch_monodgp_colab_compat.py").read_text(encoding="utf-8")
        self.assertIn("torch.serialization.safe_globals", source)
        self.assertIn("weights_only=True", source)
        self.assertNotIn("weights_only=False", source)
        self.assertIn("numpy.core.multiarray.scalar", source)

    def test_current_cuda_extension_compatibility_is_scoped(self):
        source = (ROOT / "scripts/patch_monodgp_colab_compat.py").read_text(encoding="utf-8")
        self.assertIn("value.scalar_type()", source)
        self.assertIn("active CUDA architecture", source)
        self.assertIn("from torch.nn import Linear as _LinearWithBias", source)
        self.assertIn(
            "from torch import overrides as torch_overrides",
            source,
        )
        self.assertIn("Expected MonoDGP commit", source)

    def test_reference_taxonomy_is_car_only(self):
        config = (ROOT / "configs/kitti_monodgp_m53_car_reference.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("classes: [Car]", config)
        self.assertIn("class_mapping: null", config)
        self.assertIn("require_taxonomy_manifest: false", config)

    def test_notebook_is_evaluation_only_and_fail_closed(self):
        self.assertIn("PuFanqi23/MonoDGP.git", self.code)
        self.assertIn("prepare_monodgp_m53_reference.py", self.code)
        self.assertIn("smoke_test_monodgp_m53_reference.py", self.code)
        self.assertIn("evaluate_monodgp_m53_reference.py", self.code)
        self.assertIn("training_authorized'] is False", self.code)
        self.assertNotIn("train_mobile_adas3d.py", self.code)
        self.assertNotIn("tools/train_val.py", self.code)
        self.assertLess(
            self.code.index("smoke_test_monodgp_m53_reference.py"),
            self.code.index("evaluate_monodgp_m53_reference.py"),
        )

    def test_complete_gate_authorizes_only_adaptation(self):
        source = (ROOT / "scripts/evaluate_monodgp_m53_reference.py").read_text(encoding="utf-8")
        self.assertIn("prediction_ids == set(val_ids)", source)
        self.assertIn('"two_class_adaptation_authorized": passed', source)
        self.assertIn('"product_safety_qualified": False', source)
        self.assertIn("do not start M54 adaptation", source)


if __name__ == "__main__":
    unittest.main()
