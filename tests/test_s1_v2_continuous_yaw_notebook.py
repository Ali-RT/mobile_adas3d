from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MobileADAS3D_S1_V2_Continuous_Yaw_Colab.ipynb"
CONFIG = ROOT / "configs/kitti_mobileadas3d_s1_v2_continuous_yaw.yaml"


class S1V2ContinuousYawNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_isolated_fresh_v2_contract(self):
        self.assertIn("kitti_mobileadas3d_s1_v2_continuous_yaw.yaml", self.code)
        self.assertIn("mobileadas3d_s1_v2_continuous_yaw", self.code)
        self.assertIn("manifest['yaw_encoding']=='continuous_sincos'", self.code)
        self.assertIn("Refusing incompatible resume", self.code)
        self.assertIn("manifest['distillation_enabled'] is False", self.code)

        config_text = CONFIG.read_text(encoding="utf-8")
        self.assertIn('yaw_encoding: "continuous_sincos"', config_text)
        self.assertIn("yaw_axis: false", config_text)
        self.assertIn("yaw_direction: false", config_text)
        self.assertIn("yaw_direction_weight: 0.0", config_text)
        self.assertIn("detach_yaw_in_corner3d: true", config_text)
        self.assertIn("yaw_pred_is_direct_sincos: true", config_text)
        self.assertIn("enabled: false", config_text)

    def test_gate_is_complete_and_continuation_locked(self):
        self.assertIn("gate_summary['complete_split']", self.code)
        self.assertIn("gate_summary['evaluated_images']==3769", self.code)
        self.assertIn("evaluate_3d_metrics.py", self.code)
        self.assertIn("AUTHORIZE_CONTINUATION=False", self.code)
        self.assertIn("Review S1-V2 epoch-20 gate", self.code)


if __name__ == "__main__":
    unittest.main()
