import json
import unittest
from pathlib import Path

from tools.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "MobileADAS3D_S1_V2b_Bounded_Yaw_Colab.ipynb"
CONFIG = PROJECT_ROOT / "configs" / "kitti_mobileadas3d_s1_v2b_bounded_yaw.yaml"


class S1V2bBoundedYawNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.config = load_config(str(CONFIG))

    def test_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")

    def test_v2b_is_isolated_and_bounded(self):
        self.assertEqual(self.config["logging"]["run_name"], "mobileadas3d_s1_v2b_bounded_yaw")
        self.assertEqual(self.config["loss"]["yaw_norm_floor"], 0.1)
        self.assertFalse(self.config["distillation"]["enabled"])
        self.assertIn("runtime_s1_v2b", self.source)
        self.assertIn("saved_floor!=0.1", self.source)
        self.assertIn("Do not resume S1-V1 or S1-V2", NOTEBOOK.read_text())

    def test_notebook_stops_after_epoch20_diagnostics(self):
        self.assertIn("GATE_EPOCHS=20", self.source)
        self.assertIn("kitti_r40_gate20", self.source)
        self.assertIn("geometry_diagnostic_epoch20", self.source)
        self.assertIn("STOP HERE", self.source)
        self.assertNotIn("AUTHORIZE_CONTINUATION", self.source)
        self.assertNotIn("full_resume_", self.source)


if __name__ == "__main__":
    unittest.main()
