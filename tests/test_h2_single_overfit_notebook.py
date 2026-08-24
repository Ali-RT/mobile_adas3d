import ast
import json
import unittest
from pathlib import Path

from tools.config import load_config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MobileADAS3D_H2_Single_Image_Overfit_Colab.ipynb"
CONFIG = ROOT / "configs" / "kitti_mobileadas3d_h2_single_overfit.yaml"


class H2SingleOverfitNotebookTests(unittest.TestCase):
    def test_config_preserves_v2_loss_and_uses_isolated_h2_output(self):
        config = load_config(str(CONFIG))
        self.assertEqual(config["model"]["name"], "MobileADAS3D-H2")
        self.assertEqual(config["model"]["center_offset_scale"], 0.10)
        self.assertEqual(config["training"]["epochs"], 1000)
        self.assertEqual(config["loss"]["classification_mode"], "implicit_background_softmax")
        self.assertFalse(config["distillation"]["enabled"])
        self.assertIn("mobileadas3d_h2_single", config["outputs"]["output_dir"])

    def test_notebook_is_resumable_bounded_and_runs_both_gates(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")
        self.assertIn("MobileADAS3D-H2", source)
        self.assertIn("--steps','1000", source)
        self.assertIn("--save-interval','100", source)
        self.assertIn("run_h1_single_image_overfit.py", source)
        self.assertIn("diagnose_h1_queries.py", source)
        self.assertIn("diagnose_h1_image_sensitivity.py", source)
        self.assertIn("do not run Tiny16", source)
        self.assertNotIn("mobileadas3d_h1_v2_single')", source)

    def test_runner_rejects_cross_architecture_resume(self):
        source = (ROOT / "scripts" / "run_h1_single_image_overfit.py").read_text()
        self.assertIn('checkpoint_architecture != architecture', source)
        self.assertIn('"architecture": architecture', source)


if __name__ == "__main__":
    unittest.main()
