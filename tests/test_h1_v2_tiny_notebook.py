import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MobileADAS3D_H1_V2_Tiny_Overfit_Colab.ipynb"


class H1V2TinyNotebookTests(unittest.TestCase):
    def test_notebook_is_bounded_resumable_and_gt_only(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("tiny_ids=ped[:8]+other[:8]", source)
        self.assertIn("--resume", source)
        self.assertIn("state.get('epoch')!=100", source)
        self.assertIn("diagnose_h1_queries.py", source)
        self.assertIn("distillation_enabled':False", source)
        self.assertNotIn("teacher", source.lower().replace("teacher-shaped", ""))

    def test_config_freezes_v2_objective(self):
        source = (ROOT / "configs" / "kitti_mobileadas3d_h1_v2_tiny_overfit.yaml").read_text()
        self.assertIn("classification_mode: implicit_background_softmax", source)
        self.assertIn("no_object_weight: 0.1", source)
        self.assertIn("quality_negative_weight: 0.1", source)
        self.assertIn("enabled: false", source)


if __name__ == "__main__":
    unittest.main()
