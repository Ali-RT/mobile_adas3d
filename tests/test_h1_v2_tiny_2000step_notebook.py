import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MobileADAS3D_H1_V2_Tiny_2000Step_Colab.ipynb"
CONFIG = ROOT / "configs" / "kitti_mobileadas3d_h1_v2_tiny_2000step.yaml"


class H1V2Tiny2000StepNotebookTests(unittest.TestCase):
    def test_config_changes_only_schedule_identity_and_output(self):
        source = CONFIG.read_text()
        self.assertIn("base_config: kitti_mobileadas3d_h1_v2_tiny_overfit.yaml", source)
        self.assertIn("epochs: 500", source)
        self.assertIn("save_interval: 100", source)
        for forbidden in ("loss:", "model:", "matcher:", "learning_rate:", "distillation:"):
            self.assertNotIn(forbidden, source)

    def test_notebook_is_fresh_resumable_and_diagnoses_all_milestones(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("tiny_ids=ped[:8]+other[:8]", source)
        self.assertIn("--resume", source)
        self.assertIn("state.get('epoch',0)>=500", source)
        self.assertIn("MILESTONE_EPOCHS=(100,200,300,400,500)", source)
        self.assertIn("epoch_{epoch:03d}.pt", source)
        self.assertIn("diagnose_h1_queries.py", source)
        self.assertIn("distillation_enabled':False", source)
        self.assertIn("fresh_initialization_required", source)
        self.assertNotIn("single_image", source)


if __name__ == "__main__":
    unittest.main()
