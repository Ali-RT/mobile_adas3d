import json
import unittest
from pathlib import Path

from tools.config import load_config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MobileADAS3D_H1_V2_Single_Image_Overfit_Colab.ipynb"
CONFIG = ROOT / "configs" / "kitti_mobileadas3d_h1_v2_single_overfit.yaml"


class H1V2SingleOverfitTests(unittest.TestCase):
    def test_config_is_one_image_exact_step_gt_only(self):
        config = load_config(str(CONFIG))
        splits = config["dataset"]["splits"]
        self.assertEqual(splits["expected_train_count"], 1)
        self.assertEqual(splits["expected_val_count"], 1)
        self.assertTrue(splits["require_identical_train_val"])
        self.assertEqual(config["training"]["epochs"], 1000)
        self.assertEqual(config["training"]["batch_size"], 1)
        self.assertFalse(config["validation"]["enabled"])
        self.assertFalse(config["distillation"]["enabled"])
        self.assertEqual(
            config["loss"]["classification_mode"],
            "implicit_background_softmax",
        )

    def test_notebook_selects_both_classes_and_runs_both_gates(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("names & vehicle and names & pedestrian", source)
        self.assertIn("--steps','1000", source)
        self.assertIn("--save-interval','100", source)
        self.assertIn("run_h1_single_image_overfit.py", source)
        self.assertIn("diagnose_h1_queries.py", source)
        self.assertIn("diagnose_h1_image_sensitivity.py", source)
        self.assertIn("allow_failure=True", source)
        self.assertIn("do not run full KITTI or distillation", source)

    def test_runner_is_resumable_and_rejects_wrong_data(self):
        source = (ROOT / "scripts" / "run_h1_single_image_overfit.py").read_text()
        self.assertIn('checkpoint_path.is_file()', source)
        self.assertIn('single_image_overfit', source)
        self.assertIn('len(loader.dataset) != 1', source)
        self.assertIn('required_classes.issubset(object_classes)', source)

    def test_sensitivity_gate_checks_repeat_class_and_box_outputs(self):
        source = (ROOT / "scripts" / "diagnose_h1_image_sensitivity.py").read_text()
        self.assertIn('deterministic_repeat_max_le_1e_6', source)
        self.assertIn('class_logits_mean_delta_ge_0_02', source)
        self.assertIn('box_mean_delta_ge_0_01', source)


if __name__ == "__main__":
    unittest.main()
