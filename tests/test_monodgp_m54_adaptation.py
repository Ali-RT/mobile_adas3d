from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.prepare_monodgp_m54_adaptation import (
    CHECKPOINT_SHA256,
    CLASS_MAPPING,
    OFFLINE_PRODUCT_TARGETS,
    PINNED_COMMIT,
    R0_COMPARABLE_GATES,
    SWEEP_EPOCHS,
    TRAINING_SCHEDULE,
    require_m53_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M54_Two_Class_Adaptation_Colab.ipynb"


class MonoDGPM54AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_frozen_provenance_taxonomy_and_schedule(self):
        self.assertEqual(PINNED_COMMIT, "aa059a18214aebf644510e7f0793971b403f9d14")
        self.assertEqual(
            CHECKPOINT_SHA256,
            "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d",
        )
        self.assertEqual(
            CLASS_MAPPING,
            {
                "Car": "Car",
                "Van": "Car",
                "Truck": "Car",
                "Tram": "Car",
                "Pedestrian": "Pedestrian",
                "Person_sitting": "Pedestrian",
            },
        )
        self.assertEqual(TRAINING_SCHEDULE["max_epochs"], 100)
        self.assertEqual(TRAINING_SCHEDULE["learning_rate"], 0.00005)
        self.assertEqual(TRAINING_SCHEDULE["seed"], 54054)
        self.assertEqual(SWEEP_EPOCHS, [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    def test_r0_and_product_gates_are_frozen(self):
        self.assertAlmostEqual(
            R0_COMPARABLE_GATES["vehicle_3d_moderate"], 17.634769196266316
        )
        self.assertAlmostEqual(
            R0_COMPARABLE_GATES["pedestrian_3d_moderate"], 5.721371354710236
        )
        self.assertAlmostEqual(
            R0_COMPARABLE_GATES["pedestrian_near_recall"], 0.6834215167548501
        )
        self.assertAlmostEqual(
            R0_COMPARABLE_GATES["pedestrian_localization_failure_rate_max"],
            0.24603174603174602,
        )
        self.assertEqual(
            OFFLINE_PRODUCT_TARGETS,
            {"vehicle_near_recall": 0.85, "pedestrian_near_recall": 0.80},
        )

    @mock.patch("scripts.prepare_monodgp_m54_adaptation.sha256_file")
    def test_m53_authorization_requires_complete_schema_v2_pass(self, digest):
        digest.return_value = CHECKPOINT_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.touch()
            gate_path = root / "gate.json"
            manifest_path = root / "manifest.json"
            gate = {
                "schema_version": 2,
                "complete": True,
                "reference_reproduced": True,
                "two_class_adaptation_authorized": True,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "gate_results": {"complete_prediction_set": True},
            }
            manifest = {
                "complete": True,
                "training_authorized": False,
                "upstream_commit": PINNED_COMMIT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "train_split_sha256": "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c",
                "val_split_sha256": "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8",
                "evaluation_checkpoint": str(checkpoint),
            }
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, _, resolved = require_m53_authorization(gate_path, manifest_path)
            self.assertEqual(resolved, checkpoint.resolve())
            gate["gate_results"]["complete_prediction_set"] = False
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed frozen gate"):
                require_m53_authorization(gate_path, manifest_path)
            gate["gate_results"]["complete_prediction_set"] = True
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            manifest["training_authorized"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpectedly authorizes training"):
                require_m53_authorization(gate_path, manifest_path)

    def test_training_patch_covers_both_target_paths_and_safety_guards(self):
        source = (ROOT / "scripts/patch_monodgp_m54_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("primary target mapping", source)
        self.assertIn("mixup target mapping", source)
        self.assertIn("mapped class IDs and mean-size lookups", source)
        self.assertIn("Non-finite loss", source)
        self.assertIn("Non-finite gradients", source)
        self.assertIn("explicit resume-checkpoint path", source)
        self.assertIn("evaluate_during_training", source)

    def test_notebook_stops_after_real_training_smoke_before_long_run(self):
        self.assertIn("prepare_monodgp_m54_adaptation.py", self.code)
        self.assertIn("patch_monodgp_m54_training.py", self.code)
        self.assertIn("smoke_test_monodgp_m54_training.py", self.code)
        self.assertIn("optimizer_steps']==1", self.code)
        self.assertIn("vehicle_targets']>0", self.code)
        self.assertIn("pedestrian_targets']>0", self.code)
        self.assertLess(
            self.code.index("smoke_test_monodgp_m54_training.py"),
            self.code.index("TRAIN_LOG=run_training_logged"),
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("Stop point", markdown)
        self.assertIn("m54_training_smoke.json", markdown)

    def test_notebook_has_durable_resume_and_bounded_sweep(self):
        self.assertIn("load_checkpoint_safely", self.code)
        self.assertIn("checkpoint_epoch_*.pth", self.code)
        self.assertIn("payload.get('optimizer_state') is None", self.code)
        self.assertIn("Durable combined log:", self.code)
        self.assertIn("sweep_monodgp_m54_product_checkpoints.py", self.code)
        self.assertIn("SWEEP_EPOCHS=[5,10,15,20,30,40,50,60,70,80,90,100]", self.code)

    def test_workflow_is_gt_only_and_preserves_the_m53_graph(self):
        prepare = (ROOT / "scripts/prepare_monodgp_m54_adaptation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"architecture_changed": False', prepare)
        self.assertIn('"distillation_enabled": False', prepare)
        self.assertIn('"temperature_tuning": False', prepare)
        self.assertIn('"unused_native_output_class": "Cyclist"', prepare)
        self.assertIn('"exact verified M53 official checkpoint; retain every model tensor"', prepare)

    def test_selection_runs_complete_ap_nearby_and_failure_diagnostics(self):
        source = (
            ROOT / "scripts/sweep_monodgp_m54_product_checkpoints.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pedestrian_3d_moderate", source)
        self.assertIn("m54_product_checkpoint_sweep.csv", source)
        self.assertIn("sweep_monodetr_r0_product_checkpoints.py", source)
        self.assertIn("audit_product_prediction_geometry.py", source)
        self.assertIn("diagnose_a2_pedestrian_false_negatives.py", source)
        self.assertIn('"accuracy_parent_candidate"', source)
        self.assertIn('"offline_product_gates_passed"', source)
        self.assertIn('"product_safety_qualified": False', source)


if __name__ == "__main__":
    unittest.main()
