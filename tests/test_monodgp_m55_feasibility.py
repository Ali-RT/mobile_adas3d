from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.finalize_monodgp_m55_feasibility import main as finalize_main
from scripts.prepare_monodgp_m55_feasibility import (
    EXPECTED_SWEEP_EPOCHS,
    PARENT_CHECKPOINT_SHA256,
    PARENT_METRICS,
    PRESERVATION_GATES,
    PROFILE_SETTINGS,
    validate_m54_selection,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M55_Compression_Feasibility_Colab.ipynb"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class MonoDGPM55FeasibilityTests(unittest.TestCase):
    def test_parent_metrics_and_relative_gates_are_frozen(self):
        self.assertEqual(
            PARENT_CHECKPOINT_SHA256,
            "8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847",
        )
        self.assertAlmostEqual(PARENT_METRICS["vehicle_3d_moderate"], 19.451853874133846)
        self.assertAlmostEqual(PARENT_METRICS["pedestrian_near_recall"], 0.7248677248677249)
        self.assertAlmostEqual(
            PRESERVATION_GATES["vehicle_3d_moderate"],
            0.95 * PARENT_METRICS["vehicle_3d_moderate"],
        )
        self.assertAlmostEqual(
            PRESERVATION_GATES["pedestrian_near_recall"],
            PARENT_METRICS["pedestrian_near_recall"] - 0.01,
        )
        self.assertAlmostEqual(
            PRESERVATION_GATES["pedestrian_localization_failure_rate_max"],
            PARENT_METRICS["pedestrian_localization_failure_rate"] + 0.01,
        )
        self.assertEqual(PRESERVATION_GATES["prediction_files"], 3769)
        self.assertEqual(PROFILE_SETTINGS["warmup_runs"], 5)
        self.assertEqual(PROFILE_SETTINGS["timed_runs"], 100)

    @mock.patch("scripts.prepare_monodgp_m55_feasibility.sha256_file")
    def test_m54_selection_requires_exact_parent_and_complete_sweep(self, digest):
        digest.return_value = PARENT_CHECKPOINT_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch_100.pth"
            checkpoint.touch()
            selection_path = root / "selection.json"
            sweep_path = root / "sweep.csv"
            selection = {
                "schema_version": 1,
                "complete": True,
                "selected_epoch": 100,
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "evaluated_epochs": EXPECTED_SWEEP_EPOCHS,
                "evaluated_images": 3769,
                "metrics": PARENT_METRICS,
                "r0_comparable_gate_results": {"all": True},
                "accuracy_parent_candidate": True,
                "product_safety_qualified": False,
            }
            write_json(selection_path, selection)
            with sweep_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["epoch", "rank", "complete_split"]
                )
                writer.writeheader()
                for epoch in EXPECTED_SWEEP_EPOCHS:
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "rank": 1 if epoch == 100 else 2,
                            "complete_split": True,
                        }
                    )
            resolved, selected_checkpoint = validate_m54_selection(selection_path, sweep_path)
            self.assertEqual(resolved["selected_epoch"], 100)
            self.assertEqual(selected_checkpoint, checkpoint.resolve())
            selection["evaluated_images"] = 3768
            write_json(selection_path, selection)
            with self.assertRaisesRegex(RuntimeError, "3,769"):
                validate_m54_selection(selection_path, sweep_path)

    def test_cli_entrypoints_import_from_repo_root(self):
        for script in (
            "prepare_monodgp_m55_feasibility.py",
            "profile_monodgp_m55_baseline.py",
            "finalize_monodgp_m55_feasibility.py",
        ):
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"])
            self.assertEqual(completed.returncode, 0, script)

    def test_finalizer_authorizes_offline_compression_but_not_coreml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            profile_path = root / "profile.json"
            audit_path = root / "audit.json"
            latency_path = root / "latency.csv"
            output_path = root / "gate.json"
            manifest = {
                "complete": True,
                "training_performed": False,
                "compression_performed": False,
                "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "parent_metrics": PARENT_METRICS,
                "preservation_gates": PRESERVATION_GATES,
                "runtime_config_sha256": "runtime",
            }
            write_json(manifest_path, manifest)
            profile = {
                "complete": True,
                "training_performed": False,
                "compression_performed": False,
                "manifest_sha256": __import__(
                    "scripts.finalize_monodgp_m55_feasibility", fromlist=["sha256_file"]
                ).sha256_file(manifest_path),
                "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "checkpoint_epoch": 100,
                "runtime_config_sha256": "runtime",
                "checkpoint_bytes": 100,
                "device": {"name": "test GPU"},
                "finite_outputs": True,
                "outputs": {key: {} for key in {
                    "pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth",
                    "pred_angle", "pred_depth_map_logits", "pred_region_prob",
                }},
                "parameter_inventory": {
                    "total_parameters": 10,
                    "parameter_bytes": 40,
                    "state_dict_tensor_bytes": 48,
                },
                "latency": {
                    "warmup_runs": 5,
                    "timed_runs": 100,
                    "mean_ms": 1.0,
                    "median_ms": 1.0,
                    "p95_ms": 1.1,
                    "min_ms": 0.9,
                    "max_ms": 1.2,
                },
                "memory": {
                    "loaded_allocated_bytes": 1,
                    "loaded_reserved_bytes": 1,
                    "peak_allocated_bytes": 2,
                    "peak_reserved_bytes": 2,
                },
                "compute": {
                    "complete": True,
                    "partial_lower_bound": True,
                    "profiled_flops": 0,
                    "note": "Custom kernels are not FLOP-accounted.",
                },
            }
            write_json(profile_path, profile)
            audit = {
                "complete": True,
                "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "offline_weight_compression_scope_available": True,
                "eligible_parameter_bytes": 36,
                "eligible_parameter_fraction": 0.9,
                "ordinary_weight_operator_families": ["Conv2d", "Linear"],
                "custom_cuda_extension_required": True,
                "custom_deformable_attention_modules": [{"name": "attention"}],
                "known_export_blockers": ["custom attention"],
                "direct_coreml_export_ready": False,
                "direct_coreml_conversion_authorized": False,
            }
            write_json(audit_path, audit)
            with latency_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run", "cuda_event_ms"])
                writer.writeheader()
                for index in range(100):
                    writer.writerow({"run": index + 1, "cuda_event_ms": 1.0})
            argv = [
                "finalize",
                "--manifest", str(manifest_path),
                "--profile", str(profile_path),
                "--operator-audit", str(audit_path),
                "--latency-csv", str(latency_path),
                "--output", str(output_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                finalize_main()
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["all_feasibility_gates_passed"])
            self.assertTrue(report["offline_weight_compression_authorized"])
            self.assertFalse(report["direct_coreml_conversion_authorized"])
            self.assertFalse(report["product_safety_qualified"])

    def test_profiler_audits_custom_attention_and_uses_cuda_events(self):
        source = (ROOT / "scripts/profile_monodgp_m55_baseline.py").read_text(encoding="utf-8")
        self.assertIn("torch.cuda.Event", source)
        self.assertIn("custom_deformable_attention_modules", source)
        self.assertIn("eligible_parameter_fraction", source)
        self.assertIn("partial_lower_bound", source)
        self.assertIn("direct_coreml_export_ready", source)

    def test_contract_separates_preservation_export_and_safety(self):
        contract = (ROOT / "MONODGP_M55_COMPRESSION_CONTRACT.md").read_text(encoding="utf-8")
        self.assertIn("19.451854", contract)
        self.assertIn("18.479261", contract)
        self.assertIn("exactly 3,769 / 3,769", contract)
        self.assertIn("direct Core ML conversion is not authorized", contract)
        self.assertIn("performs no training", contract)

    def test_notebook_is_self_contained_and_stops_after_feasibility(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        self.assertIn("prepare_monodgp_m55_feasibility.py", code)
        self.assertIn("profile_monodgp_m55_baseline.py", code)
        self.assertIn("finalize_monodgp_m55_feasibility.py", code)
        self.assertIn("patch_monodgp_colab_compat.py", code)
        self.assertIn("patch_monodgp_m54_training.py", code)
        self.assertNotIn("'tools/train_val.py','--config'", code)
        self.assertNotIn("optimizer.step", code)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("No training or quantization", markdown)
        self.assertIn("Stop point", markdown)


if __name__ == "__main__":
    unittest.main()
