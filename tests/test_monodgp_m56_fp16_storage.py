from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from scripts.evaluate_monodgp_m56_fp16_storage import preservation_gate_results
from scripts.prepare_monodgp_m55_feasibility import PARENT_METRICS
from scripts.prepare_monodgp_m56_fp16_storage import (
    EXPECTED_ELIGIBLE_PARAMETER_BYTES,
    MAX_MODEL_ONLY_SIZE_RATIO,
    M55_AUDIT_SHA256,
    M55_GATE_SHA256,
    M55_PROFILE_SHA256,
    collect_eligible_parameter_names,
    compress_state_dict,
    validate_m55_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M56_FP16_Parameter_Storage_Colab.ipynb"


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, bias=True)
        self.norm = torch.nn.BatchNorm2d(4)
        self.linear = torch.nn.Linear(4, 2, bias=True)


class MonoDGPM56FP16StorageTests(unittest.TestCase):
    def test_frozen_m55_evidence_and_storage_policy(self):
        self.assertEqual(
            M55_GATE_SHA256,
            "684338ae7be11f5394aff76b9e3115c22f583fb06103db02c56091c4208050fc",
        )
        self.assertEqual(
            M55_PROFILE_SHA256,
            "8ef15616614c9852ed6b51f171b69a7a0f6994a5a04b79789f0ebbdb1a3d0751",
        )
        self.assertEqual(
            M55_AUDIT_SHA256,
            "798746036e50d37729adca3e2ef9a966a65386ab000f97feb516962b3e1ee1a3",
        )
        self.assertEqual(EXPECTED_ELIGIBLE_PARAMETER_BYTES, 156331452)
        self.assertEqual(MAX_MODEL_ONLY_SIZE_RATIO, 0.60)

    def test_reviewed_m55_artifacts_match_frozen_hashes(self):
        gate, profile, audit = validate_m55_evidence(
            ROOT / "artifacts/m55_feasibility_gate_20260913.json",
            ROOT / "artifacts/m55_native_baseline_profile_20260913.json",
            ROOT / "artifacts/m55_operator_export_audit_20260913.json",
        )
        self.assertTrue(gate["all_feasibility_gates_passed"])
        self.assertTrue(profile["complete"])
        self.assertTrue(audit["complete"])

    def test_only_conv_and_linear_owned_parameters_are_stored_fp16(self):
        model = TinyModel()
        eligible, families = collect_eligible_parameter_names(model, torch)
        baseline, candidate, inventory = compress_state_dict(
            model.state_dict(), eligible, torch
        )
        self.assertEqual(
            set(eligible),
            {"conv.weight", "conv.bias", "linear.weight", "linear.bias"},
        )
        self.assertGreater(families["Conv2d"], 0)
        self.assertGreater(families["Linear"], 0)
        for name in eligible:
            self.assertEqual(baseline[name].dtype, torch.float32)
            self.assertEqual(candidate[name].dtype, torch.float16)
        for name in ("norm.weight", "norm.bias", "norm.running_mean", "norm.running_var"):
            self.assertEqual(candidate[name].dtype, baseline[name].dtype)
            self.assertTrue(torch.equal(candidate[name], baseline[name]))
        self.assertEqual(
            inventory["eligible_parameter_bytes_fp16"] * 2,
            inventory["eligible_parameter_bytes_fp32"],
        )
        self.assertLess(inventory["state_dict_tensor_size_ratio"], 1.0)

    def test_fp16_storage_round_trip_loads_into_fp32_runtime(self):
        model = TinyModel()
        eligible, _ = collect_eligible_parameter_names(model, torch)
        baseline, candidate, _ = compress_state_dict(model.state_dict(), eligible, torch)
        baseline_model = TinyModel()
        candidate_model = TinyModel()
        baseline_model.load_state_dict(baseline, strict=True)
        candidate_model.load_state_dict(candidate, strict=True)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in candidate_model.parameters()))
        source = torch.randn(2, 3, 5, 5)
        baseline_output = baseline_model.conv(source)
        candidate_output = candidate_model.conv(source)
        self.assertTrue(torch.isfinite(candidate_output).all())
        self.assertLess(
            float((candidate_output - baseline_output).abs().max().detach()), 0.01
        )

    def test_parent_metrics_pass_and_individual_failure_is_not_hidden(self):
        passed = preservation_gate_results(dict(PARENT_METRICS), 3769)
        self.assertTrue(all(passed.values()))
        failed_metrics = dict(PARENT_METRICS)
        failed_metrics["pedestrian_3d_moderate"] = 0.0
        failed = preservation_gate_results(failed_metrics, 3769)
        self.assertFalse(failed["pedestrian_3d_moderate"])
        self.assertTrue(failed["vehicle_3d_moderate"])
        self.assertFalse(all(failed.values()))

    def test_cli_entrypoints_import_from_repo_root(self):
        for script in (
            "prepare_monodgp_m56_fp16_storage.py",
            "smoke_test_monodgp_m56_fp16_storage.py",
            "evaluate_monodgp_m56_fp16_storage.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_contract_separates_storage_runtime_export_and_safety(self):
        contract = (ROOT / "MONODGP_M56_FP16_STORAGE_CONTRACT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("FP16 for storage", contract)
        self.assertIn("runtime execution remains", contract)
        self.assertIn("FP32, M56 does not predict", contract)
        self.assertIn("no larger than 60%", contract)
        self.assertIn("18.479261", contract)
        self.assertIn("direct Core ML conversion remains unauthorized", contract)
        self.assertIn("product safety remains false", contract)

    def test_notebook_has_smoke_barrier_and_complete_evaluation(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        self.assertIn("prepare_monodgp_m56_fp16_storage.py", code)
        self.assertIn("smoke_test_monodgp_m56_fp16_storage.py", code)
        self.assertIn("evaluate_monodgp_m56_fp16_storage.py", code)
        self.assertLess(
            code.index("smoke_test_monodgp_m56_fp16_storage.py"),
            code.index("evaluate_monodgp_m56_fp16_storage.py"),
        )
        self.assertIn("Stop point 1", markdown)
        self.assertIn("Do not run the complete evaluation", markdown)
        self.assertIn("No training", markdown)


if __name__ == "__main__":
    unittest.main()
