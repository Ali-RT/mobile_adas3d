from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from scripts.prepare_monodgp_m55_feasibility import PARENT_CHECKPOINT_SHA256
from scripts.prepare_monodgp_m56c_group_sensitivity import GROUP_ORDER
from scripts.prepare_monodgp_m56d_det2d_fp32_storage import (
    EXPECTED_FP16_NAMES_SHA256,
    EXPECTED_FP16_STATE_ALIASES,
    EXPECTED_FP16_UNIQUE_BYTES,
    EXPECTED_FP16_UNIQUE_PARAMETERS,
    EXPECTED_FP32_STATE_ALIASES,
    EXPECTED_FP32_UNIQUE_BYTES,
    EXPECTED_FP32_UNIQUE_PARAMETERS,
    EXPECTED_PROJECTED_SIZE_RATIO,
    FP32_GROUP,
    M56C_CSV_SHA256,
    M56C_MANIFEST_SHA256,
    M56C_REPORT_SHA256,
    M56C_SELECTED_DEPTH_MAX_ABS,
    M56C_SINGLETON_DEPTH_MAX_ABS,
    POLICY_ID,
    collect_det2d_fp32_storage_policy,
    validate_m56c_result_records,
)
from scripts.smoke_test_monodgp_m56_fp16_storage import (
    PARITY_LIMITS,
    reproduce_explicit_storage_policy,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M56D_Det2D_FP32_Storage_Colab.ipynb"


class TinyGroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        shared = torch.nn.Linear(4, 4)
        self.det2d_transformer = torch.nn.ModuleList([shared, shared])
        self.det3d_transformer = torch.nn.Linear(4, 4)
        self.class_embed = torch.nn.Linear(4, 2)
        self.norm = torch.nn.LayerNorm(4)


def valid_m56c_records():
    group_policy = {
        name: {
            "unique_parameters": 1 if name == "backbone" else 0,
            "unique_parameter_bytes": 1 if name == "backbone" else 0,
            "state_aliases": 1 if name == "backbone" else 0,
        }
        for name in GROUP_ORDER
    }
    group_policy[FP32_GROUP] = {
        "unique_parameters": EXPECTED_FP32_UNIQUE_PARAMETERS,
        "unique_parameter_bytes": EXPECTED_FP32_UNIQUE_BYTES,
        "state_aliases": EXPECTED_FP32_STATE_ALIASES,
    }
    manifest = {
        "complete": True,
        "experiment": "M56c grouped FP16-storage sensitivity isolation",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "diagnostic_authorized": True,
        "preparation_gate_results": {"complete": True},
        "candidate_count": 16,
        "full_evaluation_authorized": False,
        "offline_compression_candidate_selected": False,
        "group_policy": group_policy,
    }
    passing_summary = {
        name: {"passed": True, "limit_max_abs": limit, "max_abs": 0.0}
        for name, limit in PARITY_LIMITS.items()
    }
    passing_summary["pred_depth"]["max_abs"] = M56C_SELECTED_DEPTH_MAX_ABS
    selected = {
        "spec": {
            "name": "all_except_det2d_transformer",
            "mode": "complement",
            "fp16_groups": ["backbone"],
            "fp32_held_group": FP32_GROUP,
            "fp16_parameter_state_name_count": EXPECTED_FP16_STATE_ALIASES,
            "fp16_parameter_state_names_sha256": EXPECTED_FP16_NAMES_SHA256,
            "fp16_unique_parameters": EXPECTED_FP16_UNIQUE_PARAMETERS,
            "fp16_unique_parameter_bytes": EXPECTED_FP16_UNIQUE_BYTES,
            "projected_parameter_size_ratio": EXPECTED_PROJECTED_SIZE_RATIO,
        },
        "integrity": {"complete": True},
        "parity": {
            "all_parity_limits_passed": True,
            "failed_output_families": [],
            "parity_summary": passing_summary,
        },
    }
    singleton = {
        "spec": {
            "name": "only_det2d_transformer",
            "mode": "singleton",
            "fp16_groups": [FP32_GROUP],
        },
        "parity": {
            "all_parity_limits_passed": False,
            "failed_output_families": ["pred_depth"],
            "parity_summary": {
                "pred_depth": {"max_abs": M56C_SINGLETON_DEPTH_MAX_ABS}
            },
        },
    }
    report = {
        "complete": True,
        "manifest_sha256": M56C_MANIFEST_SHA256,
        "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "candidate_count": 16,
        "all_diagnostic_gates_passed": True,
        "diagnostic_gate_results": {"complete": True},
        "full_evaluation_authorized": False,
        "offline_compression_candidate_selected": False,
        "training_performed": False,
        "candidate_results": [singleton, selected],
        "analysis": {
            "singleton_failures": [FP32_GROUP],
            "complement_rescues": ["backbone", "input_projection", FP32_GROUP],
        },
    }
    rows = [
        {
            "candidate": "only_det2d_transformer",
            "all_parity_limits_passed": "False",
            "failed_output_families": "pred_depth",
            "pred_depth_max_abs": str(M56C_SINGLETON_DEPTH_MAX_ABS),
        },
        {
            "candidate": "all_except_det2d_transformer",
            "all_parity_limits_passed": "True",
            "failed_output_families": "",
            "pred_depth_max_abs": str(M56C_SELECTED_DEPTH_MAX_ABS),
        },
    ]
    rows.extend(
        {"candidate": f"dummy_{index}"} for index in range(14)
    )
    return manifest, report, rows


class MonoDGPM56dDet2DFP32StorageTests(unittest.TestCase):
    def test_frozen_m56c_evidence_and_selected_policy(self):
        self.assertEqual(
            M56C_MANIFEST_SHA256,
            "0c501fa2e0f13e8ad952e492ef44d45a93fa72d3655e15ee4dfca5ae77417796",
        )
        self.assertEqual(
            M56C_REPORT_SHA256,
            "7f1ca25ec2270642c508bb36a8e109bdcf8cdf09d8b05d5749a23a0129321ba1",
        )
        self.assertEqual(
            M56C_CSV_SHA256,
            "8d9eea64c0bf67eb7f4b597e76344d67ceb5e5a392074259050fa51ae32c5abe",
        )
        self.assertEqual(FP32_GROUP, "det2d_transformer")
        self.assertEqual(POLICY_ID, "m56d_det2d_transformer_fp32")
        self.assertEqual(M56C_SINGLETON_DEPTH_MAX_ABS, 0.5688076019287109)
        self.assertEqual(M56C_SELECTED_DEPTH_MAX_ABS, 0.03000640869140625)
        self.assertLessEqual(EXPECTED_PROJECTED_SIZE_RATIO, 0.60)

    def test_m56c_result_validation_is_fail_closed(self):
        manifest, report, rows = valid_m56c_records()
        validate_m56c_result_records(manifest, report, rows)
        report["analysis"]["singleton_failures"] = []
        with self.assertRaises(RuntimeError):
            validate_m56c_result_records(manifest, report, rows)

    def test_policy_preserves_every_det2d_alias(self):
        model = TinyGroupedModel()
        fp16, fp32, inventory = collect_det2d_fp32_storage_policy(model, torch)
        self.assertEqual(
            set(fp32),
            {
                "det2d_transformer.0.weight",
                "det2d_transformer.0.bias",
                "det2d_transformer.1.weight",
                "det2d_transformer.1.bias",
            },
        )
        self.assertTrue(set(fp16).isdisjoint(fp32))
        self.assertIn("backbone.weight", fp16)
        self.assertIn("det3d_transformer.weight", fp16)
        self.assertIn("class_embed.weight", fp16)
        self.assertNotIn("norm.weight", fp16)
        self.assertEqual(inventory["fp32_preserved_unique_parameters"], 2)
        self.assertEqual(inventory["fp32_preserved_state_aliases"], 4)
        self.assertEqual(
            inventory["fp16_unique_parameters"]
            + inventory["fp32_preserved_unique_parameters"],
            inventory["all_eligible_unique_parameters"],
        )

    def test_shared_smoke_reproduces_m56d_policy_independently(self):
        model = TinyGroupedModel()
        manifest = {"compression_policy": {"policy_id": POLICY_ID}}
        expected_fp16, expected_fp32, mode = reproduce_explicit_storage_policy(
            model, manifest, torch
        )
        actual_fp16, actual_fp32, _ = collect_det2d_fp32_storage_policy(model, torch)
        self.assertEqual(expected_fp16, set(actual_fp16))
        self.assertEqual(expected_fp32, set(actual_fp32))
        self.assertIn("det2d-transformer-FP32", mode)

    def test_unknown_explicit_policy_is_rejected(self):
        with self.assertRaises(RuntimeError):
            reproduce_explicit_storage_policy(
                TinyGroupedModel(),
                {"compression_policy": {"policy_id": "unreviewed"}},
                torch,
            )

    def test_cli_entrypoints_import_from_repo_root(self):
        for script in (
            "prepare_monodgp_m56d_det2d_fp32_storage.py",
            "smoke_test_monodgp_m56_fp16_storage.py",
            "evaluate_monodgp_m56_fp16_storage.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_contract_and_notebook_freeze_stop_barrier(self):
        contract = (
            ROOT / "MONODGP_M56D_DET2D_FP32_STORAGE_CONTRACT.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`det2d_transformer`", contract)
        self.assertIn("`0.5688076 > 0.50`", contract)
        self.assertIn("`0.0300064`", contract)
        self.assertIn("no larger than 60%", contract)
        self.assertIn("exactly 3,769 / 3,769", contract)
        self.assertIn("product safety remains false", contract)

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
        self.assertIn("prepare_monodgp_m56d_det2d_fp32_storage.py", code)
        self.assertIn("smoke_test_monodgp_m56_fp16_storage.py", code)
        self.assertIn("evaluate_monodgp_m56_fp16_storage.py", code)
        self.assertLess(
            code.index("smoke_test_monodgp_m56_fp16_storage.py"),
            code.index("evaluate_monodgp_m56_fp16_storage.py"),
        )
        self.assertIn("AUTHORIZE_COMPLETE_EVALUATION=False", code)
        self.assertIn("Stop point 1", markdown)
        self.assertIn("Do not run the complete evaluation", markdown)
        self.assertIn("No training", markdown)


if __name__ == "__main__":
    unittest.main()
