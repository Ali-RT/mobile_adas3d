from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.prepare_monodgp_m55_feasibility import PARENT_CHECKPOINT_SHA256
from scripts.prepare_monodgp_m56b_selective_fp16_storage import (
    FP32_HEAD_ROOTS,
    M56_CANDIDATE_SHA256,
    M56_FAILED_DEPTH_MAX_ABS,
    M56_MANIFEST_SHA256,
    collect_alias_aware_storage_policy,
    validate_m56_result_records,
)
from scripts.smoke_test_monodgp_m56_fp16_storage import (
    PARITY_LIMITS,
    summarize_depth_channels,
    validate_candidate_storage,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M56B_Selective_FP16_Storage_Colab.ipynb"


class TinyAliasedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        shared_box_head = torch.nn.Linear(4, 4)
        self.bbox_embed = torch.nn.ModuleList([shared_box_head, shared_box_head])
        self.dim_embed_3d = torch.nn.Linear(4, 3)
        self.depth_embed = torch.nn.Linear(4, 2)
        self.norm = torch.nn.LayerNorm(4)


class MonoDGPM56bSelectiveFP16StorageTests(unittest.TestCase):
    def test_frozen_m56_evidence_and_limits(self):
        self.assertEqual(
            M56_MANIFEST_SHA256,
            "a9cfbd8e41ed020b4d49ee44594b81966b27601750fe4167454cbbdecffde046",
        )
        self.assertEqual(
            M56_CANDIDATE_SHA256,
            "8da64f181e5c09b17a29e09eb41b1133159313f2d09eeb1ac8218bf1d1ea4404",
        )
        self.assertEqual(M56_FAILED_DEPTH_MAX_ABS, 0.7107391357421875)
        self.assertEqual(PARITY_LIMITS["pred_depth"], 0.50)
        self.assertEqual(
            FP32_HEAD_ROOTS, ("bbox_embed", "dim_embed_3d", "depth_embed")
        )

    def test_m56_rejection_must_match_exactly(self):
        manifest = {
            "complete": True,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "candidate_checkpoint_sha256": M56_CANDIDATE_SHA256,
        }
        smoke = {
            "complete": True,
            "manifest_sha256": M56_MANIFEST_SHA256,
            "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "candidate_checkpoint_sha256": M56_CANDIDATE_SHA256,
            "all_smoke_gates_passed": False,
            "full_evaluation_authorized": False,
            "gate_results": {
                "manifest_authorized": True,
                "raw_output_parity_within_limits": False,
            },
            "parity_summary": {
                "pred_logits": {"passed": True},
                "pred_depth": {
                    "passed": False,
                    "limit_max_abs": 0.5,
                    "max_abs": M56_FAILED_DEPTH_MAX_ABS,
                },
            },
        }
        validate_m56_result_records(manifest, smoke)
        smoke["parity_summary"]["pred_depth"]["max_abs"] = 0.70
        with self.assertRaises(RuntimeError):
            validate_m56_result_records(manifest, smoke)

    def test_alias_aware_policy_preserves_all_geometry_head_aliases(self):
        model = TinyAliasedModel()
        fp16, fp32, inventory = collect_alias_aware_storage_policy(model, torch)
        self.assertEqual(set(fp16), {"backbone.weight", "backbone.bias"})
        self.assertEqual(
            set(fp32),
            {
                "bbox_embed.0.weight",
                "bbox_embed.0.bias",
                "bbox_embed.1.weight",
                "bbox_embed.1.bias",
                "dim_embed_3d.weight",
                "dim_embed_3d.bias",
                "depth_embed.weight",
                "depth_embed.bias",
            },
        )
        self.assertEqual(inventory["all_eligible_unique_parameters"], 8)
        self.assertEqual(inventory["fp16_unique_parameters"], 2)
        self.assertEqual(inventory["fp32_preserved_unique_parameters"], 6)
        self.assertEqual(inventory["fp32_preserved_state_aliases"], 8)
        self.assertTrue(all(inventory["fp32_preserved_bytes_by_head_root"].values()))

    def test_explicit_storage_policy_is_verified_fail_closed(self):
        source = OrderedDict(
            fp16_weight=torch.tensor([1.0, 2.0], dtype=torch.float32),
            preserved_weight=torch.tensor([3.0], dtype=torch.float32),
            buffer=torch.tensor([4], dtype=torch.int64),
        )
        candidate = OrderedDict(
            fp16_weight=source["fp16_weight"].to(torch.float16),
            preserved_weight=source["preserved_weight"].clone(),
            buffer=source["buffer"].clone(),
        )
        checks = validate_candidate_storage(
            source, candidate, {"fp16_weight"}, {"preserved_weight"}, torch
        )
        self.assertTrue(checks["exact_state_dict_keys"])
        self.assertTrue(checks["all_policy_fp16_names_stored_fp16"])
        self.assertTrue(checks["all_policy_fp32_names_exact"])
        self.assertTrue(checks["all_uncompressed_state_unchanged"])
        candidate["preserved_weight"] = candidate["preserved_weight"] + 1.0
        failed = validate_candidate_storage(
            source, candidate, {"fp16_weight"}, {"preserved_weight"}, torch
        )
        self.assertFalse(failed["all_policy_fp32_names_exact"])
        self.assertFalse(failed["all_uncompressed_state_unchanged"])

    def test_depth_channel_diagnostics_separate_value_and_uncertainty(self):
        baseline = {"pred_depth": torch.tensor([[[10.0, 0.25]]])}
        candidate = {"pred_depth": torch.tensor([[[10.2, 0.30]]])}
        summary = summarize_depth_channels(baseline, candidate)
        self.assertEqual(summary["tensors"], 1)
        self.assertAlmostEqual(summary["depth_m"]["max_abs"], 0.2, places=5)
        self.assertAlmostEqual(summary["log_variance"]["max_abs"], 0.05, places=5)

    def test_cli_entrypoints_import_from_repo_root(self):
        for script in (
            "prepare_monodgp_m56b_selective_fp16_storage.py",
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
            ROOT / "MONODGP_M56B_SELECTIVE_FP16_STORAGE_CONTRACT.md"
        ).read_text(encoding="utf-8")
        for value in FP32_HEAD_ROOTS:
            self.assertIn(f"`{value}`", contract)
        self.assertIn("alias-aware", contract)
        self.assertIn("no larger than 60%", contract)
        self.assertIn("depth 0.50", contract)
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
        self.assertIn("prepare_monodgp_m56b_selective_fp16_storage.py", code)
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
