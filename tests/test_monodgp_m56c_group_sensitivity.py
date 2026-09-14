from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from scripts.diagnose_monodgp_m56c_fp16_groups import (
    compare_outputs,
    rounded_state_dict,
)
from scripts.prepare_monodgp_m55_feasibility import PARENT_CHECKPOINT_SHA256
from scripts.prepare_monodgp_m56c_group_sensitivity import (
    GROUP_ORDER,
    M56B_CANDIDATE_SHA256,
    M56B_FAILED_DEPTH_MAX_ABS,
    M56B_FP32_HEAD_ROOTS,
    M56B_MANIFEST_SHA256,
    M56B_SMOKE_SHA256,
    build_candidate_specs,
    candidate_fp16_names,
    collect_grouped_storage_policy,
    validate_m56b_result_records,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDGP_M56C_Grouped_FP16_Sensitivity_Colab.ipynb"
CONTRACT = ROOT / "MONODGP_M56C_GROUP_SENSITIVITY_CONTRACT.md"


class Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(4, 4)


class TinyGroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Stage()
        self.input_proj = torch.nn.ModuleList([Stage()])
        self.region_head = Stage()
        self.depth_predictor = Stage()
        self.det2d_transformer = Stage()
        self.det3d_transformer = Stage()
        shared_box = torch.nn.Linear(4, 4)
        self.bbox_embed = torch.nn.ModuleList([shared_box, shared_box])
        self.det2d_transformer.bbox_embed = shared_box
        self.class_embed = torch.nn.Linear(4, 3)
        self.dim_embed_3d = torch.nn.Linear(4, 3)
        self.depth_embed = torch.nn.Linear(4, 2)
        self.angle_embed = torch.nn.Linear(4, 24)
        self.unclassified = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


class MonoDGPM56cGroupSensitivityTests(unittest.TestCase):
    def test_frozen_m56b_evidence(self):
        self.assertEqual(
            M56B_MANIFEST_SHA256,
            "6251abd0ca2eb39b4d1376b9efb836221565a38b4359801efb5e2318f1a2f4ef",
        )
        self.assertEqual(
            M56B_SMOKE_SHA256,
            "c9510f67588a4998c1f84e2c309c98c3def3931627874200662fac92690fd8d8",
        )
        self.assertEqual(
            M56B_CANDIDATE_SHA256,
            "ea2d32db0836dc35d226ec17072f445368d79d4d36dbb864b4759e89c8a190f4",
        )
        self.assertEqual(M56B_FAILED_DEPTH_MAX_ABS, 0.7097339630126953)
        self.assertEqual(
            M56B_FP32_HEAD_ROOTS, ("bbox_embed", "dim_embed_3d", "depth_embed")
        )

    def test_m56b_rejection_must_match_exactly(self):
        manifest = {
            "complete": True,
            "experiment": "M56b selective FP16 storage with FP32 geometry heads",
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "candidate_checkpoint_sha256": M56B_CANDIDATE_SHA256,
            "smoke_authorized": True,
            "preparation_gate_results": {"ready": True},
        }
        smoke = {
            "complete": True,
            "manifest_sha256": M56B_MANIFEST_SHA256,
            "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "candidate_checkpoint_sha256": M56B_CANDIDATE_SHA256,
            "stored_fp16_parameter_state_name_count": 271,
            "preserved_fp32_parameter_state_name_count": 104,
            "all_smoke_gates_passed": False,
            "full_evaluation_authorized": False,
            "training_performed": False,
            "gate_results": {"ready": True, "raw_output_parity_within_limits": False},
            "parity_summary": {
                "pred_logits": {"passed": True},
                "pred_depth": {
                    "passed": False,
                    "limit_max_abs": 0.5,
                    "max_abs": M56B_FAILED_DEPTH_MAX_ABS,
                },
            },
            "pred_depth_channel_summary": {
                "depth_m": {"max_abs": M56B_FAILED_DEPTH_MAX_ABS}
            },
        }
        validate_m56b_result_records(manifest, smoke)
        smoke["parity_summary"]["pred_depth"]["max_abs"] = 0.49
        with self.assertRaises(RuntimeError):
            validate_m56b_result_records(manifest, smoke)

    def test_group_policy_is_alias_consistent_and_complete(self):
        model = TinyGroupedModel()
        grouped, m56b_fp32, inventory = collect_grouped_storage_policy(model, torch)
        self.assertTrue(all(grouped[name] for name in GROUP_ORDER))
        aliases = [name for group in grouped.values() for name in group]
        self.assertEqual(len(aliases), len(set(aliases)))
        self.assertIn("bbox_embed.0.weight", grouped["prediction_heads"])
        self.assertIn("bbox_embed.1.weight", grouped["prediction_heads"])
        self.assertIn(
            "det2d_transformer.bbox_embed.weight", grouped["prediction_heads"]
        )
        self.assertNotIn(
            "det2d_transformer.bbox_embed.weight", grouped["det2d_transformer"]
        )
        self.assertIn("bbox_embed.0.weight", m56b_fp32)
        self.assertNotIn("class_embed.weight", m56b_fp32)
        self.assertEqual(
            inventory["all_eligible_unique_parameter_bytes"],
            sum(row["unique_parameter_bytes"] for row in inventory["groups"].values()),
        )

    def test_matrix_has_references_singletons_and_complements(self):
        model = TinyGroupedModel()
        grouped, m56b_fp32, inventory = collect_grouped_storage_policy(model, torch)
        total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        specs = build_candidate_specs(grouped, m56b_fp32, inventory, total)
        self.assertEqual(len(specs), 2 + 2 * len(GROUP_ORDER))
        names = {spec["name"] for spec in specs}
        self.assertIn("reference_alias_consistent_all_eligible", names)
        self.assertIn("reference_m56b_geometry_heads_fp32", names)
        for group in GROUP_ORDER:
            self.assertIn(f"only_{group}", names)
            self.assertIn(f"all_except_{group}", names)
        m56b = next(
            spec for spec in specs if spec["name"] == "reference_m56b_geometry_heads_fp32"
        )
        fp16_names = candidate_fp16_names(m56b, grouped, m56b_fp32)
        self.assertFalse(fp16_names & m56b_fp32)

    def test_rounding_changes_only_explicit_names(self):
        source = {
            "a": torch.tensor([1.0001], dtype=torch.float32),
            "b": torch.tensor([2.0], dtype=torch.float32),
            "counter": torch.tensor([3], dtype=torch.int64),
        }
        candidate = rounded_state_dict(source, {"a"}, torch)
        self.assertEqual(candidate["a"].dtype, torch.float16)
        self.assertEqual(candidate["b"].dtype, torch.float32)
        self.assertTrue(torch.equal(candidate["b"], source["b"]))
        self.assertTrue(torch.equal(candidate["counter"], source["counter"]))

    def test_output_comparison_separates_depth_channels(self):
        baseline = {
            "pred_logits": torch.tensor([[[0.0]]]),
            "pred_boxes": torch.zeros(1, 1, 6),
            "pred_3d_dim": torch.zeros(1, 1, 3),
            "pred_depth": torch.tensor([[[10.0, 0.2]]]),
            "pred_angle": torch.zeros(1, 1, 24),
            "pred_depth_map_logits": torch.zeros(1, 1, 1, 1),
            "pred_region_prob": torch.zeros(1, 1, 1, 1),
        }
        candidate = {name: value.clone() for name, value in baseline.items()}
        candidate["pred_depth"][..., 0] += 0.6
        result = compare_outputs(baseline, candidate, torch)
        self.assertFalse(result["all_parity_limits_passed"])
        self.assertEqual(result["failed_output_families"], ["pred_depth"])
        self.assertAlmostEqual(
            result["pred_depth_channel_summary"]["depth_m"]["max_abs"], 0.6, places=5
        )
        self.assertEqual(
            result["pred_depth_channel_summary"]["log_variance"]["max_abs"], 0.0
        )

    def test_cli_entrypoints_import_from_repo_root(self):
        for script in (
            "prepare_monodgp_m56c_group_sensitivity.py",
            "diagnose_monodgp_m56c_fp16_groups.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_contract_and_notebook_lock_complete_evaluation(self):
        contract = CONTRACT.read_text(encoding="utf-8")
        for group in GROUP_ORDER:
            self.assertIn(f"`{group}`", contract)
        self.assertIn("singleton", contract)
        self.assertIn("complement", contract)
        self.assertIn("never authorizes the 3,769-image validation", contract)
        self.assertIn(M56B_MANIFEST_SHA256, contract)
        self.assertIn(M56B_SMOKE_SHA256, contract)

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
        self.assertIn("prepare_monodgp_m56c_group_sensitivity.py", code)
        self.assertIn("diagnose_monodgp_m56c_fp16_groups.py", code)
        self.assertNotIn("evaluate_monodgp", code)
        self.assertNotIn("AUTHORIZE_COMPLETE_EVALUATION", code)
        self.assertIn("no-full-evaluation", markdown)
        self.assertIn("Final stop", markdown)
        self.assertIn("selects no compressed model", markdown)


if __name__ == "__main__":
    unittest.main()
