from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.patch_monodgp_m57_deformable_attention import (
    MARKER,
    MODE_ENVIRONMENT_VARIABLE,
    patch_source,
)
from scripts.prepare_monodgp_m57_deformable_attention import (
    EXPECTED_MODULES,
    M56D_CANDIDATE_SHA256,
    M56D_COMPARISON_SHA256,
    M56D_GATE_SHA256,
    M56D_MANIFEST_SHA256,
    M56D_SMOKE_SHA256,
    M57_PATCHED_SOURCE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]


def representative_source() -> str:
    return '''from __future__ import division

import warnings
from torch.nn import Linear as _LinearWithBias
from torch import overrides as torch_overrides

def _is_power_of_2(n):
    return True


class MSDeformAttn(nn.Module):
    def __init__(self, n_points):
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(1, 1)

    def forward(self, query):
        if self.conditional:
            value = value.view(N, Len_in, self.n_heads, (self.d_model//2) // self.n_heads)
        else:
            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q)
        output = MSDeformAttnFunction.apply(
            value, shapes, starts, locations, weights, self.im2col_step)

class MSDeformAttn_cross(nn.Module):
    def __init__(self, n_points):
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(1, 1)
    def forward(self):
        output = MSDeformAttnFunction.apply(
            value, shapes, starts, locations, weights, self.im2col_step)
'''


class MonoDGPM57Tests(unittest.TestCase):
    def test_patch_is_exact_and_idempotent(self):
        patched, result = patch_source(representative_source())
        self.assertEqual(result, "patched")
        self.assertIn(MARKER, patched)
        self.assertIn("F.grid_sample(", patched)
        self.assertIn("level_points, 2", patched)
        self.assertIn(MODE_ENVIRONMENT_VARIABLE, patched)
        self.assertEqual(patched.count("MSDeformAttnFunction.apply("), 2)
        self.assertEqual(patched.count("self.use_portable_deform_attn"), 2)
        repeated, repeated_result = patch_source(patched)
        self.assertEqual(repeated_result, "already_patched")
        self.assertEqual(repeated, patched)

    def test_patch_rejects_unrecognized_source(self):
        with self.assertRaises(RuntimeError):
            patch_source("class MSDeformAttn: pass\n")

    def test_frozen_hashes_are_complete(self):
        values = (
            M56D_MANIFEST_SHA256,
            M56D_SMOKE_SHA256,
            M56D_GATE_SHA256,
            M56D_COMPARISON_SHA256,
            M56D_CANDIDATE_SHA256,
            M57_PATCHED_SOURCE_SHA256,
        )
        self.assertTrue(all(len(value) == 64 for value in values))
        self.assertEqual(len(EXPECTED_MODULES), 9)
        self.assertEqual(len(set(EXPECTED_MODULES)), 9)

    def test_prepare_fails_closed(self):
        source = (ROOT / "scripts/prepare_monodgp_m57_deformable_attention.py").read_text()
        for evidence in (
            "M56D_MANIFEST_SHA256",
            "M56D_SMOKE_SHA256",
            "M56D_GATE_SHA256",
            "M56D_COMPARISON_SHA256",
            "M57_PATCHED_SOURCE_SHA256",
            "EXPECTED_PATCHED_FILES",
        ):
            self.assertIn(evidence, source)
        self.assertIn('"full_evaluation_authorized": False', source)
        self.assertIn('"direct_coreml_conversion_authorized": False', source)

    def test_smoke_covers_native_portable_and_trace_paths(self):
        source = (ROOT / "scripts/smoke_test_monodgp_m57_deformable_attention.py").read_text()
        for marker in (
            "CountingNative",
            "ForbiddenNative",
            "module_parity",
            "torch.jit.trace",
            "aten::grid_sampler",
            "EXPECTED_TRACE_SIGNATURES",
        ):
            self.assertIn(marker, source)
        self.assertIn("range(5)", source)
        self.assertIn("range(100)", source)
        self.assertIn("MODULE_OUTPUT_MAX_ABS_LIMIT = 1e-3", source)
        self.assertIn('"direct_coreml_conversion_authorized": False', source)

    def test_contract_freezes_first_stop(self):
        contract = (ROOT / "MONODGP_M57_DEFORMABLE_ATTENTION_CONTRACT.md").read_text()
        self.assertIn("Status: frozen before M57 execution", contract)
        self.assertIn("maximum absolute delta no greater than `1e-3`", contract)
        self.assertIn("3,769-image portable-path validation", contract)
        self.assertIn("Physical-device latency", contract)

    def test_notebook_is_standalone_and_stops_after_smoke(self):
        path = ROOT / "notebooks/MonoDGP_M57_Deformable_Attention_Colab.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("patch_monodgp_m57_deformable_attention.py", code)
        self.assertIn("prepare_monodgp_m57_deformable_attention.py", code)
        self.assertIn("smoke_test_monodgp_m57_deformable_attention.py", code)
        self.assertIn("m57_deformable_attention_manifest.json", code)
        self.assertIn("m57_deformable_attention_smoke.json", code)
        self.assertNotIn("evaluate_only", code)
        self.assertIn("Stop point 1", code)


if __name__ == "__main__":
    unittest.main()
