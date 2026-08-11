from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MonoDETRCoreMLExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = (
            ROOT / "third_party/monodetr/coreml_export.patch"
        ).read_text()
        cls.patcher = (
            ROOT / "scripts/patch_monodetr_coreml_export.py"
        ).read_text()
        cls.probe = (
            ROOT / "scripts/probe_coreml_full_monodetr.py"
        ).read_text()

    def test_training_paths_remain_available(self):
        self.assertIn("if self.coreml_export:", self.patch)
        self.assertNotIn("-        output = MSDeformAttnFunction.apply(", self.patch)
        self.assertIn("else:", self.patch)

    def test_export_removes_rank_six_and_inplace_updates(self):
        self.assertIn("self.n_levels * self.n_points, 2", self.patch)
        self.assertIn("ms_deform_attn_core_coreml", self.patch)
        self.assertIn("torch.cat([tmp[..., :2] + reference", self.patch)

    def test_patcher_is_pinned_and_idempotent(self):
        self.assertIn("6994b9f512400b258c6edb75f77423beb9c126f2", self.patcher)
        self.assertIn('return "already_patched"', self.patcher)

    def test_probe_uses_locked_interface(self):
        self.assertIn("torch.rand(1, 3, 384, 1280)", self.probe)
        self.assertIn('skip_model_load=not args.compile_model', self.probe)
        self.assertIn('"mil_has_custom_op"', self.probe)


if __name__ == "__main__":
    unittest.main()
