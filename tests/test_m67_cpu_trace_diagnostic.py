from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_m67_a2_cpu_trace import (
    compare_boundaries,
    first_divergent_boundary,
    flatten_tensors,
)


class FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class M67CpuTraceDiagnosticTests(unittest.TestCase):
    def test_flatten_keeps_nested_paths_and_nested_tensor_fields(self):
        class Feature:
            tensors = FakeTensor([1.0, 2.0])
            mask = FakeTensor([False, True])

        actual = flatten_tensors({"features": [Feature()]}, FakeTorch)
        self.assertEqual(set(actual), {"root.features[0].tensors", "root.features[0].mask"})

    def test_boundary_comparison_reports_first_component_with_drift(self):
        reference = [
            {"component": "backbone", "leaves": {"root": np.array([1.0], dtype=np.float32)}},
            {"component": "transformer", "leaves": {"root": np.array([2.0], dtype=np.float32)}},
        ]
        actual = [
            {"component": "backbone", "leaves": {"root": np.array([1.0], dtype=np.float32)}},
            {"component": "transformer", "leaves": {"root": np.array([2.25], dtype=np.float32)}},
        ]
        rows = compare_boundaries(reference, actual)
        self.assertEqual(first_divergent_boundary(rows), "transformer")
        self.assertEqual(rows[1]["leaves"][0]["max_abs"], 0.25)
        self.assertTrue(rows[0]["leaves"][0]["bitwise_equal"])

    def test_boundary_comparison_detects_structure_mismatch(self):
        rows = compare_boundaries(
            [{"component": "backbone", "leaves": {"root": np.ones((1, 2))}}],
            [{"component": "backbone", "leaves": {"root": np.ones((1, 3))}}],
        )
        self.assertFalse(rows[0]["structure_match"])
        self.assertEqual(first_divergent_boundary(rows), "backbone")


if __name__ == "__main__":
    unittest.main()
