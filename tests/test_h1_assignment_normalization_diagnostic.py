import argparse
import unittest
from pathlib import Path

from scripts.diagnose_h1_assignment_normalization import assignment_summary, parse_checkpoint


class H1AssignmentNormalizationDiagnosticTests(unittest.TestCase):
    def test_checkpoint_parser_requires_positive_step(self):
        self.assertEqual(parse_checkpoint("400=/tmp/a.pt"), (400, Path("/tmp/a.pt")))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_checkpoint("bad")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_checkpoint("0=/tmp/a.pt")

    def test_assignment_summary_measures_query_churn(self):
        by_step = {
            400: {"a:0": {"query_id": 1, "iou_2d": 0.2}, "b:0": {"query_id": 3, "iou_2d": 0.3}},
            800: {"a:0": {"query_id": 1, "iou_2d": 0.4}, "b:0": {"query_id": 4, "iou_2d": 0.5}},
            1200: {"a:0": {"query_id": 1, "iou_2d": 0.6}, "b:0": {"query_id": 3, "iou_2d": 0.7}},
        }
        summary = assignment_summary(by_step)
        self.assertEqual(summary["common_objects"], 2)
        self.assertEqual(summary["adjacent_same_query_rate"], 0.5)
        self.assertEqual(summary["fully_stable_object_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
