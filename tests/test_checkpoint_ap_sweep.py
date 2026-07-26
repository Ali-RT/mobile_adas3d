import tempfile
import unittest
from pathlib import Path

from scripts.sweep_kitti_r40_checkpoints import (
    CheckpointCandidate,
    best_rows_by_metric,
    collect_checkpoint_candidates,
    summarize_checkpoint_metrics,
)


class CheckpointAPSweepTests(unittest.TestCase):
    def test_collect_checkpoint_candidates_orders_epochs_then_best_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            for filename in [
                "latest.pt",
                "epoch_010.pt",
                "notes.txt",
                "epoch_005.pt",
                "best.pt",
            ]:
                (checkpoint_dir / filename).touch()

            candidates = collect_checkpoint_candidates(checkpoint_dir)

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["epoch_005", "epoch_010", "best", "latest"],
        )

    def test_summarize_checkpoint_metrics_writes_wide_ap_columns(self):
        checkpoint = CheckpointCandidate(
            name="epoch_080",
            path=Path("/tmp/run/checkpoints/epoch_080.pt"),
            sort_key=(0, 80, "epoch_080.pt"),
        )
        evaluation_summary = {
            "evaluated_images": 3769,
            "split_images": 3769,
            "complete_split": True,
            "score_threshold": 0.001,
            "topk": 300,
            "nms_iou_threshold": 0.5,
            "metrics": [
                {
                    "metric": "3d",
                    "class_name": "Car",
                    "difficulty": "moderate",
                    "ap_r40": 2.87,
                },
                {
                    "metric": "3d",
                    "class_name": "Pedestrian",
                    "difficulty": "moderate",
                    "ap_r40": 1.07,
                },
                {
                    "metric": "3d",
                    "class_name": "Cyclist",
                    "difficulty": "moderate",
                    "ap_r40": 1.04,
                },
                {
                    "metric": "bev",
                    "class_name": "Car",
                    "difficulty": "moderate",
                    "ap_r40": 6.30,
                },
            ],
        }

        row = summarize_checkpoint_metrics(
            checkpoint=checkpoint,
            evaluation_summary=evaluation_summary,
            checkpoint_metadata={"epoch": 80, "global_step": 37120},
        )

        self.assertEqual(row["checkpoint"], "epoch_080")
        self.assertEqual(row["epoch"], 80)
        self.assertEqual(row["ap_3d_Car_moderate"], 2.87)
        self.assertEqual(row["selection_score"], 2.87)
        self.assertAlmostEqual(row["mean_3d_moderate"], (2.87 + 1.07 + 1.04) / 3)

    def test_best_rows_by_metric_selects_highest_value(self):
        rows = [
            {"checkpoint": "epoch_075", "ap_3d_Car_moderate": 2.4},
            {"checkpoint": "epoch_080", "ap_3d_Car_moderate": 2.87},
        ]

        best = best_rows_by_metric(rows, ["ap_3d_Car_moderate"])

        self.assertEqual(best["ap_3d_Car_moderate"]["checkpoint"], "epoch_080")


if __name__ == "__main__":
    unittest.main()
