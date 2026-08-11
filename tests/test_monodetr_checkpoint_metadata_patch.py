from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "third_party/monodetr/checkpoint_metadata_order.patch"


class MonoDETRCheckpointMetadataPatchTests(unittest.TestCase):
    def test_resumable_checkpoint_is_finalized_after_validation(self):
        patch = PATCH.read_text()
        metric_update = patch.index("best_epoch = self.epoch")
        final_save = patch.index(
            "+                    save_checkpoint(\n"
            "+                        get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),\n"
            "+                        ckpt_name)"
        )
        self.assertLess(metric_update, final_save)
        self.assertIn("Finalized resumable checkpoint", patch)

    def test_best_and_resumable_checkpoint_names_are_distinct(self):
        patch = PATCH.read_text()
        self.assertIn("best_ckpt_name = os.path.join", patch)
        self.assertIn("best_ckpt_name)", patch)
        self.assertIn("ckpt_name)", patch)


if __name__ == "__main__":
    unittest.main()
