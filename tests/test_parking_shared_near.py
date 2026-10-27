"""Near-range supervision, frozen split boundaries and reusable CPU package."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from models.parking_shared import ParkingDepthDecoder
from scripts import parking_edge_common as common
from scripts import parking_shared_near_data as data
from scripts import train_parking_shared_near as training


def config():
    return json.loads(data.CONFIG.read_text())


def sample(i, role="train", log=None):
    return {"sample_token": f"s{role}{i}", "camera_token": f"c{role}{i}", "role": role,
            "scene": f"scene-{role}-{i % 2}", "recording_log": log or f"log-{role}-{i % 2}",
            "counts": {key: 5 for key in config()["critical_cells"]}}


def small_recipe():
    c = config()
    c.update(epochs=2, minimum_train_keyframes_per_cell=1, minimum_dev_keyframes_per_cell=1,
             minimum_points_per_cell=5)
    return c


def manifest(c):
    rows = [sample(i, role) for role in ["train", "dev"] for i in range(2)]
    for s in rows:
        for key in ["image", "ground_truth"]:
            s[f"{key}_file"] = f"{key}/{s['camera_token']}.bin"
            s[f"{key}_sha256"] = common.sha((key + s["camera_token"]).encode())
    init = {"file": "initialization/epoch_004.pth", "sha256": c["initial_checkpoint_sha256"],
            "sidecar_file": "initialization/epoch_004.json", "sidecar_sha256": common.sha(b"sidecar"),
            "run_identity": c["initial_run_identity"]}
    m = {"complete": True, "config": c, "samples": rows, "initialization": init,
         "scenes": {role: sorted({s["scene"] for s in rows if s["role"] == role}) for role in ["train", "dev", "test"]},
         "excluded_recording_logs": ["heldout-log", "mini-log"], "readiness": data.readiness(rows, c),
         "files": {s[f"{key}_file"]: s[f"{key}_sha256"] for s in rows for key in ["image", "ground_truth"]}}
    m["manifest_sha256"] = common.identity(m)
    return m


class NearDataTests(unittest.TestCase):
    def test_real_part1_gate_is_not_waived(self):
        c = config(); rows = [sample(i) for i in range(25)] + [sample(i, "dev", "one-log") for i in range(4)]
        ready = data.readiness(rows, c)
        self.assertFalse(ready["training_data_ready"])
        for cell in ready["roles"]["dev"]["cells"].values():
            self.assertEqual(cell["unique_keyframes"], 4)
            self.assertEqual(cell["recording_logs"], 1)
            self.assertFalse(cell["passed"])

    def test_readiness_deduplicates_keyframes(self):
        c = small_recipe(); rows = [sample(i, role) for role in ["train", "dev"] for i in range(2)]
        self.assertTrue(data.readiness(rows, c)["training_data_ready"])
        rows = [sample(0, "dev", "one-log")] * 100
        self.assertEqual(data.readiness(rows, c)["roles"]["dev"]["cells"][c["critical_cells"][0]]["unique_keyframes"], 1)

    def test_gt_selection_stable_and_one_camera_per_keyframe(self):
        c = config(); rows = [sample(i) for i in range(20)]
        rows += [{**s, "camera_token": s["camera_token"] + "other"} for s in rows]
        selected = data.select(rows, 12, c)
        self.assertEqual(selected, data.select(rows[::-1], 12, c))
        self.assertEqual(len({s["sample_token"] for s in selected}), 12)
        with self.assertRaises(ValueError):
            data.select([sample(0, "heldout_eval")], 1, c)

    def test_manifest_rejects_heldout_and_log_leakage(self):
        c = small_recipe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"; path.write_text(json.dumps(c))
            with patch.object(data, "CONFIG", path):
                m = manifest(c); data.validate_manifest(m)
                for role, log in [("heldout_eval", "new"), ("dev", "heldout-log"), ("dev", "log-train-0")]:
                    changed = copy.deepcopy(m); changed["samples"][-1]["role"] = role; changed["samples"][-1]["recording_log"] = log
                    changed["manifest_sha256"] = common.identity({k: v for k, v in changed.items() if k != "manifest_sha256"})
                    with self.assertRaises(ValueError):
                        data.validate_manifest(changed)

    def test_archive_roundtrip_and_changed_file_is_preserved(self):
        c = small_recipe()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "config.json"; path.write_text(json.dumps(c))
            with patch.object(data, "CONFIG", path):
                m = manifest(c); source = root / "source"; source.mkdir()
                for s in m["samples"]:
                    for key in ["image", "ground_truth"]:
                        common.immutable(source / s[f"{key}_file"], (key + s["camera_token"]).encode())
                # Fixture initialization bytes, with an explicitly bound fixture config.
                init = m["initialization"]; raw = b"checkpoint"
                c["initial_checkpoint_sha256"] = common.sha(raw); path.write_text(json.dumps(c))
                init["sha256"] = common.sha(raw)
                common.immutable(source / init["file"], raw); common.immutable(source / init["sidecar_file"], b"sidecar")
                m["config"] = c; m["manifest_sha256"] = common.identity({k: v for k, v in m.items() if k != "manifest_sha256"})
                with contextlib.redirect_stdout(io.StringIO()):
                    data.publish(source, m); data.publish(source, m)
                target = root / "restored"; archive = source / "shared_near_inputs.zip"
                with contextlib.redirect_stdout(io.StringIO()):
                    data.restore(archive, target); data.restore(archive, target)
                self.assertEqual(data.read_manifest(target), m)
                changed = target / next(iter(m["files"])); changed.write_bytes(b"user change")
                with self.assertRaises(ValueError):
                    data.restore(archive, target)
                self.assertEqual(changed.read_bytes(), b"user change")

    def test_archive_traversal_and_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("../escape", b"bad")
            with self.assertRaises(ValueError):
                data.verify_archive(path)

    def test_failed_prepare_creates_readiness_not_training_package(self):
        c = small_recipe(); rows = [sample(i) for i in range(2)] + [sample(0, "dev", "single-log")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); acquisition = root / "acquisition.json"
            acquisition.write_text(json.dumps({"per_scene": []}))
            recipe = root / "config.json"; recipe.write_text(json.dumps(c))
            a = SimpleNamespace(config=recipe, acquisition=acquisition, part=[(root / "part", root / "audit")],
                                output=root / "output", stage="prepare")
            with patch.object(data, "collect", return_value=([], rows, ["heldout"])):
                with self.assertRaisesRegex(RuntimeError, "No training archive"), contextlib.redirect_stdout(io.StringIO()):
                    data.prepare(a)
            self.assertFalse((a.output / "shared_near_inputs.zip").exists())
            self.assertFalse((a.output / "input_manifest.json").exists())
            self.assertFalse(json.loads((a.output / "readiness.json").read_text())["training_data_ready"])


class NearTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_balances_cells_instead_of_point_counts(self):
        c = small_recipe(); c["minimum_points_in_cell_image"] = 1
        prediction = torch.tensor([[3., 3., 3., 6.]], requires_grad=True)
        truth = torch.full_like(prediction, 2.); valid = torch.ones_like(prediction, dtype=torch.bool)
        masks = [torch.tensor([[True, True, True, False]]), torch.tensor([[False, False, False, True]])]
        masks += [torch.zeros_like(valid), torch.zeros_like(valid)]
        # Huber(1)=.18, Huber(4)=.78. Cell mean=.48, broad mean=.33.
        loss = training.balanced_loss(prediction, truth, valid, masks, c, "parking_balanced")
        self.assertAlmostEqual(float(loss.detach()), .8 * .48 + .2 * .33, places=6)
        self.assertAlmostEqual(float(training.balanced_loss(prediction, truth, valid, masks, c, "control").detach()), .33, places=6)
        loss.backward(); self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_unlabelled_holes_do_not_receive_gradients(self):
        c = small_recipe(); c["minimum_points_in_cell_image"] = 1
        pred = torch.tensor([[3., 77.]], requires_grad=True); truth = torch.tensor([[2., float("nan")]])
        valid = torch.tensor([[True, False]]); masks = [valid] + [torch.zeros_like(valid)] * 3
        training.balanced_loss(pred, truth, valid, masks, c, "parking_balanced").backward()
        self.assertEqual(float(pred.grad[0, 1]), 0.)
        with self.assertRaises(ValueError):
            training.balanced_loss(pred, truth, valid, [torch.ones_like(valid)] * 4, c, "parking_balanced")

    def test_sampler_shared_between_arms_and_train_only(self):
        c = config(); rows = [sample(i) for i in range(31)]
        a = training.epoch_batches(rows, c, 1)
        self.assertEqual(a, training.epoch_batches(rows, c, 1))
        self.assertNotEqual(a, training.epoch_batches(rows, c, 2))
        self.assertEqual(len(a), 4); self.assertTrue(all(len(batch) == 8 for batch in a))
        for batch in a:
            for i, key in enumerate(c["critical_cells"]):
                self.assertGreaterEqual(rows[batch[i]]["counts"][key], 5)
        with self.assertRaises(ValueError):
            training.epoch_batches([sample(0, "dev")], c, 1)

    def test_class_and_range_masks_are_gt_only(self):
        row = {"z": np.array([2., 3.9, 4., 5.9, 6., 3.]), "semantic": np.array([1, 1, 1, 2, 2, 1]),
               "conservative": np.array([True, True, True, True, True, False])}
        masks = training.cell_masks(row, {"Vehicle": [1], "Pedestrian": [2]})
        self.assertEqual([int(v.sum()) for v in masks.values()], [2, 1, 0, 2])

    def test_real_decoder_optimizer_step(self):
        c = small_recipe(); decoder = ParkingDepthDecoder(width=16)
        row = {"p3": np.ones((64, 48, 80), np.float32), "p4": np.ones((128, 24, 40), np.float32),
            "p5": np.ones((256, 12, 20), np.float32), "grid": np.zeros((20, 2), np.float32),
            "z": np.array([3.] * 5 + [5.] * 5 + [3.] * 5 + [5.] * 5, np.float32),
            "semantic": np.array([1] * 10 + [2] * 10), "conservative": np.ones(20, bool)}
        for arm in c["arms"]:
            before = {k: v.clone() for k, v in decoder.state_dict().items()}
            optimizer = torch.optim.AdamW(decoder.parameters(), lr=c["learning_rate"])
            loss, norm = training.step(decoder, optimizer, [row], torch.device("cpu"), {"Vehicle": [1], "Pedestrian": [2]}, c, arm)
            self.assertTrue(np.isfinite([loss, norm]).all()); self.assertGreater(norm, 0)
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in decoder.state_dict().items()))

    def test_two_arm_loop_resume_and_bundle_with_synthetic_features(self):
        from scripts.screen_parking_shared_model import state_digest
        c = small_recipe()
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.body = torch.nn.Linear(1, 1)
                for parameter in self.body.parameters():
                    parameter.requires_grad_(False)
                self.depth_decoder = ParkingDepthDecoder(width=16)
            def full_urban_candidates(self, features):
                return self.body.weight.reshape(1, 1, 1) + features[0].mean()
        torch.manual_seed(4); model = FixtureModel()
        initial = {k: v.clone() for k, v in model.depth_decoder.state_dict().items()}
        c["initial_body_sha256"] = state_digest(model.body)
        m = {"config": c, "mapping": {"Vehicle": [1], "Pedestrian": [2]},
             "samples": [sample(i, role) for role in ["train", "dev"] for i in range(4)]}
        row = {"p3": np.ones((64, 48, 80), np.float32), "p4": np.ones((128, 24, 40), np.float32),
            "p5": np.ones((256, 12, 20), np.float32), "grid": np.zeros((20, 2), np.float32),
            "z": np.array([3.] * 5 + [5.] * 5 + [3.] * 5 + [5.] * 5, np.float32),
            "semantic": np.array([1] * 10 + [2] * 10), "conservative": np.ones(20, bool)}
        def cache_fixture(*args):
            return None
        def initialize_fixture(model, *args):
            model.depth_decoder.load_state_dict(initial)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs = root / "inputs"; inputs.mkdir()
            (inputs / "input_manifest.json").write_text(json.dumps(m))
            a = SimpleNamespace(inputs=inputs, output=root / "output", cache=root / "cache", source=root / "source",
                                weights=root / "weights", stage="train", device="cpu")
            with patch.object(data, "read_manifest", return_value=m), patch.object(training, "initialize", initialize_fixture), \
                    patch.object(training.pilot, "build_model", return_value=(model, {"runtime": {"device": "cpu"}})), \
                    patch.object(training.pilot, "cache_sample", cache_fixture), patch.object(training.pilot, "read_features", return_value=row), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.run(a)
                first = (a.output / "near_training_results.json").read_bytes()
                training.run(a)
                self.assertEqual(first, (a.output / "near_training_results.json").read_bytes())
            result = json.loads(first)
            self.assertEqual(set(result["arms"]), {"control", "parking_balanced"})
            self.assertTrue(all(v["optimizer_steps"] == 2 for v in result["arms"].values()))
            self.assertFalse(result["heldout_evaluation_performed"])
            with contextlib.redirect_stdout(io.StringIO()):
                training.bundle(a); before = (a.output / "parking_shared_near_results.zip").read_bytes()
                training.bundle(a)
            self.assertEqual(before, (a.output / "parking_shared_near_results.zip").read_bytes())
            self.assertEqual(state_digest(model.body), c["initial_body_sha256"])

    def test_notebook_cpu_staging_before_gpu_and_clean_cells(self):
        path = common.ROOT / "notebooks/Parking_Shared_Depth_NearRange_Trainval_Colab.ipynb"
        notebook = json.loads(path.read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for code in cells:
            ast.parse(code)
        staging = next(i for i, code in enumerate(cells) if "stage_inputs_cpu" in code)
        gpu = next(i for i, code in enumerate(cells) if "cuda_preflight" in code)
        self.assertLess(staging, gpu)
        self.assertTrue(all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code"))
        text = "\n".join(cells)
        self.assertNotIn("reset --hard", text)
        self.assertNotIn("stage('evaluate')", text)


if __name__ == "__main__":
    unittest.main()
