"""Sparse metric supervision, leakage, frozen data and safe pilot checkpoints."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
import torch

from models.parking_shared import ParkingDepthDecoder
from scripts import parking_edge_common as common
from scripts import parking_shared_depth_data as data
from scripts import train_parking_shared_depth as training
from scripts.screen_parking_shared_model import preprocess


def small_manifest():
    samples = []
    for role in ["train", "dev", "test"]:
        samples.append(dict(camera_token=role, sample_token=role, role=role, scene=role,
                            image_file=f"images/{role}.jpg", image_sha256=common.sha(role.encode()),
                            ground_truth_file=f"ground_truth/{role}.npz", ground_truth_sha256=common.sha(b"gt")))
    m = dict(complete=True, scenes={k: [k] for k in ["train", "dev", "test"]}, samples=samples,
             files={s[f"{name}_file"]: s[f"{name}_sha256"] for s in samples for name in ["image", "ground_truth"]})
    m["manifest_sha256"] = common.identity(m)
    return m


class SharedDepthTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_scene_split_deterministic_and_heldout_never_enters_train_or_dev(self):
        cfg = {"seed": 7, "dev_scenes": 2}; names = [f"scene-{i}" for i in range(8)]
        a = data.scene_split(names, ["test-a", "test-b"], cfg)
        self.assertEqual(a, data.scene_split(names[::-1], ["test-b", "test-a"], cfg))
        self.assertEqual(len(a["train"]), 6); self.assertEqual(len(a["dev"]), 2)
        with self.assertRaises(ValueError): data.scene_split(names, ["scene-0", "test-b"], cfg)

    def test_manifest_guards_scene_keyframe_and_file_inventory(self):
        original = small_manifest(); data.validate_manifest(original)
        for change in ["scene", "keyframe", "files", "digest"]:
            m = copy.deepcopy(original)
            if change == "scene": m["scenes"]["dev"] = ["train"]
            if change == "keyframe": m["samples"][1]["sample_token"] = "train"
            if change == "files": m["files"]["extra"] = "bad"
            if change != "digest": m["manifest_sha256"] = common.identity({k: v for k, v in m.items() if k != "manifest_sha256"})
            else: m["manifest_sha256"] = "bad"
            with self.assertRaises(ValueError): data.validate_manifest(m)

    def test_letterbox_pixel_centre_grid_preserves_sparse_sampling(self):
        _, metadata = preprocess(np.zeros((720, 1280, 3), np.uint8))
        xy = np.array([[0, 0], [1279, 719], [639, 359]], np.int32)
        observed = data.point_grid(xy, 1280, 720, metadata)
        expected = 2 * ((xy + .5) * [.5, .5] + [0, 12]) / [640, 384] - 1
        np.testing.assert_allclose(observed, expected, atol=1e-7)
        with self.assertRaises(ValueError): data.point_grid(np.array([[1280, 1]]), 1280, 720, metadata)

    def test_sampling_uses_prediction_bilinear_not_interpolated_truth(self):
        depth = torch.tensor([[[[2., 4.], [6., 8.]]]])
        grid = torch.tensor([[[[-.5, -.5]], [[.5, .5]], [[0., 0.]]]])
        torch.testing.assert_close(training.sample_depth(depth, grid), torch.tensor([[2., 8., 5.]]))

    def test_portable_sampler_matches_reference_values_and_depth_gradients(self):
        torch.manual_seed(9)
        depth = torch.randn(2, 1, 7, 11, requires_grad=True)
        other = depth.detach().clone().requires_grad_(True)
        grid = torch.rand(2, 31, 1, 2) * 2.4 - 1.2
        observed = training.sample_depth(depth, grid)
        expected = torch.nn.functional.grid_sample(other, grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
        weights = torch.randn_like(observed)
        (observed * weights).sum().backward(); (expected * weights).sum().backward()
        torch.testing.assert_close(depth.grad, other.grad, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_real_mps_sampling_backward(self):
        depth = torch.ones(2, 1, 8, 12, device="mps", requires_grad=True)
        grid = torch.zeros(2, 13, 1, 2, device="mps")
        training.sample_depth(depth, grid).sum().backward()
        self.assertTrue(torch.isfinite(depth.grad).all()); self.assertGreater(float(depth.grad.sum().cpu()), 0)

    def test_per_image_loss_equal_weights_and_unlabelled_holes_ignored(self):
        prediction = torch.tensor([[3., 99.], [4., 4.]], requires_grad=True)
        truth = torch.tensor([[2., float("nan")], [2., 2.]])
        valid = torch.tensor([[True, False], [True, True]])
        loss = training.sparse_loss(prediction, truth, valid, .2)
        self.assertAlmostEqual(float(loss.detach()), .28, places=6)
        loss.backward(); self.assertEqual(prediction.grad[0, 1], 0)
        with self.assertRaises(ValueError): training.sparse_loss(prediction, truth, torch.zeros_like(valid), .2)

    def test_true_optimizer_step_updates_only_new_decoder(self):
        torch.manual_seed(7); decoder = ParkingDepthDecoder(width=16)
        before = {k: v.clone() for k, v in decoder.state_dict().items()}
        row = {"p3": np.ones((64, 48, 80), np.float32), "p4": np.ones((128, 24, 40), np.float32),
               "p5": np.ones((256, 12, 20), np.float32), "grid": np.array([[0., 0.], [.5, .5]], np.float32),
               "z": np.array([3., 3.5], np.float32), "conservative": np.ones(2, bool)}
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=.001)
        loss, norm = training.step(decoder, optimizer, [row], torch.device("cpu"), {"huber_delta_m": .2, "gradient_clip_norm": 5.})
        self.assertTrue(np.isfinite([loss, norm]).all()); self.assertGreater(norm, 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in decoder.state_dict().items()))

    def test_checkpoint_weights_only_load_and_changed_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epoch_001.pth"
            training.save_checkpoint(path, {"run_identity": "same", "decoder": {"weight": torch.ones(1)}, "epoch": 1})
            self.assertEqual(training.load_checkpoint(path, "same")["epoch"], 1)
            with self.assertRaises(ValueError): training.load_checkpoint(path, "other")
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): training.load_checkpoint(path, "same")

    def test_optimizer_and_decoder_resume_preserve_next_step(self):
        torch.manual_seed(12)
        first = torch.nn.Linear(2, 1); resumed = torch.nn.Linear(2, 1)
        one = torch.optim.AdamW(first.parameters(), lr=.001)
        features = torch.ones(3, 2)
        one.zero_grad(); first(features).square().mean().backward(); one.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epoch_001.pth"
            training.save_checkpoint(path, {"run_identity": "r", "decoder": first.state_dict(), "optimizer": one.state_dict()})
            saved = training.load_checkpoint(path, "r")
            resumed.load_state_dict(saved["decoder"]); two = torch.optim.AdamW(resumed.parameters(), lr=.001)
            two.load_state_dict(saved["optimizer"])
            for model, optimizer in [(first, one), (resumed, two)]:
                optimizer.zero_grad(); model(features).square().mean().backward(); optimizer.step()
            for a, b in zip(first.parameters(), resumed.parameters()): torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_real_bundle_stage_is_repeatable_and_contains_only_declared_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); identity = {"recipe": "fixture"}; run_id = common.identity(identity)
            for name, value in {"run_identity.json": identity, "initial_dev.json": {"mae": 2.},
                "selection.json": {"selected_epoch": 1, "run_identity": run_id},
                "depth_pilot_results.json": {"complete": True, "run_identity": run_id, "selected_epoch": 1}}.items():
                common.immutable(output / name, common.encoded(value))
            training.save_checkpoint(output / "checkpoints/epoch_001.pth", {"run_identity": run_id, "decoder": {"weight": torch.ones(1)}})
            training.bundle(SimpleNamespace(output=output))
            before = (output / "parking_shared_depth_results.zip").read_bytes()
            training.bundle(SimpleNamespace(output=output))
            self.assertEqual(before, (output / "parking_shared_depth_results.zip").read_bytes())
            with zipfile.ZipFile(io.BytesIO(before)) as archive:
                self.assertIn("depth_pilot_results.json", archive.namelist())
                self.assertIn("checkpoints/epoch_001.pth", archive.namelist())
                self.assertFalse(any(name.startswith("images/") for name in archive.namelist()))

    def test_archive_restore_reuses_valid_files_and_rejects_corruption(self):
        m = small_manifest()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); source = folder / "source"; source.mkdir()
            for name, digest in m["files"].items():
                raw = b"gt" if name.startswith("ground_truth") else Path(name).stem.encode()
                common.immutable(source / name, raw)
            data.publish_archive(source, m)
            target = folder / "restored"; data.restore(source / "shared_depth_inputs.zip", target)
            before = {p: p.stat().st_mtime_ns for p in target.rglob("*") if p.is_file()}
            data.restore(source / "shared_depth_inputs.zip", target)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in target.rglob("*") if p.is_file()})
            (target / "images/train.jpg").write_bytes(b"changed")
            with self.assertRaises(ValueError): data.restore(source / "shared_depth_inputs.zip", target)

    def test_archive_traversal_and_duplicate_members_rejected(self):
        for member in ["../escape", "/absolute", "a\\b"]:
            with self.assertRaises(ValueError): data.safe_member(member)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive: archive.writestr("../escape", "bad")
            with self.assertRaises(ValueError): data.verify_archive(path)

    def test_pilot_config_has_no_kd_no_augmentation_and_separate_dev_selection(self):
        config = json.loads(data.CONFIG.read_text())
        self.assertEqual(config["epochs"], 5); self.assertEqual(config["maximum_train_images"], 512)
        self.assertFalse(config["knowledge_distillation"]); self.assertFalse(config["augmentation"])
        self.assertTrue(config["frozen_urban_weights_and_batchnorm"])
        self.assertIn("dev", config["checkpoint_selection"])

    def test_notebook_orders_cpu_staging_before_gpu_and_has_clean_code_cells(self):
        notebook = json.loads((common.ROOT / "notebooks/Parking_Shared_Depth_GT_Pilot_Colab.ipynb").read_text())
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]); ast.parse(source); code.append(source)
                self.assertEqual(cell["outputs"], []); self.assertIsNone(cell["execution_count"])
        self.assertEqual(len(code), 6)
        joined = "\n".join(code)
        self.assertLess(joined.index("stage_inputs_cpu"), joined.index("cuda_preflight"))
        self.assertLess(joined.index("stage('train')"), joined.index("stage('evaluate')"))
        self.assertIn("ultralytics==8.4.61", joined)
        self.assertNotIn("m67_", joined); self.assertNotIn("reset', '--hard", joined)
        self.assertIn("Pedestrian 2–4 m is absent", "".join("".join(c["source"]) for c in notebook["cells"]))


if __name__ == "__main__": unittest.main()
