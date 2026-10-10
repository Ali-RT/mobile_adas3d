"""Synthetic oracle tests: no downloaded weights, KITTI data or CUDA needed."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data.compact_kitti import (CompactKittiDataset, build_targets, image_affine,
                               project_keypoints, rotation_target, transform_points)
from data.kitti_parser import KittiObject
from models.compact_kitti_loss import (CompactKittiLoss, decode, decode_alpha,
                                     geometry, offsets, solve_center)
from models.compact_monocular3d import CompactMonocular3D, OUTPUT_CHANNELS
from scripts.train_compact_kitti import (atomic_checkpoint, atomic_json, config,
                                       finite_update, restore_rng, rng_state, run, setup_seed)


def fixture():
    # Different fx/fy and all translation entries exercise full P2.
    p = np.asarray([[720., 0., 620., 45.], [0., 710., 185., 0.4], [0., 0., 1., 0.002]], np.float32)
    xyz = [2.5, 1.7, 22.]
    yaw = 0.7
    alpha = yaw - math.atan2(xyz[0], xyz[2])
    obj = KittiObject("Car", 0., 0, alpha, [600., 135., 760., 265.], [1.6, 1.7, 4.1], xyz, yaw)
    return p, obj


def oracle(output_hw=(96, 320)):
    p, obj = fixture()
    affine = image_affine(1242, 375, output_hw)
    t = build_targets([obj], p, affine, output_hw=output_hw, max_objects=4)
    batch = {"targets": {k: v.unsqueeze(0) for k, v in t.items()},
             "projection": torch.from_numpy(p).unsqueeze(0),
             "inverse_affine": torch.from_numpy(np.linalg.inv(np.vstack([affine, [0, 0, 1]]))[:2].astype(np.float32)).unsqueeze(0)}
    outputs = {name: torch.zeros((1, c, *output_hw)) for name, c in OUTPUT_CHANNELS.items()}
    outputs["hm"].fill_(-15)
    outputs["hm_hp"].fill_(-15)
    i = int(t["indices"][0]); y, x = divmod(i, output_hw[1])
    outputs["hm"][0, 0, y, x] = 15
    for name in ("wh", "reg", "hps", "dim"):
        outputs[name][0, :, y, x] = t[name][0]
    for b, off in enumerate((0, 4)):
        outputs["rot"][0, off:off+2, y, x] = torch.tensor([-15., 15.]) if t["rotbin"][0, b] else torch.tensor([15., -15.])
        outputs["rot"][0, off+2:off+4, y, x] = torch.tensor([t["rotres"][0, b].sin(), t["rotres"][0, b].cos()])
    outputs["prob"][0, 0, y, x] = 15
    return batch, outputs, obj


class CompactKittiTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_projection_solve_roundtrip_full_p2(self):
        p, obj = fixture()
        keypoints = torch.tensor(project_keypoints(obj, p)).reshape(1, 1, 9, 2)
        dim = torch.tensor(obj.dimensions_3d).reshape(1, 1, 3)
        center = solve_center(keypoints.double(), dim.double(), torch.tensor([[obj.rotation_y]], dtype=torch.float64),
                              torch.tensor(p).double().unsqueeze(0))
        expected = torch.tensor(obj.location_3d, dtype=torch.float64)
        expected[1] -= obj.dimensions_3d[0]/2
        torch.testing.assert_close(center[0, 0], expected, rtol=2e-5, atol=2e-4)

    def test_rotation_covers_all_angles(self):
        for alpha in np.linspace(-math.pi, math.pi, 61):
            bins, residuals = rotation_target(alpha)
            rotation = torch.zeros(1, 1, 8)
            for b, off in enumerate((0, 4)):
                rotation[..., off:off+2] = torch.tensor([0., 10.]) if bins[b] else torch.tensor([10., 0.])
                rotation[..., off+2] = math.sin(residuals[b])
                rotation[..., off+3] = math.cos(residuals[b])
            delta = float(decode_alpha(rotation)) - alpha
            self.assertLess(abs(math.atan2(math.sin(delta), math.cos(delta))), 1e-6)

    def test_oracle_decode_is_correct_bottom_center(self):
        batch, output, obj = oracle()
        decoded = decode(output, batch["projection"], batch["inverse_affine"], topk=1)
        row = decoded[0][0]
        self.assertEqual(row["class_name"], "Vehicle")
        self.assertFalse(row["invalid_geometry"])
        np.testing.assert_allclose(row["location_3d"], obj.location_3d, atol=0.001, rtol=1e-4)
        np.testing.assert_allclose(row["bbox_2d"], obj.bbox_2d, atol=1e-3)
        self.assertAlmostEqual(row["rotation_y"], obj.rotation_y, places=4)

    def test_ignore_regions_and_empty_images(self):
        p, obj = fixture()
        obj.class_name = "DontCare"
        a = image_affine(1242, 375, (96, 320))
        t = build_targets([obj], p, a, max_objects=4)
        self.assertEqual(float(t["hm"].sum()), 0.)
        self.assertLess(float(t["hm_valid"].sum()), t["hm_valid"].numel())
        t = build_targets([], p, a, max_objects=4)
        self.assertEqual(float(t["geom_mask"].sum()), 0.)
        self.assertEqual(float(t["hm_valid"].sum()), t["hm_valid"].numel())

    def test_do_not_silently_truncate_objects(self):
        p, obj = fixture()
        with self.assertRaisesRegex(ValueError, "silently truncate"):
            build_targets([obj]*5, p, image_affine(1242, 375, (96, 320)), max_objects=4)

    def test_all_heads_have_finite_gradients_and_oracle_lower_loss(self):
        batch, output, _ = oracle((8, 32))
        criterion = CompactKittiLoss()
        for value in output.values():
            value.requires_grad_(True)
        exact, terms = criterion(output, batch)
        exact.backward()
        self.assertTrue(torch.isfinite(exact))
        self.assertEqual(set(terms), {"hm", "wh", "reg", "hps", "dim", "rot", "hm_hp", "hp_offset", "position", "prob"})
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in output.values()))
        bad = {k: torch.zeros_like(v) for k, v in output.items()}
        incorrect, _ = criterion(bad, batch)
        self.assertLess(float(exact.detach()), float(incorrect.detach()))

    def test_real_model_gradient_update_no_native_extension(self):
        batch, _, _ = oracle((8, 32))
        batch["image"] = torch.randn(2, 3, 32, 128)
        batch["targets"] = {k: v.repeat(2, *([1] * (v.ndim-1))) for k, v in batch["targets"].items()}
        batch["projection"] = batch["projection"].repeat(2, 1, 1)
        batch["inverse_affine"] = batch["inverse_affine"].repeat(2, 1, 1)
        model = CompactMonocular3D(backbone="mobilenetv4_conv_small")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        value = finite_update(model, CompactKittiLoss(), optimizer, batch, 5., 1.)
        self.assertTrue(math.isfinite(value["total"]))

    def test_full_phone_resolution_forward_backward(self):
        batch, _, _ = oracle()
        batch["image"] = torch.randn(1, 3, 384, 1280)
        model = CompactMonocular3D(backbone="mobilenetv4_conv_small")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        update = finite_update(model, CompactKittiLoss(), optimizer, batch, 5., 1.)
        self.assertTrue(math.isfinite(update["total"]))
        self.assertTrue(math.isfinite(update["gradient_norm"]))

    def test_rng_and_atomic_checkpoint_weights_only_resume(self):
        setup_seed(123)
        saved = rng_state()
        expected = (random_value(), torch.rand(3), np.random.rand(3))
        restore_rng(saved)
        actual = (random_value(), torch.rand(3), np.random.rand(3))
        self.assertEqual(expected[0], actual[0])
        torch.testing.assert_close(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"latest.pt"
            atomic_checkpoint(path, {"rng": saved, "epoch": 1, "model": {"x": torch.tensor([1.])}})
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["epoch"], 1)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_config_freezes_graph_fp32_and_no_teacher(self):
        cfg = config(Path(__file__).resolve().parents[1]/"configs/compact_mnv4_kitti_gt_r1.yaml")
        self.assertFalse(cfg["amp"])
        self.assertIsNone(cfg["external_teacher"])
        with tempfile.TemporaryDirectory() as directory:
            import yaml
            path = Path(directory)/"bad.yaml"
            cfg["external_teacher"] = "R0"
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                config(path)

    def test_dataset_full_resolution_rgb_contract(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("image_2", "label_2", "calib"):
                (root/"training"/name).mkdir(parents=True)
            # Asymmetric color catches the earlier BGR/RGB ambiguity.
            image = np.full((375, 1242, 3), [0, 64, 255], dtype=np.uint8)
            cv2.imwrite(str(root/"training/image_2/000001.png"), image)
            (root/"training/label_2/000001.txt").write_text(
                "Car 0 0 0.58 600 135 760 265 1.6 1.7 4.1 2.5 1.7 22 0.7\n")
            p, _ = fixture()
            (root/"training/calib/000001.txt").write_text("P2: " + " ".join(map(str, p.flatten())) + "\n")
            value = CompactKittiDataset(root, ["000001"], max_objects=4)[0]
            self.assertEqual(tuple(value["image"].shape), (3, 384, 1280))
            expected = (np.asarray([1., 64/255, 0.]) - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225])
            np.testing.assert_allclose(value["image"][:, 192, 640], expected, atol=1e-5)
            self.assertEqual(float(value["targets"]["geom_mask"].sum()), 1.)

    def test_train_resume_evaluate_bundle_integration(self):
        """Use a tiny synthetic loader, not a claimed KITTI accuracy run."""
        batch, _, obj = oracle((8, 32))
        batch["sample_id"] = ["000001"]
        batch["image"] = torch.randn(1, 3, 32, 128)
        cfg = config(Path(__file__).resolve().parents[1]/"configs/compact_mnv4_kitti_gt_r1.yaml")
        cfg.update(pretrained_backbone=False, epochs=1, workers=0, batch_size=1,
                   save_every=1, evaluate_every=1, topk=1, score_threshold=0.999)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"run"
            output.mkdir()
            atomic_json(output/"tiny.json", {"passed": True, "identity_sha256": "synthetic-test"})
            args = SimpleNamespace(stage="train", config=root/"config.yaml", dataset_root=root,
                                   split_dir=root, output_dir=output, device="cpu",
                                   checkpoint=None, resume=False, steps=2)
            manifest = {"identity_sha256": "synthetic-test"}
            with patch("scripts.train_compact_kitti.config", return_value=cfg), \
                    patch("scripts.train_compact_kitti.identity", return_value=(manifest, ["000001"], ["000001"])), \
                    patch("scripts.train_compact_kitti.loader", return_value=[batch]), \
                    patch("scripts.train_compact_kitti.parse_kitti_label_file", return_value=[obj]):
                run(args)
                saved = torch.load(output/"checkpoints/latest.pt", weights_only=True)
                self.assertEqual(saved["epoch"], 1)
                self.assertFalse(saved["external_teacher_used"])
                initial_hash = json.loads((output/"initialization.json").read_text())["initial_checkpoint_sha256"]
                with self.assertRaisesRegex(RuntimeError, "--resume"):
                    run(args)
                args.resume = True
                run(args)
                self.assertEqual(json.loads((output/"initialization.json").read_text())["initial_checkpoint_sha256"], initial_hash)
                args.stage = "evaluate"
                run(args)
                report = json.loads((output/"evaluations/epoch_001/accuracy.json").read_text())
                self.assertTrue(report["complete"])
                self.assertEqual(report["samples"], 1)
                self.assertFalse(report["deployment_qualified"])
                args.stage = "bundle"
                run(args)
                self.assertTrue((output/"compact_mnv4_gt_results.zip").is_file())


def random_value():
    import random
    return random.random()


if __name__ == "__main__":
    unittest.main()
