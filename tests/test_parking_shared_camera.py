"""Camera math, matched initialization, gradients and resumable pilot checks."""
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
from torch import nn

from models.parking_shared import ParkingDepthDecoder
from models.parking_shared_camera import CameraAwareDepthDecoder
from scripts import parking_edge_common as common
from scripts import train_parking_shared_camera as camera
from scripts import train_parking_shared_near as near


def sample(i=0, role="train"):
    return {"camera_token": f"{role}{i}", "role": role, "width": 1600, "height": 900,
        "intrinsics": [[1200., 0., 800.], [0., 1200., 450.], [0., 0., 1.]],
        "counts": {k: 5 for k in json.loads(camera.data.CONFIG.read_text())["critical_cells"]}}


def features(batch=1):
    return [torch.randn(batch, c, h, w) for c, h, w in [(64, 8, 16), (128, 4, 8), (256, 2, 4)]]


def row():
    rng = np.random.default_rng(123)
    return {**{f"p{i+3}": rng.normal(size=shape).astype(np.float32) for i, shape in enumerate([(64,8,16),(128,4,8),(256,2,4)])},
        "z": np.repeat([3., 5., 3., 5.], 5).astype(np.float32),
        "semantic": np.repeat([1, 1, 2, 2], 5).astype(np.uint8),
        "conservative": np.ones(20, bool), "grid": rng.uniform(-1, 1, (20, 2)).astype(np.float32)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3)).eval().requires_grad_(False)
        self.depth_decoder = ParkingDepthDecoder(width=16)

    def full_urban_candidates(self, values):
        return values[0].mean((2, 3))


class CameraPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_half_pixel_ray_invariance(self):
        for w, h in [(1600,900), (901,1601), (801,601)]:
            s = sample(); s.update(width=w, height=h)
            fx, fy, cx, cy = camera.camera_vector(s).astype(float)
            nw, nh = round(w*min(640/w,384/h)), round(h*min(640/w,384/h))
            sx, sy = nw/w, nh/h
            left, top = (640-nw)//2, (384-nh)//2
            for x, y in [(0.,0.),(37.2,143.7),(w-1.,h-1.)]:
                xx, yy = sx*(x+.5)+left-.5, sy*(y+.5)+top-.5
                self.assertAlmostEqual((xx-cx)/fx, (x-800)/1200, places=6)
                self.assertAlmostEqual((yy-cy)/fy, (y-450)/1200, places=6)

    def test_invalid_calibration_rejected(self):
        for k in [[[0.,0.,1.],[0.,2.,1.],[0.,0.,1.]], [[2.,1.,1.],[0.,2.,1.],[0.,0.,1.]],
                  [[2.,0.,float("nan")],[0.,2.,1.],[0.,0.,1.]], [[1,2],[3,4]]]:
            s = sample(); s["intrinsics"] = k
            with self.assertRaises(ValueError):
                camera.camera_vector(s)

    def test_camera_fields_at_pixel_centers(self):
        decoder = CameraAwareDepthDecoder(ParkingDepthDecoder())
        fields = decoder.camera_fields(torch.tensor([[640.,640.,319.5,191.5]]), (48,80))
        self.assertEqual(tuple(fields.shape), (1,4,48,80))
        self.assertAlmostEqual(float(fields[0,0,0,0]), (3.5-319.5)/640, places=7)
        self.assertAlmostEqual(float(fields[0,1,0,0]), (3.5-191.5)/640, places=7)
        self.assertEqual(float(fields[0,2].abs().max()), 0.)

    def test_zero_initialization_bit_exact_and_parameter_matched(self):
        torch.manual_seed(4)
        base = ParkingDepthDecoder().eval()
        adapter = CameraAwareDepthDecoder(copy.deepcopy(base)).eval()
        x = features()
        original = base(x)
        for arm in camera.configuration()["arms"]:
            self.assertTrue(torch.equal(original, adapter(x, camera.camera_batch([sample()], arm, "cpu"))))
        self.assertEqual(sum(p.numel() for p in base.parameters()), 57441)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 58994)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for name, m in adapter.named_modules() if not name.startswith("base")))

    def test_camera_input_can_change_prediction_after_learning(self):
        adapter = CameraAwareDepthDecoder(ParkingDepthDecoder()).eval()
        nn.init.normal_(adapter.residual.weight, std=.1)
        x = features()
        ref = camera.camera_batch([sample()], "reference_camera", "cpu")
        actual = camera.camera_batch([sample()], "actual_camera", "cpu")
        self.assertFalse(torch.equal(ref, actual))
        self.assertFalse(torch.equal(adapter(x, ref), adapter(x, actual)))
        value = adapter(x, actual)
        ratio = value/adapter.base(x)
        self.assertTrue((ratio >= .5).all() and (ratio <= 2).all())

    def test_finite_training_step_changes_adapter_and_base(self):
        decoder = CameraAwareDepthDecoder(ParkingDepthDecoder(width=16))
        config = {**json.loads(camera.data.CONFIG.read_text()), **camera.configuration()}
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"])
        before = copy.deepcopy(decoder.state_dict())
        for _ in range(2):
            loss, norm = camera.train_step(decoder, optimizer, [row() for _ in range(8)],
                [sample(i) for i in range(8)], "actual_camera", "cpu", {"Vehicle":[1],"Pedestrian":[2]}, config)
            self.assertTrue(np.isfinite(loss) and np.isfinite(norm))
        after = decoder.state_dict()
        self.assertFalse(torch.equal(before["residual.weight"], after["residual.weight"]))
        self.assertFalse(torch.equal(before["base.output.1.weight"], after["base.output.1.weight"]))

    def test_selection_includes_zero_and_prefers_earlier_ties(self):
        history = [{"epoch": i, "dev": {"equal_four_cell_mae_m": score}} for i, score in enumerate([.8,.9,.8])]
        self.assertEqual(camera.select_best(history)["epoch"], 0)
        with self.assertRaises(ValueError):
            camera.select_best(history[1:])

    def test_initial_replay_rejects_changed_error_or_support(self):
        reference = {"cells":{"Vehicle:2-4m":{"images":57,"points":25213,"macro_image_mae_m":.8284}}}
        camera.validate_initial_replay(copy.deepcopy(reference), reference)
        for key, value in [("images",56), ("points",25212), ("macro_image_mae_m",.84)]:
            changed = copy.deepcopy(reference); changed["cells"]["Vehicle:2-4m"][key] = value
            with self.assertRaises(ValueError):
                camera.validate_initial_replay(changed, reference)

    def test_research_checks_require_gain_and_protect_each_cell(self):
        keys = json.loads(camera.data.CONFIG.read_text())["critical_cells"]
        def metric(error):
            return {"equal_four_cell_mae_m":error,
                "cells":{k:{"macro_image_mae_m":error,"fixed_4m_mae_m":1.} for k in keys}}
        self.assertTrue(all(camera.research_gates(metric(1.), metric(.9), metric(.8), camera.configuration()).values()))
        changed = metric(.8); changed["cells"][keys[0]]["macro_image_mae_m"] = 1.1
        self.assertFalse(all(camera.research_gates(metric(1.), metric(.9), changed, camera.configuration()).values()))
        self.assertFalse(all(camera.research_gates(metric(1.), metric(.9), metric(.9), camera.configuration()).values()))

    def test_sampler_is_identical_for_both_arms_and_rejects_dev(self):
        config = {**json.loads(camera.data.CONFIG.read_text()), **camera.configuration()}
        samples = [sample(i) for i in range(16)]
        self.assertEqual(near.epoch_batches(samples, config, 8), near.epoch_batches(samples, config, 8))
        with self.assertRaises(ValueError):
            near.epoch_batches([sample(0, "dev")], config, 8)

    def test_development_evaluation_rejects_training_rows(self):
        config = {**json.loads(camera.data.CONFIG.read_text()), **camera.configuration()}
        with self.assertRaises(ValueError):
            camera.evaluate(None, [sample()], Path("unused"), "actual_camera", "cpu", {}, config)

    def test_cache_reuse_checks_identity_and_every_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); identity = {"runtime": "fixture"}
            common.immutable(path / "cache_identity.json", common.encoded(identity))
            self.assertFalse(camera.matching_cache(path, identity, [sample()]))
            for suffix in [".npz", ".json"]:
                (path / (sample()["camera_token"]+suffix)).touch()
            with patch.object(camera.pilot, "read_features", return_value=row()) as read:
                self.assertTrue(camera.matching_cache(path, identity, [sample()]))
                self.assertEqual(read.call_count, 1)
                self.assertFalse(camera.matching_cache(path, {"runtime":"other"}, [sample()]))

    def test_warmstart_hash_checked_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.zip"; path.write_bytes(b"not the reviewed archive")
            with patch("torch.load") as load:
                with self.assertRaises(ValueError):
                    camera.read_warmstart(path, load=True)
                load.assert_not_called()

    def test_synthetic_end_to_end_resume_and_bundle(self):
        """Real optimizer/checkpoints/selection, tiny fixture features; no real training."""
        config = json.loads(camera.data.CONFIG.read_text())
        manifest = {"config": config, "mapping":{"Vehicle":[1],"Pedestrian":[2]},
            "samples": [sample(i) for i in range(8)] + [sample(i, "dev") for i in range(2)]}
        def build(*args):
            torch.manual_seed(44)
            return TinyModel(), {"runtime":{"device":"cpu"}}
        torch.manual_seed(44); initial = TinyModel().depth_decoder.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = SimpleNamespace(stage="train", inputs=root/"inputs", warmstart=root/"warm.zip",
                output=root/"run", source=root/"source", weights=root/"weights", cache=root/"cache",
                reuse_cache=root/"oldcache", device="cpu")
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(camera, "prepare", return_value=manifest))
                reviewed = {k: {"images":2, "points":10, "macro_image_mae_m":0.} for k in config["critical_cells"]}
                # Bind the tiny fixture's own initialization, not the real
                # epoch-7 scores. The production guard stays unchanged.
                original_evaluate = camera.evaluate
                def fixture_evaluate(*args):
                    value = original_evaluate(*args)
                    if not reviewed[config["critical_cells"][0]]["macro_image_mae_m"]:
                        for key in reviewed:
                            reviewed[key] = copy.deepcopy(value["cells"][key])
                    return value
                stack.enter_context(patch.object(camera, "evaluate", side_effect=fixture_evaluate))
                stack.enter_context(patch.object(camera, "read_warmstart", return_value=({}, {"selected_dev":{"cells":reviewed}}, {"decoder":initial})))
                stack.enter_context(patch.object(camera.pilot, "build_model", side_effect=build))
                stack.enter_context(patch.object(camera.body_verification, "verify_warmstart_body", return_value={"passed":True}))
                stack.enter_context(patch.object(camera, "cache_identity", return_value={"fixture":True}))
                stack.enter_context(patch.object(camera, "matching_cache", return_value=True))
                stack.enter_context(patch.object(camera, "smoke", return_value={"complete":True,"optimizer_steps":0}))
                stack.enter_context(patch.object(camera.pilot, "read_features", side_effect=lambda *args:row()))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                camera.run(a)
                first = (a.output/"camera_pilot_results.json").read_bytes()
                camera.run(a)
                self.assertEqual(first, (a.output/"camera_pilot_results.json").read_bytes())
                common.immutable(a.output/"prepared_inputs.json", common.encoded({"fixture":True}))
                camera.bundle(a); camera.bundle(a)
            result = json.loads(first)
            for arm in camera.configuration()["arms"]:
                self.assertEqual(result["arms"][arm]["optimizer_steps"], 5)
                selection = json.loads((a.output/arm/"selection.json").read_text())
                self.assertEqual(len(selection["history"]), 6)
            self.assertFalse(result["deployment_qualified"])
            self.assertFalse(result["heldout_evaluated"])
            with zipfile.ZipFile(a.output/"parking_shared_camera_results.zip") as z:
                self.assertIn("camera_pilot_results.json", z.namelist())

    def test_notebook_clean_code_and_staging_order(self):
        path = camera.ROOT/"notebooks/Parking_Shared_Camera_Aware_Pilot_Colab.ipynb"
        notebook = json.loads(path.read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in [c for c in notebook["cells"] if c["cell_type"] == "code"]:
            self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]))
        self.assertIn("'restore'", cells[1])
        self.assertIn("stage('prepare')", cells[1])
        self.assertIn("stage('smoke')", cells[2])
        self.assertIn("stage('cache')", cells[3])
        self.assertIn("stage('train')", cells[4])
        self.assertIn("stage('bundle')", cells[5])
        self.assertNotIn("reset", "\n".join(cells))


if __name__ == "__main__":
    unittest.main()
