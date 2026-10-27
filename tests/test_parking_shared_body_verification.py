"""Cross-runtime fusion is bounded; source weights and run state stay exact."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import parking_edge_common as common
from scripts import parking_shared_body_verification as verification
from scripts.screen_parking_shared_model import state_digest


class TensorBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fused", torch.tensor([2., -3.]))
        self.register_buffer("untouched", torch.tensor([.25]))
        self.register_buffer("integer", torch.tensor([7], dtype=torch.int64))


def fixture():
    body = TensorBody()
    reference = copy.deepcopy(body).double()
    original = {k: v.clone() for k, v in body.state_dict().items()}
    original["fused"] = torch.ones(2)
    body.fused[0] += 1e-6
    return body, reference, original


class FusionVerificationTests(unittest.TestCase):
    def test_fp32_rounding_passes_without_replacing_weights(self):
        body, reference, original = fixture(); before = state_digest(body)
        check = verification.compare_fused_tensors(body, reference, original)
        self.assertTrue(check["passed"])
        self.assertEqual(check["tensor_values_compared"], 4)
        self.assertEqual(check["exact_source_tensors"], 2)
        self.assertEqual(check["numerically_checked_fused_tensors"], 1)
        self.assertGreater(check["maximum_fusion_abs_delta"], 0)
        self.assertEqual(state_digest(body), before)

    def test_changed_fused_weights_rejected(self):
        body, reference, original = fixture(); body.fused[0] += .001
        with self.assertRaisesRegex(ValueError, "rounding bounds"):
            verification.compare_fused_tensors(body, reference, original)

    def test_untouched_weights_require_exact_match_not_tolerance(self):
        body, reference, original = fixture(); body.untouched[0] += 1e-7
        with self.assertRaisesRegex(ValueError, "Unchanged source tensor"):
            verification.compare_fused_tensors(body, reference, original)

    def test_shape_dtype_inventory_integer_and_nonfinite_rejected(self):
        for kind in ["shape", "dtype", "inventory", "integer", "nan"]:
            body, reference, original = fixture()
            if kind == "shape":
                body.fused = torch.zeros(3)
            elif kind == "dtype":
                body.half()
            elif kind == "inventory":
                body.register_buffer("extra", torch.ones(1))
            elif kind == "integer":
                body.integer[0] += 1
            else:
                body.fused[0] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verification.compare_fused_tensors(body, reference, original)

    def test_source_mismatch_stops_before_loading_pickled_checkpoint(self):
        config = json.loads((common.ROOT / "configs/parking_shared_depth_near_v1.json").read_text())
        architecture = json.loads((common.ROOT / config["architecture_config"]).read_text())
        model_id = {"architecture": architecture, "source_checkpoint_sha256": "wrong"}
        with patch.object(verification, "fusion_reference") as loader:
            with self.assertRaisesRegex(ValueError, "source identity"):
                verification.verify_warmstart_body(SimpleNamespace(), model_id, config, Path("missing"))
            loader.assert_not_called()

    def test_changed_checkpoint_bytes_stop_before_loading(self):
        config = json.loads((common.ROOT / "configs/parking_shared_depth_near_v1.json").read_text())
        architecture = json.loads((common.ROOT / config["architecture_config"]).read_text())
        model_id = {"architecture": architecture, "source_checkpoint_sha256": verification.CHECKPOINT_SHA256}
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory); (weights / architecture["urban_checkpoint"]["file"]).write_bytes(b"changed")
            with patch.object(verification, "fusion_reference") as loader:
                with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                    verification.verify_warmstart_body(SimpleNamespace(), model_id, config, weights)
                loader.assert_not_called()

    def test_unfused_digest_checked_before_reference_fusion(self):
        body = TensorBody(); native = SimpleNamespace(model=body)
        native.float = lambda: native; native.cpu = lambda: native; native.eval = lambda: native
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda *a, **k: SimpleNamespace(model=native))}):
            with self.assertRaisesRegex(ValueError, "Unfused Urban tensors"):
                verification.fusion_reference(Path("already-hash-verified.pt"))

    def test_runtime_digest_recorded_even_when_not_bitwise_equal_to_mac(self):
        body, reference, original = fixture(); model = SimpleNamespace(body=body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = b"fixture checkpoint"; digest = common.sha(raw)
            (root / "fixture.pt").write_bytes(raw)
            architecture = {"urban_checkpoint": {"file": "fixture.pt", "sha256": digest}}
            (root / "architecture.json").write_text(json.dumps(architecture))
            config = {"architecture_config": "architecture.json", "initial_body_sha256": "reviewed-mac"}
            model_id = {"architecture": architecture, "source_checkpoint_sha256": digest}
            with patch.object(common, "ROOT", root), patch.object(verification, "CHECKPOINT_SHA256", digest), \
                    patch.object(verification, "PILOT_FUSED_BODY_SHA256", "reviewed-mac"), \
                    patch.object(verification, "fusion_reference", return_value=(reference, original)):
                check = verification.verify_warmstart_body(model, model_id, config, root)
            self.assertTrue(check["passed"])
            self.assertFalse(check["bitwise_equal_to_pilot_fusion"])
            self.assertFalse(check["weights_changed"])
            self.assertEqual(check["runtime_fused_body_sha256"], state_digest(body))

    def test_verification_cannot_mutate_even_within_rounding_tolerance(self):
        body, reference, original = fixture(); model = SimpleNamespace(body=body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = b"fixture checkpoint"; digest = common.sha(raw)
            (root / "fixture.pt").write_bytes(raw)
            architecture = {"urban_checkpoint": {"file": "fixture.pt", "sha256": digest}}
            (root / "architecture.json").write_text(json.dumps(architecture))
            config = {"architecture_config": "architecture.json", "initial_body_sha256": "reviewed-mac"}
            model_id = {"architecture": architecture, "source_checkpoint_sha256": digest}
            def changed_reference(*args):
                body.fused[0] += 1e-6
                return reference, original
            with patch.object(common, "ROOT", root), patch.object(verification, "CHECKPOINT_SHA256", digest), \
                    patch.object(verification, "PILOT_FUSED_BODY_SHA256", "reviewed-mac"), \
                    patch.object(verification, "fusion_reference", side_effect=changed_reference):
                with self.assertRaisesRegex(ValueError, "verification changed"):
                    verification.verify_warmstart_body(model, model_id, config, root)


if __name__ == "__main__":
    unittest.main()
