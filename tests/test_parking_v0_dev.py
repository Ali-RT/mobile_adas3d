"""CPU synthetic integration and interruption tests; no model downloads."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image
from scripts import evaluate_parking_v0_dev as v0
from scripts import parking_edge_common as common
from scripts import prepare_parking_v0_v1_benchmark as prepare

spec = importlib.util.spec_from_file_location("v0_metric_fixture", prepare.ROOT / "tests/test_parking_v0_v1_benchmark.py")
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def signed(value):
    return {**value, "manifest_sha256": common.identity(value)}


def make_fixture(root, count=2):
    protocol = prepare.load_protocol(); samples = []; shards = []; truth = {}
    benchmark_id = "synthetic-frozen-identity"
    for i in range(count):
        _, sample, points = fixture.metric_fixture(f"c{i}", f"log{i}", f"ann{i}")
        sample.update(role="dev", sample_token=f"frame{i}", scene=f"scene{i}", source_part=1,
            coverage_row_sha256=f"row{i}", intrinsics=[[100., 0., 49.5], [0., 100., 4.5], [0., 0., 1.]])
        image = io.BytesIO(); Image.new("RGB", (sample["width"], sample["height"]), (70, 120, 80)).save(image, format="JPEG")
        gt_raw = prepare.encode_npz(points)
        sample.update(image_member=f"images/c{i}.jpg", image_sha256=common.sha(image.getvalue()),
            ground_truth_member=f"ground_truth/c{i}.npz", ground_truth_sha256=common.sha(gt_raw))
        members = {sample["image_member"]: image.getvalue(), sample["ground_truth_member"]: gt_raw,
            "shard_manifest.json": common.encoded(signed({"benchmark_identity_sha256": benchmark_id, "samples": [sample]}))}
        raw = v0.zip_bytes(members); name = f"shards/dev/part01_{i:03d}.zip"
        common.immutable(root / name, raw)
        shards.append({"file": name, "sha256": common.sha(raw), "size_bytes": len(raw)})
        samples.append({**sample, "shard_file": name, "shard_sha256": common.sha(raw)})
        truth[sample["camera_token"]] = points
    # Its body deliberately does not exist. A dev evaluation must not open it.
    samples.append({"camera_token": "secret", "role": "heldout_eval", "shard_file": "shards/heldout_eval/secret.zip"})
    shards.append({"file": "shards/heldout_eval/secret.zip", "sha256": "not-opened", "size_bytes": 0})
    manifest = signed({"complete": True, "benchmark_identity_sha256": benchmark_id,
        "metric_protocol": protocol, "samples": samples, "shards": shards})
    config = v0.configuration(); config.update(benchmark_manifest_sha256=manifest["manifest_sha256"],
        benchmark_identity_sha256=benchmark_id, expected_images=count)
    identity = {"config": config, "model_registration": {"model": "synthetic"}}
    return manifest, config, identity, truth


class Predictor:
    def __init__(self, fail_on=None): self.calls = 0; self.fail_on = fail_on
    def predict(self, sample, image, points):
        self.calls += 1
        if self.calls == self.fail_on: raise RuntimeError("simulated interruption")
        return points["z"].copy() + np.float32(.5)
    def assert_unchanged(self): pass


class V0Tests(unittest.TestCase):
    def test_config_is_cpu_dev_only(self):
        c = v0.configuration()
        self.assertEqual((c["role"], c["device"], c["epoch"], c["expected_images"]), ("dev", "cpu", 4, 512))
        for k in ["training_authorized", "heldout_scoring_authorized", "phone_app_changed", "accuracy_pass_fail_enabled"]:
            self.assertFalse(c[k])

    def test_only_dev_shards_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            m, c, _, _ = make_fixture(Path(folder))
            inventory = v0.dev_inventory(m, c)
            self.assertEqual(len(inventory), 2)
            self.assertTrue(all(s["file"].startswith("shards/dev/") for s, _ in inventory))

    def test_changed_manifest_or_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            m, c, _, _ = make_fixture(Path(folder))
            m["samples"][0]["intrinsics"][0][0] += 1
            with self.assertRaises(ValueError): v0.dev_inventory(m, c)
            m = signed({k: v for k, v in m.items() if k != "manifest_sha256"})
            with self.assertRaises(ValueError): v0.dev_inventory(m, c)

    def test_heldout_alias_and_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            m, c, _, _ = make_fixture(Path(folder))
            c["expected_images"] = 3
            with self.assertRaises(ValueError): v0.dev_inventory(m, c)
            c["expected_images"] = 2
            m["samples"][0]["shard_file"] = "shards/heldout_eval/secret.zip"
            m = signed({k: v for k, v in m.items() if k != "manifest_sha256"}); c["benchmark_manifest_sha256"] = m["manifest_sha256"]
            with self.assertRaises(ValueError): v0.dev_inventory(m, c)

    def test_atomic_shard_resume_after_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, identity, _ = make_fixture(root / "benchmark")
            inventory = v0.dev_inventory(m, c); output = root / "output"; local = root / "cache"
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                v0.score_dev(m, inventory, root/"benchmark", output, local, Predictor(2), identity)
            self.assertFalse((output/"v0_dev_results.json").exists())
            self.assertEqual(len(list((output/"predictions").glob("*.zip"))), 1)
            predictor = Predictor()
            result = v0.score_dev(m, inventory, root/"benchmark", output, local, predictor, identity)
            self.assertEqual(predictor.calls, 1)
            self.assertEqual(result["metrics"]["conditional_equal_four_cell_mae_m"], .5)
            self.assertEqual(result["heldout_images_opened"], 0)
            predictor = Predictor(1)
            again = v0.score_dev(m, inventory, root/"benchmark", output, local, predictor, identity)
            self.assertEqual(predictor.calls, 0); self.assertEqual(result, again)

    def test_runtime_identity_change_does_not_mix_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, identity, _ = make_fixture(root/"benchmark", 1)
            inventory = v0.dev_inventory(m, c)
            v0.score_dev(m, inventory, root/"benchmark", root/"out", root/"local", Predictor(), identity)
            identity["runtime"] = "changed"
            with self.assertRaises(RuntimeError):
                v0.score_dev(m, inventory, root/"benchmark", root/"out", root/"local", Predictor(), identity)

    def test_evaluate_publishes_small_verified_result_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _, _ = make_fixture(root/"benchmark", 1)
            common.immutable(root/"benchmark/benchmark_manifest.json", common.encoded(m))
            p = Predictor(); p.registration = {"artifact_sha256": "synthetic"}
            args = SimpleNamespace(benchmark=root/"benchmark", bundle=root/"original.zip", source=root/"source",
                weights=root/"weights", output=root/"output", local=root/"local", device="cpu")
            with mock.patch.object(v0, "configuration", return_value=c), mock.patch.object(v0, "read_v0_bundle", return_value=({}, {})), mock.patch.object(v0, "V0Predictor", return_value=p):
                with mock.patch("builtins.print"):
                    v0.evaluate(args)
            with zipfile.ZipFile(root/"output/parking_v0_dev_results.zip") as z:
                result = json.loads(z.read("v0_dev_results.json"))
                self.assertTrue(result["complete"])
                self.assertEqual(result["heldout_images_opened"], 0)
                self.assertEqual(len(z.namelist()), 3)
                for item in result["prediction_shards"]:
                    self.assertEqual(common.sha(z.read(item["file"])), item["sha256"])

    def test_prediction_chunk_order_hash_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            m, c, _, truth = make_fixture(Path(folder), 1); rows = v0.dev_inventory(m, c)[0][1]
            pred = truth["c0"]["z"].copy(); pred[:5] = np.nan
            raw = v0.prediction_chunk("run", rows, {"c0": pred})
            loaded = v0.read_prediction_chunk(raw, "run", rows, truth)
            metric = v0.metrics.Metrics(m["metric_protocol"]); metric.add(rows[0], truth["c0"], loaded["c0"])
            self.assertEqual(metric.result()["cells"]["Vehicle:2-4m"]["prediction_coverage"], 0)
            with self.assertRaises(ValueError): v0.read_prediction_chunk(raw, "other-runtime", rows, truth)
            with self.assertRaises(ValueError): v0.prediction_chunk("run", rows, {})
            bad_points = copy.deepcopy(truth); bad_points["c0"]["z"] = pred[:-1]
            with self.assertRaises(ValueError): v0.read_prediction_chunk(raw, "run", rows, bad_points)

    def test_changed_shard_and_local_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _, _ = make_fixture(root/"benchmark", 1)
            shard, rows = v0.dev_inventory(m, c)[0]
            path = v0.local_shard(root/"benchmark", root/"local", shard)
            v0.load_dev_shard(path, shard, rows, m["metric_protocol"], m["benchmark_identity_sha256"])
            path.write_bytes(b"interrupted or changed")
            with self.assertRaises(ValueError): v0.local_shard(root/"benchmark", root/"local", shard)
            with self.assertRaises(ValueError): v0.load_dev_shard(path, shard, rows, m["metric_protocol"], m["benchmark_identity_sha256"])

    def test_safe_archive_names(self):
        for value in ["../escape", "/absolute", "x\\y", ""]:
            with self.assertRaises(ValueError): v0.safe_name(value)

    def test_wrong_v0_bundle_rejected_before_tensor_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"fake.zip"; p.write_bytes(b"not the frozen model")
            with self.assertRaises(ValueError): v0.read_v0_bundle(p, v0.configuration(), load=True)

    def test_notebook_cells_compile_and_are_cpu_only(self):
        notebook = json.loads((v0.ROOT/"notebooks/Parking_V0_Development_Baseline_CPU_Colab.ipynb").read_text())
        source = ""
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
                ast.parse(text); source += text
        self.assertIn("parking_shared_adapter_results.zip", source)
        self.assertIn("'evaluate-local'", source); self.assertIn("CPU runtime", source)
        self.assertNotIn("nvidia-smi", source)


if __name__ == "__main__": unittest.main()
