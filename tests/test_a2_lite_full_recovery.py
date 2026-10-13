"""Full-train scope, paired budgets, immutable pilot and recall guardrails."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]
import recover_a2_lite_full as full
import recover_a2_lite_640 as old
from test_a2_lite_recovery import ToyStudent, target, outputs


def evaluated(ap=1., vehicle=.8, pedestrian=.6):
    return dict(metrics=dict(vehicle_3d_moderate=ap, pedestrian_3d_moderate=ap,
                             mean_3d_moderate=ap, vehicle_bev_moderate=ap, pedestrian_bev_moderate=ap),
                nearby=dict(classes={name: dict(near_recall=value) for name, value in
                                     (("Vehicle", vehicle), ("Pedestrian", pedestrian))}))


class FullRecoveryTests(unittest.TestCase):
    def test_only_dataset_and_budget_change_the_training_recipe(self):
        self.assertEqual(full.RECIPE["train_images"], 3712)
        self.assertEqual(full.RECIPE["epochs"], 10)
        self.assertEqual(full.RECIPE["train_images"] * full.RECIPE["epochs"], 37120)
        for key, value in old.RECIPE.items():
            if key not in ("train_images", "epochs", "pedestrian_containing_frames", "vehicle_only_frames"):
                self.assertEqual(full.RECIPE[key], value, key)
        self.assertFalse(full.RECIPE["training_objective_changed"])
        self.assertFalse(full.GUARDRAILS["automatic_geometry_only_arm"])

    def test_recipe_context_restores_frozen_pilot_even_after_failure(self):
        original, revision = old.RECIPE, old.REVISION
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with full.recipe_context():
                self.assertIs(old.RECIPE, full.RECIPE)
                self.assertEqual(old.REVISION, full.REVISION)
                raise RuntimeError("synthetic")
        self.assertIs(old.RECIPE, original)
        self.assertEqual(old.REVISION, revision)
        self.assertEqual(old.RECIPE["train_images"], 512)

    def test_all_train_frames_used_without_class_or_validation_filter(self):
        train = [f"{n:06d}" for n in range(3712)]
        val = [f"{n:06d}" for n in range(3712, 7481)]
        self.assertEqual(full.all_train_ids(train, val), train)
        for bad_train, bad_val in ((train[:-1], val), (train + train[:1], val),
                                  (train, val[:-1]), (train, [train[0]] + val[1:])):
            with self.assertRaises(RuntimeError):
                full.all_train_ids(bad_train, bad_val)

    def test_higher_ap_cannot_hide_pedestrian_recall_loss(self):
        control = evaluated(ap=.8, pedestrian=.5855379188712522)
        kd = evaluated(ap=1.2, pedestrian=.5185185185185185)
        result = full.decision(control, kd)
        self.assertTrue(result["all_ap_metrics_nonregressing"])
        self.assertTrue(result["minimum_mean_3d_gain_passed"])
        self.assertFalse(result["nearby_recall_guardrails"]["Pedestrian"])
        self.assertFalse(result["kd_benefit_guardrail_passed"])
        self.assertIn("separately authorized", result["next_review"])

    def test_recall_boundary_is_two_absolute_percentage_points(self):
        c = evaluated()
        kd = evaluated(ap=1.1, vehicle=.78, pedestrian=.58)
        self.assertTrue(full.decision(c, kd)["kd_benefit_guardrail_passed"])
        kd["nearby"]["classes"]["Pedestrian"]["near_recall"] = .5799
        self.assertFalse(full.decision(c, kd)["kd_benefit_guardrail_passed"])

    def test_any_ap_regression_or_insufficient_mean_gain_rejects_kd(self):
        c = evaluated()
        d = evaluated(ap=1.2)
        d["metrics"]["pedestrian_bev_moderate"] = .99
        self.assertFalse(full.decision(c, d)["kd_benefit_guardrail_passed"])
        self.assertFalse(full.decision(c, evaluated(ap=1.05))["kd_benefit_guardrail_passed"])

    def test_nonfinite_or_invalid_metric_comparisons_rejected(self):
        for bad in (evaluated(ap=float("nan")), evaluated(ap=-1.), evaluated(pedestrian=1.2)):
            with self.assertRaises(RuntimeError):
                full.decision(evaluated(), bad)
        missing = evaluated(); missing["metrics"].pop("mean_3d_moderate")
        with self.assertRaises(RuntimeError):
            full.decision(evaluated(), missing)

    def test_load_rejects_changed_recipe_or_missing_train_frames(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [f"{n:06d}" for n in range(3712)]
            val = [f"{n:06d}" for n in range(3712, 7481)]
            m = dict(revision=full.REVISION, recipe=full.RECIPE, guardrails=full.GUARDRAILS,
                     implementation={"code": "same"}, cpu_environment={"packages": "same"},
                     output=str(root), initialization="init", source="source", source_sha256="same",
                     split_dir="splits", train_ids=train, val_ids=val, pilot_ids=train)
            def save(value):
                row = copy.deepcopy(value); row["signature_sha256"] = old.signature(row)
                old.write_json(root / "full_recovery_manifest.json", row)
            def split(path, name): return train if name == "train" else val
            save(m)
            with patch.object(full, "implementation", return_value={"code": "same"}), \
                 patch.object(old, "runtime_environment", return_value={"packages": "same"}), \
                 patch.object(full, "reviewed_pilot"), patch.object(old, "sha256", return_value=old.resolution.CHECKPOINT_SHA), \
                 patch.object(old.base.q, "source_hash", return_value="same"), patch.object(old.base.q, "split_ids", side_effect=split):
                self.assertEqual(full.load(root)["pilot_ids"], train)
                bad = copy.deepcopy(m); bad["pilot_ids"] = train[:-1]; save(bad)
                with self.assertRaises(RuntimeError): full.load(root)
                bad = copy.deepcopy(m); bad["recipe"]["epochs"] = 11; save(bad)
                with self.assertRaises(RuntimeError): full.load(root)

    def test_reviewed_pilot_hashes_and_both_endpoint_checks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = dict(signature_sha256="placeholder", recipe=dict(train_images=512, epochs=5),
                     initialization_sha256=old.resolution.CHECKPOINT_SHA)
            m["signature_sha256"] = old.signature({k: v for k, v in m.items() if k != "signature_sha256"})
            c = dict(complete=True, run_signature=m["signature_sha256"],
                     results={role: dict(complete=True, evaluated_images=3769, run_signature=m["signature_sha256"])
                              for role in ("control", "kd")})
            a = json.dumps(m).encode(); b = json.dumps(c).encode()
            (root / "recovery_manifest.json").write_bytes(a)
            (root / "recovery_comparison.json").write_bytes(b)
            with patch.object(full, "PILOT_MANIFEST_SHA", old.accuracy.sha_bytes(a)), \
                 patch.object(full, "PILOT_COMPARISON_SHA", old.accuracy.sha_bytes(b)), \
                 patch.object(full, "PILOT_SIGNATURE", m["signature_sha256"]):
                self.assertEqual(full.reviewed_pilot(root)[0], m)
                (root / "recovery_comparison.json").write_bytes(b + b" ")
                with self.assertRaises(RuntimeError):
                    full.reviewed_pilot(root)

    def test_cache_reuse_is_verified_rebound_train_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); pilot = root / "pilot"; output = root / "new"
            sample = "000001"; files = {sample: dict(image_2="image", calib="P2", label_2="GT")}
            old_m = dict(signature_sha256="old", pilot_ids=[sample], teacher_model_sha256="teacher")
            m = dict(signature_sha256="new", pilot_ids=[sample], data_file_sha256=files, teacher_model_sha256="teacher")
            row = dict(run_signature="old", sample_id=sample, source_sha256=files[sample],
                       target_signature="native-GT", outputs={"pred_depth": [10.]},
                       object_to_teacher_query=[42], target_labels=[1])
            row["record_sha256"] = old.signature(row)
            old.write_json(pilot / f"teacher_cache/{sample}.json", row)
            raw = (pilot / f"teacher_cache/{sample}.json").read_bytes()
            report = dict(complete=True, run_signature="old", cache_sha256={sample: old.accuracy.sha_bytes(raw)})
            report["report_sha256"] = old.signature(report)
            old.write_json(pilot / "teacher_cache_complete.json", report)
            report_raw = (pilot / "teacher_cache_complete.json").read_bytes()
            a = SimpleNamespace(pilot=pilot, output=output)
            with patch.object(full, "reviewed_pilot", return_value=(old_m, {}, b"", b"")), \
                 patch.object(full, "PILOT_CACHE_SHA", old.accuracy.sha_bytes(report_raw)):
                self.assertEqual(full.reuse_teacher_cache(a, m), 1)
                result = old.cache_record(m, output / "teacher_cache", sample)
                self.assertEqual(result["run_signature"], "new")
                self.assertEqual(result["reused_from_run_signature"], "old")
                self.assertEqual(result["object_to_teacher_query"], [42])
                self.assertEqual((pilot / f"teacher_cache/{sample}.json").read_bytes(), raw)
                self.assertEqual(full.reuse_teacher_cache(a, m), 0)
                bad = copy.deepcopy(m); bad["pilot_ids"] = ["000002"]
                with self.assertRaises(RuntimeError):
                    full.reuse_teacher_cache(a, bad)
                (pilot / f"teacher_cache/{sample}.json").write_bytes(raw + b" ")
                with self.assertRaises(RuntimeError):
                    full.reuse_teacher_cache(a, m)

    def test_both_roles_use_original_initialization_and_engine_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", output=str(root), pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: dict(image_2="test") for s in ("000001", "000002")})
            class Dataset:
                idx_list = m["pilot_ids"]
                def __len__(self): return len(self.idx_list)
            teacher = outputs((42, 47), True)
            values = {name: teacher[key].tolist() for name, key in zip(old.OUTPUT_NAMES, old.OUTPUT_KEYS)}
            hashes = {}
            for sample in m["pilot_ids"]:
                row = dict(run_signature="synthetic", sample_id=sample, source_sha256=m["data_file_sha256"][sample],
                           target_signature=old.target_signature(target()), outputs=values,
                           object_to_teacher_query=[42, 47], target_labels=[0, 1])
                row["record_sha256"] = old.signature(row)
                p = root / "teacher_cache" / f"{sample}.json"; old.write_json(p, row); hashes[sample] = old.sha256(p)
            old.save_report(root / "teacher_cache_complete.json", dict(complete=True, run_signature="synthetic",
                            kd_pilot_authorized=True, cache_sha256=hashes))
            starts = []
            def build(*args):
                model = ToyStudent(); starts.append(copy.deepcopy(model.state_dict())); return model
            with full.recipe_context(), patch.dict(full.RECIPE, epochs=2, kd_warmup_steps=1), \
                 patch.object(old, "tiny_gate"), patch.object(old, "build_model", side_effect=build), \
                 patch.object(old, "native_dataset", return_value=Dataset()), patch.object(old, "verify_files"), \
                 patch.object(old, "item", side_effect=lambda *args: ({}, target(), {})), \
                 patch.object(old, "forward", side_effect=lambda model, inputs: model.outputs()):
                for role in ("control", "kd"):
                    a = SimpleNamespace(output=root, device="cpu", role=role)
                    old.train(a, m); before = (root / role / "training_complete.json").read_bytes()
                    old.train(a, m)
                    self.assertEqual(before, (root / role / "training_complete.json").read_bytes())
            for state in starts[1:]:
                for name, value in starts[0].items(): self.assertTrue(torch.equal(value, state[name]))
            for role in ("control", "kd"):
                report = old.checked_report(root / role / "training_complete.json", m)
                self.assertEqual(report["optimizer_steps"], 4)
                self.assertFalse(report["tiny_checkpoint_used"])
                self.assertEqual(report["same_initialization_sha256"], old.resolution.CHECKPOINT_SHA)

    def test_bundle_never_includes_checkpoints_or_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "assets").mkdir()
            (root / "assets/manifest.json").write_text("{}")
            (root / "weights.pth").write_bytes(b"large")
            (root / "full_recovery_manifest.json").write_text("{}")
            full.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_full_recovery_results.zip") as z:
                self.assertEqual(z.namelist(), ["full_recovery_manifest.json"])

    def test_compare_keeps_product_qualification_separate_from_kd_gain(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", previous_pilot_metrics={})
            for role, ap in (("control", 1.), ("kd", 1.2)):
                endpoint = root / role / "last.pth"; endpoint.parent.mkdir(); endpoint.write_bytes(b"endpoint" + role.encode())
                digest = old.sha256(endpoint)
                old.save_report(root / role / "training_complete.json", dict(complete=True, run_signature="synthetic",
                    train_images=3712, optimizer_steps=37120, fixed_endpoint_epoch=10,
                    same_initialization_sha256=old.resolution.CHECKPOINT_SHA, checkpoint_sha256=digest,
                    external_teacher_used=role=="kd", tiny_checkpoint_used=False, architecture_changed=False))
                result = evaluated(ap=ap)
                result.update(complete=True, evaluated_images=3769, run_signature="synthetic", checkpoint_sha256=digest,
                              accuracy_gate_results={key: False for key in result["metrics"]})
                result["nearby"]["all_near_recall_gates_passed"] = False
                old.write_json(root / "evaluation" / role / "accuracy.json", result)
            full.compare(SimpleNamespace(output=root), m)
            result = old.checked_report(root / "full_recovery_comparison.json", m)
            self.assertTrue(result["kd_benefit_guardrail_passed"])
            self.assertFalse(result["deployment_qualified"])
            self.assertFalse(any(result["all_product_accuracy_gates_passed"].values()))
            self.assertFalse(result["geometry_only_training_authorized"])
            (root / "kd/last.pth").write_bytes(b"tampered")
            with self.assertRaises(RuntimeError): full.compare(SimpleNamespace(output=root), m)

    def test_standalone_notebook_complete_and_no_automatic_extra_arm(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_Full_Recovery_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for i, code in enumerate(codes): compile(code, f"full-recovery-{i}", "exec")
        text = "\n".join(codes)
        self.assertIn(full.REVISION, text)
        for stage in ("prepare", "smoke", "tiny", "cache", "train", "evaluate", "compare", "bundle"):
            self.assertIn(f"stage('{stage}'", text)
        self.assertLess(text.index("'pull','--ff-only'"), text.index("SCRIPT.is_file()"))
        self.assertIn("'--pilot',PILOT", text)
        self.assertIn("full_recovery_manifest.json", text)
        for forbidden in ("setup.py", "venv", "reset --hard", "coremltools", "--role','geometry"):
            self.assertNotIn(forbidden, text)
        functions = {n.name for n in ast.parse(codes[0]).body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"stage", "run_logged"} <= functions)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
