"""Full broad coverage, strict freezing, resume, packaging and notebook checks."""
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
from scripts import train_parking_shared_adapter as pilot
from scripts.screen_parking_shared_model import state_digest


def sample(i=0, role="train"):
    return {"camera_token": f"{role}-{i}", "role": role, "width": 1600, "height": 900,
        "intrinsics": [[1200., 0., 800.], [0., 1200., 450.], [0., 0., 1.]],
        "counts": {k: 5 for k in pilot.configuration()["critical_cells"]}}


def row():
    rng = np.random.default_rng(123)
    return {**{f"p{i+3}": rng.normal(size=shape).astype(np.float32)
        for i, shape in enumerate([(64,8,16), (128,4,8), (256,2,4)])},
        "z": np.repeat([3.,5.,3.,5.], 5).astype(np.float32),
        "semantic": np.repeat([1,1,2,2], 5).astype(np.uint8),
        "conservative": np.ones(20, bool), "grid": rng.uniform(-1,1,(20,2)).astype(np.float32)}


def metrics(score):
    return {"equal_four_cell_mae_m": score, "broad_macro_image_mae_m": 2.,
        "cells": {k: {"macro_image_mae_m": score, "fixed_4m_mae_m": 1., "images": 1, "points": 5}
            for k in pilot.configuration()["critical_cells"]}}


class AdapterPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_1536_broad_coverage_and_declared_budget(self):
        config = pilot.configuration(); samples = [sample(i) for i in range(1536)]
        batches = pilot.epoch_batches(samples, config, 1)
        self.assertEqual(len(batches), 384)
        self.assertEqual(sorted(i for batch in batches for i in batch[4:]), list(range(1536)))
        self.assertTrue(all(len(b) == 8 for b in batches))
        self.assertEqual(pilot.budget(samples, config)["updates_per_arm"], 1920)
        self.assertEqual(pilot.budget(samples, config)["draws_per_epoch"], 3072)
        self.assertEqual(batches, pilot.epoch_batches(samples, config, 1))
        self.assertNotEqual(batches, pilot.epoch_batches(samples, config, 2))

    def test_balanced_slots_use_only_eligible_training_images(self):
        config = pilot.configuration(); samples = [sample(i) for i in range(16)]
        for i, s in enumerate(samples):
            s["counts"] = {k: 5 if j == i%4 else 0 for j, k in enumerate(config["critical_cells"])}
        for batch in pilot.epoch_batches(samples, config, 1):
            for j, key in enumerate(config["critical_cells"]):
                self.assertGreaterEqual(samples[batch[j]]["counts"][key], 5)
        for bad in [[], samples+[sample(17,"dev")], samples+[sample(17,"heldout")], samples[:3], samples+[samples[0]]*4]:
            with self.assertRaises(ValueError):
                pilot.epoch_batches(bad, config, 1)
        for s in samples:
            s["counts"][config["critical_cells"][0]] = 0
        with self.assertRaises(ValueError):
            pilot.epoch_batches(samples, config, 1)

    def test_research_checks_protect_all_cells_and_broad_range(self):
        config = pilot.configuration()
        gates = pilot.research_gates(metrics(1.1), metrics(1.), metrics(.8), config)
        self.assertTrue(all(gates["adapter_improvement"].values()))
        self.assertTrue(gates["camera_specific_gain_at_least_2_percent"])
        changed = metrics(.8); changed["cells"][config["critical_cells"][3]]["macro_image_mae_m"] = 1.13
        self.assertFalse(all(pilot.research_gates(metrics(1.1),metrics(1.),changed,config)["adapter_improvement"].values()))
        changed = metrics(.8); changed["broad_macro_image_mae_m"] = 2.11
        self.assertFalse(all(pilot.research_gates(metrics(1.1),metrics(1.),changed,config)["adapter_improvement"].values()))
        self.assertFalse(pilot.research_gates(metrics(1.1),metrics(.8),metrics(.8),config)["camera_specific_gain_at_least_2_percent"])

    def test_epoch_zero_can_win_and_prior_failed_gate_is_preserved(self):
        history = [{"epoch":i,"dev":metrics(v)} for i,v in enumerate([.7,.8,.7])]
        self.assertEqual(pilot.camera.select_best(history)["epoch"], 0)
        self.assertFalse(pilot.configuration()["prior_evidence"]["tiny_sanity_all_checks_passed"])
        self.assertEqual(pilot.configuration()["learning_rate"], .001)
        self.assertEqual(pilot.configuration()["weight_decay"], 0.)

    def test_resume_rejects_wrong_steps_arm_sampler_and_history(self):
        config = pilot.configuration(); samples = [sample(i) for i in range(8)]
        history = [{"epoch":0,"optimizer_steps":0,"dev":metrics(1.)},
            {"epoch":1,"optimizer_steps":2,"dev":metrics(.9),
                "sampler_sha256":pilot.common.identity(pilot.epoch_batches(samples,config,1))}]
        saved = {"arm":"actual_camera","epoch":1,"optimizer_steps":2,"history":history}
        pilot.validate_resume(saved,"actual_camera",config,samples)
        for key,value in [("arm","reference_camera"),("optimizer_steps",3),("epoch",6)]:
            bad=copy.deepcopy(saved);bad[key]=value
            with self.assertRaises(ValueError):pilot.validate_resume(bad,"actual_camera",config,samples)
        bad=copy.deepcopy(saved);bad["history"][1]["sampler_sha256"]="bad"
        with self.assertRaises(ValueError):pilot.validate_resume(bad,"actual_camera",config,samples)
        bad=copy.deepcopy(saved);bad["history"][0]["optimizer_steps"]=1
        with self.assertRaises(ValueError):pilot.validate_resume(bad,"actual_camera",config,samples)

    def test_output_paths_cannot_contain_or_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=SimpleNamespace(inputs=root/"inputs",source=root/"source",weights=root/"weights",
                warmstart=root/"warm.zip",output=root/"run",cache=root/"cache")
            pilot.separate_paths(a)
            for output in [root, a.inputs/"run", a.source, a.weights/"run", a.warmstart]:
                bad=copy.copy(a);bad.output=output
                with self.assertRaises(ValueError):pilot.separate_paths(bad)
            bad=copy.copy(a);bad.cache=a.output/"cache"
            with self.assertRaises(ValueError):pilot.separate_paths(bad)

    def test_preparation_checks_frozen_input_identity_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=SimpleNamespace(inputs=root/"inputs",source=root/"source",weights=root/"weights",
                warmstart=root/"warm.zip",output=root/"run",cache=root/"cache")
            manifest={"samples":[sample(i) for i in range(1536)]+[sample(i,"dev") for i in range(256)]}
            with patch.object(pilot.camera.data,"read_manifest",return_value=manifest), \
                    patch.object(pilot.camera,"read_warmstart",return_value=({}, {"selected_dev":metrics(1.)},None)), \
                    patch.object(pilot.common,"file_sha",return_value="changed"):
                with self.assertRaisesRegex(ValueError,"unchanged reviewed"):
                    pilot.prepare(a)
            self.assertFalse((a.output/"prepared_inputs.json").exists())

    def test_real_optimizer_resume_does_not_repeat_completed_updates(self):
        config={**pilot.configuration(),"epochs":2}
        samples={"train":[sample(i) for i in range(8)],"dev":[sample(1,"dev")]}
        torch.manual_seed(55)
        decoder=CameraAwareDepthDecoder(ParkingDepthDecoder(width=16)).eval()
        pilot.sanity.freeze_base(decoder);frozen=state_digest(decoder.base)
        initial=copy.deepcopy(decoder.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            a=SimpleNamespace(output=Path(directory)/"run",arm="actual_camera")
            calls=[]; real=pilot.sanity.adapter_step
            def counted(*args):calls.append(1);return real(*args)
            # Interrupt after epoch one, then restore its checkpoint. No
            # in-memory partially trained parameters may leak into the resume.
            evaluations=[metrics(.9),RuntimeError("simulated disconnect")]
            with patch.object(pilot.camera.pilot,"read_features",side_effect=lambda *a:row()), \
                    patch.object(pilot.camera,"evaluate",side_effect=evaluations), \
                    patch.object(pilot.sanity,"adapter_step",side_effect=counted), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"disconnect"):
                    pilot.train_arm(a,decoder,initial,metrics(1.),samples,Path("unused"),
                        {"Vehicle":[1],"Pedestrian":[2]},config,"fixture",frozen)
            self.assertEqual(len(calls),4)
            self.assertFalse((a.output/a.arm/"checkpoints/epoch_002.pth").exists())
            with patch.object(pilot.camera.pilot,"read_features",side_effect=lambda *a:row()), \
                    patch.object(pilot.camera,"evaluate",return_value=metrics(.8)), \
                    patch.object(pilot.sanity,"adapter_step",side_effect=counted), \
                    contextlib.redirect_stdout(io.StringIO()):
                selected=pilot.train_arm(a,decoder,initial,metrics(1.),samples,Path("unused"),
                    {"Vehicle":[1],"Pedestrian":[2]},config,"fixture",frozen)
                before=len(calls)
                again=pilot.train_arm(a,decoder,initial,metrics(1.),samples,Path("unused"),
                    {"Vehicle":[1],"Pedestrian":[2]},config,"fixture",frozen)
            self.assertEqual(selected,again);self.assertEqual(len(calls),before)
            self.assertEqual(selected["optimizer_steps"],4)
            self.assertEqual(state_digest(decoder.base),frozen)
            self.assertFalse(torch.equal(initial["residual.weight"],decoder.state_dict()["residual.weight"]))
            self.assertTrue(all(p.grad is None for p in decoder.base.parameters()))
            # A resumed run must match uninterrupted optimization, not merely
            # have the right number of update counters.
            uninterrupted=CameraAwareDepthDecoder(ParkingDepthDecoder(width=16)).eval()
            pilot.sanity.freeze_base(uninterrupted)
            other=SimpleNamespace(output=Path(directory)/"uninterrupted",arm=a.arm)
            with patch.object(pilot.camera.pilot,"read_features",side_effect=lambda *a:row()), \
                    patch.object(pilot.camera,"evaluate",side_effect=[metrics(.9),metrics(.8)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                pilot.train_arm(other,uninterrupted,initial,metrics(1.),samples,Path("unused"),
                    {"Vehicle":[1],"Pedestrian":[2]},config,"fixture",frozen)
            for key,value in decoder.state_dict().items():
                self.assertTrue(torch.equal(value,uninterrupted.state_dict()[key]),key)

    def test_bundle_contains_initial_selected_and_last_even_if_zero_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);identity={"config":{"epochs":2},"source_sha256":{}}
            run_id=pilot.common.identity(identity)
            pilot.common.immutable(root/"run_identity.json",pilot.common.encoded(identity))
            for name in ["prepared_inputs.json","runtime_smoke.json","initial_dev.json"]:
                pilot.common.immutable(root/name,pilot.common.encoded({"fixture":True}))
            result={"complete":True,"run_identity":run_id,"arms":{}}
            for arm in pilot.configuration()["arms"]:
                arm_id=pilot.common.identity({"run_identity":run_id,"arm":arm})
                history=[{"epoch":i,"dev":metrics(.7+i*.1)} for i in range(3)]
                selection={"run_identity":arm_id,"parent_run_identity":run_id,"selected_epoch":0,
                    "selected_dev":history[0]["dev"],"history":history}
                pilot.common.immutable(root/arm/"selection.json",pilot.common.encoded(selection))
                result["arms"][arm]={"selected_epoch":0,"selected_dev":history[0]["dev"]}
                for epoch in [0,2]:
                    pilot.camera.pilot.save_checkpoint(root/arm/"checkpoints"/f"epoch_{epoch:03d}.pth",
                        {"run_identity":arm_id,"arm":arm,"epoch":epoch,"history":history[:epoch+1],"decoder":{}})
            pilot.common.immutable(root/"adapter_pilot_results.json",pilot.common.encoded(result))
            with contextlib.redirect_stdout(io.StringIO()):
                pilot.bundle(SimpleNamespace(output=root));pilot.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/"parking_shared_adapter_results.zip") as z:
                self.assertEqual(len(z.namelist()),len(set(z.namelist())))
                for arm in result["arms"]:
                    for epoch in [0,2]:
                        self.assertIn(f"{arm}/checkpoints/epoch_{epoch:03d}.pth",z.namelist())
                        self.assertIn(f"{arm}/checkpoints/epoch_{epoch:03d}.json",z.namelist())

    def test_two_arm_orchestration_resume_and_bundle_with_synthetic_features(self):
        """Real updates to the real adapter; no real dataset training."""
        config={**pilot.configuration(),"epochs":2}
        manifest={"config":json.loads(pilot.camera.data.CONFIG.read_text()),
            "mapping":{"Vehicle":[1],"Pedestrian":[2]},
            "samples":[sample(i) for i in range(8)]+[sample(i,"dev") for i in range(2)]}
        class Native(nn.Module):
            def __init__(self):
                super().__init__()
                self.body=nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3))
                self.depth_decoder=ParkingDepthDecoder()

            def full_urban_candidates(self, features):
                return features[0].mean((2,3))
        def build(*args):
            torch.manual_seed(44)
            return Native(),{"runtime":{"device":"cpu"}}
        initial=build()[0].depth_decoder.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=SimpleNamespace(stage="train",inputs=root/"inputs",warmstart=root/"warm.zip",output=root/"run",
                source=root/"source",weights=root/"weights",cache=root/"cache",reuse_cache=[root/"oldcache"],device="cpu")
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(pilot,"configuration",return_value=config))
                stack.enter_context(patch.object(pilot,"prepare",return_value=manifest))
                stack.enter_context(patch.object(pilot.camera,"read_warmstart",
                    return_value=({}, {"selected_dev":metrics(1.)}, {"decoder":initial})))
                stack.enter_context(patch.object(pilot.camera.pilot,"build_model",side_effect=build))
                stack.enter_context(patch.object(pilot.camera.body_verification,"verify_warmstart_body",return_value={"passed":True}))
                stack.enter_context(patch.object(pilot.camera,"cache_identity",return_value={"fixture":True}))
                stack.enter_context(patch.object(pilot.camera,"matching_cache",return_value=True))
                stack.enter_context(patch.object(pilot.camera.pilot,"read_features",side_effect=lambda *a:row()))
                stack.enter_context(patch.object(pilot.camera,"evaluate",return_value=metrics(1.)))
                stack.enter_context(patch.object(pilot,"smoke",return_value={"complete":True,"optimizer_steps":0}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                pilot.run(a)
                first=(a.output/"adapter_pilot_results.json").read_bytes()
                pilot.run(a)
                self.assertEqual(first,(a.output/"adapter_pilot_results.json").read_bytes())
                pilot.common.immutable(a.output/"prepared_inputs.json",pilot.common.encoded({"fixture":True}))
                pilot.bundle(a)
            result=json.loads(first)
            self.assertTrue(result["frozen_body_depth_bn_detection_unchanged"])
            self.assertFalse(result["deployment_qualified"])
            self.assertFalse(result["prior_tiny_sanity_all_checks_passed"])
            for arm in config["arms"]:
                self.assertEqual(result["arms"][arm]["selected_epoch"],0)
                self.assertEqual(result["arms"][arm]["optimizer_steps"],4)
            with zipfile.ZipFile(a.output/"parking_shared_adapter_results.zip") as archive:
                for arm in config["arms"]:
                    self.assertIn(f"{arm}/checkpoints/epoch_002.pth",archive.namelist())

    def test_notebook_code_is_clean_and_execution_order_is_explicit(self):
        path=pilot.ROOT/"notebooks/Parking_Shared_Adapter_Only_FullPool_Colab.ipynb"
        notebook=json.loads(path.read_text());code=[]
        for cell in notebook["cells"]:
            if cell["cell_type"]!="code":continue
            self.assertIsNone(cell["execution_count"]);self.assertEqual(cell["outputs"],[])
            source="".join(cell["source"]);ast.parse(source);code.append(source)
        self.assertEqual(len(code),6)
        self.assertIn("'main'",code[0]);self.assertNotIn("reset",code[0])
        for i,name in [(1,"prepare"),(2,"smoke"),(3,"cache"),(4,"train"),(5,"bundle")]:
            self.assertIn(f"stage('{name}')",code[i])
        self.assertIn("'--no-deps'",code[2]);self.assertNotIn("coreml",code[2])
        self.assertIn("384",path.read_text());self.assertIn("1,920",path.read_text())

    def test_required_sanity_helpers_and_configs_are_snapshotted(self):
        sources={str(p.relative_to(pilot.ROOT)) for p in pilot.SOURCES}
        self.assertIn("scripts/sanity_parking_shared_adapter.py",sources)
        self.assertIn("configs/parking_shared_adapter_sanity_v1.json",sources)
        self.assertIn("configs/parking_shared_adapter_pilot_v1.json",sources)
        self.assertTrue(all(p.is_file() for p in pilot.SOURCES))


if __name__=="__main__":
    unittest.main()
