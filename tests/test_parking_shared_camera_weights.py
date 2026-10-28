"""CPU-only collection, safe tensor loading and descriptive weight audit."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from models.parking_shared import ParkingDepthDecoder
from models.parking_shared_camera import CameraAwareDepthDecoder
from scripts import audit_parking_shared_camera_weights as audit
from scripts import parking_edge_common as common
from scripts.screen_parking_shared_model import state_digest


def tensor_bytes(value):
    stream = io.BytesIO(); torch.save(value, stream); return stream.getvalue()


def fixture(run):
    torch.manual_seed(17)
    model = CameraAwareDepthDecoder(ParkingDepthDecoder())
    initial = copy.deepcopy(model.state_dict())
    identity = {"source_sha256": {name: common.file_sha(audit.ROOT/name) for name in
        ["models/parking_shared.py", "models/parking_shared_camera.py"]},
        "initial_adapter_state_sha256": state_digest(model)}
    run_id = common.identity(identity)
    members = {"run_identity.json":common.encoded(identity),
        "camera_pilot_results.json":common.encoded({"complete":True,"run_identity":run_id,"initial_dev":{"fixture":True}})}
    for name in identity["source_sha256"]:
        members["source_snapshot/"+name] = (audit.ROOT/name).read_bytes()
    for i, arm in enumerate(audit.ARMS):
        arm_id = common.identity({"run_identity":run_id,"arm":arm})
        history = [{"epoch":j,"optimizer_steps":192*j,"dev":{"fixture_mae_m":.76+j*.01}}
            for j in range(6)]
        selection = {"run_identity":arm_id,"parent_run_identity":run_id,"optimizer_steps":960,
            "selected_epoch":0,"heldout_used_for_selection":False,"history":history}
        members[arm+"/selection.json"] = common.encoded(selection)
        cp = {"run_identity":arm_id,"arm":arm,"epoch":0,"optimizer_steps":0,
            "history":history[:1],"decoder":initial,"optimizer":{}}
        stem = arm+"/checkpoints/epoch_000"
        raw = tensor_bytes(cp); members[stem+".pth"] = raw
        members[stem+".json"] = common.encoded({"sha256":common.sha(raw)})
        cp = copy.deepcopy(cp); cp.update(epoch=3,optimizer_steps=576,history=history[:4])
        cp["decoder"]["residual.weight"] += .02+i*.001
        cp["decoder"]["base.output.1.weight"] += .001
        for name,value in cp["decoder"].items():
            if name.endswith("running_mean"): value += .01
            if name.endswith("num_batches_tracked"): value += 576
        raw = tensor_bytes(cp); stem = arm+"/checkpoints/epoch_003"
        common.immutable(run/(stem+".pth"), raw)
        common.immutable(run/(stem+".json"), common.encoded({"sha256":common.sha(raw)}))
    result = audit.archive_bytes(members)
    common.immutable(run/audit.RESULT_NAME,result)
    return run_id,common.sha(result)


class CameraWeightAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_collection_audit_and_bundle_repeat_without_changing_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root/"completed"; output = root/"audit"
            run_id,result_sha = fixture(run)
            before = {str(p.relative_to(run)):common.file_sha(p) for p in run.rglob("*") if p.is_file()}
            with patch.object(audit,"RUN_ID",run_id),patch.object(audit,"RESULT_SHA",result_sha),contextlib.redirect_stdout(io.StringIO()):
                archive = audit.collect(run,output)
                self.assertEqual(archive,audit.collect(run,output))
                real_load = torch.load
                with patch("torch.load",wraps=real_load) as load:
                    report = audit.audit(archive,output)
                self.assertEqual(load.call_count,4)
                for call in load.call_args_list:
                    self.assertTrue(call.kwargs["weights_only"])
                    self.assertEqual(call.kwargs["map_location"],"cpu")
                self.assertEqual(report,audit.audit(archive,output))
                audit.bundle_outputs(archive,output); audit.bundle_outputs(archive,output)
            after = {str(p.relative_to(run)):common.file_sha(p) for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before,after)
            self.assertEqual(report["optimizer_steps"],0)
            self.assertFalse(report["forward_inference_performed"])
            self.assertFalse(report["images_accessed"])
            self.assertFalse(report["training_performed"])
            for arm in audit.ARMS:
                self.assertTrue(report["arms"][arm]["adapter_weights_moved"])
                self.assertTrue(report["arms"][arm]["existing_depth_weights_moved"])
                self.assertTrue(report["arms"][arm]["existing_depth_bn_statistics_moved"])
                self.assertTrue(all(v==576 for v in report["arms"][arm]["bn_batch_counter_increments"].values()))
            self.assertGreater(report["actual_vs_reference_state_delta"]["adapter_parameters"]["delta_l2"],0)
            with zipfile.ZipFile(output/"parking_shared_camera_weight_audit_results.zip") as z:
                self.assertEqual(set(z.namelist()),{"camera_weight_audit.json","parking_shared_camera_epoch3_inputs.zip"})

    def test_hash_failure_stops_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/"run";run.mkdir()
            (run/audit.RESULT_NAME).write_bytes(b"changed")
            with patch("torch.load") as load:
                with self.assertRaises(ValueError):audit.collect(run,root/"out")
                load.assert_not_called()

    def test_missing_trained_file_does_not_trigger_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/"run";run_id,result_sha=fixture(run)
            (run/"actual_camera/checkpoints/epoch_003.pth").unlink()
            with patch.object(audit,"RUN_ID",run_id),patch.object(audit,"RESULT_SHA",result_sha):
                with self.assertRaisesRegex(FileNotFoundError,"Do not rerun training"):
                    audit.collect(run,root/"out")
                self.assertFalse((root/"out").exists())

    def test_sidecar_mismatch_preserves_completed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/"run";run_id,result_sha=fixture(run)
            path=run/"actual_camera/checkpoints/epoch_003.json"
            path.write_text(json.dumps({"sha256":"changed"}))
            with patch.object(audit,"RUN_ID",run_id),patch.object(audit,"RESULT_SHA",result_sha):
                with self.assertRaises(ValueError):audit.collect(run,root/"out")
            self.assertEqual(json.loads(path.read_text())["sha256"],"changed")

    def test_output_cannot_be_inside_completed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):audit.collect(root,root/"audit")

    def test_archive_inventory_and_paths_rejected(self):
        for members in [{"../escape":b"bad"},{"/escape":b"bad"},{"bad\\name":b"bad"}]:
            with self.assertRaises(ValueError):audit.safe_archive(audit.archive_bytes(members))
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,"w") as z:
            z.writestr("same",b"one")
            with self.assertWarns(UserWarning):z.writestr("same",b"two")
        with self.assertRaises(ValueError):audit.safe_archive(raw.getvalue())

    def test_tensor_changes_are_descriptive_and_type_checked(self):
        a={"x":torch.tensor([1.,2.]),"count":torch.tensor(1)}
        b={"x":torch.tensor([1.,3.]),"count":torch.tensor(577)}
        delta=audit.tensor_delta(a,b,{"x"})
        self.assertEqual(delta["changed_tensors"],1);self.assertEqual(delta["delta_l2"],1.)
        self.assertEqual(audit.tensor_delta(a,a,{"x"})["changed_tensors"],0)
        for value in [torch.ones(3),torch.ones(2,dtype=torch.float64),torch.tensor([1.,float("nan")])]:
            with self.assertRaises(ValueError):audit.tensor_delta(a,{**b,"x":value},{"x"})

    def test_notebook_is_cpu_only_and_self_contained(self):
        path=audit.ROOT/"notebooks/Parking_Shared_Camera_Weight_Audit_Colab.ipynb"
        notebook=json.loads(path.read_text())
        code=[]
        for cell in notebook["cells"]:
            if cell["cell_type"]=="code":
                self.assertIsNone(cell["execution_count"]);self.assertEqual(cell["outputs"],[])
                source="".join(cell["source"]);ast.parse(source);code.append(source)
        joined="\n".join(code)
        for name in ["'collect'","'audit'","'bundle'"]:self.assertIn(name,joined)
        for forbidden in ["pip install","'train'","'cache'","'cuda'","reset --hard","stage('train')"]:
            self.assertNotIn(forbidden,joined)
        self.assertIn("drive.mount",code[0]);self.assertIn("git",code[0])


if __name__=="__main__":unittest.main()
