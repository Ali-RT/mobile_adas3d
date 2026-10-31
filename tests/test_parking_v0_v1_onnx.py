"""Export contracts and tamper checks; no training or device access."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from models.parking_shared import SharedParkingNet
from models.parking_shared_camera import CameraAwareDepthDecoder
from models.parking_shared_trained_export import TrainedParkingExport
from scripts import export_parking_v0_v1_onnx as export

# File-based fixture import avoids Colab's unrelated installed `tests` package.
spec = importlib.util.spec_from_file_location("parking_shared_test_fixture", Path(__file__).with_name("test_parking_shared_model.py"))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


class TrainedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def tiny(self):
        torch.manual_seed(40)
        native = SharedParkingNet(fixture.fake_urban(), width=16).eval()
        decoder = CameraAwareDepthDecoder(native.depth_decoder, input_hw=(64,96))
        with torch.no_grad(): decoder.residual.weight.normal_(0,.05)
        return TrainedParkingExport(native, decoder)

    def test_wrapper_uses_one_body_pass_and_keeps_both_inputs(self):
        model = self.tiny(); calls = []
        handle = model.native.body[0].register_forward_hook(lambda *a: calls.append(1))
        image = torch.randn(1,3,64,96); camera = torch.tensor([[80.,80.,47.5,31.5]])
        before = export.tensor_digest(model.state_dict())
        with torch.inference_mode():
            candidates, depth = model(image,camera)
            changed_candidates, changed_depth = model(image,camera*1.2)
        handle.remove()
        self.assertEqual(len(calls),2)
        self.assertEqual(candidates.shape,(1,32,126)); self.assertEqual(depth.shape,(1,1,16,24))
        torch.testing.assert_close(candidates,changed_candidates,atol=0,rtol=0)
        self.assertGreater(float((depth-changed_depth).abs().max()),1e-6)
        self.assertEqual(export.tensor_digest(model.state_dict()),before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_calibration_uses_half_pixel_centres_not_plain_affine(self):
        sample = {"width":1600,"height":900,"intrinsics":[[1000,0,800],[0,1000,450],[0,0,1]]}
        values, grid, meta = export.input_arrays(sample,np.zeros((900,1600,3),np.uint8),{"xy":np.array([[800,450]],np.int32)})
        np.testing.assert_allclose(values["camera"],[[400,400,319.7,191.7]],atol=2e-5)
        self.assertEqual(values["image"].shape,(1,3,384,640))
        np.testing.assert_allclose(grid,[[.000625,.0010416667]],atol=1e-7)
        self.assertEqual(meta["padding_left_top"],[0,12])

    def test_point_sampling_preserves_border_and_half_pixel(self):
        depth = np.arange(16,dtype=np.float32).reshape(1,1,4,4)+1
        grid = np.array([[-1,-1],[0,0],[1,1]],np.float32)
        np.testing.assert_allclose(export.sample_points(depth,grid),[1,8.5,16],atol=0,rtol=0)

    def test_lossless_storage_is_not_model_quantization(self):
        indices=np.tile(np.arange(256,dtype=np.uint8),3).reshape(1,3,16,16)
        image=export.float_table()[indices]
        stored=export.lossless_input_indices(image)
        self.assertEqual(stored.dtype,np.uint8)
        self.assertEqual(export.float_table()[stored].tobytes(),image.tobytes())
        image[0,0,0,0]=np.float32(.123456)
        with self.assertRaises(ValueError): export.lossless_input_indices(image)

    def test_phone_accuracy_shard_preserves_inputs_truth_owners_and_masks(self):
        sample={"camera_token":"a","width":1600,"height":900,"intrinsics":[[1000,0,800],[0,1000,450],[0,0,1]],
            "role":"dev","objects":[{"class_name":"Vehicle","instance_token":"track","annotation_token":"ann"}],"recording_log":"log"}
        images={"a":np.zeros((900,1600,3),np.uint8)}
        truth={"a":{"xy":np.array([[800,450]],np.int32),"z":np.array([2.],np.float32),
            "object_index":np.array([0],np.int32),"cell_masks":np.array([[True],[False]])}}
        inputs,grid,_=export.input_arrays(sample,images["a"],truth["a"])
        checks={"a":{"preprocessed_image_sha256":export.common.sha(inputs["image"].astype("<f4").tobytes()),
            "grid_sha256":export.common.sha(grid.astype("<f4").tobytes()),"camera":inputs["camera"].tolist()}}
        values={key:{"a":np.array([2.1],np.float32)} for key in export.STREAMS}
        raw=export.accuracy_shard([sample],images,truth,values,checks,export.configuration())
        import zipfile
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            manifest=json.loads(z.read("shard_manifest.json")); entry=manifest["samples"][0]
            self.assertEqual(manifest["heldout_bodies_included"],0)
            for name,digest in manifest["files"].items(): self.assertEqual(export.common.sha(z.read(name)),digest)
            def array(descriptor):
                return np.frombuffer(z.read(descriptor["file"]),dtype=descriptor["dtype"]).reshape(descriptor["shape"])
            np.testing.assert_array_equal(export.float_table()[array(entry["image_indices"])],inputs["image"])
            np.testing.assert_array_equal(array(entry["cell_masks"]),truth["a"]["cell_masks"])
            np.testing.assert_array_equal(array(entry["object_index"]),truth["a"]["object_index"])
            np.testing.assert_array_equal(array(entry["grid"]),grid)

    def test_full_output_check_rejects_invalid_and_drifted_outputs(self):
        c=export.configuration(); c["output_shapes"]={"candidates":[1,32,2],"depth_m":[1,1,2,2]}
        reference={"candidates":np.zeros((1,32,2),np.float32),"depth_m":np.ones((1,1,2,2),np.float32)}
        self.assertTrue(export.output_checks(reference,copy.deepcopy(reference),c)["depth_m"]["passed"])
        for value in [0.,float("nan"),2.]:
            bad=copy.deepcopy(reference); bad["depth_m"][0,0,0,0]=value
            with self.assertRaises((ValueError,RuntimeError)): export.output_checks(reference,bad,c)

    def test_metric_gate_is_numerical_not_ten_centimetre_accuracy_target(self):
        cell={"reference_points":5,"valid_predicted_points":5,"prediction_coverage":1.,
              "reference_annotation_observations":1,"reference_instance_tracks":1,"reference_recording_logs":1,
              "conditional_log_track_observation_macro_mae_m":.8}
        reviewed={"images":512,"images_with_invalid_predictions":0,"conditional_equal_four_cell_mae_m":.8,"cells":{"x":cell}}
        self.assertTrue(export.check_metrics(copy.deepcopy(reviewed),reviewed,export.configuration())["passed"])
        for change in ["coverage","error","image"]:
            actual=copy.deepcopy(reviewed)
            if change=="coverage": actual["cells"]["x"]["valid_predicted_points"]=4
            if change=="error": actual["cells"]["x"]["conditional_log_track_observation_macro_mae_m"]+=.01
            if change=="image": actual["images"]=511
            with self.assertRaises(ValueError): export.check_metrics(actual,reviewed,export.configuration())

    def test_resumable_chunk_binding_and_point_order(self):
        rows=[{"camera_token":"a","ground_truth_sha256":"truth"}]; truth={"a":{"z":np.ones(3,np.float32)}}
        values={key:{"a":np.ones(3,np.float32)} for key in export.STREAMS}; checks={"a":{"passed":True}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"chunk.zip"; raw=export.chunk_bytes("run",rows,values,checks); path.write_bytes(raw)
            actual,_=export.read_chunk(path,"run",rows,truth)
            np.testing.assert_array_equal(actual["V1_onnx"]["a"],values["V1_onnx"]["a"])
            with self.assertRaises(ValueError): export.read_chunk(path,"other-run",rows,truth)
            changed={"a":{"z":np.ones(2,np.float32)}}
            with self.assertRaises(ValueError): export.read_chunk(path,"run",rows,changed)
            members=export.read_zip(path); members["V1_onnx.zip"]+=b"tamper"
            path.write_bytes(export.v0.zip_bytes(members))
            with self.assertRaises(ValueError): export.read_chunk(path,"run",rows,truth)

    def test_archive_paths_duplicates_and_size_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"unsafe.zip"
            raw=io.BytesIO()
            import zipfile
            with zipfile.ZipFile(raw,"w") as z: z.writestr("../escape",b"x")
            path.write_bytes(raw.getvalue())
            with self.assertRaises(ValueError): export.read_zip(path)
            path.write_bytes(export.v0.zip_bytes({"a":b"x"*100}))
            with self.assertRaises(ValueError): export.read_zip(path,maximum=50)

    def test_published_archive_transfer_is_atomic_and_conflict_safe(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"source.zip"; destination=Path(d)/"published.zip"
            source.write_bytes(b"immutable archive"); digest=export.common.file_sha(source)
            with self.assertRaises(ValueError): export.copy_published_archive(source,destination,"wrong-hash")
            self.assertFalse(destination.exists()); self.assertEqual(list(Path(d).glob(".onnx_transfer_*")),[])
            export.copy_published_archive(source,destination,digest)
            self.assertEqual(source.read_bytes(),destination.read_bytes())
            source.unlink()
            export.copy_published_archive(source,destination,digest)  # no source re-open
            with self.assertRaises(ValueError): export.copy_published_archive(source,destination,"changed")

    def test_fixed_fixtures_are_deterministic_and_use_no_predictions(self):
        def sample(token,count,k):
            return {"camera_token":token,"cell_names":["Vehicle:0-2m","Pedestrian:4-6m"],
                "eligible_reference_counts":dict(zip(["Vehicle:0-2m","Pedestrian:4-6m"],count)),
                "intrinsics":[[k,0,0],[0,k,0],[0,0,1]],"objects":[{"class_name":"Vehicle"}]}
        rows=[sample("b",[5,0],100),sample("a",[0,5],100),sample("c",[5,5],200)]
        first=export.choose_fixtures(rows,2); second=export.choose_fixtures(list(reversed(rows)),2)
        self.assertEqual(first,second); self.assertEqual(first[0]["camera_token"],"c")

    def test_selected_checkpoint_and_scope_are_explicit(self):
        c=export.configuration()
        for key,value in c.items():
            if key.endswith("sha256"): self.assertEqual(len(value),64,key)
        self.assertEqual(c["v1_arm"],"continuation_2_6_control"); self.assertEqual(c["v1_block"],5)
        self.assertEqual(c["input_shapes"]["camera"],[1,4]); self.assertEqual(c["development_images"],512)
        self.assertFalse(c["training_authorized"]); self.assertFalse(c["heldout_scoring_authorized"])
        self.assertFalse(c["phone_app_changes_authorized"]); self.assertFalse(c["product_accuracy_pass_fail_enabled"])

    def test_phone_package_requires_completed_reviewed_replay(self):
        with self.assertRaisesRegex(ValueError,"every frozen-development"):
            export.package(None,export.configuration(),{}, {},{}, {},{"complete":False},{},{},[])

    def test_complete_packaging_contract_with_synthetic_data_only(self):
        c=export.configuration(); c["development_images"]=16
        c["output_shapes"]={"candidates":[1,32,1],"depth_m":[1,1,2,2]}
        class Toy(torch.nn.Module):
            def forward(self,image,camera):
                return torch.zeros((1,32,1)),torch.ones((1,1,2,2))*(5+camera[0,0]*.001)
        class Session:
            def run(self,names,inputs):
                with torch.inference_mode():
                    return [v.numpy() for v in Toy()(*(torch.from_numpy(inputs[n]) for n in c["input_shapes"]))]
        rows=[]; images={}; truth={}; checks={}; values={key:{} for key in export.STREAMS}
        for i in range(16):
            token=f"view{i:02d}"
            sample={"camera_token":token,"width":1600,"height":900,
                "intrinsics":[[1000+i,0,800],[0,1000+i,450],[0,0,1]],"role":"dev",
                "cell_names":["Vehicle:2-4m"],"eligible_reference_counts":{"Vehicle:2-4m":5},
                "objects":[{"class_name":"Vehicle","instance_token":token,"annotation_token":token}],"recording_log":"log"}
            rows.append(sample); images[token]=np.zeros((900,1600,3),np.uint8)
            truth[token]={"xy":np.array([[800,450]],np.int32),"z":np.array([3.],np.float32),
                "object_index":np.array([0],np.int32),"cell_masks":np.ones((1,1),bool)}
            inputs,grid,_=export.input_arrays(sample,images[token],truth[token])
            checks[token]={"preprocessed_image_sha256":export.common.sha(inputs["image"].astype("<f4").tobytes()),
                "grid_sha256":export.common.sha(grid.astype("<f4").tobytes()),"camera":inputs["camera"].tolist()}
            for key in export.STREAMS: values[key][token]=np.array([5.4],np.float32)
        wrappers={r:Toy().eval().requires_grad_(False) for r in export.ROLES}
        graphs={r:("synthetic_"+r).encode() for r in export.ROLES}
        identity={"model_states":{r:export.tensor_digest(w.state_dict()) for r,w in wrappers.items()}}
        report={"complete":True,"run_identity":export.common.identity(identity),"comparison_identity":"synthetic",
            "development_images":16,"full_output_parity_passed":True,"reviewed_point_parity_passed":True,
            "frozen_dev_metrics_preserved":{r:{s:{"passed":True} for s in ("native","onnx")} for r in export.ROLES},
            "heldout_bodies_opened":0,"models":{r:{"sha256":export.common.sha(v)} for r,v in graphs.items()}}
        shard={"file":"shards/dev/synthetic.zip","sha256":"synthetic_source"}
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d, mock.patch.object(export.v0,"load_dev_shard",return_value=(images,truth)), \
                mock.patch.object(export,"read_chunk",return_value=(values,checks)):
            a=SimpleNamespace(output=Path(d))
            export.package(a,c,wrappers,identity,graphs,{r:Session() for r in export.ROLES},report,checks,
                {"manifest_sha256":"synthetic","benchmark_identity_sha256":"synthetic","metric_protocol":{"synthetic":True}},[(shard,rows)])
            members=export.read_zip(a.output/"portable/parking_v0_v1_phone.zip")
            index=json.loads(members["package_inventory.json"])
            self.assertEqual(index["heldout_bodies_included"],0)
            for name,record in index["files"].items(): self.assertEqual(export.common.sha(members[name]),record["sha256"])
            phone=json.loads(members["phone_manifest.json"])
            self.assertEqual(len(phone["fixtures"]),16)
            self.assertTrue(all(v["passed"] for v in phone["calibration_perturbation_checks"].values()))
            self.assertFalse(phone["phone_app_changed"]); self.assertFalse(phone["deployment_qualified"])
            dev=json.loads(members["accuracy/dev_manifest.json"])
            self.assertEqual(dev["images"],16); self.assertFalse(dev["model_quantization_performed"])

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),"optional ONNX packages")
    def test_actual_two_input_onnx_roundtrip_with_camera_perturbation(self):
        model=self.tiny(); image=torch.randn(1,3,64,96); camera=torch.tensor([[80.,80.,47.5,31.5]])
        c=export.configuration(); c["input_shapes"]={"image":[1,3,64,96],"camera":[1,4]}
        c["output_shapes"]={"candidates":[1,32,126],"depth_m":[1,1,16,24]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.onnx"
            torch.onnx.export(model,(image,camera),str(path),input_names=list(c["input_shapes"]),
                output_names=list(c["output_shapes"]),opset_version=17,dynamo=False)
            session=export.session(path.read_bytes(),c)
            for multiplier in [1.,1.1]:
                altered=camera*multiplier
                with torch.inference_mode(): ref=model(image,altered)
                actual=session.run(None,{"image":image.numpy(),"camera":altered.numpy()})
                export.output_checks(dict(zip(c["output_shapes"],[v.numpy() for v in ref])),dict(zip(c["output_shapes"],actual)),c)

    def test_notebook_is_clean_cpu_only_local_first(self):
        path=export.ROOT/"notebooks/Parking_V0_V1_Trained_ONNX_CPU_Colab.ipynb"
        book=json.loads(path.read_text()); code=[]
        for cell in book["cells"]:
            if cell["cell_type"]=="code":
                text="".join(cell["source"]); ast.parse(text); code.append(text)
                self.assertEqual(cell["outputs"],[]); self.assertIsNone(cell["execution_count"])
        combined="\n".join(code)
        self.assertEqual(len(code),6)
        self.assertIn(export.configuration()["revision"],combined)
        self.assertIn("copy_once",combined); self.assertIn("allow_failure",combined)
        for forbidden in ["nvidia-smi","--device','cuda", "reset', '--hard", "setup_m67"]:
            self.assertNotIn(forbidden,combined)


if __name__=="__main__": unittest.main()
