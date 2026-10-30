"""Synthetic CPU tests; no dataset download, real-model training or phone use."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import parking_v1_pilot_data as data
from scripts import train_parking_v1_near as pilot
from scripts import parking_edge_common as common
from scripts import parking_v0_v1_metrics as metrics


def fixture(token="train0", log="trainlog", role="train"):
    protocol = data.prepare.reference.load_protocol()
    objects = []; truth = []; semantic = []; owners = []
    for cls, sem in [("Vehicle",17),("Pedestrian",2)]:
        for z in [1.,3.,5.]:
            index = len(objects)
            objects.append({"class_name":cls,"semantic_index":sem,"instance_token":f"track{index}",
                "annotation_token":f"ann{token}_{index}"})
            truth.extend([z]*5); semantic.extend([sem]*5); owners.extend([index]*5)
    points = {"xy":np.column_stack([np.arange(30),np.zeros(30)]).astype(np.int32),
        "z":np.asarray(truth,np.float32),"semantic":np.asarray(semantic,np.uint8),"object_index":np.asarray(owners,np.int32)}
    points["cell_masks"] = metrics.make_masks(points,objects,protocol)
    sample = {"camera_token":token,"sample_token":"frame_"+token,"scene":"scene_"+log,"recording_log":log,
        "role":role,"width":1600,"height":900,"objects":objects,"channel":"CAM_BACK",
        "intrinsics":[[1200.,0.,800.],[0.,1200.,450.],[0.,0.,1.]],"cell_names":metrics.cell_names(protocol),
        "broad_reference_points":2}
    rng=np.random.default_rng(0)
    arrays={"p3":rng.normal(0,.1,(64,48,80)).astype(np.float32),"p4":np.zeros((128,24,40),np.float32),
        "p5":np.zeros((256,12,20),np.float32),"z":points["z"],"object_index":points["object_index"],
        "cell_masks":points["cell_masks"],"grid":np.column_stack([np.linspace(-.8,.8,30),np.zeros(30)]).astype(np.float32),
        "broad_z":np.asarray([3.,7.],np.float32),"broad_grid":np.asarray([[-.3,0.],[.3,0.]],np.float32)}
    return sample,points,arrays


def sampling_fixture():
    c=data.configuration();index={"near":{cell:{} for cell in c["near_cells"]},"broad":{}}
    for i in range(4):
        sample,points,_=fixture("train"+str(i),"log"+str(i%2))
        pilot.index_observations(index,sample,points,c)
    return index


class ContractTests(unittest.TestCase):
    def test_fixed_budget_and_no_holdout_or_phone(self):
        c=data.configuration()
        self.assertEqual(c["updates_per_arm"],1920)
        self.assertEqual(c["blocks"]*c["updates_per_block"]*c["batch_size"],15360)
        self.assertEqual(c["trainable_parameters"],1553)
        for key in ["heldout_evaluation_authorized","phone_update_authorized","automatic_extension_authorized"]:
            self.assertFalse(c[key])
        self.assertIsNone(c["accuracy_threshold_m"])

    def test_changed_proposal_fails(self):
        with mock.patch.object(common,"file_sha",return_value="changed"):
            with self.assertRaises(ValueError):data.configuration()

    def test_train_benchmark_split_three_levels(self):
        train,_,_=fixture();dev,_,_=fixture("dev","devlog","dev")
        benchmark={"samples":[dev]};data.disjoint([train],benchmark)
        for key in ["camera_token","sample_token","recording_log"]:
            bad=copy.deepcopy(dev);bad[key]=train[key]
            with self.assertRaises(ValueError):data.disjoint([train],{"samples":[bad]})
        bad=copy.deepcopy(train);bad["role"]="dev"
        with self.assertRaises(ValueError):data.disjoint([bad],benchmark)

    def test_nested_and_drive_work_paths_rejected(self):
        with self.assertRaises(ValueError):data.paths_separate([Path("/tmp/one"),Path("/tmp/one/cache")])
        with self.assertRaises(ValueError):data.local_path(Path("/content/drive/MyDrive/cache"))

    def test_local_archive_reuse_never_reads_drive(self):
        with tempfile.TemporaryDirectory() as f:
            dst=Path(f)/"local.zip";dst.write_bytes(b"verified")
            with mock.patch.object(data.prepare,"atomic_copy",side_effect=AssertionError("No remote read")):
                data.transfer(Path("/unreadable/drive.zip"),dst,common.sha(b"verified"))
            with self.assertRaises(ValueError):data.transfer(Path("/unreadable"),dst,"wrong")

    def test_errno5_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as f:
            with mock.patch.object(data.prepare,"atomic_copy",side_effect=OSError(5,"I/O")) as copy_fn:
                with self.assertRaisesRegex(RuntimeError,"No retry/remount"):
                    data.transfer(Path("/source"),Path(f)/"target","sha")
                self.assertEqual(copy_fn.call_count,1)

    def test_notebook_code_compiles_and_clear_revision(self):
        n=json.loads((data.ROOT/"notebooks/Parking_V1_Matched_Near_Field_GPU_Colab.ipynb").read_text())
        code=""
        for i,cell in enumerate(n["cells"]):
            if cell["cell_type"]=="code":
                text="".join(cell["source"]);compile(text,f"cell{i}","exec");code+=text
        self.assertIn(data.configuration()["revision"],code)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG",code)
        self.assertIn("PYTHONPATH",code)
        self.assertIn("stage('cache')",code)
        self.assertIn("stage('smoke')",code)
        self.assertIn("CPU_STAGE_PASSED",code)
        self.assertNotIn("reset', '--hard",code)
        self.assertNotIn("rmtree",code)


class SamplingTests(unittest.TestCase):
    def test_six_draws_plus_two_broad_identical_stream(self):
        c=data.configuration();index=sampling_fixture()
        first=pilot.batches(index,c,1)
        self.assertEqual(first,pilot.batches(index,c,1))
        self.assertNotEqual(first,pilot.batches(index,c,2))
        self.assertEqual(len(first),384)
        self.assertTrue(all([d["cell"] for d in b]==c["near_cells"]+["broad","broad"] for b in first))
        self.assertEqual(len(first[0]),8)

    def test_balancing_uses_logs_tracks_annotations_cameras(self):
        index=sampling_fixture(); c=data.configuration();cell=c["near_cells"][0]
        self.assertEqual(len(index["near"][cell]),2)
        draws=[d for batch in pilot.batches(index,c,1) for d in batch if d["cell"]==cell]
        self.assertTrue(all(d["track"] and d["annotation"] for d in draws))
        self.assertTrue(130<sum(d["log"]=="log0" for d in draws)<250)

    def test_invalid_block_and_nontrain_sampler_fail(self):
        c=data.configuration();index=sampling_fixture()
        for b in [0,6]:
            with self.assertRaises(ValueError):pilot.batches(index,c,b)
        s,p,_=fixture(role="dev")
        with self.assertRaises(ValueError):pilot.index_observations(index,s,p,c)


class CacheAndProgressTests(unittest.TestCase):
    def test_real_training_shard_reader_and_byte_tamper(self):
        from PIL import Image
        s,p,_=fixture();protocol=data.prepare.reference.load_protocol()
        image=io.BytesIO();Image.new("RGB",(s["width"],s["height"]),(50,90,120)).save(image,format="JPEG")
        broad={"xy":np.asarray([[1,1],[2,2]],np.int32),"z":np.asarray([3.,7.],np.float32)}
        payloads={"image":image.getvalue(),"ground_truth":data.prepare.reference.encode_npz(p),
            "broad":data.prepare.reference.encode_npz(broad)}
        for kind,raw in payloads.items():
            s[kind+"_member"]=kind+"/train0."+("jpg" if kind=="image" else "npz")
            s[kind+"_sha256"]=common.sha(raw)
        members={s[k+"_member"]:v for k,v in payloads.items()}
        members["shard_manifest.json"]=common.encoded(data.prepare.coverage.signed({"samples":[s]}))
        raw=data.local.v0.zip_bytes(members);sample={**s,"part_archive":"train_parts/train_part01.zip",
            "shard_member":"shards/train/part01_000.zip","shard_sha256":common.sha(raw)}
        with tempfile.TemporaryDirectory() as f:
            inputs=Path(f);path=inputs/"train"/sample["shard_member"];common.immutable(path,raw)
            loaded=data.load_train_shard(inputs,sample["shard_member"],[sample],protocol)
            np.testing.assert_array_equal(loaded["train0"]["ground_truth"]["z"],p["z"])
            path.write_bytes(raw+b"changed")
            with self.assertRaises(ValueError):data.load_train_shard(inputs,sample["shard_member"],[sample],protocol)

    def test_feature_roundtrip_and_changed_identity(self):
        s,_,a=fixture()
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/"feature.zip";p.write_bytes(pilot.feature_payload(a,s,"cache"))
            got=pilot.read_features(p,s,"cache")
            for key in a:np.testing.assert_array_equal(got[key],a[key])
            with self.assertRaises(ValueError):pilot.read_features(p,s,"other")
            bad=copy.deepcopy(s);bad["intrinsics"][0][0]+=1
            with self.assertRaises(ValueError):pilot.read_features(p,bad,"cache")

    def test_feature_tamper_fails(self):
        s,_,a=fixture()
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/"feature.zip";p.write_bytes(pilot.feature_payload(a,s,"cache"))
            with zipfile.ZipFile(p) as z:members={n:z.read(n) for n in z.namelist()}
            members["p3.npy"]=b"changed";p.write_bytes(data.local.v0.zip_bytes(members))
            with self.assertRaises(ValueError):pilot.read_features(p,s,"cache")

    def test_progress_roundtrip_more_than_64_files(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);a=SimpleNamespace(output=root/"out",publish=None);identity={"test":"pilot"}
            common.immutable(a.output/"run_identity.json",common.encoded(identity))
            for i in range(80):common.immutable(a.output/"endpoints"/f"{i:03d}.json",b"{}")
            common.immutable(a.output/"colab_logs/private.log",b"not in archive")
            p=pilot.progress(a,identity,"block")
            pilot.restore_progress(p,root/"restored")
            self.assertEqual(len(list((root/"restored/endpoints").iterdir())),80)
            self.assertFalse((root/"restored/colab_logs").exists())
            with zipfile.ZipFile(p) as z:self.assertFalse(any("portable/" in n for n in z.namelist()))

    def test_progress_different_output_preserved(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);a=SimpleNamespace(output=root/"out",publish=None);identity={"test":"pilot"}
            common.immutable(a.output/"run_identity.json",common.encoded(identity))
            p=pilot.progress(a,identity,"block")
            common.immutable(root/"restore/run_identity.json",b"keep")
            with self.assertRaises(RuntimeError):pilot.restore_progress(p,root/"restore")
            self.assertEqual((root/"restore/run_identity.json").read_bytes(),b"keep")

    def test_wrong_training_shard_role_rejected(self):
        s,_,_=fixture(role="dev");s["shard_member"]="shards/dev/test.zip"
        with self.assertRaises(ValueError):data.training_shards({"samples":[s]})


class SelectionTests(unittest.TestCase):
    def metric(self,score,coverage=1.):
        return {"conditional_equal_four_cell_mae_m":score,"cells":{"a":{"prediction_coverage":coverage},"b":{"prediction_coverage":1.}}}

    def test_baseline_can_win_ties_choose_earlier(self):
        base=self.metric(1.)
        history=[{"block":0,"dev":base},{"block":1,"dev":self.metric(1.)},{"block":2,"dev":self.metric(1.1)}]
        self.assertEqual(pilot.best_endpoint(history,base)["block"],0)
        history[2]["dev"]=self.metric(.9)
        self.assertEqual(pilot.best_endpoint(history,base)["block"],2)

    def test_lower_error_cannot_hide_lost_coverage(self):
        base=self.metric(1.);history=[{"block":0,"dev":base},{"block":1,"dev":self.metric(.5,.9)}]
        # Dict order must not affect matching cell coverage.
        history[1]["dev"]["cells"]=dict(reversed(list(history[1]["dev"]["cells"].items())))
        self.assertEqual(pilot.best_endpoint(history,base)["block"],0)

    def test_resume_budget_arm_sampler_and_history(self):
        c=data.configuration();index=sampling_fixture();arm=c["arms"][0]
        history=[{"block":0,"optimizer_steps":0},{"block":1,"optimizer_steps":384,"sampler_sha256":common.identity(pilot.batches(index,c,1))}]
        saved={"run_identity":"id","arm":arm,"block":1,"optimizer_steps":384,"history":history}
        pilot.validate_resume(saved,arm,"id",index,c)
        for key,val in [("optimizer_steps",385),("arm","other"),("block",6),("run_identity","other")]:
            bad=copy.deepcopy(saved);bad[key]=val
            with self.assertRaises(ValueError):pilot.validate_resume(bad,arm,"id",index,c)
        bad=copy.deepcopy(saved);bad["history"][1]["sampler_sha256"]="changed"
        with self.assertRaises(ValueError):pilot.validate_resume(bad,arm,"id",index,c)

    def test_signed_bias_uses_frozen_hierarchy(self):
        s,p,_=fixture(role="dev");s["objects"]=[{**o,"source_class":"test"} for o in s["objects"]]
        protocol=data.prepare.reference.load_protocol()
        bias=pilot.signed_bias([s],{s["camera_token"]:p},{s["camera_token"]:p["z"]+.5},protocol)
        self.assertTrue(all(v is None or v==.5 for v in bias.values()))


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_control_ignores_sub2_candidate_uses_sub2(self):
        import torch
        s,_,row=fixture();c=data.configuration()
        for arm,expect_zero in zip(c["arms"],[True,False]):
            pred=torch.tensor(row["z"]+.5,requires_grad=True);broad=torch.tensor(row["broad_z"]+.5,requires_grad=True)
            loss,_=pilot.loss_components([pred],[broad],[row],[s],c,arm);loss.backward()
            sub2=row["z"]<2
            self.assertEqual(bool((pred.grad[sub2]==0).all()),expect_zero)
            self.assertTrue((pred.grad[~sub2]>0).all())

    def test_huber_raw_metres_and_component_weights(self):
        import torch
        s,_,r=fixture();c=data.configuration();pred=torch.tensor(r["z"]+.5);broad=torch.tensor(r["broad_z"]+.5)
        for arm in c["arms"]:
            loss,parts=pilot.loss_components([pred],[broad],[r],[s],c,arm)
            self.assertAlmostEqual(float(loss),.08,places=6) # delta*(.5-.5*delta)
            self.assertAlmostEqual(float(parts["near_mean"]),.08,places=6)

    def test_missing_cell_and_nonfinite_fail(self):
        import torch
        s,_,r=fixture();c=data.configuration();pred=torch.tensor(r["z"]+.5);broad=torch.tensor(r["broad_z"]+.5)
        bad=copy.deepcopy(r);bad["cell_masks"][:]=False
        with self.assertRaises(ValueError):pilot.loss_components([pred],[broad],[bad],[s],c,c["arms"][1])
        pred[0]=float("nan")
        with self.assertRaises(ValueError):pilot.loss_components([pred],[broad],[r],[s],c,c["arms"][1])

    def make_decoder(self):
        import torch
        from torch import nn
        from torch.nn import functional as F
        from models.parking_shared_camera import CameraAwareDepthDecoder
        class Base(nn.Module):
            def __init__(self):
                super().__init__();self.bn=nn.BatchNorm2d(64);self.out=nn.Conv2d(64,1,1)
            def forward(self,features):
                return F.interpolate(torch.exp(self.out(self.bn(features[0]))),size=(96,160),mode="nearest")
        torch.manual_seed(0)
        decoder=CameraAwareDepthDecoder(Base());pilot.freeze_adapter(decoder)
        return decoder

    def test_real_adapter_smoke_and_step_preserve_base_and_bn(self):
        import torch
        from scripts.screen_parking_shared_model import state_digest
        decoder=self.make_decoder();s,_,row=fixture();c=data.configuration()
        params=pilot.freeze_adapter(decoder);self.assertEqual(sum(p.numel() for p in params),1553)
        optimizer=torch.optim.AdamW(params,lr=c["learning_rate"],weight_decay=0)
        before=state_digest(decoder);base=state_digest(decoder.base)
        pilot.adapter_step(decoder,optimizer,[row],[s],c,c["arms"][1],"cpu",update=False)
        self.assertEqual(state_digest(decoder),before)
        pilot.adapter_step(decoder,optimizer,[row],[s],c,c["arms"][1],"cpu")
        self.assertNotEqual(state_digest(decoder),before);self.assertEqual(state_digest(decoder.base),base)
        self.assertTrue(all(not m.training for m in decoder.base.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in decoder.base.parameters()))

    def test_atomic_checkpoint_roundtrip_and_wrong_run(self):
        import torch
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/"checkpoint.zip";value={"run_identity":"test","tensor":torch.tensor([1.])}
            pilot.checkpoint(p,value);got=pilot.read_checkpoint(p,"test")
            self.assertTrue(torch.equal(got["tensor"],value["tensor"]))
            with self.assertRaises(ValueError):pilot.read_checkpoint(p,"other")

    def test_interruption_resume_matches_uninterrupted_optimizer(self):
        import torch
        s,_,row=fixture();c=data.configuration();arm=c["arms"][1]
        decoder=self.make_decoder();initial=copy.deepcopy(decoder.state_dict())
        optimizer=torch.optim.AdamW(pilot.freeze_adapter(decoder),lr=c["learning_rate"],weight_decay=0)
        pilot.adapter_step(decoder,optimizer,[row],[s],c,arm,"cpu")
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/"saved.zip"
            pilot.checkpoint(p,{"run_identity":"resume","decoder":copy.deepcopy(decoder.state_dict()),"optimizer":optimizer.state_dict()})
            pilot.adapter_step(decoder,optimizer,[row],[s],c,arm,"cpu")
            expected=copy.deepcopy(decoder.state_dict())
            second=self.make_decoder();second.load_state_dict(initial)
            opt=torch.optim.AdamW(pilot.freeze_adapter(second),lr=c["learning_rate"],weight_decay=0)
            saved=pilot.read_checkpoint(p,"resume");second.load_state_dict(saved["decoder"]);opt.load_state_dict(saved["optimizer"])
            pilot.adapter_step(second,opt,[row],[s],c,arm,"cpu")
            self.assertTrue(all(torch.equal(v,second.state_dict()[k]) for k,v in expected.items()))

    def test_train_loop_resumes_only_completed_blocks(self):
        import torch
        from torch import nn
        from scripts.screen_parking_shared_model import state_digest
        c=data.configuration();c.update(blocks=2,updates_per_block=1,updates_per_arm=2)
        sample,points,row=fixture();index={"near":{cell:{} for cell in c["near_cells"]},"broad":{}}
        pilot.index_observations(index,sample,points,c)
        decoder=self.make_decoder();initial=copy.deepcopy(decoder.state_dict())
        native=SimpleNamespace(body=nn.Conv2d(1,1,1).eval().requires_grad_(False))
        identity={"config":c,"frozen_body_sha256":state_digest(native.body),"frozen_base_sha256":state_digest(decoder.base)}
        metric=metrics.Metrics(data.prepare.reference.load_protocol());metric.add(sample,points,points["z"]+.5)
        base=metric.result();arm=c["arms"][1]
        endpoint=copy.deepcopy(base);endpoint["conditional_equal_four_cell_mae_m"]-=.1
        def score(*args):return endpoint,[],{},{}
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);a=SimpleNamespace(output=root/"out",cache=root/"features",device="cpu")
            shared=[mock.patch.object(pilot,"read_features",return_value=row),mock.patch.object(pilot,"progress",return_value=None)]
            with shared[0],shared[1],mock.patch.object(pilot,"evaluate",side_effect=[score(),RuntimeError("interrupted endpoint")]):
                with self.assertRaisesRegex(RuntimeError,"interrupted endpoint"):
                    pilot.train_arm(a,native,decoder,identity,index,"cache",{"samples":[sample]}, {},[],initial,base,arm)
            self.assertTrue((a.output/"checkpoints"/arm/"block_001.zip").is_file())
            self.assertFalse((a.output/"checkpoints"/arm/"block_002.zip").exists())
            with mock.patch.object(pilot,"read_features",return_value=row),mock.patch.object(pilot,"progress",return_value=None),mock.patch.object(pilot,"evaluate",side_effect=score) as evaluate:
                result=pilot.train_arm(a,native,decoder,identity,index,"cache",{"samples":[sample]}, {},[],initial,base,arm)
                self.assertEqual(evaluate.call_count,1)
            self.assertEqual(result["optimizer_steps"],2)
            self.assertEqual([r["block"] for r in result["history"]],[0,1,2])
            self.assertEqual(result["selected_block"],1)
            with mock.patch.object(pilot,"read_features",side_effect=AssertionError("Do not retrain")),mock.patch.object(pilot,"evaluate",side_effect=AssertionError("Do not reevaluate")):
                repeated=pilot.train_arm(a,native,decoder,identity,index,"cache",{"samples":[sample]}, {},[],initial,base,arm)
            self.assertEqual(result,repeated)


if __name__=="__main__":unittest.main()
