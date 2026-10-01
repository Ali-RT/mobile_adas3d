import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
import pickle
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from diagnose_m63_pedestrian_tradeoff import sample_rows,compare,state_difference,validate_predictions,load_model_state_safely
from m61_common import sha256,write_json

def target():
    return dict(class_name="Pedestrian",bbox_2d=[0,0,50,100],location_3d=[0,1.7,10],
                dimensions_3d=[1.7,.6,.8],rotation_y=0.,truncated=0.,occluded=0)

def prediction():
    t=target()
    t["dimensions_3d_hwl"]=t.pop("dimensions_3d")
    t["score"]=.9
    return t

class UnsupportedMetadata:
    pass

class TradeoffTests(unittest.TestCase):
    def test_stable_matches_and_thresholds(self):
        rows,counts=sample_rows("000001",[target()],[prediction()])
        r=rows["000001:0"]
        self.assertTrue(r["matched"])
        self.assertAlmostEqual(r["depth_abs_error_m"],0)
        self.assertEqual(counts["0.5"]["matched"],1)
        self.assertEqual(r["difficulty"],"easy")

    def test_common_geometry_excludes_lost_detections(self):
        a,_=sample_rows("000001",[target()],[prediction()])
        b,_=sample_rows("000001",[target()],[])
        result=compare(a,b)["all"]
        self.assertEqual(result["lost_detections"],1)
        self.assertEqual(result["common_matched"],0)
        self.assertIsNone(result["delta_on_common"]["depth_abs_error_m"])

    def test_common_geometry_tracks_depth_change(self):
        a,_=sample_rows("000001",[target()],[prediction()])
        p=prediction();p["location_3d"][2]=12
        b,_=sample_rows("000001",[target()],[p])
        result=compare(a,b)["moderate_including_easy"]
        self.assertEqual(result["common_matched"],1)
        self.assertAlmostEqual(result["delta_on_common"]["depth_abs_error_m"],2.)

    def test_miss_categories_and_competition(self):
        p=prediction();p["class_name"]="Vehicle"
        rows,_=sample_rows("x",[target()],[p])
        self.assertEqual(rows["x:0"]["status"],"other_class_overlap_candidate")
        rows,_=sample_rows("x",[target(),target()],[prediction()])
        self.assertEqual(sum(r["matched"] for r in rows.values()),1)
        self.assertIn("one_to_one_competition",[r["status"] for r in rows.values()])
        p=prediction();p["score"]=.0001
        rows,_=sample_rows("x",[target()],[p])
        self.assertEqual(rows["x:0"]["status"],"below_score_floor_in_saved_predictions")

    def test_read_only_state_difference(self):
        a={"backbone.weight":torch.tensor([1.,2.]),"bn.running_mean":torch.tensor([0.])}
        b=copy.deepcopy(a);b["backbone.weight"][0]=2.
        result=state_difference(a,b)
        self.assertEqual(result["backbone"]["changed"],1)
        self.assertEqual(result["bn_buffers"]["changed"],0)
        self.assertEqual(float(a["backbone.weight"][0]),1.)
        with self.assertRaises(RuntimeError): state_difference(a,{})

    def test_prediction_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"native/outputs/data";data.mkdir(parents=True)
            p=data/"000001.txt";p.write_text("")
            write_json(root/"inference_manifest.json",dict(checkpoint_sha256="a",manifest_sha256="b",prediction_files={"000001":sha256(p)}))
            validate_predictions(root,["000001"],"a","b")
            p.write_text("changed")
            with self.assertRaises(RuntimeError): validate_predictions(root,["000001"],"a","b")

    def test_numpy_scalar_metadata_restricted_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"checkpoint.pth"
            state={"layer.weight":torch.tensor([1.,2.])}
            torch.save({"model_state":state,"best_result":np.float64(7.5),
                        "other_score":np.float32(1.2)},p)
            before=list(torch.serialization.get_safe_globals())
            loaded=load_model_state_safely(p,sha256(p))
            self.assertTrue(torch.equal(state["layer.weight"],loaded["layer.weight"]))
            self.assertEqual(set(torch.serialization.get_safe_globals()),set(before))

    def test_wrong_hash_stops_before_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"checkpoint.pth";p.write_bytes(b"not a checkpoint")
            with patch("torch.load") as loader:
                with self.assertRaises(RuntimeError): load_model_state_safely(p,"wrong")
                loader.assert_not_called()

    def test_unapproved_metadata_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"checkpoint.pth"
            torch.save({"model_state":{"w":torch.ones(1)},"other":UnsupportedMetadata()},p)
            with self.assertRaises(pickle.UnpicklingError):
                load_model_state_safely(p,sha256(p))

    def test_non_tensor_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"checkpoint.pth"
            torch.save({"model_state":{"w":"bad"}},p)
            with self.assertRaises(RuntimeError): load_model_state_safely(p,sha256(p))

    def test_notebook_is_analysis_only(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        code="".join(nb["cells"][-1]["source"]);ast.parse(code)
        self.assertIn("diagnose_m63_pedestrian_tradeoff.py",code)
        self.assertNotIn("--train",code);self.assertNotIn("--infer",code)
        script=(ROOT/"scripts/diagnose_m63_pedestrian_tradeoff.py").read_text()
        self.assertNotIn("build_runtime(",script)
        self.assertNotIn("optimizer.step(",script)

if __name__=="__main__": unittest.main()
