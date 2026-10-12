"""CPU synthetic tests; no downloaded models or claimed KITTI/teacher accuracy."""
import ast
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING
from models.compact_distillation import (COMPONENTS, distillation_loss, match_teacher,
                                        qualification, usable_components)
from scripts.compact_a2_teacher import (decode_teacher, teacher_inputs,
                                        preprocessing_comparison, decoded_fixture_comparison)
from scripts.distill_compact_a2 import (DistillationDataset, SETTINGS, audit_frame,
                                      freeze_bn, immutable_json, pilot_update, signature,
                                      train, read_audited_cache, unchanged_source)
# Colab can install a regular package named "tests", which shadows this
# repository's namespace directory. Load the sibling fixtures by exact path;
# do not depend on package resolution or change the frozen GT training source.
_spec = importlib.util.spec_from_file_location(
    "_compact_kitti_oracle", Path(__file__).with_name("test_compact_kitti_training.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot locate the compact GT oracle fixtures")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
oracle, fixture = _helpers.oracle, _helpers.fixture


class CompactKDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_discovery_ignores_foreign_tests_package(self):
        """Exercise notebook discovery with an already imported rival package."""
        root = Path(__file__).resolve().parents[1]
        code = ("import sys, types, unittest; "
                "foreign = types.ModuleType('tests'); foreign.__path__ = []; "
                "sys.modules['tests'] = foreign; "
                "loader = unittest.TestLoader(); "
                "suite = loader.discover('tests', pattern='test_compact_a2_distillation.py'); "
                "assert not loader.errors, '\\n'.join(loader.errors); "
                "assert suite.countTestCases() >= 16; "
                "assert 'tests.test_compact_kitti_training' not in sys.modules")
        result = subprocess.run([sys.executable, '-c', code], cwd=root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_teacher_class_order_and_native_metric_decode(self):
        p, obj = fixture()
        values = {"logits": np.full((1,50,3), -20., np.float32),
                  "boxes": np.zeros((1,50,6), np.float32),
                  "dimensions": np.zeros((1,50,3), np.float32),
                  "depth": np.zeros((1,50,2), np.float32),
                  "angle": np.zeros((1,50,24), np.float32)}
        values["logits"][0,0,1]=8
        values["boxes"][0,0]=[.5,.5,.1,.1,.1,.1]
        values["dimensions"][0,0]=[1.6,1.7,4.1]
        values["depth"][0,0]=[22.,0.]
        rows=decode_teacher(values,p,[1242,375])
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]["class_name"],"Vehicle")
        self.assertFalse(rows[0]["invalid_geometry"])
        self.assertAlmostEqual(rows[0]["location_3d"][2],22)
        values["logits"][0,0]=[-20, -20, 8]
        self.assertEqual(decode_teacher(values,p,[1242,375]),[])
        values["depth"][0,0,0]=np.nan
        with self.assertRaises(ValueError): decode_teacher(values,p,[1242,375])

    def test_teacher_rgb_pil_contract(self):
        from PIL import Image
        p,_=fixture()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"image.png"
            Image.fromarray(np.full((384,1280,3),[255,64,0],np.uint8)).save(path)
            value=teacher_inputs(path,p)
            expected=(np.asarray([1.,64/255,0])-np.asarray([.485,.456,.406]))/np.asarray([.229,.224,.225])
            np.testing.assert_allclose(value["image"][0,:,192,640],expected,atol=1e-5)
            self.assertEqual(value["image"].shape,(1,3,384,1280))
            self.assertEqual(value["image"].dtype,np.float32)
            np.testing.assert_array_equal(value["calibration"][0],p)

    def test_preprocessing_allows_only_sparse_single_rgb_level_rounding(self):
        saved=dict(image=np.zeros((1,3,384,1280),np.float32),calibration=np.zeros((1,3,4),np.float32),
                   image_size=np.asarray([[1242,375]],np.float32))
        actual={k:v.copy() for k,v in saved.items()}
        actual['image'][0,1,0,0]=1/255/.224
        self.assertTrue(preprocessing_comparison(actual,saved)['image']['passed'])
        actual['image'][0,1,0,0]*=2
        with self.assertRaises(RuntimeError):preprocessing_comparison(actual,saved)
        actual={k:v.copy() for k,v in saved.items()};actual['image'][0,1,:2,:]=1/255/.224
        with self.assertRaises(RuntimeError):preprocessing_comparison(actual,saved)
        actual={k:v.copy() for k,v in saved.items()};actual['calibration'][0,0,0]=.01
        with self.assertRaises(RuntimeError):preprocessing_comparison(actual,saved)

    def test_reconstructed_decoded_fixture_rejects_changed_guidance(self):
        p,_=fixture()
        reference=dict(logits=np.full((1,50,3),-20.,np.float32),boxes=np.zeros((1,50,6),np.float32),
                       dimensions=np.ones((1,50,3),np.float32),depth=np.ones((1,50,2),np.float32),
                       angle=np.zeros((1,50,24),np.float32))
        reference['logits'][0,0,1]=8
        reference['boxes'][0,0]=[.5,.5,.1,.1,.1,.1];reference['depth'][0,0]=[22,0]
        actual={k:v.copy() for k,v in reference.items()}
        actual['depth'][0,0,0]+=.001
        self.assertTrue(decoded_fixture_comparison(actual,reference,p,[1242,375])['passed'])
        actual['depth'][0,0,0]+=1
        with self.assertRaises(RuntimeError):decoded_fixture_comparison(actual,reference,p,[1242,375])

    def test_matching_is_class_aware_one_to_one_and_rejects_invalid(self):
        _,obj=fixture()
        pred=dict(class_name="Vehicle",bbox_2d=obj.bbox_2d,score=.9,invalid_geometry=False)
        self.assertEqual(set(match_teacher([obj,obj],[pred],KITTI_PRODUCTION_CLASS_MAPPING)),{0})
        pred["class_name"]="Pedestrian"
        self.assertEqual(match_teacher([obj],[pred],KITTI_PRODUCTION_CLASS_MAPPING),{})
        pred.update(class_name="Vehicle",invalid_geometry=True)
        self.assertEqual(match_teacher([obj],[pred],KITTI_PRODUCTION_CLASS_MAPPING),{})

    def test_class_component_audit_and_individual_worse_targets(self):
        rows=[dict(class_name="Vehicle",student_errors={k:2. for k in COMPONENTS},
                   teacher_errors={k:1. for k in COMPONENTS}) for _ in range(16)]
        report=qualification(rows)
        self.assertTrue(report["Vehicle"]["depth"]["approved"])
        self.assertFalse(report["Pedestrian"]["depth"]["approved"])
        rows[0]["teacher_errors"]["depth"]=3.
        self.assertFalse(usable_components(rows[0],report)["depth"])
        self.assertTrue(usable_components(rows[0],report)["dim"])
        self.assertFalse(qualification(rows[:15])["Vehicle"]["depth"]["approved"])

    def kd_batch(self):
        batch, output,_=oracle((8,32)); t=batch["targets"]
        batch["kd"]=dict(hps=t["hps"].clone(),dim=t["dim"].clone(),rotbin=t["rotbin"].clone(),
                         rotres=t["rotres"].clone(),depth=t["location"][...,2].clone(),
                         **{k+"_mask":torch.zeros_like(t["geom_mask"]) for k in COMPONENTS})
        return batch,output

    def test_zero_masks_produce_zero_kd_and_no_teacher_gradient(self):
        batch,output=self.kd_batch()
        output={k:v.requires_grad_(True) for k,v in output.items()}
        total,_=distillation_loss(output,batch)
        self.assertEqual(float(total.detach()),0.)
        total.backward()
        for k in ("hps","dim","rot"):
            self.assertTrue(torch.isfinite(output[k].grad).all())
            self.assertEqual(float(output[k].grad.abs().sum()),0.)

    def test_depth_kd_has_finite_nonzero_geometry_gradients(self):
        batch,output=self.kd_batch()
        batch["kd"]["depth"][0,0]+=2
        batch["kd"]["depth_mask"][0,0]=1
        output={k:v.requires_grad_(True) for k,v in output.items()}
        total,_=distillation_loss(output,batch);total.backward()
        self.assertGreater(float(total.detach()),0.)
        self.assertGreater(float(output["hps"].grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(output["hps"].grad).all())

    def test_oracle_audit_frame_and_matching(self):
        batch,outputs,obj=oracle()
        item=dict(image=torch.zeros(3,384,1280),projection=batch["projection"][0],
                  inverse_affine=batch["inverse_affine"][0],targets={k:v[0] for k,v in batch["targets"].items()})
        class Oracle(torch.nn.Module):
            def forward(self,image): return outputs
        pred=dict(class_name="Vehicle",score=.9,bbox_2d=obj.bbox_2d,dimensions_3d=obj.dimensions_3d,
                  location_3d=obj.location_3d,rotation_y=obj.rotation_y,alpha=obj.alpha,invalid_geometry=False)
        rows,checks=audit_frame(Oracle(),item,[obj],{0:pred},"cpu")
        self.assertEqual(len(rows),1)
        self.assertLess(checks[0]["double_relative_error"],.001)
        self.assertLess(checks[0]["fp32_relative_error"],.005)
        self.assertLess(rows[0]["teacher_errors"]["hps"],5e-5)

    def test_bn_freeze_and_inference_parameter_shapes_unchanged(self):
        from models.compact_monocular3d import CompactMonocular3D
        model=CompactMonocular3D(backbone="mobilenetv4_conv_small")
        before={k:v.shape for k,v in model.state_dict().items()}
        model.train();freeze_bn(model)
        self.assertTrue(all(not x.training for x in model.modules() if isinstance(x,torch.nn.BatchNorm2d)))
        self.assertEqual(before,{k:v.shape for k,v in model.state_dict().items()})
        self.assertEqual(SETTINGS["gt_weight"],1.)
        self.assertFalse(SETTINGS["architecture_changed"])

    def test_real_compact_kd_update_full_resolution(self):
        from models.compact_monocular3d import CompactMonocular3D
        batch, _, _ = oracle()
        t = batch["targets"]
        batch["image"] = torch.randn(1,3,384,1280)
        batch["kd"] = dict(hps=t["hps"].clone(),dim=t["dim"].clone(),
                           rotbin=t["rotbin"].clone(),rotres=t["rotres"].clone(),
                           depth=t["location"][...,2].clone(),
                           **{k+"_mask": t["geom_mask"].clone() for k in COMPONENTS})
        model = CompactMonocular3D(backbone="mobilenetv4_conv_small")
        model.train();freeze_bn(model)
        before={k:v.clone() for k,v in model.state_dict().items() if "running_" in k}
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        row=pilot_update(model,batch,optimizer,.5)
        self.assertTrue(all(math.isfinite(v) for v in row.values()))
        self.assertGreater(row["kd"],0.)
        for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)

    def test_augmented_kd_targets_use_current_affine(self):
        batch, _, obj = oracle()
        p,_=fixture()
        from data.compact_kitti import project_keypoints
        row=dict(gt_location=obj.location_3d,gt_dimensions=obj.dimensions_3d,class_name="Vehicle",
                 teacher=dict(dimensions_3d=obj.dimensions_3d,location_3d=obj.location_3d,alpha=obj.alpha),
                 projected_keypoints=project_keypoints(obj,p).tolist(),
                 student_errors={k:1. for k in COMPONENTS},teacher_errors={k:0. for k in COMPONENTS})
        approved=qualification([row]*16)
        item=dict(sample_id="000001",image=torch.zeros(3,384,1280),inverse_affine=batch["inverse_affine"][0],
                  targets={k:v[0] for k,v in batch["targets"].items()})
        dataset=DistillationDataset(Path('/unused'),['000001'],records={'000001':[row]},approved=approved)
        with patch('scripts.distill_compact_a2.CompactKittiDataset.__getitem__',return_value=item):
            result=dataset[0]
        torch.testing.assert_close(result["kd"]["hps"][0],item["targets"]["hps"][0],atol=5e-5,rtol=1e-5)
        self.assertEqual(float(result["kd"]["depth_mask"].sum()),1.)

    def test_immutable_reports_reject_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"receipt.json"
            immutable_json(path,{"x":1});immutable_json(path,{"x":1})
            with self.assertRaises(RuntimeError):immutable_json(path,{"x":2})
            self.assertEqual(json.loads(path.read_text()),{"x":1})

    def test_resume_cache_identity_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'train_cache').mkdir()
            record=dict(identity_sha256='test',sample_id='000001',rows=[],source_stats={},source_sha256={})
            checksum=signature(record); record['record_sha256']=checksum
            (root/'train_cache/000001.json').write_text(json.dumps(record))
            audit=dict(pilot_authorized=True,identity_sha256='test',
                       train_cache_inventory_sha256=signature({'000001':checksum}))
            audit['audit_sha256']=signature(audit)
            (root/'teacher_audit.json').write_text(json.dumps(audit))
            m=dict(output=str(root),identity_sha256='test',train_ids=['000001'],dataset_root=str(root))
            with patch('scripts.distill_compact_a2.unchanged_source',return_value=True):
                records,_=read_audited_cache(m)
                self.assertEqual(records,{'000001':[]})
                record['rows']=[{'changed':True}]
                (root/'train_cache/000001.json').write_text(json.dumps(record))
                with self.assertRaises(RuntimeError):read_audited_cache(m)

    def test_train_both_roles_resume_and_compare(self):
        """One synthetic epoch, not a real teacher or KITTI AP result."""
        from models.compact_monocular3d import CompactMonocular3D
        from types import SimpleNamespace
        from scripts.train_compact_kitti import config
        import copy
        batch, _,_=oracle((8,32));t=batch['targets']
        item=dict(sample_id='000001',image=torch.randn(3,32,128),projection=batch['projection'][0],
                  inverse_affine=batch['inverse_affine'][0],targets={k:v[0] for k,v in t.items()})
        item['kd']=dict(hps=item['targets']['hps'],dim=item['targets']['dim'],
                        rotbin=item['targets']['rotbin'],rotres=item['targets']['rotres'],
                        depth=item['targets']['location'][...,2],
                        **{k+'_mask':item['targets']['geom_mask'] for k in COMPONENTS})
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):return 2
            def __getitem__(self,index):return copy.deepcopy(item)
        cfg=config(Path(__file__).resolve().parents[1]/'configs/compact_mnv4_kitti_gt_r1.yaml')
        cfg.update(workers=0,batch_size=1)
        metrics={k:1. for k in ('vehicle_3d_moderate','pedestrian_3d_moderate','mean_3d_moderate',
                               'vehicle_bev_moderate','pedestrian_bev_moderate')}
        template=CompactMonocular3D(backbone='mobilenetv4_conv_small')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m=dict(output=str(root),identity_sha256='test',dataset_root=str(root),split_dir=str(root),
                   train_ids=['000001'],val_ids=['000002'],config=cfg,baseline_metrics=metrics)
            a=SimpleNamespace(role='control',device='cpu')
            with patch.dict(SETTINGS,epochs=1,batch_size=1,workers=0,evaluate_every=1), \
                 patch('scripts.distill_compact_a2.read_audited_cache',return_value=({'000001':[{}]}, {'class_component_audit':{}})), \
                 patch('scripts.distill_compact_a2.usable_components',return_value={'hps':True}), \
                 patch('scripts.distill_compact_a2.DistillationDataset',return_value=Dataset()), \
                 patch('scripts.distill_compact_a2.frozen_student',side_effect=lambda *args:copy.deepcopy(template)), \
                 patch('scripts.distill_compact_a2.evaluate',return_value={'metrics':metrics}):
                for role in ('control','kd'):
                    a.role=role
                    train(a,m,smoke=True)
                    train(a,m)
                    train(a,m)  # Resume completed epoch, do not apply another update.
                    saved=torch.load(root/role/'checkpoints/latest.pt',weights_only=True)
                    self.assertEqual(len(saved['history']),1)
                    self.assertEqual(saved['external_teacher_used'],role=='kd')
                    self.assertFalse(saved['architecture_changed'])
                comparison=json.loads((root/'comparison.json').read_text())
                self.assertTrue(comparison['complete'])
                self.assertEqual(comparison['kd_minus_control']['mean_3d_moderate'],0.)

    def test_notebook_compiles_and_is_self_contained(self):
        path=Path(__file__).resolve().parents[1]/"notebooks/Compact_MobileNetV4_A2_Distillation_Pilot_Colab.ipynb"
        nb=json.loads(path.read_text());codes=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
        for code in codes:compile(code,"compact-kd-cell","exec")
        names={n.name for n in ast.parse(codes[0]).body if isinstance(n,ast.FunctionDef)}
        self.assertIn("run_logged",names);self.assertIn("stage",names)
        text="\n".join(codes)
        self.assertNotIn("reset --hard",text);self.assertNotIn("setup_m67_runtime",text)
        self.assertNotIn("MultiScaleDeformableAttention",text)
        self.assertIn("stage('train', '--role', 'control')",text)
        self.assertIn("stage('train', '--role', 'kd')",text)
        self.assertLess(text.index("stage('cache-audit')"),text.index("stage('train'"))
        for c in nb["cells"]:
            if c["cell_type"]=="code":self.assertEqual(c["outputs"],[])


if __name__=="__main__":unittest.main()
