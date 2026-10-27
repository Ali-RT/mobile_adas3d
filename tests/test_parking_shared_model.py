"""Shared-feature execution, frozen BN, raw-metre loss and runtime contracts."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from models.parking_shared import SharedParkingNet, ParkingDepthDecoder, metric_depth_loss
from scripts import screen_parking_shared_model as screen


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__();self.f=[16,19,22];self.i=23;self.nc=80;self.kpt_shape=[8,3];self.end2end=True
        self.projections=nn.ModuleList(nn.Conv2d(c,108,1) for c in [64,128,256])
    @property
    def one2one(self):return {}
    def forward_head(self,features):
        return {'dense':torch.cat([p(x).flatten(2) for p,x in zip(self.projections,features)],dim=2)}
    def _inference(self,raw):return raw['dense']


def fake_urban():
    modules=[]
    convolutions={0:(3,16,2),3:(16,32,2),5:(32,64,2),7:(64,128,2),16:(128,64,1),
        17:(64,128,2),20:(128,256,2)}
    for index in range(23):
        if index in convolutions:
            cin,cout,stride=convolutions[index]
            layer=nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1),nn.BatchNorm2d(cout),nn.SiLU())
        else:layer=nn.Upsample(scale_factor=2,mode='nearest') if index==15 else nn.Identity()
        layer.i=index;layer.f=-1;modules.append(layer)
    modules.append(FakeHead())
    urban=nn.Module();urban.model=nn.Sequential(*modules);urban.save=[4,6,10,13,16,19,22]
    return urban


class ParkingSharedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_one_backbone_pass_and_shared_features(self):
        model=SharedParkingNet(fake_urban(),width=16).eval()
        calls=[]
        handle=model.body[0].register_forward_hook(lambda *args:calls.append(1))
        image=torch.randn(1,3,64,96)
        candidates,depth=model(image);handle.remove()
        self.assertEqual(len(calls),1)
        self.assertEqual(candidates.shape,(1,32,126))
        self.assertEqual(depth.shape,(1,1,16,24))
        self.assertTrue(torch.isfinite(depth).all() and (depth>0).all())

    def test_product_channels_are_exact_slices_not_relearned_predictions(self):
        model=SharedParkingNet(fake_urban(),width=16).eval();image=torch.randn(1,3,64,96)
        full=model.full_urban_candidates(model.image_features(image))
        selected,_=model(image)
        indices=[0,1,2,3,4,6,9,11]+list(range(84,108))
        torch.testing.assert_close(selected,full[:,indices],rtol=0,atol=0)

    def test_frozen_body_weights_and_bn_remain_unchanged_during_depth_gradient_check(self):
        model=SharedParkingNet(fake_urban(),width=16)
        before=copy.deepcopy(model.body.state_dict());model.train()
        self.assertFalse(model.body.training)
        self.assertTrue(model.depth_decoder.training)
        _,depth=model(torch.randn(2,3,64,96))
        metric_depth_loss(depth,torch.full_like(depth,3.),torch.ones_like(depth,dtype=torch.bool)).backward()
        self.assertTrue(all(p.grad is None for p in model.body.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.depth_decoder.parameters()))
        self.assertTrue(any(torch.count_nonzero(p.grad) for p in model.depth_decoder.parameters()))
        for name,value in model.body.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_metric_loss_does_not_fit_scale_and_ignores_unlabelled_holes(self):
        prediction=torch.tensor([2.,4.,99.],requires_grad=True)
        truth=torch.tensor([2.,2.,float('nan')]);valid=torch.tensor([True,True,False])
        loss=metric_depth_loss(prediction,truth,valid)
        self.assertAlmostEqual(float(loss.detach()),.19,places=6)
        loss.backward();self.assertEqual(prediction.grad[-1],0)
        no_labels=metric_depth_loss(prediction,truth,torch.zeros(3,dtype=torch.bool))
        self.assertEqual(no_labels,0)

    def test_invalid_depth_targets_and_weights_rejected(self):
        pred=torch.ones(2);truth=torch.ones(2);valid=torch.ones(2,dtype=torch.bool)
        for target in [torch.tensor([0.,1.]),torch.tensor([float('nan'),1.])]:
            with self.assertRaises(ValueError):metric_depth_loss(pred,target,valid)
        with self.assertRaises(ValueError):metric_depth_loss(pred,truth,valid,weights=torch.ones(3))
        with self.assertRaises(ValueError):metric_depth_loss(pred,truth,valid,weights=torch.zeros(2))
        with self.assertRaises(ValueError):metric_depth_loss(pred,truth,valid.float())

    def test_incompatible_upstream_contract_rejected(self):
        urban=fake_urban();urban.model[-1].nc=2
        with self.assertRaises(ValueError):SharedParkingNet(urban)
        with self.assertRaises(ValueError):ParkingDepthDecoder(width=7)

    def test_letterbox_affine_and_intrinsics(self):
        rgb=np.zeros((720,1280,3),np.uint8);rgb[:,:,0]=255
        k=np.array([[1000,0,640],[0,1000,360],[0,0,1.]])
        value,metadata=screen.preprocess(rgb,intrinsics=k)
        self.assertEqual(value.shape,(1,3,384,640));self.assertEqual(value.dtype,np.float32)
        self.assertEqual(metadata['padding_left_top'],[0,12])
        np.testing.assert_allclose(metadata['input_intrinsics'],[[500,0,320],[0,500,192],[0,0,1]])
        self.assertEqual(value[0,0,20,10],1.);self.assertEqual(value[0,2,20,10],0.)
        self.assertAlmostEqual(float(value[0,0,0,0]),114/255.,places=6)

    def test_decoder_restores_original_image_and_never_displays_untrained_depth(self):
        _,metadata=screen.preprocess(np.zeros((720,1280,3),np.uint8))
        raw=np.zeros((1,32,1),np.float32);raw[0,:4,0]=[100,100,200,200];raw[0,5,0]=.9
        rows=screen.decode_candidates(raw,metadata)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['class_name'],'Vehicle')
        np.testing.assert_allclose(rows[0]['bbox'],[200,176,400,376])
        self.assertIsNone(rows[0]['distance_m']);self.assertTrue(rows[0]['depth_head_untrained'])
        with self.assertRaises(ValueError):screen.decode_candidates(np.zeros((1,30,1),np.float32),metadata)

    def test_strict_parity_rejects_wrong_shapes_or_large_errors(self):
        reference=np.ones(3,np.float32)
        self.assertTrue(screen.compare(reference,reference.copy(),.001,.001)['passed'])
        with self.assertRaises(RuntimeError):screen.compare(reference,reference*2,.001,.001)
        with self.assertRaises(ValueError):screen.compare(reference,np.ones(4,np.float32),.001,.001)

    def test_bundle_is_reproducible_and_keeps_diagnostics_separate(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);(output/'shared_model_screen.json').write_text('{}')
            screen.publish_bundle(output)
            before=(output/'parking_shared_runtime_results.zip').read_bytes()
            (output/'errors').mkdir();(output/'errors/failure.json').write_text('{"diagnostic":true}')
            screen.publish_bundle(output)
            self.assertEqual(before,(output/'parking_shared_runtime_results.zip').read_bytes())

    def test_config_is_runtime_only_and_protects_evaluation_data(self):
        config=json.loads(screen.CONFIG.read_text())
        self.assertFalse(config['depth_head_usable_for_measurement'])
        self.assertFalse(config['training_performed'])
        self.assertFalse(config['held_out_evaluation_set_used_for_training'])
        self.assertEqual(config['input_shape'],[1,3,384,640])

    def test_notebook_is_clean_cpu_only_and_matches_source_revision(self):
        path=screen.ROOT/'notebooks/Parking_Shared_Urban_Depth_Runtime_Colab.ipynb'
        notebook=json.loads(path.read_text())
        sources=[]
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']);ast.parse(source);sources.append(source)
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        combined='\n'.join(sources)
        self.assertEqual(len(sources),4)
        self.assertIn('PARKING-SHARED-NANO-2026-10-27-r1',combined)
        self.assertIn("scripts/screen_parking_shared_model.py",combined)
        self.assertIn("ultralytics==8.4.61",combined)
        self.assertNotIn('nvidia-smi',combined)
        self.assertNotIn('reset\',\'--hard',combined)
        self.assertNotIn('m67_',combined)


if __name__=='__main__':unittest.main()
