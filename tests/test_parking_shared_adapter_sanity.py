"""Adapter-only gradients, frozen original BN, and train-only subset guards."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from models.parking_shared import ParkingDepthDecoder
from models.parking_shared_camera import CameraAwareDepthDecoder
from scripts import sanity_parking_shared_adapter as sanity
from scripts.screen_parking_shared_model import state_digest


def recipe():
    return {**json.loads(sanity.camera.data.CONFIG.read_text()),**sanity.configuration()}


def sample(i,role='train'):
    return {'camera_token':f'{role}-{i}','role':role,'width':1600,'height':900,
        'intrinsics':[[1200.,0.,800.],[0.,1200.,450.],[0.,0.,1.]],
        'counts':{k:5 for k in recipe()['critical_cells']}}


def row():
    rng=np.random.default_rng(30)
    return {**{f'p{i+3}':rng.normal(size=shape).astype(np.float32) for i,shape in enumerate([(64,8,16),(128,4,8),(256,2,4)])},
        'z':np.repeat([3.,5.,3.,5.],5).astype(np.float32),'semantic':np.repeat([1,1,2,2],5).astype(np.uint8),
        'conservative':np.ones(20,bool),'grid':rng.uniform(-1,1,(20,2)).astype(np.float32)}


class AdapterSanityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_deterministic_train_only_unique_balanced_selection(self):
        rows=[sample(i) for i in range(24)]+[sample(i,'dev') for i in range(100)]+[sample(i,'heldout_eval') for i in range(100)]
        a,assign=sanity.select_training_samples(rows,recipe(),recipe()['critical_cells'])
        b,other=sanity.select_training_samples(rows[::-1],recipe(),recipe()['critical_cells'])
        self.assertEqual(a,b);self.assertEqual(assign,other)
        self.assertEqual(len(a),16);self.assertEqual(len({s['camera_token'] for s in a}),16)
        self.assertEqual({s['role'] for s in a},{'train'})
        for cell in recipe()['critical_cells']:self.assertEqual(assign.count(cell),4)

    def test_missing_pool_fails_instead_of_using_dev(self):
        rows=[sample(i) for i in range(24)]
        for s in rows:s['counts'][recipe()['critical_cells'][0]]=0
        with self.assertRaises(ValueError):
            sanity.select_training_samples(rows+[sample(i,'dev') for i in range(100)],recipe(),recipe()['critical_cells'])

    def test_fixed_two_batch_cycles_shared_by_arms(self):
        stream=sanity.batch_indices(16,recipe())
        self.assertEqual(len(stream),300)
        self.assertEqual(stream,sanity.batch_indices(16,recipe()))
        for i in range(0,300,2):self.assertEqual(sorted(stream[i]+stream[i+1]),list(range(16)))
        with self.assertRaises(ValueError):sanity.batch_indices(17,recipe())

    def test_only_adapter_parameters_enter_optimizer(self):
        model=CameraAwareDepthDecoder(ParkingDepthDecoder())
        params=sanity.freeze_base(model)
        self.assertEqual(sum(p.numel() for p in params),1553)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        ids={id(p) for p in params}
        self.assertFalse(ids & {id(p) for p in model.base.parameters()})

    def test_adapter_updates_but_base_weights_and_all_bn_buffers_stay_exact(self):
        model=CameraAwareDepthDecoder(ParkingDepthDecoder(width=16))
        params=sanity.freeze_base(model);frozen=state_digest(model.base)
        before=copy.deepcopy(model.state_dict())
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0.)
        probes=[]
        for _ in range(3):
            probes.append(sanity.adapter_step(model,optimizer,[row() for _ in range(8)],
                [sample(i) for i in range(8)],'actual_camera','cpu',{'Vehicle':[1],'Pedestrian':[2]},recipe(),frozen))
        self.assertEqual(state_digest(model.base),frozen)
        for k,v in before.items():
            if k.startswith('base.'):self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        self.assertFalse(torch.equal(before['residual.weight'],model.state_dict()['residual.weight']))
        self.assertGreater(probes[0]['gradient_l2']['residual'],0)
        self.assertEqual(probes[0]['gradient_l2']['appearance'],0)
        self.assertGreater(probes[1]['gradient_l2']['appearance'],0)

    def test_unfrozen_bn_or_parameter_is_rejected(self):
        model=CameraAwareDepthDecoder(ParkingDepthDecoder(width=16))
        sanity.freeze_base(model);frozen=state_digest(model.base)
        model.base.train()
        with self.assertRaises(ValueError):sanity.frozen_check(model,frozen)
        model.base.eval();next(model.base.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):sanity.frozen_check(model,frozen)

    def test_initial_metrics_and_prediction_change(self):
        model=CameraAwareDepthDecoder(ParkingDepthDecoder(width=16)).eval()
        sanity.freeze_base(model)
        rows=[row() for _ in range(2)];samples=[sample(i) for i in range(2)]
        with torch.no_grad():baseline=[sanity.predict(model,[r],[s],'actual_camera','cpu')[0][0].numpy() for r,s in zip(rows,samples)]
        metrics,_=sanity.tiny_metrics(model,rows,samples,'actual_camera','cpu',{'Vehicle':[1],'Pedestrian':[2]},recipe(),baseline)
        self.assertEqual(metrics['equal_four_cell_prediction_change_m'],0.)
        self.assertGreater(metrics['loss'],0)
        self.assertEqual(len(metrics['cells']),4)
        with torch.no_grad():model.residual.bias.fill_(.1)
        changed,_=sanity.tiny_metrics(model,rows,samples,'actual_camera','cpu',{'Vehicle':[1],'Pedestrian':[2]},recipe(),baseline)
        self.assertGreater(changed['equal_four_cell_prediction_change_m'],.05)


if __name__=='__main__':unittest.main()
