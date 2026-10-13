"""Train-only selection, object-aligned KD, finite losses and resume contracts."""
import ast
from collections import deque
import copy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import recover_a2_lite_640 as r
from models import a2_lite_recovery_loss as k


def target():
    return dict(labels=torch.tensor([0,1]),
        boxes_3d=torch.tensor([[.3,.5,.03,.03,.15,.15],[.7,.5,.08,.08,.07,.07]]),
        size_3d=torch.tensor([[1.7,.6,.8],[1.5,1.7,4.1]]),depth=torch.tensor([[15.],[30.]]),
        heading_bin=torch.tensor([[0],[6]]),heading_res=torch.tensor([[.1],[.1]]))


def outputs(queries=(0,1),perfect=False):
    t=target();v={key:torch.zeros(shape) for key,shape in
        [('pred_logits',(1,50,3)),('pred_boxes',(1,50,6)),('pred_3d_dim',(1,50,3)),
         ('pred_depth',(1,50,2)),('pred_angle',(1,50,24))]}
    v['pred_logits'].fill_(-10);v['pred_boxes'][:]=torch.tensor([.95,.95,.01,.01,.01,.01])
    v['pred_3d_dim'].fill_(1);v['pred_depth'][:,:,0]=20.
    for j,query in enumerate(queries):
        label=int(t['labels'][j]);v['pred_logits'][0,query,label]=10 if perfect else 1
        v['pred_boxes'][0,query]=t['boxes_3d'][j]+(0 if perfect else .02)
        v['pred_3d_dim'][0,query]=t['size_3d'][j]+(0 if perfect else .2)
        v['pred_depth'][0,query,0]=t['depth'][j,0]+(0 if perfect else 5)
        anglebin=int(t['heading_bin'][j,0]) if perfect else 3
        v['pred_angle'][0,query,:12]=-10;v['pred_angle'][0,query,anglebin]=10
        v['pred_angle'][0,query,12+anglebin]=.1
    return v


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name,value in outputs().items():self.register_parameter(name,torch.nn.Parameter(value.clone()))
    def outputs(self):return {name:p for name,p in self.named_parameters()}


class RecoveryTests(unittest.TestCase):
    def test_exact_architecture_and_bounded_budget(self):
        self.assertEqual(r.RECIPE['input_hw'],[192,640]);self.assertEqual(r.RECIPE['queries'],50)
        self.assertEqual(r.RECIPE['epochs'],5);self.assertEqual(r.RECIPE['train_images'],512)
        self.assertEqual(r.RECIPE['temperature'],2.);self.assertFalse(r.RECIPE['architecture_changed'])
        self.assertFalse(r.RECIPE['validation_selection']);self.assertFalse(r.RECIPE['training_query_groups'])
        self.assertEqual(r.RECIPE['gt_weights'],k.GT_WEIGHTS)

    def test_perfect_gt_geometry_is_finite_zero_not_dimension_zero_over_zero(self):
        v={name:value.requires_grad_() for name,value in outputs(perfect=True).items()}
        total,parts,metrics=k.gt_loss(v,target())
        self.assertTrue(torch.isfinite(total));total.backward()
        for name in ('loss_center','loss_bbox','loss_giou','loss_depth','loss_dim'):
            self.assertLess(abs(float(parts[name].detach())),1e-5)
        self.assertLess(float(parts['loss_angle'].detach()),1e-5)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in v.values()))
        for name in ('boxes','dimensions','depth','angle'):self.assertLess(metrics[name],1e-6)

    def test_empty_targets_and_wrong_contracts(self):
        empty={name:v[:0] for name,v in target().items()}
        loss,_,_=k.gt_loss(outputs(),empty);self.assertTrue(torch.isfinite(loss))
        bad=target();bad['labels'][0]=2
        with self.assertRaises(ValueError):k.gt_loss(outputs(),bad)
        bad=target();bad['size_3d'][0,0]=0
        with self.assertRaises(ValueError):k.gt_loss(outputs(),bad)
        wrong=outputs();wrong['pred_logits']=wrong['pred_logits'].double()
        with self.assertRaises(ValueError):k.gt_loss(wrong,target())

    def test_hungarian_object_alignment_not_query_numbers(self):
        teacher=outputs((42,47),True);mapping=k.teacher_matches(teacher,target())
        self.assertEqual(mapping.tolist(),[42,47])
        student={name:value.requires_grad_() for name,value in outputs().items()}
        loss,parts,counts=k.kd_loss(student,teacher,target(),mapping)
        self.assertGreater(float(loss.detach()),0.);loss.backward()
        for name,value in student.items():
            self.assertIsNotNone(value.grad);self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(counts['depth']['Pedestrian'],0);self.assertGreater(counts['angle']['Vehicle'],0)

    def test_teacher_is_detached_and_worse_components_are_masked(self):
        teacher={name:value.requires_grad_() for name,value in outputs((42,47),True).items()}
        teacher['pred_depth'].data[0,42,0]=100.
        student={name:value.requires_grad_() for name,value in outputs().items()}
        mapping=k.teacher_matches(teacher,target());loss,_,counts=k.kd_loss(student,teacher,target(),mapping)
        loss.backward()
        self.assertEqual(counts['depth']['Pedestrian'],0)
        self.assertEqual(counts['depth']['Vehicle'],1)
        self.assertTrue(all(v.grad is None for v in teacher.values()))

    def test_teacher_match_rejects_wrong_class_low_confidence_and_geometry(self):
        teacher=outputs((42,47),True);teacher['pred_logits'][0,42]=torch.tensor([-10.,10.,-10.])
        self.assertEqual(k.teacher_matches(teacher,target())[0].item(),-1)
        teacher=outputs((42,47),True);teacher['pred_logits'][0,42]=torch.tensor([-4.,-10.,-10.])
        self.assertEqual(k.teacher_matches(teacher,target())[0].item(),-1)
        teacher=outputs((42,47),True);teacher['pred_depth'][0,42,0]=-1
        self.assertEqual(k.teacher_matches(teacher,target())[0].item(),-1)

    def test_no_matches_produce_zero_differentiable_kd(self):
        student={name:value.requires_grad_() for name,value in outputs().items()}
        loss,_,counts=k.kd_loss(student,outputs(perfect=True),target(),torch.tensor([-1,-1]))
        self.assertEqual(float(loss.detach()),0.);loss.backward()
        self.assertEqual(float(student['pred_logits'].grad.abs().sum()),0.)
        self.assertTrue(all(sum(v.values())==0 for v in counts.values()))

    def test_class_balanced_kd_means_not_object_frequency(self):
        value=torch.tensor([1.,1.,1.,9.]);labels=torch.tensor([1,1,1,0]);keep=torch.ones(4,dtype=torch.bool)
        result=k.class_mean(value,labels,keep,torch.tensor(0.))
        self.assertEqual(float(result),5.)

    def test_train_only_selection_deterministic_disjoint_and_not_equal_objects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=[]
            for index in range(600):
                sample=f'{index:06d}';ids.append(sample)
                car='Car 0 0 0 100 100 200 200 1.5 1.7 4.1 0 1.5 20 0\n'
                text=car+(('Pedestrian 0 0 0 300 100 340 200 1.7 .6 .8 3 1.7 15 0\n') if index<300 else car)
                (root/f'{sample}.txt').write_text(text)
            selected=r.choose_ids(ids,root);self.assertEqual(selected,r.choose_ids(ids,root))
            self.assertEqual(len(selected),len(set(selected)));self.assertEqual(len(selected),512)
            self.assertTrue(set(selected)<=set(ids));self.assertNotIn('999999',selected)
            for i in range(0,len(selected),2):
                self.assertLess(int(selected[i]),300);self.assertGreaterEqual(int(selected[i+1]),300)

    def test_portable_import_block_without_loading_native_extension(self):
        finder=r.PortableOnly()
        with self.assertRaises(ModuleNotFoundError):finder.find_spec('MultiScaleDeformableAttention')
        self.assertIsNone(finder.find_spec('ordinary_module'))

    def test_resume_model_optimizer_cursor_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=dict(signature_sha256='synthetic');dataset=SimpleNamespace(idx_list=['000001','000002'])
            class Dataset:
                idx_list=dataset.idx_list
                def __len__(self):return 2
            full=ToyStudent();resumed=ToyStudent();a=r.optimizer(full);b=r.optimizer(resumed)
            with patch.object(r,'item',side_effect=lambda *args:({},target(),{})),patch.object(r,'verify_files'),\
                 patch.object(r,'forward',side_effect=lambda model,inputs:model.outputs()):
                r.run_steps(m,full,a,Dataset(),4,root/'full.pth','control')
                r.run_steps(m,resumed,b,Dataset(),2,root/'resume.pth','control')
                state=torch.load(root/'resume.pth',weights_only=True);self.assertEqual(state['step'],2)
                self.assertIn('active_kd_counts',state)
                fresh=ToyStudent();r.run_steps(m,fresh,r.optimizer(fresh),Dataset(),4,root/'resume.pth','control')
            for name,value in full.state_dict().items():self.assertTrue(torch.equal(value,fresh.state_dict()[name]))
            self.assertEqual(torch.load(root/'resume.pth',weights_only=True)['step'],4)

    def test_cache_tamper_and_validation_target_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=dict(signature_sha256='same',pilot_ids=['000001'],data_file_sha256={'000001':{'image_2':'original'}})
            row=dict(run_signature='same',sample_id='000001',source_sha256=m['data_file_sha256']['000001'])
            row['record_sha256']=r.signature(row);r.write_json(root/'000001.json',row)
            self.assertEqual(r.cache_record(m,root,'000001')['sample_id'],'000001')
            with self.assertRaises(RuntimeError):r.cache_record(m,root,'000002')
            row['source_sha256']['image_2']='changed';(root/'000001.json').write_text(json.dumps(row))
            with self.assertRaises(RuntimeError):r.cache_record(m,root,'000001')

    def test_bundle_excludes_checkpoints_models_images_and_export_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();(root/'assets/manifest.json').write_text('{}')
            for name in ('init.pth','model.onnx','frame.png'):(root/name).write_bytes(b'large')
            (root/'tiny_fit.json').write_text('{}');r.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_recovery_results.zip') as z:self.assertEqual(z.namelist(),['tiny_fit.json'])

    def test_changed_quality_gate_cannot_silently_authorize_training(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tiny_fit.json';m=dict(signature_sha256='test')
            r.save_report(path,dict(run_signature='test',complete=True,passed=False))
            self.assertFalse(r.checked_report(path,m)['passed'])
            row=json.loads(path.read_text());row['passed']=True;path.write_text(json.dumps(row))
            with self.assertRaises(RuntimeError):r.checked_report(path,m)

    def test_both_arms_restart_same_weights_and_completed_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=dict(signature_sha256='synthetic',output=str(root),pilot_ids=['000001','000002'],
                               data_file_sha256={sample:{'image_2':'test'} for sample in ('000001','000002')})
            class Dataset:
                idx_list=m['pilot_ids']
                def __len__(self):return len(self.idx_list)
            teacher=outputs((42,47),True)
            raw={name:teacher[key].tolist() for name,key in zip(r.OUTPUT_NAMES,r.OUTPUT_KEYS)}
            cache={}
            for sample in m['pilot_ids']:
                row=dict(run_signature='synthetic',sample_id=sample,source_sha256=m['data_file_sha256'][sample],
                    target_signature=r.target_signature(target()),outputs=raw,object_to_teacher_query=[42,47],target_labels=[0,1])
                row['record_sha256']=r.signature(row);path=root/'teacher_cache'/f'{sample}.json';r.write_json(path,row)
                cache[sample]=r.sha256(path)
            r.save_report(root/'teacher_cache_complete.json',dict(complete=True,run_signature='synthetic',kd_pilot_authorized=True,cache_sha256=cache))
            starts=[]
            def build(*args):
                model=ToyStudent();starts.append({k:v.clone() for k,v in model.state_dict().items()});return model
            with patch.object(r,'tiny_gate'),patch.object(r,'build_model',side_effect=build),\
                 patch.object(r,'native_dataset',return_value=Dataset()),patch.object(r,'verify_files'),\
                 patch.object(r,'item',side_effect=lambda *args:({},target(),{})),\
                 patch.object(r,'forward',side_effect=lambda model,inputs:model.outputs()),\
                 patch.dict(r.RECIPE,epochs=2,kd_warmup_steps=1):
                for role in ('control','kd'):
                    a=SimpleNamespace(output=root,device='cpu',role=role)
                    r.train(a,m);before=(root/role/'training_complete.json').read_bytes();r.train(a,m)
                    self.assertEqual(before,(root/role/'training_complete.json').read_bytes())
            for value in starts[1:]:
                for name,v in starts[0].items():self.assertTrue(torch.equal(v,value[name]))
            gt=r.checked_report(root/'control/training_complete.json',m)
            kd=r.checked_report(root/'kd/training_complete.json',m)
            self.assertEqual(gt['optimizer_steps'],kd['optimizer_steps']);self.assertEqual(gt['optimizer_steps'],4)
            self.assertFalse(gt['external_teacher_used']);self.assertTrue(kd['external_teacher_used'])
            self.assertFalse(kd['tiny_checkpoint_used']);self.assertGreater(kd['active_kd_counts']['angle']['Pedestrian'],0)

    def test_notebook_complete_cpu_teacher_gpu_student_no_old_runtime(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_640_Recovery_KD_Colab.ipynb').read_text())
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];self.assertEqual(len(codes),8)
        for i,code in enumerate(codes):compile(code,f'recovery-cell-{i}','exec')
        text='\n'.join(codes);self.assertIn(r.REVISION,text)
        for stage in ('smoke','tiny','cache','train','evaluate','compare','bundle'):self.assertIn(f"stage('{stage}'",text)
        self.assertLess(text.index("'pull','--ff-only'"),text.index('SCRIPT.is_file()'))
        for forbidden in ('setup.py','setup_m67','venv','reset --hard','coremltools'):self.assertNotIn(forbidden,text)
        self.assertIn("'timm':'1.0.27'",text);self.assertIn("saved[name]",text)
        self.assertIn("if TINY_PASSED:",text);self.assertIn("if KD_ALLOWED:",text)
        self.assertIn("'--initialization',INITIALIZATION",text)
        functions={n.name for n in ast.parse(codes[0]).body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'stage','run_logged'}<=functions)
        for c in nb['cells']:
            if c['cell_type']=='code':self.assertEqual(c['outputs'],[])

    def test_notebook_failure_log_keeps_both_streams(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_640_Recovery_KD_Colab.ipynb').read_text())
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        funcs=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
        with tempfile.TemporaryDirectory() as d:
            ns=dict(Path=Path,datetime=datetime,timezone=timezone,deque=deque,shlex=shlex,subprocess=subprocess,LOGS=Path(d))
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'helpers','exec'),ns)
            with self.assertRaises(RuntimeError):ns['run_logged']([sys.executable,'-c',"import sys;print('out');print('err',file=sys.stderr);sys.exit(3)"],'failure')
            text=next(Path(d).glob('*.log')).read_text()
            for marker in ('out','err','EXIT CODE: 3'):self.assertIn(marker,text)


if __name__=='__main__':unittest.main()
