"""Synthetic scoring, immutable resume and standalone Colab contracts."""
import ast
from collections import deque
from datetime import datetime,timezone
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
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_a2_lite_accuracy as e


def prediction():
    return dict(class_name='Vehicle',alpha=0.,bbox_2d=[100.,100.,200.,200.],dimensions_3d=[1.6,1.7,4.1],
                location_3d=[0.,.8,20.],rotation_y=0.,score=.9,invalid_geometry=False)


def outputs():
    values={name:np.zeros(shape,np.float32) for name,shape in e.OUTPUT_SHAPES.items()}
    values['logits'][:]=-20.;values['logits'][0,0,1]=4.
    values['boxes'][:]=[.5,.5,.1,.1,.1,.1]
    values['dimensions'][:]=[1.6,1.7,4.1];values['depth'][:,:,0]=20.
    return values


class AccuracyTests(unittest.TestCase):
    def test_frozen_graphs_same_checkpoint_different_resolutions(self):
        self.assertEqual(set(e.MODELS),{'full_resolution_control','compact_640'})
        self.assertEqual(e.MODELS['full_resolution_control']['hw'],[384,1280])
        self.assertEqual(e.MODELS['compact_640']['hw'],[192,640])
        with self.assertRaises(RuntimeError):e.validate_package(b'{}','compact_640')

    def test_serializer_native_fields_and_invalid_retained(self):
        good=prediction();bad={**good,'dimensions_3d':[-1.,1.,2.],'score':2.,'invalid_geometry':True}
        raw,count=e.prediction_text([bad,good]);ap,apcount=e.prediction_text([bad,good],True)
        self.assertEqual(count,1);self.assertEqual(apcount,1)
        self.assertEqual(len(raw.splitlines()),2);self.assertEqual(len(ap.splitlines()),2)
        self.assertEqual(len(ap.splitlines()[0].split()),16)
        self.assertEqual(ap.splitlines()[0].split()[8:11],['0.00','0.00','0.00'])
        self.assertIn('-1.00',raw)
        with self.assertRaises(RuntimeError):e.prediction_text([{**good,'class_name':'Cyclist'}])
        with self.assertRaises(RuntimeError):e.prediction_text([{**good,'score':float('nan')}])

    def test_decoder_native_operation_order_and_confidence_threshold(self):
        values=outputs();values['depth'][0,0,1]=2.
        projection=np.asarray([[700,0,620,44],[0,700,180,.2],[0,0,1,0]],np.float32)
        rows=e.decode_predictions(values,projection,[1242,375])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['class_name'],'Vehicle')
        corners=torch.tensor([.4,.4,.6,.6]);xywh=torch.stack(((corners[0]+corners[2])/2,
                 (corners[1]+corners[3])/2,corners[2]-corners[0],corners[3]-corners[1])).numpy()
        x,y,w,h=xywh*np.asarray([1242,375,1242,375],np.float32)
        self.assertEqual(rows[0]['bbox_2d'],[float(v) for v in (x-w/2,y-h/2,x+w/2,y+h/2)])
        self.assertAlmostEqual(rows[0]['score'],float(torch.tensor(4.).sigmoid()*torch.tensor(-2.).exp()))
        values['logits'][0,0,1]=-7.
        self.assertEqual(e.decode_predictions(values,projection,[1242,375]),[])
        values['logits'][0,0,1]=4.;values['depth'][0,0,0]=-20.
        self.assertTrue(e.decode_predictions(values,projection,[1242,375])[0]['invalid_geometry'])
        values['angle'][0,0,0]=np.nan
        with self.assertRaises(RuntimeError):e.decode_predictions(values,projection,[1242,375])

    def test_zero_volume_false_positive_counts_in_ap(self):
        good=prediction();bad={**good,'dimensions_3d':[-1.,1.,2.],'score':2.,'invalid_geometry':True}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);text,_=e.prediction_text([bad,good],True);(root/'000001.txt').write_text(text)
            raw=e.load_kitti_prediction_directory(root,['000001'],allowed_classes=['Car'])
            pred={'000001':e.map_objects(raw['000001'],e.KITTI_PRODUCTION_CLASS_MAPPING)}
            gt={'000001':[dict(class_name='Vehicle',truncated=0.,occluded=0,alpha=0.,bbox_2d=good['bbox_2d'],
                  dimensions_3d=good['dimensions_3d'],location_3d=good['location_3d'],rotation_y=0.)]}
            rows=e.evaluate_kitti_r40(gt,pred,['Vehicle'],e.PRODUCT_IOU_THRESHOLDS,e.PRODUCT_NEIGHBOR_CLASSES)
            result=next(r for r in rows if r.metric=='3d' and r.difficulty=='moderate')
            self.assertAlmostEqual(result.ap_r40,50.);self.assertEqual(result.num_false_positives,1)

    def test_nearby_is_two_dimensional_not_depth_accuracy(self):
        good=prediction();pred=dict(class_name='Vehicle',bbox_2d=good['bbox_2d'],dimensions_3d_hwl=good['dimensions_3d'],
              location_3d=[0.,.8,80.],rotation_y=0.,score=.9)
        gt=dict(class_name='Vehicle',bbox_2d=good['bbox_2d'],dimensions_3d=good['dimensions_3d'],
                location_3d=good['location_3d'],rotation_y=0.)
        report,matched,_=e.audit({'000001':[pred]},{'000001':[gt]},.001,.5)
        self.assertEqual(report['classes']['Vehicle']['near_recall'],1.)
        self.assertEqual(matched[0]['depth_abs_error_m'],60.)

    def test_data_stage_val_only_aliases_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';target=root/'kitti_a2_lite_accuracy_test'
            for name in ('image_02','calib','label_02'):
                folder=source/'training'/name;folder.mkdir(parents=True)
                for sample in ('000001','000002'):
                    suffix='.png' if name=='image_02' else '.txt';(folder/(sample+suffix)).write_bytes((name+sample).encode())
            result=e.stage_data(target,[source],['000001','000002'])
            self.assertEqual(set(result),{'000001','000002'})
            self.assertFalse((target/'ImageSets/train.txt').exists())
            self.assertEqual((source/'training/image_02/000001.png').read_bytes(),b'image_02000001')
            self.assertEqual((target/'training/image_2/000001.png').read_bytes(),b'image_02000001')

    def test_missing_data_stops_before_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'kitti_a2_lite_accuracy_test'
            with self.assertRaises(RuntimeError):e.stage_data(target,[root/'missing'],['000001'])
            self.assertFalse(target.exists())

    def test_immutable_files_preserve_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'prediction.txt';e.immutable_bytes(path,b'original');e.immutable_bytes(path,b'original')
            with self.assertRaises(RuntimeError):e.immutable_bytes(path,b'changed')
            self.assertEqual(path.read_bytes(),b'original')
            path=Path(d)/'record.json';e.immutable_json(path,dict(complete=True))
            with self.assertRaises(RuntimeError):e.immutable_json(path,dict(complete=False))

    def test_signed_cache_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cache.json';identity=dict(run_signature='exact',sample_id='000001',input_sha256={'image':'x'})
            record=dict(identity=identity,predictions={'compact_640':{'raw_text':'text'}})
            record['record_sha256']=e.signature(record);e.write_json(path,record)
            self.assertEqual(e.cache_read(path,identity)['predictions'],record['predictions'])
            with self.assertRaises(RuntimeError):e.cache_read(path,{**identity,'run_signature':'different'})
            record['predictions']['compact_640']['raw_text']='edited';path.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError):e.cache_read(path,identity)
            path.write_text('partial-json')
            with self.assertRaises(json.JSONDecodeError):e.cache_read(path,identity)

    def test_partial_split_cannot_be_reported_as_full(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'val.txt';path.write_text('000001\n')
            with self.assertRaises(RuntimeError):e.q.split_ids(path,'val')

    def test_paired_inference_resume_restores_text_without_rerunning_models(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'dataset';output=root/'output'
            for name,suffix in (('image_2','.png'),('calib','.txt'),('label_2','.txt')):
                p=dataset/'training'/name/('000001'+suffix);p.parent.mkdir(parents=True);p.write_bytes(name.encode())
            hashes={name:e.sha256(dataset/'training'/name/('000001'+('.png' if name=='image_2' else '.txt')))
                    for name in ('image_2','calib','label_2')}
            models={}
            for role in e.MODELS:
                p=root/role;p.mkdir();(p/'manifest.json').write_text('{}');models[role]=dict(root=str(p))
            m=dict(dataset=str(dataset),sample_ids=['000001'],data_file_sha256={'000001':hashes},
                   models=models,signature_sha256='synthetic-test')
            class FakeSession:
                calls=0
                def run(self,names,inputs):
                    self.calls+=1;v=outputs();return [v[n] for n in names]
            sessions={role:FakeSession() for role in e.MODELS}
            image={n:np.zeros(shape,np.float32) for n,shape in e.INPUT_SHAPES.items()}
            projection=np.asarray([[700,0,620,0],[0,700,180,0],[0,0,1,0]],np.float32)
            image['image_size'][:]=[1242,375]
            with patch.object(e,'load',return_value=m),patch.object(e,'session',side_effect=lambda root,p,role:sessions[role]),\
                 patch.object(e,'teacher_inputs',return_value=image),patch.object(e,'parse_kitti_calib_file',return_value={'P2':projection}):
                e.infer(SimpleNamespace(output=output))
                missing=output/'predictions/compact_640/raw/000001.txt';original=missing.read_bytes();missing.unlink()
                e.infer(SimpleNamespace(output=output))
                self.assertEqual(missing.read_bytes(),original)
                self.assertEqual([s.calls for s in sessions.values()],[1,1])
                done=json.loads((output/'prediction_complete.json').read_text());self.assertEqual(done['images'],1)
                with self.assertRaises(RuntimeError):e.evaluate(SimpleNamespace(output=output))
                (dataset/'training/label_2/000001.txt').write_text('changed')
                with self.assertRaises(RuntimeError):e.infer(SimpleNamespace(output=output))

    def test_bundle_excludes_models_and_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();(root/'assets/manifest.json').write_text('{}')
            (root/'model.onnx').write_bytes(b'model');(root/'frame.png').write_bytes(b'image')
            (root/'a2_lite_accuracy_comparison.json').write_text('{}')
            e.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_accuracy_results.zip') as z:
                self.assertEqual(z.namelist(),['a2_lite_accuracy_comparison.json'])

    def test_failed_accuracy_gates_saved_without_execution_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'dataset';output=root/'output';sample='000001'
            label=dataset/'training/label_2'/f'{sample}.txt';label.parent.mkdir(parents=True)
            label.write_text('Car 0 0 0 100 100 200 200 1.6 1.7 4.1 0 .8 20 0\n'
                             'Pedestrian 0 0 0 300 100 350 200 1.7 .6 .8 5 .85 20 0\n')
            hashes={'label_2':e.sha256(label)};text,_=e.prediction_text([prediction()])
            models={role:dict(root='not used by synthetic scoring') for role in e.MODELS}
            m=dict(dataset=str(dataset),sample_ids=[sample],data_file_sha256={sample:hashes},
                   models=models,signature_sha256='synthetic-test')
            record=dict(identity=dict(run_signature=m['signature_sha256'],sample_id=sample,input_sha256=hashes),
                predictions={role:dict(raw_text=text,ap_text=text,invalid_geometry_count=0) for role in models})
            record['record_sha256']=e.signature(record);e.write_json(output/'cache'/f'{sample}.json',record)
            files={}
            for role in models:
                files[role]={}
                for kind in ('raw','ap'):
                    path=output/'predictions'/role/kind/f'{sample}.txt';e.immutable_bytes(path,text.encode())
                    files[role][kind]={sample:e.sha256(path)}
            e.write_json(output/'prediction_complete.json',dict(complete=True,run_signature=m['signature_sha256'],images=1,prediction_files=files))
            with patch.object(e,'load',return_value=m),patch.object(e,'VAL_COUNT',1):
                e.evaluate(SimpleNamespace(output=output))
            report=json.loads((output/'a2_lite_accuracy_comparison.json').read_text())
            self.assertEqual(report['evaluated_images'],1)
            self.assertTrue(report['complete']);self.assertFalse(report['deployment_qualified'])
            self.assertFalse(report['training_performed']);self.assertFalse(report['distillation_authorized'])
            for result in report['models'].values():
                self.assertFalse(result['all_accuracy_gates_passed'])
                self.assertEqual(result['metrics']['pedestrian_3d_moderate'],0.)
            label.write_text('changed')
            with patch.object(e,'load',return_value=m),patch.object(e,'VAL_COUNT',1):
                with self.assertRaises(RuntimeError):e.evaluate(SimpleNamespace(output=output))

    def test_notebook_standalone_cpu_only_and_visible_revision(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_640_Accuracy_Colab.ipynb').read_text())
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];self.assertEqual(len(codes),5)
        for i,c in enumerate(codes):compile(c,f'accuracy-cell-{i}','exec')
        names={n.name for n in ast.parse(codes[0]).body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'run_logged','stage'}<=names)
        text='\n'.join(codes)
        self.assertIn(e.REVISION,text);self.assertIn('onnxruntime==1.30.0',text)
        self.assertLess(text.index("'pull','--ff-only'"),text.index('SCRIPT.is_file()'))
        self.assertIn("stage('infer')",text);self.assertIn("stage('evaluate')",text)
        for forbidden in ('torch.cuda','coremltools','reset --hard','setup_m68','build_m64'):
            self.assertNotIn(forbidden,text)
        for c in nb['cells']:
            if c['cell_type']=='code':self.assertEqual(c['outputs'],[])

    def test_notebook_captures_both_failure_streams(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_640_Accuracy_Colab.ipynb').read_text())
        code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        functions=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
        with tempfile.TemporaryDirectory() as d:
            namespace=dict(Path=Path,datetime=datetime,timezone=timezone,deque=deque,subprocess=subprocess,
                           sys=sys,shlex=shlex,LOGS=Path(d))
            exec(compile(ast.Module(body=functions,type_ignores=[]),'helpers','exec'),namespace)
            with self.assertRaises(RuntimeError):namespace['run_logged']([sys.executable,'-c',"import sys; print('out'); print('err',file=sys.stderr); sys.exit(3)"],'failed')
            text=next(Path(d).glob('*.log')).read_text()
            for marker in ('out','err','EXIT CODE: 3'):self.assertIn(marker,text)


if __name__=='__main__':unittest.main()
