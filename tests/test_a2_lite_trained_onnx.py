"""Inference-only trained export selection, immutable provenance and gates."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trained_export',ROOT/'scripts/export_a2_lite_trained_onnx.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class TrainedExportTests(unittest.TestCase):
    def metric(self):
        return dict(metrics=dict(vehicle_3d_moderate=3.959, pedestrian_3d_moderate=2.141,
            mean_3d_moderate=3.050, vehicle_bev_moderate=8.085, pedestrian_bev_moderate=2.936),
            nearby=dict(classes={c:dict(near_recall=v) for c,v in [('Vehicle',.834),('Pedestrian',.560)]}))

    def records(self):
        m=dict(signature_sha256=r.RUN_SIGNATURE,train_ids=['000001'],val_ids=[f'{i:06}' for i in range(1000,4769)])
        t=dict(run_signature=r.RUN_SIGNATURE,complete=True,role='control',external_teacher_used=False,
            train_images=3712,fixed_endpoint_epoch=10,optimizer_steps=37120,tiny_checkpoint_used=False,
            checkpoint_sha256=r.CHECKPOINT_SHA,report_sha256='synthetic')
        a=dict(run_signature=r.RUN_SIGNATURE,complete=True,checkpoint_sha256=r.CHECKPOINT_SHA,evaluated_images=3769)
        return dict(zip(r.REVIEWED,[m,t,a]))

    def test_reviewed_real_bundle_identity_when_available(self):
        source=Path(os.environ.get('A2_LITE_REVIEWED_RUN', '/Users/ral3ply/Downloads/a2_lite_full_recovery_results.zip'))
        if not source.exists(): self.skipTest('Reviewed archive is not on this host')
        m,t,a=r.reviewed(source)
        self.assertEqual(t['optimizer_steps'],37120)
        self.assertEqual(t['checkpoint_sha256'],r.CHECKPOINT_SHA)
        self.assertAlmostEqual(a['metrics']['vehicle_3d_moderate'],3.959141930351357)

    def test_identity_and_schema_reject_kd_wrong_steps_overlap_and_missing_validation(self):
        base=self.records()
        def check(values):
            blobs={k:json.dumps(v).encode() for k,v in values.items()}
            hashes={k:r.digest(v) for k,v in blobs.items()}
            with patch.object(r,'REVIEWED',hashes),patch.object(r,'read_record',side_effect=lambda _,n:blobs[n]),patch.object(r,'checked_signature',side_effect=lambda v,_:v):
                return r.reviewed(Path('synthetic'))
        check(base)
        changes=[('control/training_complete.json','role','kd'),('control/training_complete.json','external_teacher_used',True),
                 ('control/training_complete.json','optimizer_steps',2560),('control/training_complete.json','checkpoint_sha256','other'),
                 ('evaluation/control/accuracy.json','evaluated_images',16),('full_recovery_manifest.json','train_ids',['001000'])]
        for name,key,value in changes:
            altered=copy.deepcopy(base);altered[name][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):check(altered)

    def test_modified_reviewed_bytes_rejected_before_signature(self):
        with patch.object(r,'read_record',return_value=b'{}'),self.assertRaisesRegex(RuntimeError,'exact reviewed'):
            r.reviewed(Path('synthetic'))

    def test_payload_requires_completed_control_not_tiny_or_kd(self):
        value=dict(run_signature=r.RUN_SIGNATURE,role='control',step=37120,architecture_changed=False,model_state={})
        r.check_payload(value)
        for field,bad in [('role','kd'),('step',128),('run_signature','other'),('architecture_changed',True),('model_state',None)]:
            with self.subTest(field=field),self.assertRaises(RuntimeError):r.check_payload({**value,field:bad})

    def test_drift_gate_accepts_exact_and_boundary_but_rejects_loss_or_gain_outside(self):
        a=self.metric();self.assertTrue(r.decision(a,a)['export_accuracy_preserved'])
        for change in (-.05,.05):
            b=copy.deepcopy(a);b['metrics']['vehicle_3d_moderate']+=change
            self.assertTrue(r.decision(a,b)['ap_preserved'])
        for change in (-.051,.051):
            b=copy.deepcopy(a);b['metrics']['vehicle_3d_moderate']+=change
            self.assertFalse(r.decision(a,b)['export_accuracy_preserved'])
        b=copy.deepcopy(a);b['nearby']['classes']['Pedestrian']['near_recall']-=.006
        self.assertFalse(r.decision(a,b)['export_accuracy_preserved'])

    def test_missing_nonfinite_or_invalid_metrics_rejected(self):
        a=self.metric()
        b=copy.deepcopy(a);del b['metrics']['mean_3d_moderate']
        with self.assertRaises(RuntimeError):r.decision(a,b)
        b=copy.deepcopy(a);b['metrics']['mean_3d_moderate']=float('nan')
        with self.assertRaises(RuntimeError):r.decision(a,b)
        b=copy.deepcopy(a);b['nearby']['classes']['Vehicle']['near_recall']=2
        with self.assertRaises(RuntimeError):r.decision(a,b)

    def test_signatures_and_immutable_writes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.json';value=r.signed_json(path,dict(complete=True))
            r.checked_signature(value,'report_sha256');r.immutable_json(path,value)
            changed={**value,'complete':False}
            with self.assertRaises(RuntimeError):r.checked_signature(changed,'report_sha256')
            with self.assertRaises(RuntimeError):r.immutable_json(path,changed)

    def test_failed_json_publication_leaves_no_truncated_record(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.json'
            with patch.object(Path,'replace',side_effect=OSError('interrupted publication')):
                with self.assertRaises(OSError):r.immutable_json(path,dict(complete=True))
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(d).iterdir()),[])
            r.immutable_json(path,dict(complete=True))
            self.assertTrue(json.loads(path.read_text())['complete'])

    def test_completed_export_can_be_reused_without_model_reconstruction(self):
        from types import SimpleNamespace
        import recover_a2_lite_640 as engine
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'training/control/last.pth'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic')
            output=root/'export';output.mkdir();(output/'trained_onnx_export.json').write_text('{}')
            m=dict(output=str(root/'training'),student_export=str(root/'inputs'),
                   initialization=str(root/'init/initialization.pth'),cpu_environment={})
            args=SimpleNamespace(full_run=root/'training',checkpoint=checkpoint,input_package=None,output=output,repo=root/'repo')
            with patch.object(r,'reviewed',return_value=(m,{},{})),patch.object(r,'file_hash',return_value=r.CHECKPOINT_SHA), \
                 patch.object(r,'environment',return_value={}),patch.object(r,'load_export',return_value=({},{})) as load, \
                 patch.object(engine.base,'prepare_repo') as prepare:
                r.export(args);load.assert_called_once_with(output);prepare.assert_not_called()

    def test_partial_export_is_preserved_before_source_or_model_work(self):
        from types import SimpleNamespace
        import recover_a2_lite_640 as engine
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'training/control/last.pth'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic')
            output=root/'export';output.mkdir();(output/r.RESOURCE).mkdir()
            m=dict(output=str(root/'training'),student_export=str(root/'inputs'),
                   initialization=str(root/'init/initialization.pth'),cpu_environment={})
            args=SimpleNamespace(full_run=root/'training',checkpoint=checkpoint,input_package=None,output=output,repo=root/'repo')
            with patch.object(r,'reviewed',return_value=(m,{},{})),patch.object(r,'file_hash',return_value=r.CHECKPOINT_SHA), \
                 patch.object(r,'environment',return_value={}),patch.object(engine.base,'prepare_repo') as prepare:
                with self.assertRaisesRegex(RuntimeError,'Preserve partial'):r.export(args)
                prepare.assert_not_called()
            self.assertTrue((output/r.RESOURCE).is_dir())

    def test_archive_reader_rejects_duplicate_ambiguous_records(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'records.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('a/manifest.json','{}');z.writestr('b/manifest.json','{}')
            with self.assertRaises(RuntimeError):r.read_record(path,'manifest.json')
            self.assertEqual(r.read_record(path,'a/manifest.json'),b'{}')

    def test_bundle_keeps_failed_reports_and_package_but_excludes_weights(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'export_failure.json').write_text('{}');(root/'last.pth').write_text('secret')
            package=root/r.RESOURCE;package.mkdir();(package/'model.onnx').write_text('graph');(package/'input.bin').write_bytes(b'fixture')
            r.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_trained_onnx_results.zip') as z:
                self.assertIn('export_failure.json',z.namelist());self.assertIn(r.RESOURCE+'/model.onnx',z.namelist())
                self.assertNotIn('last.pth',z.namelist())

    def test_cli_bundle_needs_no_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/export_a2_lite_trained_onnx.py'),
                'bundle','--output',d],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_notebook_cells_compile_and_use_absolute_script_and_no_training(self):
        path=ROOT/'notebooks/MonoDETR_A2_Lite_Trained_ONNX_Colab.ipynb'
        if not path.exists():self.skipTest('Notebook not written yet')
        cells=json.loads(path.read_text())['cells'];code=[]
        for index,c in enumerate(cells):
            if c['cell_type']=='code':
                text=''.join(c['source']);compile(text,str(path)+':'+str(index),'exec');code.append(text)
                self.assertIsNone(c['execution_count']);self.assertFalse(c['outputs'])
        whole='\n'.join(code)
        self.assertIn('export_a2_lite_trained_onnx.py',whole)
        self.assertNotIn("stage('train'",whole)
        self.assertNotIn('setup_m67_runtime',whole)
        self.assertIn("'--branch','main'",whole)
        self.assertIn("stage('validate'",whole)


if __name__=='__main__':unittest.main()
