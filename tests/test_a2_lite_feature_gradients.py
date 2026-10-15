"""Exact class partitions, gradient geometry and immutable endpoint audit."""
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import audit_a2_lite_feature_gradients as audit

# Explicit file loading avoids installed third-party packages named tests.
spec = importlib.util.spec_from_file_location("feature_gradient_fixtures", ROOT / "tests/test_a2_lite_features.py")
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


class FeatureGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_gt_class_loss_and_all_output_gradients_reconstruct_original(self):
        for labels in (None, [0], [1], []):
            with self.subTest(labels=labels):
                target = fixtures.target()
                if labels is not None:
                    mask = torch.tensor([int(v) in labels for v in target['labels']])
                    target = {k: v[mask] for k, v in target.items()}
                outputs = {k: v.requires_grad_() for k, v in fixtures.outputs().items()}
                terms = audit.gt_partition(outputs, target)
                original, _, _ = audit.engine.gt_loss(outputs, target)
                self.assertTrue(torch.allclose(sum(terms.values()), original, rtol=1e-6, atol=1e-5))
                parameters = list(outputs.values())
                g = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
                h = torch.autograd.grad(sum(terms.values()), parameters, allow_unused=True)
                for a, b in zip(g, h):
                    a = torch.zeros_like(b) if a is None and b is not None else a
                    b = torch.zeros_like(a) if b is None and a is not None else b
                    if a is not None: self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))

    def test_missing_class_retains_its_negative_classification_signal(self):
        target = {k: v[1:] for k, v in fixtures.target().items()}
        outputs = {k: v.requires_grad_() for k, v in fixtures.outputs().items()}
        values = audit.gt_partition(outputs, target)
        gradient = torch.autograd.grad(values['gt/Pedestrian'], outputs['pred_logits'])[0]
        self.assertGreater(float(gradient[..., 0].abs().sum()), 0)
        self.assertEqual(float(gradient[..., 1:].abs().sum()), 0)
        self.assertGreater(float(values['gt/unused_channel'].detach()), 0)

    def test_feature_class_shares_preserve_original_loss_and_gradient(self):
        for eligible in ([True, True], [False, True], [True, False], [False, False]):
            target = fixtures.target(); student = torch.randn(2, 4, 256, 3, 3, requires_grad=True)
            row = dict(features=torch.randn_like(student), eligible=torch.tensor(eligible))
            outputs = {k: v.requires_grad_() for k, v in fixtures.outputs().items()}
            values, counts = audit.losses(outputs, student, row, target, 7.3)
            combined = values['kd/Pedestrian'] + values['kd/Vehicle']
            original, _ = audit.feature.feature_kd_loss(student, row['features'], target['labels'], row['eligible'])
            self.assertTrue(torch.allclose(combined, original * 7.3, atol=1e-6, rtol=1e-5))
            g = torch.autograd.grad(combined, student, retain_graph=True)[0]
            h = torch.autograd.grad(original * 7.3, student)[0]
            self.assertTrue(torch.allclose(g, h, atol=1e-6, rtol=1e-5))
            self.assertEqual(counts, dict(Pedestrian=int(eligible[0]), Vehicle=int(eligible[1])))

    def test_gram_uses_identical_coordinates_with_none_as_zero(self):
        gradients = {k: (torch.tensor([float(i + 1), -1.]), None) for i, k in enumerate(audit.TERMS)}
        gradients[audit.TERMS[-1]] = (None, torch.tensor([2.]))
        matrices = audit.gram(('visual_encoder', 'late_backbone'), gradients)
        flat = torch.stack([torch.cat([v[0] if v[0] is not None else torch.zeros(2), v[1] if v[1] is not None else torch.zeros(1)]) for v in gradients.values()]).double()
        self.assertTrue(torch.equal(torch.tensor(matrices['shared'], dtype=torch.double), flat @ flat.T))
        measurement = audit.measurements(matrices['shared'])
        self.assertAlmostEqual(measurement['norms']['gt_all'], float(flat[:3].sum(0).norm()))

    def test_cosines_expose_class_conflict_not_scalar_loss_size(self):
        v = torch.tensor([[1., 0.], [-1., 1.], [0., 0.], [-2., 0.], [0., .1]], dtype=torch.double)
        result = audit.measurements((v @ v.T).tolist())
        self.assertEqual(result['comparisons']['kd/Pedestrian vs gt/Pedestrian']['cosine'], -1)
        self.assertEqual(result['comparisons']['kd/Pedestrian vs gt/Pedestrian']['norm_ratio'], 2)
        rows = audit.summarize([dict(gradients={'visual_encoder': result})])
        row = next(r for r in rows if r['comparison']=='kd/Pedestrian vs gt/Pedestrian')
        self.assertEqual(row['negative_cosine_fraction'], 1)
        self.assertEqual(row['norm_ratio_above_one_fraction'], 1)

    def test_zero_teacher_has_no_fake_conflict_or_division_by_zero(self):
        m = torch.zeros(5, 5).tolist(); m[0][0] = 1
        result = audit.measurements(m)
        self.assertIsNone(result['comparisons']['kd_all vs gt/Pedestrian']['cosine'])
        row = audit.summarize([dict(gradients={'visual_encoder': result})])[0]
        self.assertEqual(row['active_frames'], 0); self.assertIsNone(row['median_cosine'])
        with self.assertRaises(ValueError): audit.measurements([[float('nan')] * 5] * 5)

    def test_fixed_train_subset_rejects_validation_and_duplicates(self):
        m = dict(pilot_ids=[f'{i:06d}' for i in range(32)], train_ids=[f'{i:06d}' for i in range(32)], val_ids=['999999'])
        self.assertEqual(len(audit.subset(m)), 32)
        m['val_ids'] = ['000001']
        with self.assertRaisesRegex(RuntimeError, 'TRAIN'): audit.subset(m)
        m['val_ids'] = []; m['pilot_ids'][1] = m['pilot_ids'][0]
        with self.assertRaisesRegex(RuntimeError, 'TRAIN'): audit.subset(m)

    def test_local_copy_is_hash_checked_and_never_clobbers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; destination = root/'local/weight.pth'
            source.write_bytes(b'weight'); expected = audit.engine.sha256(source)
            audit.local_copy(source, destination, expected)
            source.unlink(); audit.local_copy(source, destination, expected)
            destination.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Preserve changed'): audit.local_copy(source, destination, expected)
            self.assertEqual(destination.read_bytes(), b'changed')

    def test_reviewed_byte_pin_rejects_changed_run_before_source_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'feature_manifest.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'exact reviewed'): audit.reviewed(root)

    def prepare_fixture(self, root):
        root=root.resolve()
        a=SimpleNamespace(output=root/'audit',feature_run=root/'history',features_dir=root/'history/teacher_features',
            initialization=root/'initial/initial.pth',control_checkpoint=root/'history/control/last.pth',
            kd_checkpoint=root/'history/kd/last.pth',repo=root/'MonoDETR_A2LITE_FEATURE_AUDIT_test',
            dataset=root/'kitti_a2_lite_accuracy_feature_audit_test',weights_dir=root/'local_weights',
            candidate=[root/'dataset_source'],feature_candidate=[],device='cpu')
        weights={}
        for role,path in zip(audit.ROLES,(a.initialization,a.control_checkpoint,a.kd_checkpoint)):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(role.encode());weights[role]=audit.engine.sha256(path)
        ids=[f'{i:06d}' for i in range(32)]
        historical=dict(signature_sha256=audit.RUN_SIGNATURE,output=str(a.feature_run),config={'model':{}},source_sha256='source',
            pilot_ids=ids,train_ids=ids,val_ids=['999999'],data_file_sha256={s:{'image_2':'image','label_2':'label','calib':'calib'} for s in ids},
            teacher_checkpoint_sha256='teacher',cpu_environment={'torch':'historical'},execution_identity={'gpu':'historical'})
        hashes={}
        for i,s in enumerate(ids):
            labels=torch.tensor([0] if i<16 else [1]);row=dict(**audit.feature.feature_identity(historical,s),
                target_signature='target',labels=labels,eligible=torch.ones(1,dtype=torch.bool),object_to_teacher_query=torch.tensor([0]),
                features=torch.ones(1,4,256,3,3))
            path=a.features_dir/f'{s}.pt';audit.engine.atomic_checkpoint(path,row);hashes[s]=audit.engine.sha256(path)
        records={'feature_cache_complete.json':dict(complete=True,kd_pilot_authorized=True,cache_sha256=hashes),
                 'feature_kd_smoke.json':dict(calibration={'loss_scale':29.235752538530086})}
        return a,historical,records,weights

    @contextmanager
    def preparation_dependencies(self,a,historical,records,weights):
        with patch.object(audit,'reviewed',return_value=(historical,records)), \
             patch.object(audit,'WEIGHTS',weights),patch.object(audit,'runtime',return_value={'gpu':'test'}), \
             patch.object(audit,'implementation',return_value={'audit':'frozen'}), \
             patch.object(audit.engine.base,'prepare_repo',side_effect=lambda p:p.mkdir(parents=True,exist_ok=True)) as restore, \
             patch.object(audit.engine.base.q,'source_hash',return_value='source'), \
             patch.object(audit.staging,'stage') as stage, \
             patch.object(audit.engine,'read_record',return_value=b'{}'),redirect_stdout(io.StringIO()):
            yield restore,stage

    def test_prepare_stages_only32_and_cold_reset_restores_before_load(self):
        with tempfile.TemporaryDirectory() as d:
            a,h,r,w=self.prepare_fixture(Path(d))
            original={p:p.read_bytes() for p in a.feature_run.rglob('*') if p.is_file()}
            with self.preparation_dependencies(a,h,r,w) as (restore,stage):
                audit.prepare(a);saved=(a.output/'feature_gradient_manifest.json').read_bytes()
                self.assertEqual(len(stage.call_args.args[2]),32)
                self.assertEqual(set(stage.call_args.args[3]),set(h['pilot_ids']))
                self.assertEqual(stage.call_args.kwargs['workers'],4)
                m,_=audit.load(a.output,'cpu')
                self.assertEqual(m['staged_validation_frames'],0);self.assertFalse(m['training_authorized'])
                shutil.rmtree(a.repo);shutil.rmtree(a.dataset);shutil.rmtree(a.weights_dir)
                audit.prepare(a);audit.load(a.output,'cpu')
                self.assertTrue(a.repo.exists());self.assertEqual(restore.call_count,2)
            self.assertEqual((a.output/'feature_gradient_manifest.json').read_bytes(),saved)
            self.assertTrue(all(p.read_bytes()==b for p,b in original.items()))

    def test_changed_diagnostic_runtime_stops_before_any_source_or_staging(self):
        with tempfile.TemporaryDirectory() as d:
            a,h,r,w=self.prepare_fixture(Path(d))
            with self.preparation_dependencies(a,h,r,w) as (restore,stage):
                audit.prepare(a);restore.reset_mock();stage.reset_mock()
                before=(a.output/'feature_gradient_manifest.json').read_bytes()
                with patch.object(audit,'runtime',return_value={'gpu':'changed'}):
                    with self.assertRaisesRegex(RuntimeError,'runtime/code/paths'):audit.prepare(a)
                restore.assert_not_called();stage.assert_not_called()
            self.assertEqual((a.output/'feature_gradient_manifest.json').read_bytes(),before)

    def test_corrupted_local_feature_stops_before_restoring_or_staging(self):
        with tempfile.TemporaryDirectory() as d:
            a,h,r,w=self.prepare_fixture(Path(d))
            path=a.weights_dir/'teacher_features'/f"{h['pilot_ids'][0]}.pt"
            path.parent.mkdir(parents=True);path.write_bytes(b'corrupted')
            with self.preparation_dependencies(a,h,r,w) as (restore,stage):
                with self.assertRaisesRegex(RuntimeError,'bytes changed'):audit.prepare(a)
                restore.assert_not_called();stage.assert_not_called()
            self.assertEqual(path.read_bytes(),b'corrupted')

    def test_completed_run_cannot_be_used_as_diagnostic_destination(self):
        with tempfile.TemporaryDirectory() as d:
            a,h,r,w=self.prepare_fixture(Path(d));a.output=a.feature_run
            with self.preparation_dependencies(a,h,r,w) as (restore,stage):
                with self.assertRaisesRegex(RuntimeError,'read-only'):audit.prepare(a)
                restore.assert_not_called();stage.assert_not_called()
            self.assertFalse((a.output/'feature_gradient_manifest.json').exists())

    def test_load_rejects_resigned_model_config_and_changed_local_weights(self):
        with tempfile.TemporaryDirectory() as d:
            a,h,r,w=self.prepare_fixture(Path(d))
            with self.preparation_dependencies(a,h,r,w):
                audit.prepare(a);path=a.output/'feature_gradient_manifest.json';saved=path.read_bytes()
                m=json.loads(saved);m['view']['config']['model']['num_queries']=100
                m['signature_sha256']=audit.engine.signature({k:v for k,v in m.items() if k!='signature_sha256'})
                audit.engine.write_json(path,m)
                with self.assertRaisesRegex(RuntimeError,'model config changed'):audit.load(a.output,'cpu')
                path.write_bytes(saved);(a.weights_dir/'control_epoch5.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'checkpoint changed'):audit.load(a.output,'cpu')

    def test_probe_has_zero_updates_and_resumes_verified_frames(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=fixtures.ToyModel(); before=copy.deepcopy(model.state_dict())
            view=dict(pilot_ids=fixtures.Dataset.idx_list)
            m=dict(signature_sha256='audit', view=view, weights={}, feature_root=str(root/'cache'),
                   feature_sha256={s:'sha' for s in fixtures.Dataset.idx_list}, final_feature_coefficient=7.3)
            a=SimpleNamespace(output=root, role='initial', device='cpu')
            row=dict(features=torch.randn(2,4,256,3,3),eligible=torch.tensor([True,True]))
            with patch.object(audit.engine,'build_model',return_value=model), \
                 patch.object(audit.engine,'native_dataset',return_value=fixtures.Dataset()), \
                 patch.object(audit.engine,'verify_files'), patch.object(audit.engine,'item',side_effect=lambda *args:({},fixtures.target(),{})), \
                 patch.object(audit.engine,'forward',side_effect=lambda model, inputs:model.result()), \
                 patch.object(audit.feature,'read_feature',return_value=row), redirect_stdout(io.StringIO()):
                audit.probe(a,m,{})
                with patch.object(audit.engine,'forward',side_effect=AssertionError('Completed frames must resume without forward')):
                    audit.probe(a,m,{})
            report=audit.engine.checked_report(root/'initial_gradient_report.json',m)
            self.assertEqual(report['optimizer_steps'],0); self.assertTrue(report['weights_and_buffers_unchanged'])
            self.assertEqual(report['frames'],2)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
            self.assertEqual(len(model.depthaware_transformer.encoder._forward_hooks),0)

    def test_wrong_endpoint_identity_rejected_with_safe_load_and_no_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'wrong.pth';torch.save(dict(run_signature='wrong',role='kd',step=2560),path)
            m=dict(view={},weights={'feature_epoch5':dict(path=str(path))},final_feature_coefficient=7.3)
            a=SimpleNamespace(role='feature_epoch5',device='cpu',output=root)
            with patch.object(audit.engine,'build_model',return_value=fixtures.ToyModel()):
                with self.assertRaisesRegex(RuntimeError,'Wrong completed endpoint'):audit.probe(a,m,{})

    def test_valid_endpoint_payload_uses_strict_weights_only_and_keeps_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'endpoint.pth';model=fixtures.ToyModel()
            state=copy.deepcopy(model.state_dict());state['pred_logits']+=.1
            coefficient=7.3089381346325215
            torch.save(dict(run_signature=audit.RUN_SIGNATURE,role='kd',step=2560,
                feature_loss_scale=coefficient/.25,model_state=state),path)
            m=dict(signature_sha256='audit',view=dict(pilot_ids=fixtures.Dataset.idx_list),
                weights={'feature_epoch5':dict(path=str(path))},feature_root=str(root/'cache'),
                feature_sha256={s:'sha' for s in fixtures.Dataset.idx_list},final_feature_coefficient=coefficient)
            a=SimpleNamespace(output=root,role='feature_epoch5',device='cpu')
            row=dict(features=torch.randn(2,4,256,3,3),eligible=torch.tensor([True,True]))
            with patch.object(audit.engine,'build_model',return_value=model), \
                 patch.object(audit.engine,'native_dataset',return_value=fixtures.Dataset()), \
                 patch.object(audit.engine,'verify_files'),patch.object(audit.engine,'item',side_effect=lambda *args:({},fixtures.target(),{})), \
                 patch.object(audit.engine,'forward',side_effect=lambda model,inputs:model.result()), \
                 patch.object(audit.feature,'read_feature',return_value=row), \
                 patch.object(audit.torch,'load',wraps=torch.load) as reader,redirect_stdout(io.StringIO()):
                audit.probe(a,m,{})
            self.assertTrue(reader.call_args.kwargs['weights_only'])
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_comparison_requires_three_complete_unchanged_states_and_frame_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=[f'{i:06d}' for i in range(32)]
            m=dict(signature_sha256='audit',view=dict(pilot_ids=ids),final_feature_coefficient=7.3)
            for role in audit.ROLES:
                hashes={}
                for sample in ids:
                    path=root/'frames'/role/f'{sample}.json';audit.engine.write_json(path,{'evidence':sample})
                    hashes[sample]=audit.engine.sha256(path)
                audit.engine.save_report(root/f'{role}_gradient_report.json',dict(complete=True,run_signature='audit',
                    role=role,frames=32,optimizer_steps=0,weights_and_buffers_unchanged=True,
                    fully_ramped_coefficient=7.3,frame_sha256=hashes,coordinate_names_sha256='same',summary=[]))
            a=SimpleNamespace(output=root)
            with redirect_stdout(io.StringIO()):audit.compare(a,m)
            value=audit.engine.checked_report(root/'feature_gradient_comparison.json',m)
            self.assertFalse(value['training_authorized']);self.assertFalse(value['validation_inference_performed'])
            (root/'frames/feature_epoch5/000001.json').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'complete, unchanged-weight'):audit.compare(a,m)

    def test_bundle_excludes_weights_features_and_images(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('report.json','colab_logs/probe.log','weights/model.pth','features/000001.pt','training/image_2/000001.png'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
            with redirect_stdout(io.StringIO()):audit.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_feature_gradient_results.zip') as z:
                self.assertEqual(set(z.namelist()),{'report.json','colab_logs/probe.log'})

    def test_notebook_is_standalone_train_only_no_unsafe_loader_or_training(self):
        path=ROOT/'notebooks/MonoDETR_A2_Lite_Feature_Endpoint_Gradient_Audit_Colab.ipynb'
        notebook=json.loads(path.read_text());codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codes),6)
        for i, code in enumerate(codes):compile(code,f'feature-gradient-cell-{i}','exec')
        source='\n'.join(codes)
        self.assertIn(audit.REVISION,source)
        self.assertIn("'--control-checkpoint'",source);self.assertIn("'--kd-checkpoint'",source)
        self.assertIn("stage('compare')",source)
        for bad in ('weights_only=False','reset --hard','setup_m67_runtime',"stage('train'",'pip install torch','from tests.'):
            self.assertNotIn(bad,source)
        self.assertLess(source.index("['git','pull','--ff-only']"),source.index('if not SCRIPT.is_file()'))


if __name__=='__main__':
    unittest.main()
