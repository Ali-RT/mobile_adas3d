"""Per-group caps, class conflict rejection, unchanged GT and exact resume."""
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import recover_a2_lite_protected_features as r
from models import a2_lite_protected_feature_kd as guard

spec=importlib.util.spec_from_file_location('protected_feature_fixtures',ROOT/'tests/test_a2_lite_features.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def gram(vectors):
    v=torch.tensor(vectors,dtype=torch.double)
    return (v@v.T).tolist()


class ProtectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):torch.manual_seed(123)

    def test_dynamic_cap_does_not_amplify_scheduled_teacher_weight(self):
        m=gram([[1,0],[1,0],[1,0],[10,0]])
        d=guard.choose_coefficient({'all_trainable':m,'visual_encoder':m},7.3)
        self.assertTrue(d['accepted']);self.assertTrue(d['scaled_for_cap'])
        self.assertLessEqual(d['coefficient']*10,.05)
        self.assertAlmostEqual(d['coefficient'],.005,places=8)
        self.assertEqual(guard.choose_coefficient({'all_trainable':m},.001)['coefficient'],.001)

    def test_class_conflict_rejected_even_when_total_gt_aligns(self):
        m=gram([[1,1],[-1,1],[2,0],[1,0]])
        d=guard.choose_coefficient({'all_trainable':m},7.3)
        self.assertFalse(d['accepted']);self.assertEqual(d['coefficient'],0.)
        self.assertEqual(d['conflicts'],['all_trainable:gt/Pedestrian'])

    def test_group_conflict_not_hidden_by_other_large_aligned_layers(self):
        good=gram([[10,0],[5,0],[5,0],[10,0]])
        bad=gram([[1,0],[1,0],[1,0],[-1,0]])
        total=(torch.tensor(good)+torch.tensor(bad)).tolist()
        d=guard.choose_coefficient({'all_trainable':total,'visual_encoder':good,'late_backbone':bad},1.)
        self.assertEqual(d['coefficient'],0.)
        self.assertTrue(any(x.startswith('late_backbone:') for x in d['conflicts']))

    def test_strictest_group_controls_cap_including_other_parameters(self):
        large=gram([[100,0],[100,0],[100,0],[1,0]])
        small=gram([[.1,0],[.1,0],[.1,0],[1,0]])
        d=guard.choose_coefficient({'all_trainable':large,'other_trainable':small},1.)
        self.assertLessEqual(d['coefficient'],.005)
        self.assertTrue(all(v['applied_teacher_to_gt_norm_ratio']<=.05 for v in d['checks'].values()))

    def test_zero_gt_or_teacher_does_not_divide_by_zero(self):
        m=gram([[0,0],[0,0],[0,0],[1,0]])
        self.assertEqual(guard.choose_coefficient({'all_trainable':m},1.)['coefficient'],0.)
        m=gram([[1,0],[1,0],[1,0],[0,0]])
        d=guard.choose_coefficient({'all_trainable':m},1.)
        self.assertFalse(d['accepted']);self.assertTrue(d['no_teacher_gradient'])

    def test_nonfinite_and_missing_gradient_coordinates_rejected(self):
        with self.assertRaises(ValueError):guard.choose_coefficient({},1.)
        with self.assertRaises(ValueError):guard.choose_coefficient({'all_trainable':[[float('nan')]*4]*4},1.)
        with self.assertRaises(ValueError):guard.choose_coefficient({'all_trainable':gram([[1],[1],[1],[1]])},float('inf'))
        gradients={k:[torch.ones(2)] for k in guard.TERMS};gradients['kd']=[torch.tensor([float('nan'),1.])]
        with self.assertRaises(RuntimeError):guard.gradient_grams(['visual_encoder'],gradients)

    def test_gram_shared_coordinates_include_disconnected_zero_terms(self):
        gradients={k:[torch.tensor([i+1.]),None] for i,k in enumerate(guard.TERMS)}
        gradients['kd']=[None,torch.tensor([2.])]
        matrices=guard.gradient_grams(['visual_encoder','other_trainable'],gradients)
        self.assertEqual(matrices['all_trainable'][3][3],4.)
        self.assertEqual(matrices['all_trainable'][0][3],0.)

    def test_numerical_roundoff_tolerance_is_explicit_not_large_conflict_slack(self):
        tiny=gram([[1,0],[1,0],[1,0],[-1e-8,1]])
        self.assertTrue(guard.choose_coefficient({'all_trainable':tiny},.01)['accepted'])
        bad=gram([[1,0],[1,0],[1,0],[-1e-3,1]])
        self.assertFalse(guard.choose_coefficient({'all_trainable':bad},.01)['accepted'])

    def test_all_trainable_parameters_have_exactly_one_group(self):
        model=f.ToyModel();model.backbone=torch.nn.Linear(2,2);model.backbone.bias.requires_grad_(False)
        names,parameters,groups=guard.coordinates(model)
        self.assertEqual(len(parameters),len({id(p) for p in parameters}))
        self.assertEqual(len(parameters),sum(p.requires_grad for p in model.parameters()))
        self.assertNotIn('backbone.bias',names)
        self.assertEqual(set(groups),{'visual_encoder','late_backbone','other_trainable'})

    def test_recipe_preserves_architecture_gt_budget_and_one_fixed_policy(self):
        self.assertEqual(r.RECIPE['input_hw'],[192,640]);self.assertEqual(r.RECIPE['epochs']*r.RECIPE['train_images'],2560)
        self.assertEqual(r.RECIPE['gt_weights'],r.feature.RECIPE['gt_weights'])
        self.assertFalse(r.RECIPE['architecture_changed']);self.assertEqual(r.RECIPE['prediction_kd_components'],[])
        self.assertEqual(r.RECIPE['feature_gradient_cap'],.05);self.assertFalse(r.RECIPE['adamw_update_safety_guaranteed'])
        self.assertTrue(r.RECIPE['teacher_features_reused']);self.assertTrue(r.RECIPE['validation_staged_after_training'])
        old=r.engine.RECIPE,r.engine.REVISION
        with self.assertRaises(RuntimeError):
            with r.recipe_context():
                self.assertIs(r.engine.RECIPE,r.RECIPE);raise RuntimeError('synthetic')
        self.assertEqual((r.engine.RECIPE,r.engine.REVISION),old)

    def test_rejected_teacher_gives_exact_original_gt_parameter_gradients(self):
        model=f.ToyModel();target=f.target();_,parameters,groups=guard.coordinates(model)
        row=dict(features=torch.randn(2,4,256,3,3),eligible=torch.ones(2,dtype=torch.bool))
        with r.feature.EncoderFeatureTap(model) as tap:
            outputs=model.result();gt,_,_=r.engine.gt_loss(outputs,target)
            expected=torch.autograd.grad(gt,tuple(model.parameters()),retain_graph=True,allow_unused=True)
            with patch.object(r,'choose_coefficient',return_value={'coefficient':0.,'accepted':False}):
                total,_,_,_,_=r.guarded_objective(model,outputs,tap.take(),row,target,7.3,parameters,groups)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            total.backward()
        for p,g in zip(model.parameters(),expected):
            if g is not None:self.assertTrue(torch.equal(p.grad,g))

    def test_actual_controller_changes_no_state_and_enforces_all_group_caps(self):
        model=f.ToyModel();before=copy.deepcopy(model.state_dict());target=f.target()
        _,parameters,groups=guard.coordinates(model)
        row=dict(features=torch.randn(2,4,256,3,3),eligible=torch.ones(2,dtype=torch.bool))
        with r.feature.EncoderFeatureTap(model) as tap:
            outputs=model.result()
            total,gt,kd,counts,d=r.guarded_objective(model,outputs,tap.take(),row,target,7.3,parameters,groups)
        self.assertTrue(torch.allclose(total,gt+d['coefficient']*kd))
        self.assertEqual(counts,{'Pedestrian':1,'Vehicle':1})
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(v['applied_teacher_to_gt_norm_ratio']<=.05+1e-8 for v in d['checks'].values()))
        if d['accepted']:
            self.assertTrue(all(c is None or c>=-guard.COSINE_TOLERANCE for v in d['checks'].values() for c in v['cosines'].values()))

    def mocks(self,model):
        return (patch.object(r.engine,'verify_files'),patch.object(r.engine,'item',side_effect=lambda *args:({},f.target(),{})),
                patch.object(r.engine,'forward',side_effect=lambda model,inputs:model.result()))

    def manifest(self,root):return dict(signature_sha256='synthetic',runtime={'test':True},scheduled_max_coefficient=7.3,output=str(root))

    def test_fresh_control_matches_original_feature_gt_training_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.manifest(root);old=f.ToyModel();new=copy.deepcopy(old)
            p1,p2,p3=self.mocks(old)
            with p1,p2,p3,redirect_stdout(io.StringIO()),r.recipe_context():
                r.feature.run_steps(m,old,r.engine.optimizer(old),f.Dataset(),4,root/'old.pth','control','cpu')
                r.run_steps(m,{},new,r.engine.optimizer(new),f.Dataset(),4,root/'new.pth','control','cpu')
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
            checkpoint=torch.load(root/'new.pth',weights_only=True)
            self.assertEqual(checkpoint['guard_statistics'],{})

    def test_kd_resume_preserves_coefficients_statistics_and_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.manifest(root);full=f.ToyModel();interrupted=copy.deepcopy(full)
            row=dict(features=torch.randn(2,4,256,3,3),eligible=torch.ones(2,dtype=torch.bool))
            recipe={**r.RECIPE,'checkpoint_every_steps':1}
            p1,p2,p3=self.mocks(full)
            with p1,p2,p3,patch.object(r,'RECIPE',recipe),patch.object(r,'read_row',return_value=row),redirect_stdout(io.StringIO()),r.recipe_context():
                r.run_steps(m,{},full,r.engine.optimizer(full),f.Dataset(),4,root/'full.pth','kd','cpu')
                calls=[0]
                def item(*args):
                    calls[0]+=1
                    if calls[0]==3:raise RuntimeError('interrupt')
                    return {},f.target(),{}
                with patch.object(r.engine,'item',side_effect=item):
                    with self.assertRaisesRegex(RuntimeError,'interrupt'):
                        r.run_steps(m,{},interrupted,r.engine.optimizer(interrupted),f.Dataset(),4,root/'resumed.pth','kd','cpu')
                resumed=f.ToyModel()
                r.run_steps(m,{},resumed,r.engine.optimizer(resumed),f.Dataset(),4,root/'resumed.pth','kd','cpu')
            self.assertTrue(all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items()))
            a=torch.load(root/'full.pth',weights_only=True);b=torch.load(root/'resumed.pth',weights_only=True)
            self.assertEqual(a['guard_statistics'],b['guard_statistics']);self.assertEqual(a['history'],b['history'])

    def test_reviewed_evidence_byte_pins_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'feature_gradient_manifest.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'exact reviewed'):r.reviewed_audit(path)

    def test_validation_staging_requires_both_finished_arms(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.manifest(root);m.update(dataset=str(root/'data'),val_ids=['999999'],data_file_sha256={})
            with patch.object(r.engine,'checked_report',return_value={'complete':True,'optimizer_steps':128}),patch.object(r.staging,'stage') as stage:
                with self.assertRaisesRegex(RuntimeError,'Finish both'):r.stage_validation(SimpleNamespace(output=root,candidate=[]),m)
                stage.assert_not_called()

    def test_guard_smoke_rejection_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(output=Path(d),device='cpu',role='kd')
            with patch.object(r.engine,'tiny_gate'),patch.object(r.engine,'checked_report',return_value={'passed':False,'optimizer_steps':0}),patch.object(r.engine,'optimizer') as optimizer:
                with self.assertRaisesRegex(RuntimeError,'Passed zero-update'):r.train(a,self.manifest(Path(d)),{})
                optimizer.assert_not_called()

    def prepare_fixture(self,root):
        root=root.resolve();history=root/'history';history.mkdir()
        initialization=root/'original_weights/initial.pth';initialization.parent.mkdir();initialization.write_bytes(b'initial')
        a=SimpleNamespace(output=root/'new_run',feature_run=history,gradient_audit=root/'audit_history',
            features_dir=history/'teacher_features',initialization=initialization,
            repo=root/'MonoDETR_A2LITE_PROTECTED_test',dataset=root/'kitti_a2_lite_accuracy_protected_test',
            compact=None,device='cpu',candidate=[],feature_candidate=[])
        ids=[f'{i:06d}' for i in range(512)];val=[f'{i:06d}' for i in range(1000,4769)]
        parent=dict(output=str(history),source=str(root/'old_source'),dataset=str(root/'old_data'),source_sha256='source',
            pilot_ids=ids,val_ids=val,config={'model':{'hidden_dim':256}},student_export=str(history/'student'),
            data_file_sha256={s:{'image_2':'image','calib':'calib','label_2':'label'} for s in ids+val})
        # Model the512-file inventory without creating thousands of tiny test
        # files. Byte-safe copy is covered independently by the audit tests.
        template=a.features_dir/'template.pt';template.parent.mkdir(parents=True);template.write_bytes(b'feature')
        class FeatureSource:
            def __truediv__(self,name):return template
        a.feature_candidate=[FeatureSource()]
        digest=r.engine.sha256(template);hashes=dict.fromkeys(ids,digest)
        records={'feature_cache_complete.json':{'cache_sha256':hashes},'feature_kd_smoke.json':{'calibration':{'loss_scale':4.}}}
        return a,parent,records

    @contextmanager
    def preparation_dependencies(self,a,parent,records):
        original_copy=r.audit.local_copy
        def copy_input(source,destination,expected):
            if destination.suffix!='.pt':original_copy(source,destination,expected)
        with patch.object(r.audit,'reviewed',return_value=(parent,records)),patch.object(r,'reviewed_audit'), \
             patch.object(r.audit,'runtime',return_value={'gpu':'test'}),patch.object(r,'implementation',return_value={'pilot':'source'}), \
             patch.object(r.engine.resolution,'CHECKPOINT_SHA',r.engine.sha256(a.initialization)), \
             patch.object(r.engine.base,'prepare_repo',side_effect=lambda p:p.mkdir(parents=True,exist_ok=True)) as restore, \
             patch.object(r.engine.base.q,'source_hash',return_value='source'),patch.object(r.staging,'stage') as stage, \
             patch.object(r.engine.accuracy,'checked_package',return_value=(a.output/'assets/student',{})), \
             patch.object(r.feature,'read_feature'),patch.object(r.audit,'local_copy',side_effect=copy_input), \
             patch.object(r.engine,'read_record',return_value=b'{}'),redirect_stdout(io.StringIO()):
            yield restore,stage

    def test_prepare_defers_val_and_cold_reset_restores_before_saved_load(self):
        with tempfile.TemporaryDirectory() as d:
            a,parent,records=self.prepare_fixture(Path(d));originals={p:p.read_bytes() for p in a.feature_run.rglob('*') if p.is_file()}
            with self.preparation_dependencies(a,parent,records) as (restore,stage):
                r.prepare(a);snapshot=(a.output/'protected_manifest.json').read_bytes()
                self.assertEqual(stage.call_args.args[2],parent['pilot_ids']);self.assertEqual(stage.call_args.kwargs['workers'],4)
                m,_=r.load(a.output,'cpu')
                self.assertEqual(m['train_ids'],parent['pilot_ids']);self.assertEqual(len(m['val_ids']),3769)
                self.assertFalse((a.output/'validation_staging.json').exists())
                shutil.rmtree(a.repo);shutil.rmtree(a.dataset)
                r.prepare(a);r.load(a.output,'cpu')
                self.assertEqual(restore.call_count,2)
            self.assertEqual((a.output/'protected_manifest.json').read_bytes(),snapshot)
            self.assertTrue(all(p.read_bytes()==value for p,value in originals.items()))

    def test_changed_runtime_stops_before_staging_or_source_restore(self):
        with tempfile.TemporaryDirectory() as d:
            a,parent,records=self.prepare_fixture(Path(d))
            with self.preparation_dependencies(a,parent,records) as (restore,stage):
                r.prepare(a);restore.reset_mock();stage.reset_mock()
                with patch.object(r.audit,'runtime',return_value={'gpu':'different'}):
                    with self.assertRaisesRegex(RuntimeError,'runtime/code/paths changed'):r.prepare(a)
                restore.assert_not_called();stage.assert_not_called()

    def test_completed_run_cannot_be_protected_pilot_output(self):
        with tempfile.TemporaryDirectory() as d:
            a,parent,records=self.prepare_fixture(Path(d));a.output=a.feature_run
            with self.preparation_dependencies(a,parent,records) as (restore,stage):
                with self.assertRaisesRegex(RuntimeError,'read-only'):r.prepare(a)
                restore.assert_not_called();stage.assert_not_called()
            self.assertFalse((a.output/'protected_manifest.json').exists())

    def test_promotion_requires_accuracy_and_actual_safeguard_use(self):
        for violation in (0,1):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);m=self.manifest(root)
                for role in ('control','kd'):
                    path=root/role/'last.pth';path.parent.mkdir();path.write_bytes(role.encode())
                    stats={} if role=='control' else dict(checked_steps=2560,cap_violations=violation,accepted_Pedestrian=100,accepted_Vehicle=100)
                    t=dict(complete=True,run_signature=m['signature_sha256'],role=role,optimizer_steps=2560,train_images=512,
                        fixed_endpoint_epoch=5,tiny_checkpoint_used=False,architecture_changed=False,runtime=m['runtime'],
                        same_initialization_sha256=r.engine.resolution.CHECKPOINT_SHA,external_teacher_used=role=='kd',
                        checkpoint_sha256=r.engine.sha256(path),guard_statistics=stats)
                    r.engine.save_report(path.parent/'training_complete.json',t)
                    metrics=dict(vehicle_3d_moderate=5.,pedestrian_3d_moderate=5.,mean_3d_moderate=5.,vehicle_bev_moderate=5.,pedestrian_bev_moderate=5.)
                    if role=='kd':metrics.update(vehicle_3d_moderate=5.4,mean_3d_moderate=5.2)
                    value=dict(complete=True,run_signature=m['signature_sha256'],checkpoint_sha256=t['checkpoint_sha256'],evaluated_images=3769,
                        metrics=metrics,nearby={'classes':{cls:{'near_recall':.8} for cls in ('Pedestrian','Vehicle')}})
                    r.engine.write_json(root/'evaluation'/role/'accuracy.json',value)
                with redirect_stdout(io.StringIO()):r.compare(SimpleNamespace(output=root),m)
                c=r.engine.checked_report(root/'protected_comparison.json',m)
                self.assertEqual(c['benefit_guardrail_passed'],violation==0)
                self.assertFalse(c['full_training_authorized']);self.assertFalse(c['deployment_qualified'])

    def test_bundle_excludes_features_weights_and_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('report.json','colab_logs/run.log','weights/checkpoint.pth','teacher_features/000001.pt','assets/student/manifest.json'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            with redirect_stdout(io.StringIO()):r.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_protected_feature_results.zip') as z:self.assertEqual(set(z.namelist()),{'report.json','colab_logs/run.log'})

    def test_notebook_is_self_contained_and_only_defers_validation_not_training_labels(self):
        n=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_Protected_Feature_KD_Colab.ipynb').read_text())
        codes=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codes),7)
        for i,code in enumerate(codes):compile(code,f'protected-cell-{i}','exec')
        source='\n'.join(codes)
        self.assertIn(r.REVISION,source)
        self.assertLess(source.index("stage('train'"),source.index("stage('stage-validation'"))
        self.assertLess(source.index("['git','pull','--ff-only']"),source.index('if not SCRIPT.is_file()'))
        for bad in ('weights_only=False','reset --hard','setup_m67_runtime','from tests.',"stage('cache'"):
            self.assertNotIn(bad,source)


if __name__=='__main__':unittest.main()
