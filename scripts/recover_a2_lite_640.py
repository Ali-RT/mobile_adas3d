"""Bounded A2-Lite640 recovery: smoke, tiny fit, matched GT/A2-KD pilot.

No historical runtime restore, CUDA extension compilation or architecture
sweep. Fixed batch1/50-query portable inference path with autograd, frozen
BN/dropout, instance GT losses. Both pilot arms restart from exact reviewed
pruned weights; the tiny-fit checkpoint is never their initialization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import importlib.abc
import importlib.metadata
import json
import random
from pathlib import Path
import sys
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]

import numpy as np
import torch

import a2_lite_onnx as base
import a2_lite_resolution_onnx as resolution
import evaluate_a2_lite_accuracy as accuracy
from compact_a2_teacher import A2Teacher, materialize_export, teacher_inputs, preprocessing_comparison
from m68_onnx_common import INPUT_NAMES, OUTPUT_KEYS, OUTPUT_NAMES, OUTPUT_SHAPES, compare_outputs, read_f32, sha256, signature, write_json
from models.a2_lite_recovery_loss import (COMPONENTS, GT_WEIGHTS, COSTS, TEMPERATURE,
                                         gt_loss, assignment, component_errors, teacher_matches, kd_loss)

REVISION = 'A2-LITE-640-RECOVERY-KD-2026-10-12-r1'
SCREEN_SHA = '1c197d6f9588ced4b3ac8e65641308b3e15af7104a38aac058fc82c1ded94274'
ACCURACY_SHA = '9a3b6703458da696552ed0855dedb0b472679f4f7aaccf6161eedb90c6261788'
RECIPE = dict(seed=123, train_images=512, pedestrian_containing_frames=256,
    vehicle_only_frames=256, tiny_images=16, tiny_steps=128, tiny_geometry_ratio=.9,
    tiny_positive_gt_ratio=.7, epochs=5, batch_size=1, queries=50, input_hw=[192,640],
    learning_rate=1e-4, backbone_learning_rate=1e-5, weight_decay=1e-4,
    clip_gradient_norm=1., checkpoint_every_steps=128, kd_max_weight=.25,
    kd_warmup_steps=128, temperature=TEMPERATURE, teacher_class_threshold=.1,
    teacher_2d_iou=.5, teacher_error_ratio=.9, minimum_teacher_improvement=1e-6,
    minimum_teacher_matches_per_class=16,
    gt_weights=GT_WEIGHTS, matching_costs=COSTS, dense_depth_map_auxiliary_loss=False,
    augmentation=False, training_query_groups=False, freeze_batchnorm=True,
    dropout=False, amp=False, validation_selection=False, architecture_changed=False)


class PortableOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='MultiScaleDeformableAttention':
            raise ModuleNotFoundError('Native attention extension intentionally blocked: portable tensor training only')


def runtime_environment():
    return {**accuracy.environment(), **{name:importlib.metadata.version(name) for name in
            ('timm','scipy','opencv-python-headless')}}


def implementation():
    return {**accuracy.implementation(), **base.implementation(),
            'scripts/recover_a2_lite_640.py':sha256(__file__),
            'models/a2_lite_recovery_loss.py':sha256(ROOT/'models/a2_lite_recovery_loss.py')}


def read_record(source, name):
    if source.is_dir():return (source/name).read_bytes()
    with zipfile.ZipFile(source) as z:
        names=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
        if len(names)!=1:raise RuntimeError(f'Expected one {name} in ZIP')
        return z.read(names[0])


def signed_record(raw, expected, field='signature_sha256'):
    if accuracy.sha_bytes(raw)!=expected:raise RuntimeError('Use the exact reviewed baseline record')
    value=json.loads(raw)
    if field and signature({k:v for k,v in value.items() if k!=field})!=value[field]:
        raise RuntimeError('Baseline record signature changed')
    return value


def save_report(path, value):
    value=dict(value);value['report_sha256']=signature(value)
    accuracy.immutable_json(path,value)


def checked_report(path, m):
    value=json.loads(path.read_text())
    if (signature({k:v for k,v in value.items() if k!='report_sha256'})!=value.get('report_sha256')
            or value['run_signature']!=m['signature_sha256']):raise RuntimeError(f'Changed recovery report: {path}')
    return value


def choose_ids(train_ids, label_dir):
    """Stratify using train labels only, never validation performance."""
    ped,vehicle=[],[]
    for sample in train_ids:
        classes=set()
        for obj in accuracy.parse_kitti_label_file(label_dir/f'{sample}.txt'):
            name=accuracy.KITTI_PRODUCTION_CLASS_MAPPING.get(obj.class_name)
            if (name and obj.occluded<=2 and obj.truncated<=.5 and obj.bbox_2d[3]-obj.bbox_2d[1]>=25
                    and 2<=obj.location_3d[2]<=65 and min(obj.dimensions_3d)>0):classes.add(name)
        if 'Pedestrian' in classes:ped.append(sample)
        elif 'Vehicle' in classes:vehicle.append(sample)
    rng=random.Random(RECIPE['seed']);rng.shuffle(ped);rng.shuffle(vehicle)
    count=RECIPE['pedestrian_containing_frames']
    if len(ped)<count or len(vehicle)<count:raise RuntimeError('Need256 eligible Pedestrian-containing and256 Vehicle-only TRAIN frames')
    ids=[value for pair in zip(ped[:count],vehicle[:count]) for value in pair]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(train_ids):raise RuntimeError('Invalid train-only selection')
    return ids


def prepare(a):
    screen=signed_record(read_record(a.baseline,'a2_lite_manifest.json'),SCREEN_SHA)
    reviewed=signed_record(read_record(a.accuracy,'a2_lite_accuracy_comparison.json'),ACCURACY_SHA,field=None)
    if (not reviewed['complete'] or reviewed['evaluated_images']!=3769 or reviewed['training_performed']
            or screen['signature_sha256']!=resolution.SCREEN_SHA):raise RuntimeError('Wrong reviewed untrained Lite baseline')
    checkpoint=a.initialization or a.baseline/'a2_lite_initialization.pth'
    if not checkpoint.is_file() or sha256(checkpoint)!=resolution.CHECKPOINT_SHA:
        raise RuntimeError(f'Missing/changed reviewed pruned initialization: {checkpoint}. Use its Drive folder; the screen ZIP excludes checkpoints.')
    # New receipt only: never write inside completed screen/accuracy/teacher runs.
    for path in (a.baseline,a.accuracy,a.teacher_export,a.compact):
        if path.is_dir() and a.output.is_relative_to(path.resolve()):raise RuntimeError('Use a separate recovery output folder')
    base.prepare_repo(a.repo)
    if base.q.source_hash(a.repo)!=screen['source_sha256']:raise RuntimeError('Fresh portable source differs from the reviewed A2-Lite source')
    train=base.q.split_ids(a.split_dir/'train.txt','train');val=base.q.split_ids(a.split_dir/'val.txt','val')
    if set(train)&set(val):raise RuntimeError('Train/validation overlap')
    for attempt in range(3):
        try:
            files=accuracy.stage_data(a.dataset,a.candidate,train+val);break
        except OSError:
            if attempt==2:raise
            print('Transient data read failed; retrying while preserving staged files.',flush=True);time.sleep(attempt+1)
    for split,ids in (('train',train),('val',val)):
        accuracy.immutable_bytes(a.dataset/'ImageSets'/f'{split}.txt',('\n'.join(ids)+'\n').encode())
    selected=choose_ids(train,a.dataset/'training/label_2')
    student_root,package=accuracy.checked_package(a.compact,'compact_640',a.output/'assets/student')
    teacher_root=materialize_export(a.teacher_export,a.output/'assets/teacher')
    teacher=A2Teacher(teacher_root)
    for f in teacher.package['fixtures']:
        inputs={n:read_f32(teacher.root,f['inputs'][n]) for n in INPUT_NAMES}
        refs={n:read_f32(teacher.root,f['cpu_reference_outputs'][n]) for n in OUTPUT_NAMES}
        if not compare_outputs(teacher.raw(inputs),refs)['passed']:raise RuntimeError('Frozen A2 ONNX fixture parity failed')
    config=copy.deepcopy(screen['original_config'])
    if config['dataset']['meanshape'] is not False:raise RuntimeError('Expected metric H/W/L, meanshape=False')
    fields={'loss_ce':'cls_loss_coef','loss_center':'3dcenter_loss_coef','loss_bbox':'bbox_loss_coef',
            'loss_giou':'giou_loss_coef','loss_depth':'depth_loss_coef','loss_dim':'dim_loss_coef','loss_angle':'angle_loss_coef'}
    if any(config['model'][fields[k]]!=v for k,v in GT_WEIGHTS.items()):raise RuntimeError('Inherited A2 GT weights differ')
    m=dict(revision=REVISION,recipe=RECIPE,implementation=implementation(),cpu_environment=runtime_environment(),
        source=str(a.repo.resolve()),source_sha256=screen['source_sha256'],config=config,
        initialization=str(checkpoint.resolve()),initialization_sha256=resolution.CHECKPOINT_SHA,
        teacher_export=str(teacher_root.resolve()),teacher_model_sha256=teacher.package['model']['sha256'],
        student_export=str(student_root.resolve()),dataset=str(a.dataset.resolve()),
        data_file_sha256=files,split_dir=str(a.split_dir.resolve()),train_ids=train,val_ids=val,
        pilot_ids=selected,reviewed_accuracy_sha256=ACCURACY_SHA,
        original_compact_metrics=reviewed['models']['compact_640'],output=str(a.output.resolve()),
        target_selection_train_only=True,full_training_authorized=False,deployment_qualified=False)
    m['signature_sha256']=signature(m)
    accuracy.immutable_json(a.output/'recovery_manifest.json',m)
    print('Prepared: same E1/D1 weights, 640x192, balanced-frame512 pilot, no validation target selection.',flush=True)


def load(output):
    m=json.loads((output/'recovery_manifest.json').read_text())
    if (signature({k:v for k,v in m.items() if k!='signature_sha256'})!=m['signature_sha256']
            or m['revision']!=REVISION or m['recipe']!=RECIPE or m['implementation']!=implementation()
            or m['cpu_environment']!=runtime_environment()):raise RuntimeError('Recovery code/CPU package versions changed; preserve run and use a new RUN_ID')
    if sha256(m['initialization'])!=resolution.CHECKPOINT_SHA or base.q.source_hash(Path(m['source']))!=m['source_sha256']:
        raise RuntimeError('Frozen initialization or portable source changed')
    for split,key in (('train','train_ids'),('val','val_ids')):
        if base.q.split_ids(Path(m['split_dir'])/f'{split}.txt',split)!=m[key]:raise RuntimeError('Split changed')
    if (set(m['train_ids'])&set(m['val_ids']) or not set(m['pilot_ids'])<=set(m['train_ids'])
            or len(m['pilot_ids'])!=len(set(m['pilot_ids'])) or len(m['pilot_ids'])!=RECIPE['train_images']):
        raise RuntimeError('Train-only pilot identity failed')
    return m


def native_dataset(m, ids, split='train'):
    repo=Path(m['source']);sys.path.insert(0,str(repo))
    # Dataset encoding does not need the upstream CUDA/Numba AP evaluator.
    # Block only its two unused entry points in this isolated subprocess;
    # all label/image/affine encoding still comes from frozen native source.
    name='lib.datasets.kitti.kitti_eval_python.eval'
    if name not in sys.modules:
        unused=types.ModuleType(name)
        def unavailable(*args,**kwargs):raise RuntimeError('Use the project product evaluator, not upstream CUDA AP')
        unused.get_official_eval_result=unavailable;unused.get_distance_eval_result=unavailable
        sys.modules[name]=unused
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    module=sys.modules[KITTI_Dataset.__module__]
    if not Path(module.__file__).resolve().is_relative_to(repo):raise RuntimeError('Wrong KITTI module resolved')
    cfg=copy.deepcopy(m['config']['dataset']);cfg.update(root_dir=m['dataset'],aug_pd=False,aug_crop=False,random_flip=0.,random_crop=0.)
    result=KITTI_Dataset(split,cfg);result.idx_list=list(ids);result.data_augmentation=False
    allowed=m['train_ids'] if split=='train' else m['val_ids']
    if not set(ids)<=set(allowed):raise RuntimeError('Forbidden split IDs')
    return result


def item(dataset,index,device='cpu'):
    image,p,raw,info=dataset[index]
    original=dict(image=np.ascontiguousarray(np.asarray(image,np.float32)[None]),
                  calibration=np.asarray(p,np.float32)[None],image_size=np.asarray(raw['img_size'],np.float32)[None])
    inputs=resolution.resize_inputs(original,'r640x192')
    target={k:torch.as_tensor(raw[k][raw['mask_2d']],device=device) for k in base.q.TARGET_KEYS}
    target['labels']=target['labels'].long()
    return inputs,target,original


def build_model(m,device='cpu',trainable=False):
    if 'MultiScaleDeformableAttention' in sys.modules:raise RuntimeError('Use a fresh stage subprocess: native attention was already imported')
    if not any(isinstance(v,PortableOnly) for v in sys.meta_path):sys.meta_path.insert(0,PortableOnly())
    torch.manual_seed(RECIPE['seed'])
    model=base.inference_model(Path(m['source']),m['config']['model'])
    base.prune_a2_layers(model);resolution.configure_attention(model,192,640)
    saved=torch.load(m['initialization'],map_location='cpu',weights_only=True)
    if (saved.get('epoch')!=0 or saved.get('source_checkpoint_sha256')!=base.q.A2_SHA
            or saved.get('source_epoch')!=130 or saved.get('training_performed') is not False):raise RuntimeError('Wrong untrained pruned checkpoint lineage')
    model.load_state_dict(saved['model_state'],strict=True)
    model.requires_grad_(trainable)
    if trainable:
        for name,parameter in model.named_parameters():
            if name.startswith('backbone.0.body.') and not name.startswith(('backbone.0.body.blocks.2.','backbone.0.body.blocks.3.','backbone.0.body.blocks.4.')):
                parameter.requires_grad_(False)
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                for p in module.parameters(recurse=False):p.requires_grad_(False)
    # eval mode disables dropout and selects the deployed first50 queries.
    # It does NOT disable autograd or optimizer updates.
    return model.to(device).eval()


def forward(model,inputs):
    device=next(model.parameters()).device
    data={n:torch.from_numpy(inputs[n]).to(device) for n in INPUT_NAMES}
    return model(data['image'],data['calibration'],None,data['image_size'])


def tensor_arrays(output):
    return {name:output[key].detach().cpu().numpy().astype(np.float32) for name,key in zip(OUTPUT_NAMES,OUTPUT_KEYS)}


def native_arrays(values,device='cpu'):
    return {key:torch.tensor(values[name],dtype=torch.float32,device=device) for name,key in zip(OUTPUT_NAMES,OUTPUT_KEYS)}


def buffers_signature(model):
    return signature({n:sha_bytes_tensor(v) for n,v in model.named_buffers()})


def head_gradients(model):
    result={}
    for name in base.HEADS:
        gradients=[p.grad for p in getattr(model,name)[0].parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(v).all() for v in gradients):raise RuntimeError(f'Missing/nonfinite {name} gradients')
        result[name]=sum(float(v.detach().abs().sum()) for v in gradients)
        if result[name]<=0:raise RuntimeError(f'No learning signal in {name}')
    return result


def sha_bytes_tensor(value):
    return accuracy.sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def optimizer(model):
    late,other=[],[]
    for name,p in model.named_parameters():
        if p.requires_grad:(late if name.startswith('backbone.') else other).append(p)
    return torch.optim.AdamW([dict(params=other,lr=RECIPE['learning_rate']),
                             dict(params=late,lr=RECIPE['backbone_learning_rate'])],weight_decay=RECIPE['weight_decay'])


def setup_device(device):
    torch.set_num_threads(2);torch.manual_seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);random.seed(RECIPE['seed'])
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Choose a GPU Colab runtime. No extension compilation is required.')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.manual_seed_all(RECIPE['seed'])
    torch.set_float32_matmul_precision('highest')


def verify_files(m,sample):
    root=Path(m['dataset'])
    found={name:sha256(root/'training'/name/(sample+('.png' if name=='image_2' else '.txt'))) for name in ('image_2','calib','label_2')}
    if found!=m['data_file_sha256'][sample]:raise RuntimeError(f'Dataset bytes changed: {sample}')


def endpoint_metrics(model,dataset):
    losses=[];components={name:[] for name in COMPONENTS};counts=Counter()
    per_class={0:[],1:[]}
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs,target,_=item(dataset,i,str(next(model.parameters()).device))
            output=forward(model,inputs);loss,_,metrics=gt_loss(output,target);losses.append(float(loss))
            counts.update(target['labels'].cpu().tolist())
            for name,v in metrics.items():
                if v is not None:components[name].append(v)
            queries,objects=assignment(output,target)
            errors=component_errors(output,target,queries,objects)
            geometry=sum(errors[k] for k in ('boxes','dimensions','depth','angle'))
            for cls in (0,1):per_class[cls].extend(geometry[target['labels'][objects]==cls].cpu().tolist())
    result={name:float(np.mean(values)) if values else None for name,values in components.items()}
    result.update(gt_loss=float(np.mean(losses)),geometry_error=sum(result[k] for k in ('boxes','dimensions','depth','angle')),
                  encoded_gt_counts=dict(Pedestrian=counts[0],Vehicle=counts[1]),
                  geometry_error_per_class={name:float(np.mean(per_class[cls])) if per_class[cls] else None
                                            for name,cls in [('Pedestrian',0),('Vehicle',1)]})
    return result


def smoke(a,m):
    if (a.output/'smoke.json').exists():
        old=checked_report(a.output/'smoke.json',m)
        if old['complete'] and old['run_signature']==m['signature_sha256']:
            for sample in old['tiny_ids']:verify_files(m,sample)
            print('Unchanged passed smoke reused; no extra optimizer steps.',flush=True);return
        raise RuntimeError('Preserve changed smoke record')
    setup_device(a.device)
    cpu=build_model(m)
    root=Path(m['student_export']);package=accuracy.validate_package((root/'manifest.json').read_bytes(),'compact_640')
    parity=[]
    with torch.no_grad():
        for f in package['fixtures']:
            inputs={n:read_f32(root,f['inputs'][n]) for n in INPUT_NAMES}
            refs={n:read_f32(root,f['cpu_reference_outputs'][n]) for n in OUTPUT_NAMES}
            comparison=compare_outputs(tensor_arrays(forward(cpu,inputs)),refs)
            if not comparison['passed']:raise RuntimeError('CPU initial student does not reproduce the reviewed640 graph')
            parity.append(dict(sample_id=f['sample_id'],comparison=comparison))
    del cpu
    dataset=native_dataset(m,m['pilot_ids']);ped,vehicle=[],[]
    for i,sample in enumerate(m['pilot_ids']):
        inputs,target,original=item(dataset,i)
        source=teacher_inputs(Path(m['dataset'])/'training/image_2'/f'{sample}.png',original['calibration'][0])
        preprocessing_comparison(original,source)
        labels=set(target['labels'].tolist())
        if 0 in labels and len(ped)<8:ped.append(sample)
        elif 0 not in labels and 1 in labels and len(vehicle)<8:vehicle.append(sample)
        if len(ped)==len(vehicle)==8:break
    if len(ped)!=8 or len(vehicle)!=8:raise RuntimeError('Need8 encoded Pedestrian-containing and8 Vehicle-only tiny train frames')
    tiny=[x for pair in zip(ped,vehicle) for x in pair]
    small=native_dataset(m,tiny);model=build_model(m,a.device,True);opt=optimizer(model)
    before=buffers_signature(model);checks=[]
    for index in range(2):
        inputs,target,_=item(small,index,a.device);opt.zero_grad(set_to_none=True)
        output=forward(model,inputs);loss,parts,_=gt_loss(output,target)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite GT smoke loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE['clip_gradient_norm'])
        if not torch.isfinite(norm) or float(norm)<=0:raise RuntimeError('Invalid/zero GT gradients')
        gradients=head_gradients(model)
        opt.step();checks.append(dict(loss=float(loss.detach()),gradient_norm=float(norm),head_gradients=gradients,
                                    components={k:float(v.detach()) for k,v in parts.items()}))
    if buffers_signature(model)!=before:raise RuntimeError('Frozen BN/buffers changed')
    record=dict(complete=True,revision=REVISION,run_signature=m['signature_sha256'],tiny_ids=tiny,
                cpu_initial_parity=parity,finite_nonzero_gradients=True,buffers_unchanged=True,optimizer_steps=2,
                device=a.device,checks=checks,student_architecture_changed=False)
    save_report(a.output/'smoke.json',record)
    print('Smoke passed: all16 initial CPU references, deployed50 queries, two finite gradient steps, frozen buffers.',flush=True)


def atomic_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


def run_steps(m,model,opt,dataset,total_steps,path,role,teacher_cache=None):
    """Deterministic frame order; save model+optimizer+cursor atomically."""
    step=0;history=[];before=buffers_signature(model)
    counts={name:Counter() for name in COMPONENTS}
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['run_signature']!=m['signature_sha256'] or saved['role']!=role:raise RuntimeError('Resume checkpoint identity differs')
        model.load_state_dict(saved['model_state'],strict=True);opt.load_state_dict(saved['optimizer_state'])
        step=saved['step'];history=saved['history'];before=saved['buffers_signature']
        counts={name:Counter(saved['active_kd_counts'][name]) for name in COMPONENTS}
    if step>total_steps:raise RuntimeError('Checkpoint cursor exceeds fixed budget')
    device=str(next(model.parameters()).device);epoch_order=[]
    for cursor in range(step,total_steps):
        epoch,offset=divmod(cursor,len(dataset))
        if not epoch_order or offset==0:
            epoch_order=list(range(len(dataset)));random.Random(RECIPE['seed']+epoch).shuffle(epoch_order)
        index=epoch_order[offset];sample=dataset.idx_list[index];verify_files(m,sample)
        inputs,target,_=item(dataset,index,device);opt.zero_grad(set_to_none=True)
        outputs=forward(model,inputs);gt,parts,_=gt_loss(outputs,target)
        kd=outputs['pred_logits'].sum()*0;weight=0.
        if teacher_cache is not None:
            row=cache_record(m,teacher_cache,sample);teacher=native_arrays(row['outputs'],device)
            if row['target_signature']!=target_signature(target):raise RuntimeError('Native GT encoding changed')
            mapping=torch.tensor(row['object_to_teacher_query'],device=device)
            kd,_,coverage=kd_loss(outputs,teacher,target,mapping)
            for name in COMPONENTS:counts[name].update(coverage[name])
            weight=RECIPE['kd_max_weight']*min(1.,(cursor+1)/RECIPE['kd_warmup_steps'])
        total=gt+weight*kd
        if not torch.isfinite(total):raise RuntimeError(f'Nonfinite {role} loss at step{cursor}')
        total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE['clip_gradient_norm'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite optimization gradients')
        opt.step();step=cursor+1
        if step%32==0 or step==total_steps:
            history.append(dict(step=step,epoch=epoch+1,sample_id=sample,gt=float(gt.detach()),kd=float(kd.detach()),kd_weight=weight,
                                gradient_norm=float(norm),components={k:float(v.detach()) for k,v in parts.items()}))
            print(f'{role} {step}/{total_steps}: GT={float(gt.detach()):.4f} KD={float(kd.detach()):.4f} weight={weight:.3f}',flush=True)
        if step%RECIPE['checkpoint_every_steps']==0 or step==total_steps:
            if buffers_signature(model)!=before:raise RuntimeError('Frozen buffers changed during training')
            atomic_checkpoint(path,dict(run_signature=m['signature_sha256'],role=role,step=step,model_state=model.state_dict(),
                optimizer_state=opt.state_dict(),history=history,buffers_signature=before,
                active_kd_counts={k:dict(v) for k,v in counts.items()},architecture_changed=False))
    return step,history,{k:dict(v) for k,v in counts.items()}


def tiny(a,m):
    if (a.output/'tiny_fit.json').exists():
        old=checked_report(a.output/'tiny_fit.json',m)
        if old['run_signature']!=m['signature_sha256']:raise RuntimeError('Tiny-fit identity differs')
        print('Completed tiny diagnostic reused:',old['passed'],flush=True);return
    setup_device(a.device);checked=checked_report(a.output/'smoke.json',m)
    if not checked['complete'] or checked['run_signature']!=m['signature_sha256']:raise RuntimeError('Passed smoke required')
    dataset=native_dataset(m,checked['tiny_ids']);model=build_model(m,a.device,True)
    initial=endpoint_metrics(model,dataset);step,history,_=run_steps(m,model,optimizer(model),dataset,RECIPE['tiny_steps'],a.output/'tiny/last.pth','tiny')
    final=endpoint_metrics(model,dataset)
    ratio=final['gt_loss']/max(initial['gt_loss'],1e-6)
    geometry=final['geometry_error']/max(initial['geometry_error'],1e-6)
    per_class={name:final['geometry_error_per_class'][name]/max(initial['geometry_error_per_class'][name],1e-6)
               for name in ('Pedestrian','Vehicle')}
    passed=bool(ratio<=RECIPE['tiny_positive_gt_ratio'] and geometry<=RECIPE['tiny_geometry_ratio']
                and max(per_class.values())<=RECIPE['tiny_geometry_ratio']
                and min(final['encoded_gt_counts'].values())>=8)
    record=dict(complete=True,passed=passed,run_signature=m['signature_sha256'],initial=initial,final=final,
                gt_loss_ratio=ratio,geometry_error_ratio=geometry,geometry_ratio_per_class=per_class,
                optimizer_steps=step,history=history,
                evaluation_scope='same16 TRAIN frames only; overfit diagnostic, not validation/generalization',
                tiny_checkpoint_is_pilot_initialization=False,paired_pilot_authorized=passed,deployment_qualified=False)
    save_report(a.output/'tiny_fit.json',record)
    print(json.dumps({k:record[k] for k in ('passed','gt_loss_ratio','geometry_error_ratio','evaluation_scope')},indent=2),flush=True)


def target_signature(target):
    return signature({k:sha_bytes_tensor(v) for k,v in sorted(target.items())})


def cache_record(m,root,sample):
    if sample not in m['pilot_ids']:raise RuntimeError('Teacher cache access is restricted to the selected TRAIN pilot IDs')
    row=json.loads((root/f'{sample}.json').read_text());saved=row.pop('record_sha256')
    if (signature(row)!=saved or row['run_signature']!=m['signature_sha256'] or row['sample_id']!=sample
            or row['source_sha256']!=m['data_file_sha256'][sample]):raise RuntimeError('Teacher cache identity changed')
    return row


def tiny_gate(m):
    value=checked_report(Path(m['output'])/'tiny_fit.json',m)
    if value['run_signature']!=m['signature_sha256'] or not value['complete'] or not value['passed']:
        raise RuntimeError('Tiny learnability check did not pass. Bundle results; do not start paired training.')


def cache(a,m):
    tiny_gate(m);torch.set_num_threads(2);teacher=A2Teacher(m['teacher_export'])
    dataset=native_dataset(m,m['pilot_ids']);root=a.output/'teacher_cache';counts=Counter();encoded=Counter()
    for index,sample in enumerate(m['pilot_ids']):
        verify_files(m,sample);path=root/f'{sample}.json'
        if path.exists():row=cache_record(m,root,sample)
        else:
            _,target,original=item(dataset,index)
            rebuilt=teacher_inputs(Path(m['dataset'])/'training/image_2'/f'{sample}.png',original['calibration'][0])
            preprocessing_comparison(original,rebuilt)
            values=teacher.raw(original);native=native_arrays(values)
            mapping=teacher_matches(native,target)
            row=dict(run_signature=m['signature_sha256'],sample_id=sample,source_sha256=m['data_file_sha256'][sample],
                target_signature=target_signature(target),outputs={k:v.tolist() for k,v in values.items()},
                object_to_teacher_query=mapping.tolist(),target_labels=target['labels'].tolist())
            row['record_sha256']=signature(row);write_json(path,row)
        for label,index_teacher in zip(row['target_labels'],row['object_to_teacher_query']):
            encoded[label]+=1
            if index_teacher>=0:counts[label]+=1
        if (index+1)%32==0 or index+1==len(dataset):print(f'TRAIN-only frozen A2 cache verified {index+1}/{len(dataset)}',flush=True)
    allowed=min(counts[0],counts[1])>=RECIPE['minimum_teacher_matches_per_class']
    report=dict(complete=True,run_signature=m['signature_sha256'],images=len(dataset),
                encoded_gt_counts=dict(Pedestrian=encoded[0],Vehicle=encoded[1]),
                qualified_teacher_matches=dict(Pedestrian=counts[0],Vehicle=counts[1]),
                kd_pilot_authorized=allowed,selection_train_only=True,
                cache_sha256={sample:sha256(root/f'{sample}.json') for sample in m['pilot_ids']})
    save_report(a.output/'teacher_cache_complete.json',report)
    print(json.dumps({k:report[k] for k in ('encoded_gt_counts','qualified_teacher_matches','kd_pilot_authorized')},indent=2),flush=True)


def train(a,m):
    tiny_gate(m);setup_device(a.device)
    cache_root=None
    if a.role=='kd':
        report=checked_report(a.output/'teacher_cache_complete.json',m)
        if not report['complete'] or report['run_signature']!=m['signature_sha256'] or not report['kd_pilot_authorized']:
            raise RuntimeError('Qualified train-only teacher cache required for KD')
        cache_root=a.output/'teacher_cache'
        if {sample:sha256(cache_root/f'{sample}.json') for sample in m['pilot_ids']}!=report['cache_sha256']:raise RuntimeError('Teacher cache bytes changed')
    model=build_model(m,a.device,True);dataset=native_dataset(m,m['pilot_ids'])
    endpoint=a.output/a.role/'last.pth'
    step,history,counts=run_steps(m,model,optimizer(model),dataset,len(dataset)*RECIPE['epochs'],endpoint,a.role,cache_root)
    report=dict(complete=True,run_signature=m['signature_sha256'],role=a.role,checkpoint_sha256=sha256(endpoint),
        optimizer_steps=step,fixed_endpoint_epoch=RECIPE['epochs'],train_images=len(dataset),
        same_initialization_sha256=resolution.CHECKPOINT_SHA,tiny_checkpoint_used=False,
        external_teacher_used=a.role=='kd',history=history,active_kd_counts=counts,
        architecture_changed=False,deployment_qualified=False)
    save_report(a.output/a.role/'training_complete.json',report)


def evaluate(a,m):
    setup_device(a.device);role=a.role
    complete=checked_report(a.output/role/'training_complete.json',m);checkpoint=a.output/role/'last.pth'
    if not complete['complete'] or complete['run_signature']!=m['signature_sha256'] or sha256(checkpoint)!=complete['checkpoint_sha256']:
        raise RuntimeError('Completed fixed-budget endpoint required')
    model=build_model(m,a.device);payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model_state'],strict=True)
    destination=a.output/'evaluation'/role;predictions={};ground_truth={}
    for index,sample in enumerate(m['val_ids']):
        verify_files(m,sample);path=destination/'cache'/f'{sample}.json'
        identity=dict(run_signature=m['signature_sha256'],sample_id=sample,checkpoint_sha256=complete['checkpoint_sha256'])
        row=accuracy.cache_read(path,identity)
        if row is None:
            p=accuracy.parse_kitti_calib_file(Path(m['dataset'])/'training/calib'/f'{sample}.txt')['P2']
            original=teacher_inputs(Path(m['dataset'])/'training/image_2'/f'{sample}.png',p)
            with torch.no_grad():values=tensor_arrays(forward(model,resolution.resize_inputs(original,'r640x192')))
            rows=accuracy.decode_predictions(values,p,original['image_size'][0])
            raw,_=accuracy.prediction_text(rows);ap,bad=accuracy.prediction_text(rows,True)
            row=dict(identity=identity,raw_text=raw,ap_text=ap,invalid_geometry=bad);row['record_sha256']=signature(row);write_json(path,row)
        for kind in ('raw','ap'):accuracy.immutable_bytes(destination/kind/f'{sample}.txt',row[kind+'_text'].encode())
        ground_truth[sample]=accuracy.map_objects([asdict(v) for v in accuracy.parse_kitti_label_file(Path(m['dataset'])/'training/label_2'/f'{sample}.txt')],accuracy.KITTI_PRODUCTION_CLASS_MAPPING)
        if (index+1)%100==0 or index+1==len(m['val_ids']):print(f'{role} VAL endpoint inferred/verified {index+1}/{len(m["val_ids"])}',flush=True)
    for kind in ('raw','ap'):
        raw=accuracy.load_kitti_prediction_directory(destination/kind,m['val_ids'],allowed_classes=list(accuracy.KITTI_PRODUCTION_CLASS_MAPPING))
        predictions[kind]={sample:accuracy.map_objects(rows,accuracy.KITTI_PRODUCTION_CLASS_MAPPING) for sample,rows in raw.items()}
    scored=accuracy.evaluate_kitti_r40(ground_truth,predictions['ap'],['Vehicle','Pedestrian'],accuracy.PRODUCT_IOU_THRESHOLDS,accuracy.PRODUCT_NEIGHBOR_CLASSES)
    metrics={f'{v.class_name.lower()}_{v.metric}_moderate':float(v.ap_r40) for v in scored if v.difficulty=='moderate'}
    metrics['mean_3d_moderate']=(metrics['vehicle_3d_moderate']+metrics['pedestrian_3d_moderate'])/2
    nearby,matched,missed=accuracy.audit(predictions['raw'],ground_truth,.001,.5)
    report=dict(complete=True,run_signature=m['signature_sha256'],checkpoint_sha256=complete['checkpoint_sha256'],
                evaluated_images=len(m['val_ids']),metrics=metrics,all_ap_metrics=[v.to_dict() for v in scored],nearby=nearby,
                accuracy_gate_results={k:metrics[k]>=v for k,v in accuracy.GATES.items()},deployment_qualified=False)
    accuracy.immutable_json(destination/'accuracy.json',report)
    accuracy.write_csv(matched,destination/'matched_geometry.csv');accuracy.write_csv(missed,destination/'false_negatives.csv')


def compare(a,m):
    results={role:json.loads((a.output/'evaluation'/role/'accuracy.json').read_text()) for role in ('control','kd')}
    for role,value in results.items():
        if not value['complete'] or value['evaluated_images']!=3769 or value['run_signature']!=m['signature_sha256']:raise RuntimeError('Both complete fixed-endpoint evaluations required')
    report=dict(complete=True,revision=REVISION,run_signature=m['signature_sha256'],recipe=RECIPE,
        original_untrained_640=m['original_compact_metrics'],results=results,
        kd_minus_control={k:results['kd']['metrics'][k]-v for k,v in results['control']['metrics'].items()},
        control_minus_untrained={k:results['control']['metrics'][k]-v for k,v in m['original_compact_metrics']['metrics'].items()},
        same_initialization_sha256=resolution.CHECKPOINT_SHA,tiny_checkpoint_used=False,
        training_scope='512 train-only stratified frames, five fixed epochs per arm; not a full training run',
        deployment_qualified=False,phone_retest_required=True,full_training_authorized=False,
        interpretation='GT-only gain is recovery; KD benefit requires improvement over matched GT-only control. Review before longer training/export.')
    accuracy.immutable_json(a.output/'recovery_comparison.json',report)
    print(json.dumps({k:report[k] for k in ('kd_minus_control','control_minus_untrained','interpretation')},indent=2),flush=True)


def bundle(a):
    path=a.output/'a2_lite_recovery_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(a.output.rglob('*')):
            if source.is_file() and 'assets' not in source.relative_to(a.output).parts and source.suffix in ('.json','.csv','.txt','.log'):
                z.write(source,source.relative_to(a.output))
    print(path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','smoke','tiny','cache','train','evaluate','compare','bundle'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--role',choices=('control','kd'));p.add_argument('--baseline',type=Path);p.add_argument('--accuracy',type=Path)
    p.add_argument('--compact',type=Path);p.add_argument('--teacher-export',type=Path);p.add_argument('--initialization',type=Path)
    p.add_argument('--repo',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--split-dir',type=Path)
    p.add_argument('--candidate',type=Path,action='append',default=[])
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    for key in ('baseline','accuracy','compact','teacher_export','initialization','repo','dataset','split_dir'):
        if getattr(a,key) is not None:setattr(a,key,getattr(a,key).resolve())
    a.candidate=[path.resolve() for path in a.candidate]
    if a.stage=='prepare' and any(getattr(a,key) is None for key in ('baseline','accuracy','compact','teacher_export','repo','dataset','split_dir')):p.error('prepare requires all source/data paths')
    if a.stage in ('train','evaluate') and a.role is None:p.error('train/evaluate require --role')
    try:
        if a.stage=='prepare':prepare(a)
        elif a.stage=='bundle':bundle(a)
        else:dict(smoke=smoke,tiny=tiny,cache=cache,train=train,evaluate=evaluate,compare=compare)[a.stage](a,load(a.output))
    except Exception as error:
        import traceback
        write_json(a.output/'failures'/f'{a.stage}_{time.time_ns()}.json',dict(complete=False,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
