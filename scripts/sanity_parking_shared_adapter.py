"""Tiny fixed-train-subset adapter fit check, never a deployment candidate.

Freeze the complete reviewed epoch-7 depth head including every BN buffer.
Train only the new 1553-parameter camera adapter. No dev/held-out evaluation,
teacher, source-run edit, automatic larger training, export or phone change.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import train_parking_shared_camera as camera

CONFIG = ROOT/'configs/parking_shared_adapter_sanity_v1.json'


def configuration():
    return json.loads(CONFIG.read_text())


def select_training_samples(samples, config, cells):
    """Do not inspect dev images, predictions or per-image errors for selection."""
    rows = [s for s in samples if s['role']=='train']
    ranked = sorted(rows, key=lambda s:(common.sha(f"{config['seed']}:{s['camera_token']}".encode()),s['camera_token']))
    selected = []; used = set(); assignments = []
    for _ in range(config['unique_images_per_critical_pool']):
        for cell in cells:
            s = next((s for s in ranked if s['camera_token'] not in used and s['counts'][cell]>=5),None)
            if s is None:
                raise ValueError('Insufficient unique training-only images for the sanity subset')
            selected.append(s); used.add(s['camera_token']); assignments.append(cell)
    if len(selected)!=config['images'] or len(used)!=len(selected):
        raise ValueError('Fixed sanity-subset budget changed')
    return selected,assignments


def batch_indices(count, config):
    import numpy as np
    if count!=config['images'] or count%config['batch_size']:
        raise ValueError('Use the fixed complete tiny-subset batches')
    stream = []; cycle = 0
    while len(stream)<config['updates_per_arm']:
        order = np.random.default_rng(config['seed']+cycle).permutation(count).tolist()
        stream.extend(order[i:i+config['batch_size']] for i in range(0,count,config['batch_size']))
        cycle += 1
    return stream[:config['updates_per_arm']]


def freeze_base(decoder):
    decoder.base.requires_grad_(False)
    decoder.base.eval()
    decoder.base.zero_grad(set_to_none=True)
    return [p for name,p in decoder.named_parameters() if not name.startswith('base.') and p.requires_grad]


def frozen_check(decoder, expected):
    from scripts.screen_parking_shared_model import state_digest
    if (state_digest(decoder.base)!=expected or decoder.base.training
            or any(p.requires_grad or p.grad is not None for p in decoder.base.parameters())
            or any(m.training for m in decoder.base.modules())):
        raise ValueError('Original depth weights/BN/gradient protection failed')


def predict(decoder, rows, samples, arm, device):
    features,grid,truth,valid = camera.pilot.batch_arrays(rows,device)
    prediction = camera.pilot.sample_depth(decoder(features,camera.camera_batch(samples,arm,device)),grid)
    return prediction,truth,valid


def tiny_metrics(decoder, rows, samples, arm, device, mapping, config, initial_predictions):
    import numpy as np
    import torch
    decoder.eval(); cells={k:[] for k in config['critical_cells']}; broad=[]; changes={k:[] for k in cells}
    objective = []; all_predictions=[]
    with torch.no_grad():
        for row,s,baseline in zip(rows,samples,initial_predictions):
            prediction,truth,valid = predict(decoder,[row],[s],arm,device)
            value = prediction[0].cpu().numpy()
            if not np.isfinite(value).all() or not (value>0).all():raise ValueError('Invalid positive metre predictions')
            all_predictions.append(value)
            error=np.abs(value.astype(np.float64)-row['z'])
            shift=np.abs(value.astype(np.float64)-baseline)
            broad.append(float(error[row['conservative']].mean()))
            masks = camera.data.cell_masks(row,mapping)
            for key,mask in masks.items():
                if int(mask.sum())>=config['minimum_points_in_cell_image']:
                    cells[key].append(float(error[mask].mean()));changes[key].append(float(shift[mask].mean()))
        # Same equal-image/equal-cell objective over the entire fixed subset,
        # rather than averaging changing mini-batch cell availability.
        prediction,truth,valid=predict(decoder,rows,samples,arm,device)
        loss=camera.near.balanced_loss(prediction,truth,valid,camera.masks_for(rows,valid,mapping,config),config,'parking_balanced')
    if any(not v for v in cells.values()):raise ValueError('Tiny subset lacks a critical training cell')
    return {'loss':float(loss.cpu()),'equal_four_cell_mae_m':float(np.mean([np.mean(v) for v in cells.values()])),
        'cells':{k:{'images':len(v),'macro_image_mae_m':float(np.mean(v))} for k,v in cells.items()},
        'broad_macro_image_mae_m':float(np.mean(broad)),
        'equal_four_cell_prediction_change_m':float(np.mean([np.mean(v) for v in changes.values()]))},all_predictions


def adapter_step(decoder,optimizer,rows,samples,arm,device,mapping,config,frozen):
    import torch
    decoder.train();decoder.base.eval();frozen_check(decoder,frozen)
    optimizer.zero_grad(set_to_none=True)
    prediction,truth,valid=predict(decoder,rows,samples,arm,device)
    loss=camera.near.balanced_loss(prediction,truth,valid,camera.masks_for(rows,valid,mapping,config),config,'parking_balanced')
    if not torch.isfinite(loss):raise ValueError('Non-finite adapter sanity loss')
    loss.backward()
    params=[p for n,p in decoder.named_parameters() if not n.startswith('base.')]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/non-finite adapter gradient')
    gradients={name:float(sum(p.grad.detach().double().square().sum() for n,p in decoder.named_parameters()
        if n.startswith(name+'.')).sqrt().cpu()) for name in ['appearance','fusion','spatial','residual']}
    norm=torch.nn.utils.clip_grad_norm_(params,config['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step();frozen_check(decoder,frozen)
    if any(not torch.isfinite(p).all() for p in params):raise ValueError('Non-finite adapter weights')
    return {'batch_loss':float(loss.detach().cpu()),'pre_clip_gradient_norm':float(norm.cpu()),'gradient_l2':gradients}


def run(a):
    import numpy as np
    import torch
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts.screen_parking_shared_model import state_digest
    manifest=camera.data.read_manifest(a.inputs)
    if common.file_sha(a.inputs/'input_manifest.json')!=camera.INPUT_SHA:raise ValueError('Reviewed input package changed')
    _,_,warm=camera.read_warmstart(a.warmstart,load=True)
    config={**manifest['config'],**configuration()}
    samples,assignments=select_training_samples(manifest['samples'],config,config['critical_cells'])
    for path in [a.output,a.cache]:
        if path.is_symlink() or any(path.resolve().is_relative_to(p.resolve()) for p in [a.inputs,a.source,a.weights,a.warmstart.parent]):
            raise ValueError('Use separate new sanity output/cache folders')
    if (a.output/'run_identity.json').exists():
        raise ValueError('Preserve the completed/interrupted sanity run; use a new output folder rather than mixing or extending budgets')
    (a.output/'runtime_settings').mkdir(parents=True,exist_ok=True)
    native,model_id=camera.pilot.build_model(a,manifest['config'])
    verification=camera.body_verification.verify_warmstart_body(native,model_id,manifest['config'],a.weights)
    native.depth_decoder.load_state_dict(warm['decoder'],strict=True);native.eval()
    device=next(native.parameters()).device
    frozen_body=state_digest(native.body);frozen_depth=state_digest(native.depth_decoder)
    cache_id=camera.cache_identity(a.inputs/'input_manifest.json',native,model_id)
    common.immutable(a.cache/'cache_identity.json',common.encoded(cache_id))
    rows=[]
    for i,s in enumerate(samples,1):
        camera.pilot.cache_sample(native,s,a.inputs,a.cache,device)
        rows.append(camera.pilot.read_features(s,a.cache))
        print(f'Train-only frozen features: {i}/{len(samples)}',flush=True)
    assert state_digest(native.body)==frozen_body
    torch.manual_seed(config['seed'])
    decoder=CameraAwareDepthDecoder(copy.deepcopy(native.depth_decoder)).to(device).eval()
    params=freeze_base(decoder)
    if sum(p.numel() for p in params)!=1553:raise ValueError('Only the 1553 adapter parameters may train')
    initial={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
    sources=[Path(__file__),CONFIG,ROOT/'models/parking_shared_camera.py',ROOT/'models/parking_shared.py',
        Path(camera.__file__),Path(camera.pilot.__file__),Path(camera.near.__file__)]
    identity={'revision':config['revision'],'config':config,'input_manifest_sha256':camera.INPUT_SHA,
        'warmstart_sha256':camera.CHECKPOINT_SHA,'runtime':model_id['runtime'],'backbone_verification':verification,
        'frozen_body_sha256':frozen_body,'frozen_depth_sha256':frozen_depth,
        'initial_adapter_state_sha256':state_digest(decoder),'subset':samples,'critical_pool_assignments':assignments,
        'source_sha256':{str(p.relative_to(ROOT)):common.file_sha(p) for p in sources}}
    common.immutable(a.output/'run_identity.json',common.encoded(identity));run_id=common.identity(identity)
    for p in sources:common.immutable(a.output/'source_snapshot'/p.relative_to(ROOT),p.read_bytes())
    # Same original predictions for both arms, without gradients or new fitting.
    baseline=[]
    with torch.no_grad():
        for row,s in zip(rows,samples):
            features,grid,_,_=camera.pilot.batch_arrays([row],device)
            native_value=camera.pilot.sample_depth(native.depth_decoder(features),grid)[0]
            for arm in config['arms']:
                value=decoder(features,camera.camera_batch([s],arm,device))
                if not torch.equal(native_value,camera.pilot.sample_depth(value,grid)[0]):raise ValueError('Initial parity failed')
            baseline.append(native_value.cpu().numpy())
    stream=batch_indices(len(samples),config);results={}
    for arm in config['arms']:
        decoder.load_state_dict(initial,strict=True);params=freeze_base(decoder)
        optimizer=torch.optim.AdamW(params,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in params}:
            raise ValueError('Optimizer parameter identity changed')
        initial_metrics,_=tiny_metrics(decoder,rows,samples,arm,device,manifest['mapping'],config,baseline)
        history=[{'update':0,'metrics':initial_metrics}];gradients=[];began=time.perf_counter()
        arm_id=common.identity({'run_identity':run_id,'arm':arm})
        camera.pilot.save_checkpoint(a.output/arm/'update_000.pth',{'run_identity':arm_id,'arm':arm,'updates':0,'decoder':initial})
        for update,indices in enumerate(stream,1):
            info=adapter_step(decoder,optimizer,[rows[i] for i in indices],[samples[i] for i in indices],
                arm,device,manifest['mapping'],config,frozen_depth)
            if update in [1,2,10,25,100,300]:gradients.append({'update':update,**info})
            if update%config['evaluate_every_updates']==0 or update==config['updates_per_arm']:
                metrics,_=tiny_metrics(decoder,rows,samples,arm,device,manifest['mapping'],config,baseline)
                history.append({'update':update,'metrics':metrics})
                print(json.dumps({'arm':arm,'update':update,**metrics,'seconds':time.perf_counter()-began}),flush=True)
        frozen_check(decoder,frozen_depth)
        if state_digest(native.body)!=frozen_body:raise ValueError('Frozen Urban body changed')
        final=history[-1]['metrics']
        gates={'training_loss_reduced_at_least_10_percent':final['loss']<=(1-config['minimum_relative_training_loss_reduction'])*initial_metrics['loss'],
            'training_mae_reduced_at_least_10_percent':final['equal_four_cell_mae_m']<=(1-config['minimum_relative_training_mae_reduction'])*initial_metrics['equal_four_cell_mae_m'],
            'predictions_changed_at_least_5cm':final['equal_four_cell_prediction_change_m']>=config['minimum_macro_image_prediction_change_m'],
            'base_weights_and_bn_exactly_frozen':state_digest(decoder.base)==frozen_depth}
        camera.pilot.save_checkpoint(a.output/arm/f"update_{config['updates_per_arm']:03d}.pth",{'run_identity':arm_id,
            'arm':arm,'updates':config['updates_per_arm'],'decoder':{k:v.detach().cpu() for k,v in decoder.state_dict().items()}})
        results[arm]={'updates':config['updates_per_arm'],'initial':initial_metrics,'final':final,'history':history,
            'gradient_probes':gradients,'sanity_checks':gates,'all_sanity_checks_passed':all(gates.values()),
            'seconds':time.perf_counter()-began,'base_state_unchanged':True}
    result={'complete':True,'run_identity':run_id,'training_images':len(samples),'arms':results,
        'scope':config['scope'],'training_performed':True,'optimizer_steps_per_arm':config['updates_per_arm'],
        'trainable_parameters':1553,'frozen_depth_parameters':sum(p.numel() for p in decoder.base.parameters()),
        'frozen_body_and_depth_bn_unchanged':True,'development_used':False,'heldout_used':False,
        'knowledge_distillation':False,'larger_run_authorized':False,'phone_app_changed':False,'deployment_qualified':False,
        'next_step':'Review train-only fit and per-layer gradient/correction scales before proposing any larger development pilot; do not promote the overfit diagnostic weights'}
    common.immutable(a.output/'adapter_sanity_results.json',common.encoded(result))
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','warmstart','source','weights','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    run(p.parse_args())


if __name__=='__main__':main()
