"""Paired full Chen validation: full-resolution A2-Lite vs its 640x192 graph.

ONNX CPU only. No training, native source clone, checkpoint load or CUDA build.
Stage val data locally, verify fixed16 preprocessing/output contracts, resume
paired per-frame predictions safely, then score with existing product AP/recall.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]

import numpy as np
import torch

from compact_a2_teacher import (teacher_inputs, preprocessing_comparison,
                                decoded_fixture_comparison)
from a2_lite_resolution_onnx import resize_inputs
from m68_onnx_common import (INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES,
                             compare_outputs, read_f32, sha256, signature, write_json)
from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING, map_objects
from data.kitti_parser import parse_kitti_calib_file, parse_kitti_label_file
from data.kitti_prediction_parser import load_kitti_prediction_directory
from data.kitti_r40 import evaluate_kitti_r40, PRODUCT_IOU_THRESHOLDS, PRODUCT_NEIGHBOR_CLASSES
from audit_product_prediction_geometry import audit, write_csv, NEAR_RECALL_GATES
import m64_teacher_qualification as q

REVISION = "A2-LITE-640-PAIRED-ACCURACY-2026-10-12-r1"
MODELS = {
    "full_resolution_control": dict(resource="A2_LITE_ONNX", prefix="export/A2_LITE_ONNX/",
        manifest_sha="b6a424947b8ec0fc427ce27b0f69712648e8b807ce8047d94d29001d24d67f7b",
        signature="bf7580b5a57d782a6e0ced8b4fe02a7181b35edc61800b02371b0d1f8b727cb0",
        model_sha="8e9c8472b03fcece8ed90e873f4b484d10c2e77bc6a2dcecd2f3473a870819a4", hw=[384,1280]),
    "compact_640": dict(resource="A2_LITE_640_ONNX", prefix="exports/r640x192/A2_LITE_640_ONNX/",
        manifest_sha="27b794cc2b6c2de2edcf81469e2b48fa461c17b12a4369676ecf47cf73525a29",
        signature="53a09353c016026c5ef5976fe4168190c6081c2f3b1e7b43dd03dabfed6cbbff",
        model_sha="3b04d26936af7676e026b3f4a5a1922068732144602fa2077722771c783677ac", hw=[192,640]),
}
CHECKPOINT_SHA = "43e0e84d95de649a5cee645d7826baf6bf7ba338895940160ad94bcdd5d3281d"
FIXED_IDS = ["000001","000002","000004","000005","000006","000008","000015","000019",
             "000020","000021","000023","000024","000025","000027","000028","000031"]
GATES = dict(vehicle_3d_moderate=15.8713,pedestrian_3d_moderate=5.1493,mean_3d_moderate=10.5103,
             vehicle_bev_moderate=21.3134,pedestrian_bev_moderate=5.9365)
SCORE_THRESHOLD = .001
VAL_COUNT = q.SPLITS['val'][0]
SERIALIZATION = "native MonoDETR KITTI fields to 2 decimals, including score; class threshold .001 before uncertainty reweighting; top50 class/query pairs; meanshape=False"
INVALID_POLICY = "retain raw predictions; zero-volume dimensions for invalid geometry in AP scoring, never silently drop false positives; nearby recall uses raw serialized 2D boxes"


def implementation():
    names = ["scripts/evaluate_a2_lite_accuracy.py","scripts/compact_a2_teacher.py",
             "scripts/a2_lite_resolution_onnx.py","scripts/m68_onnx_common.py",
             "scripts/audit_product_prediction_geometry.py","data/kitti_r40.py",
             "scripts/m64_teacher_qualification.py","scripts/train_compact_kitti.py",
             "data/kitti_parser.py","data/kitti_prediction_parser.py","data/class_taxonomy.py",
             "data/matching.py"]
    return {name:sha256(ROOT/name) for name in names}


def environment():
    # CPU interpolation/version matters; unrelated GPU allocation/CUDA wheels do not.
    return dict(python=f"{sys.version_info.major}.{sys.version_info.minor}",
                torch=str(torch.__version__).split('+')[0],numpy=np.__version__,
                onnxruntime=importlib.metadata.version('onnxruntime'),
                pillow=importlib.metadata.version('pillow'),
                opencv=__import__('cv2').__version__)


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise RuntimeError(f"Preserve changed record and use a new RUN_ID: {path}")
    else:write_json(path,value)


def immutable_bytes(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError(f"Preserve conflicting file: {path}")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)


def checked_package(source,role,destination):
    spec=MODELS[role];source=Path(source)
    if source.is_dir():
        root=source if source.name==spec['resource'] else source/spec['prefix']
        def read(relative):
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()):raise RuntimeError('Unsafe package path')
            return path.read_bytes()
        raw=read('manifest.json')
    else:
        with zipfile.ZipFile(source) as z:
            entries=[n for n in z.namelist() if n.endswith(spec['resource']+'/manifest.json')]
            if len(entries)!=1:raise RuntimeError('Expected one reviewed package in ZIP')
            prefix=entries[0][:-len('manifest.json')]
            def read(relative):
                p=Path(relative)
                if p.is_absolute() or '..' in p.parts or '\\' in relative:raise RuntimeError('Unsafe ZIP path')
                name=prefix+relative
                if z.namelist().count(name)!=1:raise RuntimeError('Missing/duplicate ZIP asset')
                return z.read(name)
            raw=read('manifest.json');m=json.loads(raw)
            files=['manifest.json',m['model']['file']]
            files += [v['file'] for f in m['fixtures'] for kind in ('inputs','cpu_reference_outputs') for v in f[kind].values()]
            # Validate metadata before copying only its frozen declared assets.
            validate_package(raw,role)
            for relative in files:immutable_bytes(destination/relative,read(relative))
        root=destination
        def read(relative):return (root/relative).read_bytes()
    m=validate_package(raw,role)
    if sha256(root/m['model']['file'])!=spec['model_sha']:raise RuntimeError('Changed model bytes')
    for f in m['fixtures']:
        for kind in ('inputs','cpu_reference_outputs'):
            for record in f[kind].values():read_f32(root,record)
    return root,m


def validate_package(raw,role):
    spec=MODELS[role];m=json.loads(raw)
    if (sha_bytes(raw)!=spec['manifest_sha'] or m['manifest_sha256']!=spec['signature']
            or signature({k:v for k,v in m.items() if k!='manifest_sha256'})!=spec['signature']
            or m['model']['sha256']!=spec['model_sha'] or m['checkpoint_sha256']!=CHECKPOINT_SHA
            or not m['complete'] or m['providers']!=['CPUExecutionProvider'] or m['precision']!='FP32'
            or m['sample_ids']!=FIXED_IDS or [f['sample_id'] for f in m['fixtures']]!=FIXED_IDS):
        raise RuntimeError('Use the exact reviewed A2-Lite export packages')
    return m


def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def session(root,package,role):
    import onnxruntime as ort
    if ort.__version__!='1.30.0':raise RuntimeError('Use ONNX Runtime1.30.0, matching the phone')
    options=ort.SessionOptions();options.intra_op_num_threads=2
    s=ort.InferenceSession(str(root/package['model']['file']),options,providers=['CPUExecutionProvider'])
    shapes={**INPUT_SHAPES,'image':(1,3,*MODELS[role]['hw'])}
    if (s.get_providers()!=['CPUExecutionProvider'] or {v.name:tuple(v.shape) for v in s.get_inputs()}!=shapes
            or {v.name:tuple(v.shape) for v in s.get_outputs()}!=OUTPUT_SHAPES):raise RuntimeError('Wrong CPU tensor contract')
    return s


def stage_data(dataset,candidates,ids):
    if not dataset.name.startswith('kitti_a2_lite_accuracy_'):raise RuntimeError('Use a dedicated kitti_a2_lite_accuracy_<RUN_ID> folder')
    sources={};errors=[]
    for name,suffix in (('image_2','.png'),('calib','.txt'),('label_2','.txt')):
        aliases=[name]+([name[:-1]+'02'] if name in ('image_2','label_2') else [])
        choices=[root/'training'/alias for root in [dataset,*candidates] for alias in aliases]
        source=next((p for p in choices if p.is_dir() and all((p/(i+suffix)).is_file() for i in ids)),None)
        if source is None:errors.append(f'Missing complete val {name}; candidates: {choices}')
        else:sources[name]=source
    if errors:raise RuntimeError('Data preflight failed; no files copied:\n'+'\n'.join(errors))
    records={}
    for index,sample in enumerate(ids):
        records[sample]={}
        for name,suffix in (('image_2','.png'),('calib','.txt'),('label_2','.txt')):
            source=sources[name]/(sample+suffix);dest=dataset/'training'/name/source.name
            data=source.read_bytes();records[sample][name]=sha_bytes(data)
            if source.resolve()!=dest.resolve():immutable_bytes(dest,data)
        if (index+1)%100==0 or index+1==len(ids):print(f'Val data staged/hashed {index+1}/{len(ids)}',flush=True)
    return records


def decode_predictions(values,projection,image_size,threshold=SCORE_THRESHOLD):
    """CPU reproduction of native extract/decode operation order, meanshape=False.

    In particular, corners -> cxcywh -> pixel corners is intentionally retained:
    simplifying that arithmetic changes native two-decimal rounding at ties.
    """
    for name in OUTPUT_NAMES:
        a=np.asarray(values[name])
        if a.shape!=OUTPUT_SHAPES[name] or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise RuntimeError(f'Invalid FP32 {name} output')
    tensors={name:torch.from_numpy(value) for name,value in values.items()}
    scores,indexes=torch.topk(tensors['logits'].sigmoid().view(1,-1),50,dim=1)
    queries=(indexes//3)[0].numpy();classes=(indexes%3)[0].numpy()
    boxes=tensors['boxes'][0,queries]
    xc,yc,left,right,top,bottom=boxes.unbind(-1)
    x0,y0,x1,y1=xc-left,yc-top,xc+right,yc+bottom
    xywh=torch.stack(((x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0),dim=-1).numpy()
    confidence=scores[0].numpy()
    uncertainty=torch.exp(-tensors['depth'][0,queries,1]).numpy()
    p=np.asarray(projection,np.float32);width,height=np.asarray(image_size,np.float32)
    rows=[]
    for j,(query,cls) in enumerate(zip(queries,classes)):
        if cls==2 or confidence[j]<threshold:continue
        x,y,w,h=xywh[j]*np.asarray([width,height,width,height],np.float32)
        box=np.asarray([x-w/2,y-h/2,x+w/2,y+h/2])
        dimensions=values['dimensions'][0,query].copy()
        depth=values['depth'][0,query,0]
        u=values['boxes'][0,query,0]*width;v=values['boxes'][0,query,1]*height
        location=np.asarray([((u-p[0,2])*depth)/p[0,0]-p[0,3]/p[0,0],
                             ((v-p[1,2])*depth)/p[1,1]-p[1,3]/p[1,1],depth])
        location[1]+=dimensions[0]/2
        heading=values['angle'][0,query];bin_id=np.argmax(heading[:12])
        alpha=bin_id*(2*np.pi/12.)+heading[12+bin_id]
        if alpha>np.pi:alpha-=2*np.pi
        yaw=alpha+np.arctan2(x-p[0,2],p[0,0])
        if yaw>np.pi:yaw-=2*np.pi
        if yaw<-np.pi:yaw+=2*np.pi
        score=confidence[j]*uncertainty[j]
        if not np.isfinite([*box,*dimensions,*location,alpha,yaw,score]).all():
            raise RuntimeError('Nonfinite native decoded geometry/score')
        invalid=bool(min(dimensions)<=0 or depth<=.1 or (box[2:]<=box[:2]).any())
        rows.append(dict(class_name='Pedestrian' if cls==0 else 'Vehicle',score=float(score),
            query_index=int(query),class_confidence=float(confidence[j]),bbox_2d=box.tolist(),
            dimensions_3d=dimensions.tolist(),location_3d=location.tolist(),alpha=float(alpha),
            rotation_y=float(yaw),invalid_geometry=invalid))
    return rows


def fixed16(root,package,s,dataset,role,parent=None):
    checks=[]
    for f in package['fixtures']:
        sample=f['sample_id'];p=parse_kitti_calib_file(dataset/'training/calib'/f'{sample}.txt')['P2']
        original=teacher_inputs(dataset/'training/image_2'/f'{sample}.png',p)
        rebuilt=resize_inputs(original,'r640x192') if role=='compact_640' else original
        saved={n:read_f32(root,f['inputs'][n]) for n in INPUT_NAMES}
        if role=='compact_640':
            if parent is None:raise RuntimeError('Compact preprocessing requires signed full-resolution parent fixtures')
            parent_root,parent_package=parent
            parent_fixture=next(v for v in parent_package['fixtures'] if v['sample_id']==sample)
            signed_parent={n:read_f32(parent_root,parent_fixture['inputs'][n]) for n in INPUT_NAMES}
            parent_check=preprocessing_comparison(original,signed_parent)
            resized_parent=resize_inputs(signed_parent,'r640x192')
            resize_deltas={n:float(np.abs(resized_parent[n]-saved[n]).max()) for n in INPUT_NAMES}
            if any(v>1e-6 for v in resize_deltas.values()):raise RuntimeError('Signed parent resize does not reproduce compact fixture')
            # Interpolation spreads sparse 8-bit rounding noise to adjacent pixels.
            # Prove the original sparse bound and exact signed resize separately.
            preprocessing=dict(parent=parent_check,signed_resize_max_abs=resize_deltas,passed=True,
                reconstructed_image_max_abs=float(np.abs(rebuilt['image']-saved['image']).max()))
        else:preprocessing=preprocessing_comparison(rebuilt,saved)
        refs={n:read_f32(root,f['cpu_reference_outputs'][n]) for n in OUTPUT_NAMES}
        actual=dict(zip(OUTPUT_NAMES,s.run(list(OUTPUT_NAMES),saved)))
        raw=compare_outputs(actual,refs)
        if not raw['passed']:raise RuntimeError('Saved-input raw parity failed')
        recreated=dict(zip(OUTPUT_NAMES,s.run(list(OUTPUT_NAMES),rebuilt)))
        decoded=decoded_fixture_comparison(recreated,refs,p,original['image_size'][0])
        checks.append(dict(sample_id=sample,preprocessing=preprocessing,raw=raw,decoded=decoded))
    return checks


def prepare(args):
    ids=q.split_ids(args.split_dir/'val.txt','val')
    torch.set_num_threads(2)
    validated={}
    for role,source in (('full_resolution_control',args.baseline),('compact_640',args.compact)):
        if source.is_dir() and args.output.resolve().is_relative_to(source.resolve()):
            raise RuntimeError('Use a separate accuracy output, never an export source folder')
        root,p=checked_package(source,role,args.output/'assets'/role)
        validated[role]=(root,p)
    files=stage_data(args.dataset,args.candidate,ids)
    packages={}
    for role,(root,p) in validated.items():
        s=session(root,p,role);checks=fixed16(root,p,s,args.dataset,role,validated['full_resolution_control'])
        packages[role]=dict(root=str(root.resolve()),manifest_sha256=MODELS[role]['manifest_sha'],
                            model_sha256=MODELS[role]['model_sha'],fixed16_checks=checks)
    value=dict(revision=REVISION,dataset=str(args.dataset.resolve()),split_dir=str(args.split_dir.resolve()),
        sample_ids=ids,data_file_sha256=files,models=packages,implementation=implementation(),environment=environment(),
        class_mapping=KITTI_PRODUCTION_CLASS_MAPPING,score_threshold=SCORE_THRESHOLD,topk=50,
        serialization=SERIALIZATION,invalid_geometry_policy=INVALID_POLICY,near_iou_threshold=.5,
        training_performed=False,optimizer_steps=0)
    value['signature_sha256']=signature(value)
    immutable_json(args.output/'a2_lite_accuracy_manifest.json',value)
    print('Paired accuracy evaluation prepared: both fixed16 pipelines passed; no training.',flush=True)


def load(output):
    m=json.loads((output/'a2_lite_accuracy_manifest.json').read_text())
    if (m['signature_sha256']!=signature({k:v for k,v in m.items() if k!='signature_sha256'})
            or m['revision']!=REVISION or m['implementation']!=implementation() or m['environment']!=environment()
            or q.split_ids(Path(m['split_dir'])/'val.txt','val')!=m['sample_ids']
            or set(m['models'])!=set(MODELS) or set(m['data_file_sha256'])!=set(m['sample_ids'])
            or m['class_mapping']!=KITTI_PRODUCTION_CLASS_MAPPING or m['topk']!=50
            or m['score_threshold']!=SCORE_THRESHOLD or m['serialization']!=SERIALIZATION
            or m['invalid_geometry_policy']!=INVALID_POLICY):
        raise RuntimeError('Run code/CPU environment/split changed; preserve outputs and use a new RUN_ID')
    for role,record in m['models'].items():
        root=Path(record['root']);p=validate_package((root/'manifest.json').read_bytes(),role)
        if sha256(root/p['model']['file'])!=MODELS[role]['model_sha']:raise RuntimeError('Model changed')
    return m


def prediction_text(rows,zero_invalid=False):
    lines=[];invalid=0
    for row in rows:
        if row['class_name'] not in ('Vehicle','Pedestrian'):raise RuntimeError('Unexpected decoded product class')
        dimensions=list(row['dimensions_3d'])
        bad=bool(row['invalid_geometry'] or min(round(v,2) for v in dimensions)<=0
                 or round(row['location_3d'][2],2)<=.1)
        invalid+=int(bad)
        if zero_invalid and bad:dimensions=[0.,0.,0.]
        name='Car' if row['class_name']=='Vehicle' else 'Pedestrian'
        values=[row['alpha'],*row['bbox_2d'],*dimensions,*row['location_3d'],row['rotation_y'],row['score']]
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite decoded prediction')
        lines.append(name+' 0.0 0'+''.join(f' {float(v):.2f}' for v in values)+'\n')
    return ''.join(lines),invalid


def cache_read(path,identity):
    if not path.exists():return None
    value=json.loads(path.read_text());saved=value.pop('record_sha256',None)
    if saved!=signature(value) or value.get('identity')!=identity:raise RuntimeError(f'Changed paired cache: {path}; preserve it')
    return value


def infer(args):
    m=load(args.output);torch.set_num_threads(2);dataset=Path(m['dataset'])
    sessions={}
    for role,record in m['models'].items():
        root=Path(record['root']);p=json.loads((root/'manifest.json').read_text());sessions[role]=session(root,p,role)
    for index,sample in enumerate(m['sample_ids']):
        current={name:sha256(dataset/'training'/name/(sample+('.png' if name=='image_2' else '.txt')))
                 for name in ('image_2','calib','label_2')}
        if current!=m['data_file_sha256'][sample]:raise RuntimeError(f'Val input changed: {sample}')
        identity=dict(run_signature=m['signature_sha256'],sample_id=sample,input_sha256=current)
        path=args.output/'cache'/f'{sample}.json';record=cache_read(path,identity)
        if record is None:
            p=parse_kitti_calib_file(dataset/'training/calib'/f'{sample}.txt')['P2']
            original=teacher_inputs(dataset/'training/image_2'/f'{sample}.png',p);predictions={}
            for role,s in sessions.items():
                data=resize_inputs(original,'r640x192') if role=='compact_640' else original
                values=dict(zip(OUTPUT_NAMES,s.run(list(OUTPUT_NAMES),data)))
                rows=decode_predictions(values,p,original['image_size'][0])
                raw,_=prediction_text(rows);ap,invalid=prediction_text(rows,True)
                predictions[role]=dict(raw_text=raw,ap_text=ap,invalid_geometry_count=invalid)
            record=dict(identity=identity,predictions=predictions);record['record_sha256']=signature(record)
            write_json(path,record)
        for role,value in record['predictions'].items():
            for kind in ('raw','ap'):
                immutable_bytes(args.output/'predictions'/role/kind/f'{sample}.txt',value[kind+'_text'].encode())
        if (index+1)%100==0 or index+1==len(m['sample_ids']):print(f"Paired inferred/verified {index+1}/{len(m['sample_ids'])}",flush=True)
    if {p.stem for p in (args.output/'cache').glob('*.json')}!=set(m['sample_ids']):raise RuntimeError('Incomplete/extra paired cache IDs')
    complete=dict(complete=True,run_signature=m['signature_sha256'],images=len(m['sample_ids']),
                  prediction_files={role:{kind:{sample:sha256(args.output/'predictions'/role/kind/f'{sample}.txt')
                      for sample in m['sample_ids']} for kind in ('raw','ap')} for role in MODELS})
    immutable_json(args.output/'prediction_complete.json',complete)


def evaluate(args):
    m=load(args.output);dataset=Path(m['dataset']);ids=m['sample_ids']
    done=json.loads((args.output/'prediction_complete.json').read_text())
    if (not done['complete'] or len(ids)!=VAL_COUNT or done['images']!=len(ids)
            or done['run_signature']!=m['signature_sha256']):raise RuntimeError('Complete paired predictions required')
    for sample in ids:
        if sha256(dataset/'training/label_2'/f'{sample}.txt')!=m['data_file_sha256'][sample]['label_2']:
            raise RuntimeError(f'Ground truth changed after inference: {sample}')
    gt={sample:map_objects([asdict(obj) for obj in parse_kitti_label_file(dataset/'training/label_2'/f'{sample}.txt')],
                          KITTI_PRODUCTION_CLASS_MAPPING) for sample in ids}
    results={}
    for role in MODELS:
        predictions={}
        for kind in ('raw','ap'):
            directory=args.output/'predictions'/role/kind
            if {p.stem for p in directory.glob('*.txt')}!=set(ids):raise RuntimeError('Missing/extra predictions')
            if {sample:sha256(directory/f'{sample}.txt') for sample in ids}!=done['prediction_files'][role][kind]:raise RuntimeError('Prediction bytes changed')
            raw=load_kitti_prediction_directory(directory,ids,allowed_classes=list(KITTI_PRODUCTION_CLASS_MAPPING))
            predictions[kind]={sample:map_objects(rows,KITTI_PRODUCTION_CLASS_MAPPING) for sample,rows in raw.items()}
        print(f'Scoring {role}: product AP_R40 and nearby recall on all {len(ids)} images…',flush=True)
        scored=evaluate_kitti_r40(gt,predictions['ap'],['Vehicle','Pedestrian'],PRODUCT_IOU_THRESHOLDS,PRODUCT_NEIGHBOR_CLASSES)
        metrics={f'{r.class_name.lower()}_{r.metric}_moderate':float(r.ap_r40) for r in scored if r.difficulty=='moderate'}
        metrics['mean_3d_moderate']=(metrics['vehicle_3d_moderate']+metrics['pedestrian_3d_moderate'])/2
        nearby,matched,missed=audit(predictions['raw'],gt,SCORE_THRESHOLD,.5)
        invalid=sum(cache_read(args.output/'cache'/f'{sample}.json',dict(run_signature=m['signature_sha256'],sample_id=sample,
                    input_sha256=m['data_file_sha256'][sample]))['predictions'][role]['invalid_geometry_count'] for sample in ids)
        results[role]=dict(metrics=metrics,all_ap_metrics=[r.to_dict() for r in scored],nearby=nearby,
            invalid_geometry_predictions=invalid,accuracy_gates=GATES,gate_results={k:metrics[k]>=v for k,v in GATES.items()},
            all_accuracy_gates_passed=all(metrics[k]>=v for k,v in GATES.items()))
        destination=args.output/'evaluation'/role
        immutable_json(destination/'accuracy.json',results[role])
        write_csv(matched,destination/'matched_geometry.csv');write_csv(missed,destination/'false_negatives.csv')
    control,candidate=results['full_resolution_control'],results['compact_640']
    report=dict(complete=True,revision=REVISION,run_signature=m['signature_sha256'],evaluated_images=len(ids),
        protocol='Chen product-taxonomy KITTI-difficulty AP_R40; not official leaderboard',models=results,
        compact_minus_control_ap={k:candidate['metrics'][k]-v for k,v in control['metrics'].items()},
        compact_minus_control_near_recall={k:candidate['nearby']['classes'][k]['near_recall']-control['nearby']['classes'][k]['near_recall'] for k in NEAR_RECALL_GATES},
        historical_full_A2_context=dict(metrics=q.BASELINE,nearby=q.BASELINE_NEARBY,not_recomputed_in_this_run=True),
        historical_context_not_a_paired_pruning_ablation=True,serialization=SERIALIZATION,invalid_geometry_policy=INVALID_POLICY,
        nearby_definition='2D IoU>=0.5, serialized score>=0.001; GT range Vehicle<40m, Pedestrian<30m; not 3D safety qualification',
        training_performed=False,optimizer_steps=0,phone_runtime_measured_in_this_run=False,distillation_authorized=False,
        deployment_qualified=False,next_step='Review the accuracy/recall gap before choosing recovery training or selective A2 distillation')
    immutable_json(args.output/'a2_lite_accuracy_comparison.json',report)
    print(json.dumps({k:report[k] for k in ('compact_minus_control_ap','compact_minus_control_near_recall')},indent=2),flush=True)


def bundle(args):
    target=args.output/'a2_lite_accuracy_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path!=target and 'assets' not in path.relative_to(args.output).parts and path.suffix in ('.json','.csv','.txt','.log'):
                z.write(path,path.relative_to(args.output))
    print(target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','infer','evaluate','bundle'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path)
    p.add_argument('--split-dir',type=Path);p.add_argument('--candidate',type=Path,action='append',default=[])
    p.add_argument('--baseline',type=Path);p.add_argument('--compact',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    for key in ('dataset','split_dir','baseline','compact'):
        if getattr(a,key) is not None:setattr(a,key,getattr(a,key).resolve())
    a.candidate=[path.resolve() for path in a.candidate]
    if a.stage=='prepare' and any(getattr(a,k) is None for k in ('dataset','split_dir','baseline','compact')):p.error('prepare requires dataset, split-dir, baseline and compact')
    try:dict(prepare=prepare,infer=infer,evaluate=evaluate,bundle=bundle)[a.stage](a)
    except Exception as error:
        import traceback
        write_json(a.output/'failures'/f'{a.stage}_{time.time_ns()}.json',dict(complete=False,error=str(error),traceback=traceback.format_exc(),training_performed=False))
        raise


if __name__=='__main__':main()
