"""Export and validate the reviewed full-data GT-only A2-Lite endpoint.

No training, CUDA extension, M67 runtime restoration, or teacher is needed.
Completed experiments are read-only. A successful export is not deployment
qualification: full validation must preserve the recovered model's accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
REVISION = 'A2-LITE-TRAINED-GT-ONNX-2026-10-15-r1'
RUN_SIGNATURE = 'b5239955208c794cd3f0ab9bc5564b2d656a9fcec1397212a33cfb0a94890cab'
CHECKPOINT_SHA = '0cd8dd0d0c2e1e6acb5980806084264fa83eb588b1bde763746b14d47e176930'
REVIEWED = {
    'full_recovery_manifest.json': '94216f852fa40a8d18dc38075da6c06348b9887a8bf1ce1ec3dd10d219a99d4e',
    'control/training_complete.json': 'dd58f45c490044cb961072ed40bbae6a83b9f049fa1cf18876d9e0e8a1062572',
    'evaluation/control/accuracy.json': 'a93bc768aa52850013df691402e3147287400d874d9996800a103a1c43e0c176',
}
# Export-preservation tolerances, NOT relaxed product-accuracy targets.
AP_DRIFT = .05
RECALL_DRIFT = .005
RESOURCE = 'A2_LITE_TRAINED_ONNX'
MODEL_FILE = 'A2_Lite_GT_Epoch10_640x192_FP32.onnx'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_record(source, name):
    source = Path(source)
    if source.is_dir():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise RuntimeError('Unsafe source path')
        return path.read_bytes()
    with zipfile.ZipFile(source) as z:
        candidates = [n for n in z.namelist() if n == name or n.endswith('/' + name)]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one reviewed record: {name}')
        return z.read(candidates[0])


def checked_signature(value, field):
    if value.get(field) != signature({k: v for k, v in value.items() if k != field}):
        raise RuntimeError(f'Invalid {field}')
    return value


def reviewed(source):
    values = {}
    for name, expected in REVIEWED.items():
        raw = read_record(source, name)
        if digest(raw) != expected:
            raise RuntimeError(f'Use the exact reviewed full-data GT run: {name}')
        values[name] = json.loads(raw)
    m, t, a = [values[n] for n in REVIEWED]
    checked_signature(m, 'signature_sha256'); checked_signature(t, 'report_sha256')
    if (m['signature_sha256'] != RUN_SIGNATURE or t['run_signature'] != RUN_SIGNATURE
            or a['run_signature'] != RUN_SIGNATURE or not t['complete'] or not a['complete']
            or t['role'] != 'control' or t['external_teacher_used']
            or t['train_images'] != 3712 or t['fixed_endpoint_epoch'] != 10
            or t['optimizer_steps'] != 37120 or t['tiny_checkpoint_used']
            or t['checkpoint_sha256'] != CHECKPOINT_SHA or a['checkpoint_sha256'] != CHECKPOINT_SHA
            or a['evaluated_images'] != 3769 or len(m['val_ids']) != 3769
            or len(set(m['val_ids'])) != 3769 or set(m['train_ids']) & set(m['val_ids'])):
        raise RuntimeError('Wrong full-data label-only endpoint')
    return m, t, a


def check_payload(payload):
    if (payload.get('run_signature') != RUN_SIGNATURE or payload.get('role') != 'control'
            or payload.get('step') != 37120 or payload.get('architecture_changed') is not False
            or not isinstance(payload.get('model_state'), dict)):
        raise RuntimeError('Checkpoint is not the reviewed completed GT-only endpoint')


def decision(baseline, candidate):
    if set(baseline['metrics']) != set(candidate['metrics']):
        raise RuntimeError('Changed accuracy metric inventory')
    delta = {k: float(candidate['metrics'][k]) - float(v) for k, v in baseline['metrics'].items()}
    if any(not math.isfinite(v) for v in delta.values()):
        raise RuntimeError('Nonfinite AP comparison')
    recall = {}
    for cls in ('Vehicle', 'Pedestrian'):
        old = float(baseline['nearby']['classes'][cls]['near_recall'])
        new = float(candidate['nearby']['classes'][cls]['near_recall'])
        if not (math.isfinite(old) and math.isfinite(new) and 0 <= old <= 1 and 0 <= new <= 1):
            raise RuntimeError('Invalid nearby recall')
        recall[cls] = new - old
    ap_passed = all(abs(d) <= AP_DRIFT + 1e-12 for d in delta.values())
    recall_passed = all(abs(d) <= RECALL_DRIFT + 1e-12 for d in recall.values())
    return dict(ap_delta=delta, nearby_recall_delta=recall,
        maximum_absolute_ap_drift=AP_DRIFT, maximum_absolute_recall_drift=RECALL_DRIFT,
        ap_preserved=ap_passed, nearby_recall_preserved=recall_passed,
        export_accuracy_preserved=ap_passed and recall_passed)


def immutable_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Preserve conflicting output: {path}; use a new RUN_ID')
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.trained_export_', delete=False) as stream:
                temporary = Path(stream.name); stream.write(data); stream.flush()
            # Stages run serially. Publish only a complete JSON file so a
            # runtime interruption cannot leave a truncated resume record.
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise RuntimeError(f'Preserve conflicting output: {path}')
            else:
                temporary.replace(path)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)


def signed_json(path, value):
    value = dict(value); value['report_sha256'] = signature(value)
    immutable_json(path, value)
    return value


def implementation():
    import recover_a2_lite_640 as engine
    return {**engine.implementation(),
            'scripts/export_a2_lite_trained_onnx.py': file_hash(__file__),
            'scripts/stage_a2_lite_data.py': file_hash(ROOT / 'scripts/stage_a2_lite_data.py')}


def environment():
    import recover_a2_lite_640 as engine
    return {**engine.runtime_environment(), 'onnx': importlib.metadata.version('onnx')}


def input_fixtures(source, destination):
    import evaluate_a2_lite_accuracy as accuracy
    from m68_onnx_common import INPUT_NAMES, read_f32
    raw = read_record(source, 'manifest.json') if Path(source).is_dir() else read_record(source, 'A2_LITE_640_ONNX/manifest.json')
    package = accuracy.validate_package(raw, 'compact_640')
    fixtures = []
    for f in package['fixtures']:
        for record in f['inputs'].values():
            relative = record['file']
            content = read_record(source, relative if Path(source).is_dir() else 'A2_LITE_640_ONNX/' + relative)
            if digest(content) != record['sha256']:
                raise RuntimeError('Changed signed input fixture')
            accuracy.immutable_bytes(destination / relative, content)
        fixtures.append((f['sample_id'], {n: read_f32(destination, f['inputs'][n]) for n in INPUT_NAMES}))
    if len(fixtures) != 16:
        raise RuntimeError('Need all 16 original 640x192 input fixtures')
    return fixtures


def build_model(repo, config, payload):
    import torch
    import recover_a2_lite_640 as engine
    if 'MultiScaleDeformableAttention' in sys.modules:
        raise RuntimeError('Run in a fresh subprocess; native CUDA attention must not load')
    sys.meta_path.insert(0, engine.PortableOnly())
    model = engine.base.inference_model(repo, config['model'])
    inventory = engine.base.prune_a2_layers(model)
    engine.resolution.configure_attention(model, 192, 640)
    check_payload(payload)
    model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise RuntimeError('Nonfinite checkpoint tensors')
    return model.cpu().eval().requires_grad_(False), inventory


def export(a):
    import shutil
    import torch
    import recover_a2_lite_640 as engine
    from compact_a2_teacher import decoded_fixture_comparison
    from m68_onnx_common import INPUT_NAMES, OUTPUT_NAMES, OUTPUT_SHAPES, RAW_LIMITS, compare_outputs, write_f32
    m, t, baseline = reviewed(a.full_run)
    checkpoint = a.checkpoint or Path(m['output']) / 'control/last.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: weights stay on Drive and are excluded from result ZIPs')
    if file_hash(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError('Wrong trained checkpoint bytes; do not use a KD or initialization checkpoint')
    package_source = a.input_package or Path(m['student_export'])
    protected = [Path(m['output']).resolve(), checkpoint.parent.resolve(),
                 Path(m['initialization']).parent.resolve()]
    if Path(a.full_run).is_dir(): protected.append(Path(a.full_run).resolve())
    if package_source.is_dir(): protected.append(package_source.resolve())
    if any(a.output.resolve().is_relative_to(p) for p in protected):
        raise RuntimeError('Export into a separate run; completed training data is read-only')
    current = environment()
    if {k: current[k] for k in m['cpu_environment']} != m['cpu_environment']:
        raise RuntimeError('CPU preprocessing/export package versions differ from the reviewed run: ' + json.dumps(current))
    if (a.output / 'trained_onnx_export.json').exists():
        load_export(a.output)
        print('Verified completed trained export; reusing without model reconstruction.', flush=True)
        return
    if a.output.exists() and any(p.name != 'colab_logs' for p in a.output.iterdir()):
        raise RuntimeError('Preserve partial export; choose a fresh RUN_ID')
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != m['source_sha256']:
        raise RuntimeError('Fresh portable inference source differs from the reviewed student')
    torch.set_num_threads(2)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model, inventory = build_model(a.repo, m['config'], payload); del payload
    before = engine.resolution.state_signature(model)
    package_root = a.output / RESOURCE
    package_root.mkdir(parents=True)
    fixtures = input_fixtures(package_source, package_root)
    wrapper = engine.base.Wrapper(model).eval()
    session, graph = engine.resolution.graph_export(wrapper, fixtures[0][1], package_root / MODEL_FILE, 'r640x192')
    checks, records = [], []
    for sample, data in fixtures:
        with torch.inference_mode():
            reference = engine.base.tensors_to_arrays(wrapper(*(torch.from_numpy(data[n]) for n in INPUT_NAMES)))
        actual = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), data)))
        raw = compare_outputs(actual, reference)
        decoded = decoded_fixture_comparison(actual, reference, data['calibration'][0], data['image_size'][0])
        def save(values, kind):
            result = {}
            for name, value in values.items():
                path = package_root / 'fixtures' / f'{sample}_{kind}_{name}.bin'
                record = write_f32(path, value); record['file'] = str(path.relative_to(package_root)); result[name] = record
            return result
        records.append(dict(sample_id=sample, inputs=save(data, 'input'), cpu_reference_outputs=save(actual, 'reference')))
        checks.append(dict(sample_id=sample, raw=raw, decoded=decoded))
        print(f'Trained checkpoint export parity {len(checks)}/16: {raw["passed"] and decoded["passed"]}', flush=True)
        if not raw['passed'] or not decoded['passed']:
            signed_json(a.output / 'export_failure.json', dict(complete=False, checks=checks, training_performed=False))
            raise RuntimeError('Trained FP32 export parity failed; do not relax limits or proceed to phone')
    after = engine.resolution.state_signature(model)
    if before != after or file_hash(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError('Weights changed during export')
    shutil.copyfile(a.repo / 'LICENSE.txt', package_root / 'LICENSE_MONODETR.txt')
    package = dict(schema_version=1, revision=REVISION, complete=True, resource_name=RESOURCE,
        model=dict(file=MODEL_FILE, sha256=file_hash(package_root / MODEL_FILE),
                   size_bytes=(package_root / MODEL_FILE).stat().st_size, **graph),
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_kind='full_data_GT_epoch10',
        parent_training_signature=RUN_SIGNATURE, source_checkpoint_sha256=engine.base.q.A2_SHA,
        model_state_sha256=before, precision='FP32', providers=['CPUExecutionProvider'],
        onnxruntime_version='1.30.0', inputs={n:dict(shape=list(s),dtype='float32') for n,s in engine.resolution.input_shapes('r640x192').items()},
        outputs={n:dict(shape=list(s),dtype='float32') for n,s in OUTPUT_SHAPES.items()},
        sample_ids=[s for s,_ in fixtures], fixtures=records, raw_limits=RAW_LIMITS,
        architecture={**inventory, 'input_resolution_changed':True}, image_height=192, image_width=640,
        preprocessing=engine.resolution.PREPROCESSING, training_performed=True,
        training_in_this_workflow=False, optimizer_steps=0, historical_training_steps=37120,
        accuracy_evaluated=False, accuracy_qualified=False, deployment_qualified=False,
        warmups=5, timed_predictions=100, sustain_seconds=60,
        reference_environment={k:str(v) for k,v in current.items()},
        reference_scope='trained ONNX CPU versus same trained PyTorch CPU on exact signed inputs; not native CUDA tensor parity',
        must_not_replace_original_A2_assets=True)
    package['manifest_sha256'] = signature(package); immutable_json(package_root / 'manifest.json', package)
    report = dict(complete=True, revision=REVISION, full_run=str(a.full_run.resolve()),
        checkpoint_sha256=CHECKPOINT_SHA, run_signature=RUN_SIGNATURE, environment=current,
        implementation=implementation(), source_sha256=m['source_sha256'],
        package_manifest_sha256=package['manifest_sha256'], onnx_sha256=package['model']['sha256'],
        checks=checks, model_weights_unchanged=True, training_performed=False, optimizer_steps=0,
        native_cuda_tensor_parity_measured=False, full_validation_required=True,
        iphone_performance_measured=False, deployment_qualified=False)
    signed_json(a.output / 'trained_onnx_export.json', report)


def load_export(output):
    r = checked_signature(json.loads((output / 'trained_onnx_export.json').read_text()), 'report_sha256')
    p = checked_signature(json.loads((output / RESOURCE / 'manifest.json').read_text()), 'manifest_sha256')
    if (not r['complete'] or r['revision'] != REVISION or r['implementation'] != implementation()
            or r['environment'] != environment() or r['checkpoint_sha256'] != CHECKPOINT_SHA
            or p['checkpoint_sha256'] != CHECKPOINT_SHA or p['manifest_sha256'] != r['package_manifest_sha256']
            or file_hash(output / RESOURCE / p['model']['file']) != r['onnx_sha256']):
        raise RuntimeError('Export code, environment, model or provenance changed; preserve this run')
    return r, p


def validate(a):
    import recover_a2_lite_640 as engine
    import stage_a2_lite_data as staging
    from compact_a2_teacher import teacher_inputs
    from m68_onnx_common import OUTPUT_NAMES
    m, _, baseline = reviewed(a.full_run)
    report, package = load_export(a.output)
    engine.torch.set_num_threads(2)
    staging_report = staging.stage(a.dataset, a.candidate, m['val_ids'], m['data_file_sha256'], workers=4)
    staging_path = a.output / 'validation_staging.json'
    if not staging_path.exists():
        signed_json(staging_path, staging_report)
    else:
        old_staging = checked_signature(json.loads(staging_path.read_text()), 'report_sha256')
        if not old_staging['complete'] or old_staging['frames'] != 3769:
            raise RuntimeError('Invalid previous validation staging receipt')
    session = engine.resolution.cpu_session(a.output / RESOURCE / MODEL_FILE, engine.resolution.input_shapes('r640x192'))
    destination = a.output / 'validation'; ground_truth = {}
    invalid = 0
    for index, sample in enumerate(m['val_ids']):
        expected = m['data_file_sha256'][sample]
        for kind,suffix in staging.KINDS:
            if file_hash(a.dataset / 'training' / kind / (sample+suffix)) != expected[kind]:
                raise RuntimeError('Staged validation data changed')
        identity = dict(sample_id=sample, onnx_sha256=report['onnx_sha256'], run_signature=RUN_SIGNATURE,
                        implementation=signature(report['implementation']), environment=report['environment'])
        path = destination / 'cache' / f'{sample}.json'
        row = engine.accuracy.cache_read(path, identity)
        if row is None:
            projection = engine.accuracy.parse_kitti_calib_file(a.dataset / 'training/calib' / f'{sample}.txt')['P2']
            original = teacher_inputs(a.dataset / 'training/image_2' / f'{sample}.png', projection)
            inputs = engine.resolution.resize_inputs(original, 'r640x192')
            values = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), inputs)))
            predictions = engine.accuracy.decode_predictions(values, projection, original['image_size'][0])
            raw,_ = engine.accuracy.prediction_text(predictions)
            ap,bad = engine.accuracy.prediction_text(predictions, True)
            row = dict(identity=identity, raw_text=raw, ap_text=ap, invalid_geometry=bad)
            row['record_sha256'] = signature(row); immutable_json(path,row)
        invalid += row['invalid_geometry']
        for kind in ('raw','ap'):
            engine.accuracy.immutable_bytes(destination / kind / f'{sample}.txt', row[kind+'_text'].encode())
        ground_truth[sample] = engine.accuracy.map_objects([asdict(v) for v in engine.accuracy.parse_kitti_label_file(
            a.dataset / 'training/label_2' / f'{sample}.txt')], engine.accuracy.KITTI_PRODUCTION_CLASS_MAPPING)
        if (index+1)%100==0 or index+1==len(m['val_ids']):
            print(f'Trained ONNX VAL inferred/verified {index+1}/{len(m["val_ids"])}', flush=True)
    predictions = {}
    for kind in ('raw','ap'):
        loaded = engine.accuracy.load_kitti_prediction_directory(destination / kind, m['val_ids'],
            allowed_classes=list(engine.accuracy.KITTI_PRODUCTION_CLASS_MAPPING))
        predictions[kind] = {s:engine.accuracy.map_objects(rows,engine.accuracy.KITTI_PRODUCTION_CLASS_MAPPING) for s,rows in loaded.items()}
    scored = engine.accuracy.evaluate_kitti_r40(ground_truth,predictions['ap'],['Vehicle','Pedestrian'],
        engine.accuracy.PRODUCT_IOU_THRESHOLDS, engine.accuracy.PRODUCT_NEIGHBOR_CLASSES)
    metrics = {f'{v.class_name.lower()}_{v.metric}_moderate':float(v.ap_r40) for v in scored if v.difficulty=='moderate'}
    metrics['mean_3d_moderate'] = (metrics['vehicle_3d_moderate']+metrics['pedestrian_3d_moderate'])/2
    nearby, matched, missed = engine.accuracy.audit(predictions['raw'],ground_truth,.001,.5)
    candidate = dict(complete=True,evaluated_images=3769,metrics=metrics,nearby=nearby)
    checks = decision(baseline,candidate)
    result = dict(complete=True,revision=REVISION,checkpoint_sha256=CHECKPOINT_SHA,onnx_sha256=report['onnx_sha256'],
        evaluated_images=3769,baseline=baseline,candidate=candidate,export_preservation=checks,
        all_ap_metrics=[v.to_dict() for v in scored], invalid_geometry_predictions=invalid,
        product_accuracy_gates={k:metrics[k]>=v for k,v in engine.accuracy.GATES.items()},
        phone_benchmark_candidate=checks['export_accuracy_preserved'],
        training_performed=False,optimizer_steps=0,iphone_performance_measured=False,deployment_qualified=False)
    signed_json(a.output / 'trained_onnx_accuracy.json',result)
    engine.accuracy.write_csv(matched,destination / 'matched_geometry.csv')
    engine.accuracy.write_csv(missed,destination / 'false_negatives.csv')
    print(json.dumps(dict(metrics=metrics,export_preservation=checks,phone_benchmark_candidate=result['phone_benchmark_candidate']),indent=2),flush=True)
    if not checks['export_accuracy_preserved']:
        raise RuntimeError('Export accuracy not preserved; bundle evidence and stop before phone staging')


def bundle(a):
    # Bundle logs and partial reports even if export/validation failed. No PTH,
    # images, dataset or historical experiment directory is modified/included.
    a.output.mkdir(parents=True,exist_ok=True)
    path = a.output / 'a2_lite_trained_onnx_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(a.output.rglob('*')):
            if p.is_file() and p != path and p.suffix in ('.json','.csv','.txt','.log','.onnx','.bin'):
                z.write(p,p.relative_to(a.output))
    print(path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('export','validate','bundle'))
    p.add_argument('--full-run',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--input-package',type=Path)
    p.add_argument('--repo',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--candidate',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage != 'bundle' and a.full_run is None:p.error('--full-run is required')
    if a.stage=='export' and a.repo is None:p.error('--repo is required for export')
    if a.stage=='validate' and a.dataset is None:p.error('--dataset is required for validation')
    dict(export=export,validate=validate,bundle=bundle)[a.stage](a)


if __name__=='__main__':main()
