"""Restore one coherent OpenCV4.14 wheel for trained A2-Lite export.

Historical manifests are read-only. Download a compatible wheel before
removing conflicting cv2 distributions. No Torch, NumPy or CUDA changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
REVISION = 'A2-LITE-OPENCV414-SETUP-2026-10-15-r1'
CV_VERSION = '4.14.0'
WHEEL_VERSION = '4.14.0.94'
PACKAGES = ('opencv-python', 'opencv-python-headless',
            'opencv-contrib-python', 'opencv-contrib-python-headless')


def probe():
    code = '''import importlib.metadata as m, json
packages={}
for name in ('opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless'):
    try: packages[name]=m.version(name)
    except m.PackageNotFoundError: pass
result={'distributions':packages}
try:
    import cv2, hashlib
    result.update(cv2=cv2.__version__, module=cv2.__file__, build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest())
except Exception as error: result['import_error']=repr(error)
print(json.dumps(result))'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('OpenCV probe failed:\n' + result.stdout + result.stderr)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError('Unexpected OpenCV probe output:\n' + result.stdout + result.stderr) from exc


def coherent(value):
    return (value.get('cv2') == CV_VERSION
            and value.get('distributions') == {'opencv-python-headless': WHEEL_VERSION})


def run(command):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def repair(full_run, receipt):
    import export_a2_lite_trained_onnx as export
    m, _, _ = export.reviewed(full_run)
    protected = [Path(m['output']).resolve()]
    if Path(full_run).is_dir(): protected.append(Path(full_run).resolve())
    if any(receipt.resolve().is_relative_to(p) for p in protected):
        raise RuntimeError('Keep the operational receipt outside completed training runs')
    frozen = m['cpu_environment']
    if frozen.get('opencv') != CV_VERSION or frozen.get('opencv-python-headless') != '5.0.0.93':
        raise RuntimeError('This repair only covers the reviewed inconsistent OpenCV metadata')
    before = probe(); wheel_record = None
    if not coherent(before):
        pip = [sys.executable, '-m', 'pip']
        with tempfile.TemporaryDirectory(prefix='a2_lite_opencv414_') as directory:
            directory = Path(directory)
            # Obtain a wheel for THIS Python/platform before any uninstall.
            run(pip + ['download', '--only-binary=:all:', '--no-deps', '--dest', directory,
                       'opencv-python-headless==' + WHEEL_VERSION])
            wheels = list(directory.glob('opencv_python_headless-*.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected exactly one compatible OpenCV wheel; no packages removed')
            wheel = wheels[0]
            wheel_record = dict(file=wheel.name, sha256=export.file_hash(wheel))
            run(pip + ['uninstall', '-y', *PACKAGES])
            run(pip + ['install', '--no-deps', wheel])
    after = probe()
    if not coherent(after):
        raise RuntimeError('OpenCV4.14 restoration failed; do not start export:\n' + json.dumps(after, indent=2))
    value = dict(complete=True, revision=REVISION, before=before, after=after,
        downloaded_wheel=wheel_record, frozen_recorded_opencv=frozen['opencv'],
        frozen_recorded_headless_distribution=frozen['opencv-python-headless'],
        frozen_manifest_modified=False, model_weights_modified=False,
        torch_numpy_or_cuda_modified=False, optimizer_steps=0,
        signed_input_preprocessing_check_still_required=True)
    export.signed_json(receipt, value)
    print(json.dumps(value, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-run', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args(); repair(a.full_run, a.receipt)


if __name__ == '__main__': main()
