"""Generate the checked-in M68 Google Colab runbook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M68-A2-ONNX-ORT-CPU-IPHONE-FEASIBILITY-2026-10-08-r1"

cells = [
    ("markdown", f"""# M68 — run the original A2 model with ONNX Runtime CPU

Revision: `{REVISION}`. This is the deployment-first check for the unchanged A2 epoch-130 checkpoint. It exports the already-reviewed M67 portable graph to ONNX, checks it on 16 fixed validation frames, then compares native CUDA and ONNX Runtime CPU on the complete 3,769-frame Chen validation split.

No training, distillation, quantization, or architecture changes happen here. If CPU inference does not preserve validation quality, stop and investigate that backend before considering a smaller student or a stronger teacher. If quality holds, copy the phone bundle into the existing iOS benchmark app and measure ONNX Runtime CPU on the iPhone. This is fixed-input model timing—not camera-to-label timing."""),
    ("markdown", """## 1. Mount Drive, update the project, and restore M67's temporary Colab files

Run top-to-bottom. Colab clears `/content` after a runtime reset, so the M67 Python environment, MonoDETR checkout, and compiled attention extension may need to be reconstructed. This cell uses the signed M67 manifest and saved runtime receipt to rebuild only missing temporary files, then verifies the exact CUDA/GPU/package identity, source, checkpoint, and extension before continuing. It never resets or deletes a checkout. If exact identity cannot be reproduced, it stops and preserves the saved run. If an M68 output from an interrupted run exists, preserve it and change `RUN_ID` before starting a new export."""),
    ("code", r'''from pathlib import Path
from collections import deque
import json, shlex, shutil, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = Path('/content/mobile_adas3d')
PROJECT_URL = 'https://github.com/Ali-RT/mobile_adas3d.git'
if not PROJECT_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', 'main', '--single-branch', PROJECT_URL, str(PROJECT_DIR)], check=True)
else:
    branch = subprocess.check_output(['git','-C',str(PROJECT_DIR),'branch','--show-current'],text=True).strip()
    if branch != 'main': raise RuntimeError(f'Expected main branch, found {branch!r}; do not switch branches automatically')
    dirty = subprocess.run(['git','-C',str(PROJECT_DIR),'diff','--quiet'],check=False).returncode or subprocess.run(['git','-C',str(PROJECT_DIR),'diff','--cached','--quiet'],check=False).returncode
    if dirty: raise RuntimeError('Tracked Colab repo edits exist. Preserve them; do not pull over them.')
    subprocess.run(['git','-C',str(PROJECT_DIR),'pull','--ff-only'],check=True)

RUN_ID = 'm68_a2_onnx_cpu_r1'  # Use a new ID after any partial/failed export.
M66_MANIFEST = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_m66_r0_a2_feature/m66_r0_a2_vehicle_feature_r1/m66_manifest.json')
M67_REPO = Path('/content/MonoDETR_M67_A2')
M67_DATASET = Path('/content/kitti_m67')
M67_VENV = Path('/content/m67_a2_fp32_export_r1_cuda130_venv')
PYTHON = M67_VENV / 'bin/python'
M67_OUTPUT = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_a2_m67_export/m67_a2_fp32_export_r1')
M67_MANIFEST = M67_OUTPUT / 'm67_manifest.json'
SPLIT_DIR = Path('/content/drive/MyDrive/mobile_adas3d_splits/kitti_chen')
DATASET_ROOT = Path('/content/kitti_m68_onnx_fullval')
OUTPUT = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_a2_m68_onnx_cpu') / RUN_ID
LOG_DIR = OUTPUT.parent / 'colab_logs'
SCRIPT_EXPORT = PROJECT_DIR / 'scripts/export_monodetr_a2_onnx.py'
SCRIPT_FULLVAL = PROJECT_DIR / 'scripts/evaluate_monodetr_a2_onnx_cpu_fullval.py'
SCRIPT_RESTORE_M67 = PROJECT_DIR / 'scripts/restore_m67_ephemeral_for_m68.py'

def run_logged(command, name, allow_failure=False):
    command = [str(item) for item in command]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{name}.log'
    print('+', shlex.join(command), '\nDurable log:', log_path, flush=True)
    tail = deque(maxlen=80)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush(); tail.append(line.rstrip())
        code = process.wait(); log.write(f'\nEXIT CODE: {code}\n')
    if code and not allow_failure: raise RuntimeError(f'Exit {code}; log={log_path}\n'+'\n'.join(tail))
    return code, log_path

for required in (M66_MANIFEST, M67_MANIFEST, SPLIT_DIR/'val.txt', SCRIPT_EXPORT, SCRIPT_FULLVAL, SCRIPT_RESTORE_M67):
    if not required.is_file(): raise FileNotFoundError(required)
restore_command = [sys.executable, '-u', SCRIPT_RESTORE_M67,
                   '--mobile-repo', PROJECT_DIR, '--m66-manifest', M66_MANIFEST,
                   '--m67-output', M67_OUTPUT, '--repo', M67_REPO,
                   '--venv', M67_VENV, '--splits', SPLIT_DIR]
for candidate in ('/content/kitti_m66','/content/kitti_m65','/content/kitti_m64',
                  '/content/kitti_m62','/content/kitti_m61','/content/kitti',
                  '/content/monodetr_kitti_a2',
                  '/content/drive/MyDrive/datasets/kitti'):
    restore_command += ['--candidate', candidate]
RESTORE_CODE, RESTORE_LOG = run_logged(restore_command, 'm68_restore_ephemeral_m67', allow_failure=True)
if RESTORE_CODE:
    raise RuntimeError(f'Could not safely restore the saved M67 runtime after the Colab reset. No M68 export started; inspect {RESTORE_LOG}. Preserve the M67 manifest and receipt.')
if not PYTHON.is_file(): raise FileNotFoundError(f'M67 runtime restore reported success but Python is missing: {PYTHON}')
print('Frozen M67 manifest:', M67_MANIFEST)
print('M68 output:', OUTPUT)
print('M68 uses original A2; no teacher or student changes.')'''),
    ("markdown", """## 2. Restore the full KITTI split without changing the M67 fixed-16 inputs

M67 prepared 16 export examples; this separate folder restores all train/validation files using existing dataset locations. The restore helper validates both split counts and all required files before creating links. It does not download or rewrite KITTI data."""),
    ("code", r'''if not shutil.which('nvidia-smi'): raise RuntimeError('A CUDA GPU runtime is required for the native A2 reference pass')
subprocess.run(['nvidia-smi'],check=True)
CANDIDATES = [Path(p) for p in ('/content/kitti_m67','/content/kitti_m66','/content/kitti_m65','/content/kitti_m64','/content/kitti_m62','/content/kitti_m61','/content/kitti','/content/monodetr_kitti_a2','/content/drive/MyDrive/datasets/kitti')]
command = [PYTHON, PROJECT_DIR/'scripts/restore_m63_data.py', '--dataset', DATASET_ROOT, '--splits', SPLIT_DIR]
for candidate in CANDIDATES: command += ['--candidate', candidate]
run_logged(command,'m68_restore_full_kitti')

probe = ('import sys,json; sys.path.insert(0,sys.argv[1]); import audit_m67_a2_coreml as m; '
         'x=m.load_manifest(__import__("pathlib").Path(sys.argv[2])); '
         'print(json.dumps({"repo":x["repo"],"checkpoint":x["checkpoint"],"sample_ids":len(x["sample_ids"])}))')
subprocess.run([str(PYTHON),'-c',probe,str(PROJECT_DIR/'scripts'),str(M67_MANIFEST)],check=True)
print('Full dataset root:', DATASET_ROOT)'''),
    ("markdown", """## 3. Install ONNX tools in the existing isolated M67 environment

This uses the existing isolated Python—not Colab's global Python. After installation the cell reloads the signed M67 manifest; if its original environment identity changed, it stops before export."""),
    ("code", r'''run_logged([PYTHON,'-m','pip','install','onnx==1.20.1','onnxruntime==1.30.0'],'m68_onnx_dependencies')
probe = ('import sys; sys.path.insert(0,sys.argv[1]); import onnx,onnxruntime as ort; '
         'import audit_m67_a2_coreml as m; m.load_manifest(__import__("pathlib").Path(sys.argv[2])); '
         'print("ONNX",onnx.__version__,"ORT",ort.__version__,"M67 identity unchanged")')
subprocess.run([str(PYTHON),'-c',probe,str(PROJECT_DIR/'scripts'),str(M67_MANIFEST)],check=True)'''),
    ("markdown", """## 4. Export the unchanged A2 checkpoint and test ONNX Runtime CPU on the fixed 16

This writes a fixed-shape FP32 ONNX model plus 16 input/reference fixtures. It compares CUDA portable/native A2 first, then records CPU-vs-CUDA raw-head differences. The complete decoded validation result—not a relaxed raw tolerance—decides whether phone timing is worth doing."""),
    ("code", r'''if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError(f'Preserve this partial output and choose a new RUN_ID: {OUTPUT}')
EXPORT_CODE, EXPORT_LOG = run_logged([PYTHON,'-u',SCRIPT_EXPORT,'--m67-manifest',M67_MANIFEST,'--output-dir',OUTPUT],'m68_export',allow_failure=True)
EXPORT_REPORT = OUTPUT/'m68_onnx_export.json'
if not EXPORT_REPORT.is_file(): raise RuntimeError(f'No export report; inspect {EXPORT_LOG}')
export = json.loads(EXPORT_REPORT.read_text())
print(json.dumps({k:export.get(k) for k in ('complete','onnx_sha256','onnx_size_bytes','onnxruntime_provider','fixed16_portable_vs_native_passed','fixed16_ort_cpu_vs_native_passed','fixed16_ort_cpu_vs_portable_passed')},indent=2))
if EXPORT_CODE: print('Export stopped; preserve logs/report and do not continue to full validation.')'''),
    ("markdown", """## 5. Compare native CUDA A2 with ONNX Runtime CPU on all 3,769 validation images

This is the decision gate. It writes resumable per-image records and prediction files, computes the same product AP_R40 and nearby-recall diagnostics for both paths, and requires AP drift ≤0.5 points and nearby-recall drift ≤0.02 for each product class. It does not qualify a phone deployment or measure iPhone speed."""),
    ("code", r'''FULLVAL_READY = False
if EXPORT_CODE != 0 or not export.get('complete'): raise RuntimeError('Section 4 export did not complete; stop here and preserve the artifacts.')
FULLVAL_CODE, FULLVAL_LOG = run_logged([PYTHON,'-u',SCRIPT_FULLVAL,'--m67-manifest',M67_MANIFEST,'--output-dir',OUTPUT,'--dataset-root',DATASET_ROOT,'--split-dir',SPLIT_DIR],'m68_fullval',allow_failure=True)
FULLVAL_REPORT = OUTPUT/'m68_onnx_fullval.json'
if FULLVAL_REPORT.is_file():
    report=json.loads(FULLVAL_REPORT.read_text())
    FULLVAL_READY = bool(report.get('complete') and report.get('quality_gate_passed'))
    print(json.dumps({k:report.get(k) for k in ('evaluated_images','complete_split','native_cuda_metrics','onnx_cpu_metrics','absolute_ap_drift','ap_drift_limit','native_cuda_near_recall','onnx_cpu_near_recall','absolute_near_recall_drift','near_recall_drift_limit','quality_gate_passed','next_step')},indent=2))
else:
    print(f'Full-validation report was not written. Preserve and inspect {FULLVAL_LOG}')
if FULLVAL_CODE and not FULLVAL_REPORT.is_file(): raise RuntimeError(f'Full validation failed before writing its report; inspect {FULLVAL_LOG}')
if not FULLVAL_READY: print('STOP: do not stage this model for the phone until the CPU accuracy result is reviewed.')'''),
    ("markdown", """## 6. Save a review bundle; only provide a phone bundle if the quality gate passes

If the full-val AP and nearby-recall gates pass, download the separate phone bundle and extract/replace its `A2_ONNX` folder at `mobile_adas3d/ios/M60Benchmark/A2_ONNX` on the Mac. Then install the Podfile dependency, open `ios/M60Benchmark/M60Benchmark.xcworkspace`, build/run on the iPhone, and choose **Run A2 MonoDETR — ONNX Runtime CPU**. Keep the phone unlocked and the app foregrounded for the 60-second run. The app uses only ONNX Runtime CPU; it does not use Core ML/Neural Engine.

If the full-val gate fails, the bundle is for diagnosis only: inspect CPU backend drift first. Do not jump to compression or a new teacher yet."""),
    ("code", r'''RESULTS_ZIP = OUTPUT/'m68_results.zip'
include = [EXPORT_REPORT, FULLVAL_REPORT, OUTPUT/'A2_ONNX/manifest.json', M67_MANIFEST, EXPORT_LOG, FULLVAL_LOG]
for role in ('native_cuda','onnx_cpu'):
    include += [OUTPUT/'fullval/metrics'/role/'kitti_r40_summary.json', OUTPUT/'fullval/nearby'/role/'nearby_geometry_summary.json']
include = [p for p in include if p.is_file()]
if RESULTS_ZIP.exists(): raise RuntimeError(f'Preserve existing result ZIP: {RESULTS_ZIP}')
with zipfile.ZipFile(RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in include: archive.write(path,path.relative_to(OUTPUT) if path.is_relative_to(OUTPUT) else Path('provenance')/path.name)
print('Review bundle:',RESULTS_ZIP)
if FULLVAL_READY:
    PHONE_BUNDLE=OUTPUT/'m68_a2_onnx_phone_bundle.zip'
    if not PHONE_BUNDLE.is_file(): raise FileNotFoundError(PHONE_BUNDLE)
    print('Quality gate passed. Phone package:',PHONE_BUNDLE,'bytes=',PHONE_BUNDLE.stat().st_size)
else:
    print('No phone package approved; return the review bundle so backend quality can be reviewed.')
print('Logs:',EXPORT_LOG,FULLVAL_LOG)'''),
]

notebook = {
    "cells": [
        {"cell_type": kind, "metadata": {},
         **({"execution_count": None, "outputs": []} if kind == "code" else {}),
         "source": source.splitlines(keepends=True)}
        for kind, source in cells
    ],
    "metadata": {
        "colab": {"name": "MonoDETR_A2_M68_ONNX_Runtime_iPhone_Colab", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "m68_revision": REVISION,
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/MonoDETR_A2_M68_ONNX_Runtime_iPhone_Colab.ipynb"
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(path)
