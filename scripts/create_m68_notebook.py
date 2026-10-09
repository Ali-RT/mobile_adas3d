"""Generate the checked-in M68 Google Colab runbook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M68-A2-STANDALONE-ONNX-RUNTIME-2026-10-09-r3"

cells = [
    ("markdown", f"""# M68 — run the original A2 model with ONNX Runtime CPU

Revision: `{REVISION}`. This is the deployment-first check for the unchanged A2 epoch-130 checkpoint. It exports the already-reviewed M67 portable graph to ONNX, checks it on 16 fixed validation frames, then compares native CUDA and ONNX Runtime CPU on the complete 3,769-frame Chen validation split.

No training, distillation, quantization, or architecture changes happen here. If CPU inference does not preserve validation quality, stop and investigate that backend before considering a smaller student or a stronger teacher. If quality holds, copy the phone bundle into the existing iOS benchmark app and measure ONNX Runtime CPU on the iPhone. This is fixed-input model timing—not camera-to-label timing."""),
    ("markdown", """## 1. Mount Drive, update main, and select the standalone M68 run

Run top-to-bottom on a CUDA 13.0 GPU runtime. M68 creates its own Python environment and MonoDETR checkout. The M67 manifest and receipt are read as historical model/source evidence; M68 records the current execution environment separately. The failed M67 environment can stay in place. This r3 notebook fixes NumPy inputs in the exporter and uses a fresh run ID; leave r2 records and logs untouched. Use a new `RUN_ID` after a completed or failed export. An interrupted full-validation pass can resume with the same run and environment."""),
    ("code", r'''from pathlib import Path
from collections import deque
import json, os, shlex, shutil, subprocess, sys, zipfile
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

NOTEBOOK_REVISION = 'M68-A2-STANDALONE-ONNX-RUNTIME-2026-10-09-r3'
print('Latest notebook:', NOTEBOOK_REVISION, flush=True)
RUN_ID = 'm68_a2_onnx_cpu_r3'  # Preserve r2; its manifest binds the old exporter.
M66_MANIFEST = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_m66_r0_a2_feature/m66_r0_a2_vehicle_feature_r1/m66_manifest.json')
REPO = Path(f'/content/MonoDETR_M68_A2_{RUN_ID}')
VENV = Path(f'/content/{RUN_ID}_cuda130_venv')
PYTHON = VENV / 'bin/python'
M67_OUTPUT = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_a2_m67_export/m67_a2_fp32_export_r1')
M67_MANIFEST = M67_OUTPUT / 'm67_manifest.json'
SPLIT_DIR = Path('/content/drive/MyDrive/mobile_adas3d_splits/kitti_chen')
FIXTURE_ROOT = Path(f'/content/kitti_m68_fixed16_{RUN_ID}')
DATASET_ROOT = Path(f'/content/kitti_m68_fullval_{RUN_ID}')
OUTPUT_ROOT = Path('/content/drive/MyDrive/mobile_adas3d_outputs/students/monodetr_a2_m68_onnx_cpu') / RUN_ID
OUTPUT = OUTPUT_ROOT / 'export'
MANIFEST = OUTPUT_ROOT / 'prepared/m68_a2_manifest.json'
RUNTIME_RECEIPT = OUTPUT_ROOT / 'prepared/m68_runtime_receipt.json'
LOG_DIR = OUTPUT_ROOT / 'colab_logs'
ENV = os.environ.copy()
SETUP_READY = False
SCRIPT_EXPORT = PROJECT_DIR / 'scripts/export_monodetr_a2_onnx.py'
SCRIPT_FULLVAL = PROJECT_DIR / 'scripts/evaluate_monodetr_a2_onnx_cpu_fullval.py'
SCRIPT_PREPARE = PROJECT_DIR / 'scripts/prepare_m68_a2_onnx.py'

def run_logged(command, name, allow_failure=False):
    command = [str(item) for item in command]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{name}.log'
    print('+', shlex.join(command), '\nDurable log:', log_path, flush=True)
    tail = deque(maxlen=80)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush(); tail.append(line.rstrip())
        code = process.wait(); log.write(f'\nEXIT CODE: {code}\n')
    if code and not allow_failure: raise RuntimeError(f'Exit {code}; log={log_path}\n'+'\n'.join(tail))
    return code, log_path

for required in (M66_MANIFEST, M67_MANIFEST, SPLIT_DIR/'train.txt', SPLIT_DIR/'val.txt', SCRIPT_EXPORT, SCRIPT_FULLVAL, SCRIPT_PREPARE):
    if not required.is_file(): raise FileNotFoundError(required)
print('Historical A2 source record:', M67_MANIFEST)
print('M68 output:', OUTPUT)
print('M68 uses original A2; no teacher or student changes.')'''),
    ("markdown", """## 2. Build M68's Python environment and attention extension

This installs the pinned PyTorch CUDA 13, ONNX and ONNX Runtime packages in M68's isolated environment. It restores the exact reviewed A2 source and fixed 16 inputs, compiles attention for the current GPU, then signs a new M68 execution record. Core ML tools are not installed. This setup can take several minutes."""),
    ("code", r'''if not shutil.which('nvidia-smi'): raise RuntimeError('A CUDA GPU runtime is required for the native A2 reference pass')
CANDIDATES = [Path(p) for p in ('/content/kitti_m67','/content/kitti_m66','/content/kitti_m65','/content/kitti_m64','/content/kitti_m62','/content/kitti_m61','/content/kitti','/content/monodetr_kitti_a2','/content/drive/MyDrive/datasets/kitti')]
command = [sys.executable, '-u', SCRIPT_PREPARE, '--m67-manifest', M67_MANIFEST,
           '--m66-manifest', M66_MANIFEST, '--repo', REPO, '--venv', VENV,
           '--output-root', OUTPUT_ROOT, '--dataset-root', FIXTURE_ROOT, '--splits', SPLIT_DIR]
for candidate in CANDIDATES: command += ['--candidate', candidate]
_, PREPARE_LOG = run_logged(command, 'm68_prepare')
if not PYTHON.is_file() or not MANIFEST.is_file(): raise RuntimeError('M68 preparation did not create its interpreter and manifest')
sys.path.insert(0, str(PROJECT_DIR/'scripts'))
from setup_m64_runtime import runtime_env
prepared = json.loads(MANIFEST.read_text())
ENV = runtime_env(Path(prepared['environment']['cuda_home']), PYTHON.parent)
run_logged([PYTHON, '-m', 'unittest', 'discover', '-s', 'tests',
            '-p', 'test_m68_export_inputs.py', '-v'], 'm68_input_regression_tests')
SETUP_READY = True
print('M68 standalone runtime ready:', PYTHON)
print('M68 execution record:', MANIFEST)'''),
    ("markdown", """## 3. Restore the full KITTI split and verify M68 inputs

This separate data folder restores all train/validation files using existing dataset locations. The restore helper validates both split counts and all required files before creating links. It does not download or rewrite KITTI data."""),
    ("code", r'''if not SETUP_READY: raise RuntimeError('Run section 2 first')
command = [PYTHON, PROJECT_DIR/'scripts/restore_m63_data.py', '--dataset', DATASET_ROOT, '--splits', SPLIT_DIR]
for candidate in CANDIDATES: command += ['--candidate', candidate]
run_logged(command,'m68_restore_full_kitti')

probe = ('import sys,json; sys.path.insert(0,sys.argv[1]); import m68_a2_runtime as m; '
         'x=m.load_manifest(__import__("pathlib").Path(sys.argv[2])); '
         'print(json.dumps({"repo":x["repo"],"checkpoint":x["checkpoint"],"sample_ids":len(x["sample_ids"])}))')
subprocess.run([str(PYTHON),'-c',probe,str(PROJECT_DIR/'scripts'),str(MANIFEST)],env=ENV,check=True)
print('Full dataset root:', DATASET_ROOT)'''),
    ("markdown", """## 4. Export the unchanged A2 checkpoint and test ONNX Runtime CPU on the fixed 16

This writes a fixed-shape FP32 ONNX model plus 16 input/reference fixtures. It compares CUDA portable/native A2 first, then records CPU-vs-CUDA raw-head differences. The complete decoded validation result—not a relaxed raw tolerance—decides whether phone timing is worth doing."""),
    ("code", r'''if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError(f'Preserve this partial output and choose a new RUN_ID: {OUTPUT}')
EXPORT_CODE, EXPORT_LOG = run_logged([PYTHON,'-u',SCRIPT_EXPORT,'--manifest',MANIFEST,'--output-dir',OUTPUT],'m68_export',allow_failure=True)
EXPORT_REPORT = OUTPUT/'m68_onnx_export.json'
if not EXPORT_REPORT.is_file(): raise RuntimeError(f'No export report; inspect {EXPORT_LOG}')
export = json.loads(EXPORT_REPORT.read_text())
print(json.dumps({k:export.get(k) for k in ('complete','onnx_sha256','onnx_size_bytes','onnxruntime_provider','fixed16_portable_vs_native_passed','fixed16_ort_cpu_vs_native_passed','fixed16_ort_cpu_vs_portable_passed')},indent=2))
if EXPORT_CODE: print('Export stopped; preserve logs/report and do not continue to full validation.')'''),
    ("markdown", """## 5. Compare native CUDA A2 with ONNX Runtime CPU on all 3,769 validation images

This is the decision gate. It writes resumable per-image records and prediction files, computes the same product AP_R40 and nearby-recall diagnostics for both paths, and requires AP drift ≤0.5 points and nearby-recall drift ≤0.02 for each product class. It does not qualify a phone deployment or measure iPhone speed."""),
    ("code", r'''FULLVAL_READY = False
if EXPORT_CODE != 0 or not export.get('complete'): raise RuntimeError('Section 4 export did not complete; stop here and preserve the artifacts.')
FULLVAL_CODE, FULLVAL_LOG = run_logged([PYTHON,'-u',SCRIPT_FULLVAL,'--manifest',MANIFEST,'--output-dir',OUTPUT,'--dataset-root',DATASET_ROOT,'--split-dir',SPLIT_DIR],'m68_fullval',allow_failure=True)
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
    ("code", r'''RESULTS_ZIP = OUTPUT_ROOT/'m68_results.zip'
include = [EXPORT_REPORT, FULLVAL_REPORT, OUTPUT/'A2_ONNX/manifest.json', M67_MANIFEST,
           MANIFEST, RUNTIME_RECEIPT, PREPARE_LOG, EXPORT_LOG, FULLVAL_LOG]
for role in ('native_cuda','onnx_cpu'):
    include += [OUTPUT/'fullval/metrics'/role/'kitti_r40_summary.json', OUTPUT/'fullval/nearby'/role/'nearby_geometry_summary.json']
include = [p for p in include if p.is_file()]
if RESULTS_ZIP.exists(): raise RuntimeError(f'Preserve existing result ZIP: {RESULTS_ZIP}')
with zipfile.ZipFile(RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in include: archive.write(path,path.relative_to(OUTPUT_ROOT) if path.is_relative_to(OUTPUT_ROOT) else Path('provenance')/path.name)
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
