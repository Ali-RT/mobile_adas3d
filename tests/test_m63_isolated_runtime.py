import ast
import json
from pathlib import Path
import sys
import subprocess
import signal
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from setup_m63_isolated_runtime import validate_inventory, ensure_private_python
from smoke_m63_isolated_runtime import SmokeTrace

class IsolatedRuntimeTests(unittest.TestCase):
    def test_partial_venv_bootstraps_only_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'bin').mkdir()
            (root/'bin/python').touch()
            (root/'pyvenv.cfg').write_text('include-system-site-packages = false')
            calls=[]
            def run(args,**kwargs):
                calls.append(args)
                return SimpleNamespace(returncode=1 if kwargs.get('capture_output') else 0)
            with patch('setup_m63_isolated_runtime.subprocess.run',side_effect=run), patch('setup_m63_isolated_runtime.venv.EnvBuilder') as builder:
                self.assertEqual(ensure_private_python(root),root/'bin/python')
                builder.assert_not_called()
            boot=next(c for c in calls if '--python' in c)
            self.assertEqual(boot[boot.index('--python')+1],str(root/'bin/python'))
            self.assertIn('pip==25.2',boot)

    def test_fresh_venv_never_calls_ensurepip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'env'
            def create(path):
                (path/'bin').mkdir(parents=True)
                (path/'bin/python').touch()
                (path/'pyvenv.cfg').write_text('include-system-site-packages = false')
            with patch('setup_m63_isolated_runtime.venv.EnvBuilder') as builder, patch('setup_m63_isolated_runtime.subprocess.run',return_value=SimpleNamespace(returncode=0)):
                builder.return_value.create.side_effect=create
                ensure_private_python(root)
                builder.assert_called_once_with(with_pip=False,system_site_packages=False,symlinks=True)

    def test_unsafe_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'keep').write_text('preserve')
            with self.assertRaises(RuntimeError): ensure_private_python(root)
            self.assertEqual((root/'keep').read_text(),'preserve')
            (root/'pyvenv.cfg').write_text('include-system-site-packages = true')
            with self.assertRaises(RuntimeError): ensure_private_python(root)

    def test_inventory_rejects_shared_and_cuda13(self):
        good=dict(prefix='/private/venv',base_prefix='/usr',packages={'numba':'0.61.2','llvmlite':'0.44.0'})
        validate_inventory(good)
        for name in ('numba-cuda','cuda-core','cuda-python','nvidia-nvvm','cupy-cu13'):
            with self.assertRaises(RuntimeError):
                validate_inventory(dict(good,packages={**good['packages'],name:'1.0'}))
        with self.assertRaises(RuntimeError): validate_inventory(dict(good,prefix='/usr'))

    def test_notebook_private_python_and_path_arguments(self):
        n=json.loads((ROOT/'notebooks/MonoDETR_M63h_Reproducibility_Colab.ipynb').read_text())
        codes=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        for code in codes: ast.parse(code)
        tree=ast.parse(codes[0])
        pilot=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pilot')
        calls=[]
        scope=dict(PYTHON=Path('/private/venv/bin/python'),MOBILE_REPO=ROOT,MANIFEST=Path('/drive/m.json'),run=lambda *args:calls.append(args))
        exec(compile(ast.Module(body=[pilot],type_ignores=[]),'pilot','exec'),scope)
        scope['pilot']('diagnose.py','--runtime-receipt',Path('/drive/receipt.json'))
        self.assertEqual(calls[0][0][0],'/private/venv/bin/python')
        self.assertNotIn('/',calls[0][1])
        self.assertIn('smoke_m63_isolated_runtime.py',codes[1])
        self.assertLess(codes[1].index('smoke_m63_isolated_runtime.py'),codes[1].index('PREFLIGHT_READY = True'))
        self.assertIn('-X\',\'faulthandler',codes[1])
        self.assertLess(codes[1].index('PREFLIGHT_READY = False'),codes[1].index('restore_m63_data.py'))
        self.assertEqual(n['metadata']['colab']['name'],'MonoDETR_M63h_Reproducibility_Colab.ipynb')
        self.assertIn('--runtime-receipt',codes[2])
        self.assertIn('diagnostics_m63h_isolated',codes[3])

    def test_smoke_is_real_kernel_and_no_updates(self):
        code=(ROOT/'scripts/smoke_m63_isolated_runtime.py').read_text()
        self.assertIn('rotate_iou_gpu_eval(box, box)',code)
        self.assertIn('KITTI_Dataset',code)
        self.assertIn('optimizer_steps=0',code)
        self.assertNotIn('optimizer.step',code)


    def test_smoke_trace_records_python_failure_before_reraising(self):
        with tempfile.TemporaryDirectory() as d, patch('builtins.print'):
            path = Path(d) / 'smoke.progress.json'
            trace = SmokeTrace(path)
            with self.assertRaisesRegex(ValueError, 'probe failed'):
                with trace.stage('native_import'):
                    raise ValueError('probe failed')
            progress = json.loads(path.read_text())
            self.assertFalse(progress['complete'])
            self.assertEqual(progress['last_stage'], 'native_import')
            self.assertEqual(progress['last_status'], 'failed')
            self.assertEqual(progress['events'][0]['status'], 'running')
            self.assertEqual(progress['events'][-1]['error_type'], 'ValueError')
            self.assertEqual(progress['optimizer_steps'], 0)

    def test_smoke_trace_marks_success_only_after_finish(self):
        with tempfile.TemporaryDirectory() as d, patch('builtins.print'):
            path = Path(d) / 'smoke.progress.json'
            trace = SmokeTrace(path)
            with trace.stage('kernel'):
                pass
            self.assertFalse(json.loads(path.read_text())['complete'])
            trace.finish()
            progress = json.loads(path.read_text())
            self.assertTrue(progress['complete'])
            self.assertEqual(progress['last_stage'], 'complete')
            self.assertEqual(progress['events'][1]['status'], 'passed')

    @unittest.skipUnless(sys.platform != 'win32', 'requires POSIX signals')
    def test_native_crash_preserves_stage_and_fault_handler_output(self):
        # Deliberate signal in an isolated child; no invalid memory or GPU needed.
        code = """
import faulthandler, os, resource, signal, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.path.insert(0, sys.argv[1])
from smoke_m63_isolated_runtime import SmokeTrace
faulthandler.enable(all_threads=True)
trace = SmokeTrace(sys.argv[2])
with trace.stage('native_crash_probe'):
    os.kill(os.getpid(), signal.SIGSEGV)
"""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'smoke.progress.json'
            result = subprocess.run([sys.executable, '-u', '-c', code,
                                     str(ROOT / 'scripts'), str(path)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, -signal.SIGSEGV)
            self.assertIn('native_crash_probe: running', result.stdout)
            self.assertIn('Fatal Python error: Segmentation fault', result.stderr)
            progress = json.loads(path.read_text())
            self.assertFalse(progress['complete'])
            self.assertEqual(progress['last_stage'], 'native_crash_probe')
            self.assertEqual(progress['last_status'], 'running')
            self.assertEqual(progress['optimizer_steps'], 0)

if __name__=='__main__': unittest.main()
