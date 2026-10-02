import ast
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from setup_m63_isolated_runtime import validate_inventory

class IsolatedRuntimeTests(unittest.TestCase):
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
        self.assertIn('--runtime-receipt',codes[2])
        self.assertIn('diagnostics_m63h_isolated',codes[3])

    def test_smoke_is_real_kernel_and_no_updates(self):
        code=(ROOT/'scripts/smoke_m63_isolated_runtime.py').read_text()
        self.assertIn('rotate_iou_gpu_eval(box,box)',code)
        self.assertIn('KITTI_Dataset',code)
        self.assertIn('optimizer_steps=0',code)
        self.assertNotIn('optimizer.step',code)

if __name__=='__main__': unittest.main()
