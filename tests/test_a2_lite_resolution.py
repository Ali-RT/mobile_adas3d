"""Resolution-only contracts: no changed weights or camera geometry conventions."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import a2_lite_resolution_onnx as screen


class ResolutionTests(unittest.TestCase):
    def test_only_two_aspect_preserving_resolutions(self):
        self.assertEqual(set(screen.VARIANTS), {"r960x288", "r640x192"})
        for key, value in screen.VARIANTS.items():
            self.assertEqual(value["width"]*384, value["height"]*1280)
            self.assertEqual(screen.input_shapes(key)["image"], (1,3,value["height"],value["width"]))
            self.assertNotEqual(value["resource"], "A2_LITE_ONNX")
        self.assertEqual(screen.feature_shapes(288,960), ((36,120),(18,60),(9,30),(5,15)))
        self.assertEqual(screen.feature_shapes(192,640), ((24,80),(12,40),(6,20),(3,10)))
        with self.assertRaises(ValueError): screen.feature_shapes(256,832)

    def test_resize_preserves_original_calibration_and_size(self):
        values = dict(image=np.ones((1,3,384,1280),np.float32),
                      calibration=np.arange(12,dtype=np.float32).reshape(1,3,4),
                      image_size=np.asarray([[1242,375]],np.float32))
        for key in screen.VARIANTS:
            actual=screen.resize_inputs(values,key)
            np.testing.assert_allclose(actual["image"],1.,atol=1e-6)
            self.assertEqual(actual["image"].shape,screen.input_shapes(key)["image"])
            for name in ("calibration","image_size"):
                np.testing.assert_array_equal(actual[name],values[name])
                self.assertFalse(np.shares_memory(actual[name],values[name]))

    def test_pinhole_geometry_stays_in_original_coordinates(self):
        p=np.asarray([[[721,0,609,0],[0,721,172,0],[0,0,1,0]]],np.float32)
        data=dict(image=np.zeros((1,3,384,1280),np.float32),calibration=p,
                  image_size=np.asarray([[1242,375]],np.float32))
        def geometry(value):
            width,height=value["image_size"][0]; c=value["calibration"][0]
            return ((.6*width-c[0,2])*20/c[0,0],(.5*height-c[1,2])*20/c[1,1])
        for key in screen.VARIANTS:
            self.assertEqual(geometry(screen.resize_inputs(data,key)),geometry(data))

    def test_invalid_nonfinite_or_wrong_source_shapes_rejected(self):
        data=dict(image=np.zeros((1,3,384,1280),np.float32),
                  calibration=np.zeros((1,3,4),np.float32),image_size=np.ones((1,2),np.float32))
        for name in screen.INPUT_NAMES:
            broken={k:v.copy() for k,v in data.items()}; broken[name].flat[0]=np.nan
            with self.assertRaises(ValueError): screen.resize_inputs(broken,"r640x192")
        broken=dict(data,image=np.zeros((1,3,192,640),np.float32))
        with self.assertRaises(ValueError): screen.resize_inputs(broken,"r640x192")

    def test_attention_grids_updated_without_changing_state(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,4))
        for module in model:
            module.coreml_spatial_shapes=screen.feature_shapes(384,1280)
            module.n_levels=4; module.coreml_export=True
        before=screen.state_signature(model)
        for h,w in ((288,960),(192,640)):
            shapes=screen.configure_attention(model,h,w)
            for module in model: self.assertEqual(module.coreml_spatial_shapes,shapes)
            self.assertEqual(screen.state_signature(model),before)
        model[1].n_levels=3
        with self.assertRaises(RuntimeError):screen.configure_attention(model,288,960)

    def test_manifest_signature_change_fails_closed(self):
        value=dict(checkpoint="exact",training=False)
        digest=screen.signature(value); value["signature"]=digest
        screen.signed(value,"signature",digest)
        value["training"]=True
        with self.assertRaises(RuntimeError):screen.signed(value,"signature",digest)

    def test_bundle_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"model.pth").write_bytes(b"do-not-copy")
            (root/"report.json").write_text('{}'); (root/"LICENSE_MONODETR.txt").write_text('license')
            screen.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/"a2_lite_resolution_results.zip") as z:
                self.assertEqual(set(z.namelist()),{"report.json","LICENSE_MONODETR.txt"})

    def test_script_no_training_and_separate_native_resolution_checks(self):
        source=(ROOT/"scripts/a2_lite_resolution_onnx.py").read_text()
        self.assertIn('model.load_state_dict(payload["model_state"], strict=True)',source)
        self.assertIn('state_before != state_after',source)
        self.assertIn('decoded_fixture_comparison(actual, native',source)
        self.assertIn('register_forward_pre_hook(check_features)',source)
        tree=ast.parse(source)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('cuda',calls); self.assertNotIn('backward',calls); self.assertNotIn('step',calls)

    def test_original_folder_and_partial_export_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(RuntimeError):
                screen.prepare(SimpleNamespace(output=root/'child',baseline=root,repo=root/'repo'))
            target=root/'exports/r640x192'; target.mkdir(parents=True)
            sentinel=target/'keep.json'; sentinel.write_text('preserve')
            with patch.object(screen,'load',return_value={}), self.assertRaises(RuntimeError):
                screen.export(SimpleNamespace(output=root,variant='r640x192'))
            self.assertEqual(sentinel.read_text(),'preserve')

    def test_safe_strict_load_rejects_mismatched_payload(self):
        model=torch.nn.Linear(4,4)
        inventory=dict(example='pruned')
        payload=dict(epoch=0,source_epoch=130,source_checkpoint_sha256=screen.base.q.A2_SHA,
                     training_performed=False,optimizer_steps=0,pruning=inventory,optimizer_state=None,
                     model_state=model.state_dict())
        with patch.object(screen.base,'inference_model',return_value=model), \
             patch.object(screen.base,'prune_a2_layers',return_value=inventory), \
             patch.object(screen.base.q,'safe_payload',return_value=payload):
            loaded,_=screen.load_model(Path('repo'),{'model':{}},Path('checkpoint'),{'architecture':inventory})
            self.assertIs(loaded,model)
            payload['source_epoch']=129
            with self.assertRaises(RuntimeError):screen.load_model(Path('repo'),{'model':{}},Path('checkpoint'),{'architecture':inventory})
            payload['source_epoch']=130; payload['model_state']={}
            with self.assertRaises(RuntimeError):screen.load_model(Path('repo'),{'model':{}},Path('checkpoint'),{'architecture':inventory})

    def test_notebook_self_contained_with_exact_revision(self):
        nb=json.loads((ROOT/"notebooks/MonoDETR_A2_Lite_Resolution_Screen_Colab.ipynb").read_text())
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codes),5)
        for i,code in enumerate(codes):compile(code,f'resolution-cell-{i}','exec')
        names={n.name for n in ast.parse(codes[0]).body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'run_logged','stage'}<=names)
        text='\n'.join(codes)
        self.assertIn(screen.REVISION,text)
        self.assertIn('onnxruntime==1.30.0',text)
        self.assertLess(text.index("'pull','--ff-only'"),text.index('SCRIPT.is_file()'))
        self.assertIn("('r960x288','r640x192')",text)
        self.assertIn("stage('prepare'",text); self.assertIn("stage('export'",text)
        self.assertNotIn('setup_m68',text); self.assertNotIn('torch.cuda',text)
        self.assertNotIn('coremltools',text); self.assertNotIn('reset --hard',text)
        for cell in nb['cells']:
            if cell['cell_type']=='code': self.assertEqual(cell['outputs'],[])

    def test_notebook_combined_failure_log(self):
        from collections import deque
        from datetime import datetime,timezone
        import shlex,subprocess
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_Resolution_Screen_Colab.ipynb').read_text())
        code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        functions=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
        with tempfile.TemporaryDirectory() as d:
            namespace=dict(Path=Path,datetime=datetime,timezone=timezone,deque=deque,
                           subprocess=subprocess,sys=sys,shlex=shlex,LOGS=Path(d))
            exec(compile(ast.Module(body=functions,type_ignores=[]),'helpers','exec'),namespace)
            with self.assertRaises(RuntimeError):
                namespace['run_logged']([sys.executable,'-c',"import sys; print('out'); print('err',file=sys.stderr); sys.exit(3)"],'failure')
            text=next(Path(d).glob('*.log')).read_text()
            for marker in ('out','err','EXIT CODE: 3'):self.assertIn(marker,text)


if __name__=='__main__':unittest.main()
