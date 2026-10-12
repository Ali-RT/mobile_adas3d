"""Layer-pruning and standalone notebook contracts; no KITTI/AP claims."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from a2_lite_onnx import HEADS, REVISION, Wrapper, prune_a2_layers, prepare_repo, bundle
from m68_onnx_common import OUTPUT_KEYS, OUTPUT_NAMES


def synthetic_a2():
    m = torch.nn.Module()
    m.depthaware_transformer = torch.nn.Module()
    for name in ("encoder", "decoder"):
        stack = torch.nn.Module()
        stack.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        stack.num_layers = 3
        setattr(m.depthaware_transformer, name, stack)
    for name in HEADS:
        setattr(m, name, torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)]))
    m.depthaware_transformer.decoder.bbox_embed = m.bbox_embed
    m.depthaware_transformer.decoder.dim_embed = m.dim_embed_3d
    m.depth_predictor = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    m.backbone = torch.nn.Linear(4, 4)
    m.query_embed = torch.nn.Embedding(50, 4)
    m.two_stage = m.use_dab = m.two_stage_dino = False
    m.with_box_refine = True
    return m


class A2LiteTests(unittest.TestCase):
    def test_pruning_preserves_stage_zero_and_depth_backbone_query_weights(self):
        m = synthetic_a2()
        saved = {k: v.clone() for k, v in m.state_dict().items()}
        depth, backbone, queries = m.depth_predictor, m.backbone, m.query_embed
        encoder = m.depthaware_transformer.encoder.layers[0]
        decoder = m.depthaware_transformer.decoder.layers[0]
        heads = {name:getattr(m,name)[0] for name in HEADS}
        result = prune_a2_layers(m)
        self.assertIs(m.depth_predictor,depth); self.assertIs(m.backbone,backbone); self.assertIs(m.query_embed,queries)
        self.assertIs(m.depthaware_transformer.encoder.layers[0],encoder)
        self.assertIs(m.depthaware_transformer.decoder.layers[0],decoder)
        self.assertEqual(m.depthaware_transformer.encoder.num_layers,1)
        self.assertEqual(m.depthaware_transformer.decoder.num_layers,1)
        for name, head in heads.items():
            self.assertEqual(len(getattr(m,name)),1); self.assertIs(getattr(m,name)[0],head)
        for k,v in m.state_dict().items(): torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
        self.assertGreater(result['removed_parameters'],0)
        self.assertFalse(result['depth_module_changed'])
        self.assertEqual(result['retained_head_indices'],[0])

    def test_refinement_aliases_are_rebound_not_left_with_removed_heads(self):
        m=synthetic_a2(); old_bbox=m.bbox_embed
        prune_a2_layers(m)
        self.assertIs(m.depthaware_transformer.decoder.bbox_embed,m.bbox_embed)
        self.assertIs(m.depthaware_transformer.decoder.dim_embed,m.dim_embed_3d)
        self.assertIsNot(m.bbox_embed,old_bbox)
        self.assertNotIn('depthaware_transformer.decoder.bbox_embed.2.weight',m.state_dict())

    def test_wrong_layer_or_head_layout_rejected_before_mutation(self):
        for field in ('encoder','decoder','class_embed'):
            m=synthetic_a2()
            if field=='class_embed':m.class_embed=torch.nn.ModuleList([m.class_embed[0]])
            else:getattr(m.depthaware_transformer,field).num_layers=2
            with self.assertRaises(RuntimeError):prune_a2_layers(m)
            self.assertEqual(len(m.depthaware_transformer.encoder.layers),3)

    def test_two_stage_and_repeated_pruning_are_rejected(self):
        m=synthetic_a2();m.two_stage=True
        with self.assertRaises(RuntimeError):prune_a2_layers(m)
        m.two_stage=False;prune_a2_layers(m)
        with self.assertRaises(RuntimeError):prune_a2_layers(m)

    def test_wrapper_keeps_native_input_order_and_all_five_outputs(self):
        class Native(torch.nn.Module):
            def forward(self,image,calibration,targets,size):
                if targets is not None:raise AssertionError('No training targets')
                return {k:torch.tensor([float(i)]) for i,k in enumerate(OUTPUT_KEYS)}
        out=Wrapper(Native())(torch.zeros(1),torch.zeros(1),torch.zeros(1))
        self.assertEqual(len(out),len(OUTPUT_NAMES))
        self.assertEqual([float(v) for v in out],list(map(float,range(5))))

    def test_historical_checkout_name_rejected_without_git_actions(self):
        with self.assertRaises(RuntimeError):prepare_repo(Path('/content/MonoDETR_M68_A2'))

    def test_inference_constructor_never_builds_the_cuda_training_criterion(self):
        source=(ROOT/'scripts/a2_lite_onnx.py').read_text()
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inference_model')
        calls=[n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertIn('MonoDETR',calls); self.assertNotIn('build',calls)
        self.assertNotIn('cuda',calls)

    def test_notebook_self_contained_and_cpu_only_with_visible_revision(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_ONNX_Speed_Screen_Colab.ipynb').read_text())
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for i,code in enumerate(codes):compile(code,f'a2-lite-cell-{i}','exec')
        names={n.name for n in ast.parse(codes[0]).body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'run_logged','stage'}<=names)
        text='\n'.join(codes)
        self.assertIn(REVISION,text)
        self.assertIn('onnxruntime==1.30.0',text)
        self.assertLess(text.index("'pull','--ff-only'"),text.index('SCRIPT.is_file()'))
        self.assertNotIn('setup_m68_onnx_runtime',text)
        self.assertNotIn('build_m64_attention',text)
        self.assertNotIn('reset --hard',text)
        self.assertNotIn('torch.cuda',text)
        self.assertIn("stage('prepare'",text); self.assertIn("stage('export')",text)
        for c in nb['cells']:
            if c['cell_type']=='code':self.assertEqual(c['outputs'],[])

    def test_notebook_failure_log_contains_both_streams(self):
        from collections import deque
        from datetime import datetime,timezone
        nb=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_ONNX_Speed_Screen_Colab.ipynb').read_text())
        code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        functions=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
        with tempfile.TemporaryDirectory() as d:
            namespace=dict(Path=Path,datetime=datetime,timezone=timezone,deque=deque,
                           subprocess=subprocess,sys=sys,LOGS=Path(d))
            import shlex
            namespace['shlex']=shlex
            exec(compile(ast.Module(body=functions,type_ignores=[]),'helpers','exec'),namespace)
            command=[sys.executable,'-c',"import sys; print('stdout-marker'); print('stderr-marker',file=sys.stderr); sys.exit(3)"]
            with self.assertRaises(RuntimeError):namespace['run_logged'](command,'expected_failure')
            text=next(Path(d).glob('*.log')).read_text()
            self.assertIn('stdout-marker',text); self.assertIn('stderr-marker',text)
            self.assertIn('EXIT CODE: 3',text)

    def test_bundle_excludes_teacher_and_initialization_checkpoint(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'teacher_export').mkdir(); (root/'export').mkdir()
            (root/'a2_lite_manifest.json').write_text('{}')
            (root/'teacher_export/manifest.json').write_text('{}')
            (root/'a2_lite_initialization.pth').write_bytes(b'checkpoint')
            (root/'export/LICENSE_MONODETR.txt').write_text('test-license')
            bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root/'a2_lite_screen_results.zip') as z:
                self.assertIn('a2_lite_manifest.json',z.namelist())
                self.assertIn('export/LICENSE_MONODETR.txt',z.namelist())
                self.assertNotIn('teacher_export/manifest.json',z.namelist())
                self.assertNotIn('a2_lite_initialization.pth',z.namelist())


if __name__=='__main__':unittest.main()
