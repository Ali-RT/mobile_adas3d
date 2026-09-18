import ast
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.export_monodgp_m59e_encoder_internals import (
    TAP_NAMES, instrument_model, internal_probe, verify_source,
)
from scripts.validate_monodgp_m59e_macos import summarize
from scripts.diagnose_monodgp_m59e_position_layout import analyze_layout


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(4, 4)
        self.sampling_offsets = nn.Linear(4, 4)
        self.attention_weights = nn.Linear(4, 4)
        self.output_proj = nn.Linear(4, 4)

    def forward(self, query, tensor, reference_points):
        value = self.value_proj(tensor)
        offsets = self.sampling_offsets(query)
        weights = self.attention_weights(query).softmax(-1)
        sampling_locations = reference_points + offsets
        return self.output_proj(value * weights + sampling_locations)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.norm1, self.norm2 = nn.LayerNorm(4), nn.LayerNorm(4)
        self.linear1, self.linear2 = nn.Linear(4, 8), nn.Linear(8, 4)

    def forward(self, tensor, pos, reference_points):
        query = tensor + pos
        value = self.norm1(tensor + self.self_attn(query, tensor, reference_points))
        return self.norm2(value + self.linear2(torch.relu(self.linear1(value))))


class TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = nn.Module()
        self.wrapped.det2d_transformer = nn.Module()
        self.wrapped.det2d_transformer.encoder = nn.Module()
        self.wrapped.det2d_transformer.encoder.layers = nn.ModuleList([TinyEncoder()])

    def forward(self, image, calibration, image_size):
        tensor, pos, reference_points = image.clone(), calibration.clone(), image_size.clone()
        value = self.wrapped.det2d_transformer.encoder.layers[0](tensor, pos, reference_points)
        return (value,)


class M59eTests(unittest.TestCase):
    def test_notebook_is_self_contained_cpu_replay(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/MonoDGP_M59e_Encoder_Internals_Colab.ipynb'
        notebook = json.loads(path.read_text())
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(code), 3)
        for cell in code:
            ast.parse(cell)
        self.assertIn("REVISION = 'M59e-2026-09-18-r1'", code[0])
        self.assertIn('def run_logged(', code[0])
        self.assertIn("'coremltools==9.0'", code[1])
        self.assertIn("'numpy>=2.0,<2.4'", code[1])
        self.assertIn('export_monodgp_m59e_encoder_internals.py', code[1])
        self.assertIn("'--m59d-dir', M59D_DIR", code[2])
        self.assertNotIn('MonoDETR.git', ''.join(code))
        self.assertNotIn('setup.py', ''.join(code))

    def test_grouped_position_layout_isolated_after_level_bias(self):
        reference = np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 32
        embedding = np.arange(16, dtype=np.float32).reshape(2, 8) / 7
        candidate = reference.copy()
        permutation = [0, 2, 1, 3, 4, 6, 5, 7]
        candidate[:, 2:] = (reference[:, 2:] - embedding[1])[..., permutation] + embedding[1]
        result = analyze_layout(reference, candidate, embedding, shapes=((1, 2), (1, 1)))
        self.assertEqual(result['explained_failing_levels'], [1])
        self.assertTrue(result['scales'][0]['original_order']['passed'])
        self.assertLess(result['scales'][1]['grouped_sine_cosine_hypothesis']['max_abs_delta'], 1e-6)
        identity = analyze_layout(reference, reference, embedding, shapes=((1, 2), (1, 1)))
        self.assertEqual(identity['explained_failing_levels'], [])
        candidate[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            analyze_layout(reference, candidate, embedding, shapes=((1, 2), (1, 1)))

    def test_probe_preserves_original_and_exposes_causal_stages(self):
        torch.manual_seed(17)
        model = TinyWrapper().eval()
        inputs = tuple(torch.randn(1, 3, 4) for _ in range(3))
        original = model(*inputs)[0]
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.trace(model, inputs), buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer).eval()
        count, graph = instrument_model(loaded, torch)
        self.assertEqual(count, 1)
        probe = torch._C._create_function_from_graph('test_m59e', graph)
        outputs = probe(loaded, *inputs)
        self.assertEqual(len(outputs), count + len(TAP_NAMES))
        torch.testing.assert_close(outputs[0], original, atol=0, rtol=0)
        taps = dict(zip(TAP_NAMES, outputs[count:]))
        torch.testing.assert_close(taps['enc0_src'], inputs[0], atol=0, rtol=0)
        torch.testing.assert_close(taps['enc0_pos'], inputs[1], atol=0, rtol=0)
        torch.testing.assert_close(taps['enc0_output'], original, atol=0, rtol=0)
        reduced = internal_probe(graph, count, torch)(loaded, *inputs)
        for left, right in zip(outputs[count:], reduced):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_first_failure_and_missing_or_nonfinite_tensor(self):
        reference = {n: np.zeros((2,), dtype=np.float32) for n in TAP_NAMES}
        candidate = {n: v.copy() for n, v in reference.items()}
        candidate['enc0_norm1'][0] = 0.25
        rows, first = summarize(reference, candidate)
        self.assertEqual(first, 'enc0_norm1')
        self.assertTrue(rows['enc0_residual1']['passed'])
        candidate['enc0_src'][0] = np.nan
        self.assertEqual(summarize(reference, candidate)[1], 'enc0_src')
        del candidate['enc0_src']
        self.assertEqual(summarize(reference, candidate)[1], 'enc0_src')

    def test_source_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = {}
            for name, key in [('MonoDGP_M59d_2d_transformer_fp32.pt', 'torchscript_sha256'),
                              ('m59d_2d_transformer_reference_io.npz', 'reference_io_sha256')]:
                (root / name).write_bytes(b'fixture')
                artifacts[key] = hashlib.sha256(b'fixture').hexdigest()
            gate = {'complete': True, 'all_export_gates_passed': True,
                    'source_commit': 'aa059a18214aebf644510e7f0793971b403f9d14', 'artifacts': artifacts}
            (root / 'm59d_2d_transformer_export_gate.json').write_text(json.dumps(gate))
            verify_source(root)
            (root / 'MonoDGP_M59d_2d_transformer_fp32.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'changed artifact'):
                verify_source(root)


if __name__ == '__main__':
    unittest.main()
