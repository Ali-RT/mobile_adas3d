import ast
import io
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.export_monodgp_m59f_position_interleave import rewrite_positions
from scripts.validate_monodgp_m59f_macos import compare_stage
from scripts.validate_monodgp_m58_macos_parity import OUTPUT_NAMES, OUTPUT_SHAPES


class PositionEmbeddingSine(nn.Module):
    def forward(self, mask):
        valid = ~mask
        y, x = valid.cumsum(1, dtype=torch.float32), valid.cumsum(2, dtype=torch.float32)
        y = y / (y[:, -1:, :] + 1e-6) * (2 * math.pi)
        x = x / (x[:, :, -1:] + 1e-6) * (2 * math.pi)
        frequencies = torch.arange(128, dtype=torch.float32)
        frequencies = 10000 ** (2 * (frequencies // 2) / 128)
        x, y = x[..., None] / frequencies, y[..., None] / frequencies
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=4).flatten(3)
        y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((y, x), dim=3).permute(0, 3, 1, 2)


class PositionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = PositionEmbeddingSine()

    def forward(self, a, b, c, d):
        return self.position(a), self.position(b), self.position(c), self.position(d)


class M59fTests(unittest.TestCase):
    def test_notebook_is_standalone_cpu_export(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/MonoDGP_M59f_Position_Interleave_Colab.ipynb'
        notebook = json.loads(path.read_text())
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        for code in cells:
            ast.parse(code)
        self.assertIn("REVISION = 'M59f-2026-09-18-r1'", cells[0])
        self.assertIn('def run_logged(', cells[0])
        self.assertIn("'numpy>=2.0,<2.4'", cells[1])
        self.assertIn("('full', M58_DIR)", cells[2])
        self.assertNotIn('MonoDETR.git', ''.join(cells))
        self.assertNotIn('setup.py', ''.join(cells))

    def test_rewrite_preserves_all_four_scales_and_masks_exactly(self):
        torch.manual_seed(59)
        inputs = tuple(torch.rand(1, h, w) < 0.2 for h, w in ((16, 40), (8, 20), (4, 10), (2, 5)))
        eager = PositionFixture().eval()
        expected = eager(*inputs)
        stream = io.BytesIO()
        torch.jit.save(torch.jit.trace(eager, inputs), stream)
        stream.seek(0)
        model = torch.jit.load(stream)
        changes = rewrite_positions(model, torch)
        self.assertEqual(len(changes), 4)
        self.assertEqual(sum(changes.values()), 8)
        for actual, reference in zip(model(*inputs), expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        for method in model.position._c._method_names():
            graph = str(model.position._c._get_method(method).graph)
            self.assertNotIn('aten::stack', graph)
            self.assertEqual(graph.count('aten::index_select'), 2)
        with self.assertRaisesRegex(RuntimeError, 'Expected two positional pairs'):
            rewrite_positions(model, torch)

    def test_unknown_position_structure_fails_closed(self):
        model = torch.jit.trace(nn.Identity(), torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'one PositionEmbeddingSine'):
            rewrite_positions(model, torch)
        with self.assertRaisesRegex(ValueError, 'Unknown stage'):
            compare_stage('unknown', {}, {}, ['anything'])
        with self.assertRaisesRegex(ValueError, 'empty diagnostic'):
            compare_stage('layers', {}, {}, [])

    def test_raw_limits_do_not_override_strict_decoded_gate(self):
        reference = {name: np.zeros(shape, dtype=np.float32) for name, shape in OUTPUT_SHAPES.items()}
        actual = {name: value.copy() for name, value in reference.items()}
        result = compare_stage('full', reference, actual, OUTPUT_NAMES)
        self.assertTrue(result['all_parity_gates_passed'])
        actual['pred_depth'][..., 0] += 0.001
        result = compare_stage('full', reference, actual, OUTPUT_NAMES)
        self.assertTrue(result['raw_output_parity']['pred_depth']['passed'])
        self.assertFalse(result['decoded_candidate_parity']['passed'])
        self.assertFalse(result['decoded_candidate_parity']['fields']['depth_m']['passed'])
        self.assertEqual(result['decoded_candidate_parity']['topk_flat_index_changes'], 0)
        self.assertFalse(result['all_parity_gates_passed'])
        actual['pred_boxes'][0, 0, 0] = np.nan
        self.assertFalse(compare_stage('full', reference, actual, OUTPUT_NAMES)['all_parity_gates_passed'])

    def test_positional_scale_error_is_not_hidden(self):
        reference = {'enc0_pos': np.zeros((1, 10200, 256), dtype=np.float32)}
        actual = {'enc0_pos': reference['enc0_pos'].copy()}
        actual['enc0_pos'][0, -1, 0] = 1
        result = compare_stage('internals', reference, actual, list(reference))
        self.assertEqual(result['first_diverging_tensor'], 'enc0_pos')
        self.assertEqual([scale['passed'] for scale in result['positional_scales']], [True, True, True, False])


if __name__ == '__main__':
    unittest.main()
