import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_m62_r0_a2 import COMPONENTS, require_identity, retry_read, summarize


def rows(n=120):
    return [dict(depth=10 if i % 2 else 30,
                 teacher={c: 0.1 for c in COMPONENTS},
                 student={c: 1.0 for c in COMPONENTS}) for i in range(n)]


class M62Tests(unittest.TestCase):
    def test_empty_report_does_not_authorize_candidate(self):
        components, proposal = summarize([])
        self.assertIsNone(proposal)
        self.assertTrue(all(c['paired_objects'] == 0 for c in components.values()))

    def test_predeclared_tie_order(self):
        components, proposal = summarize(rows())
        self.assertEqual(proposal, 'depth')
        self.assertEqual(components['depth']['improved_objects'], 120)

    def test_does_not_require_better_global_teacher_mean(self):
        data = rows()
        data.append(dict(depth=30, teacher={c: 1000. for c in COMPONENTS},
                         student={c: 1. for c in COMPONENTS}))
        components, proposal = summarize(data)
        self.assertEqual(proposal, 'depth')
        self.assertGreater(components['depth']['teacher_mean_error'], components['depth']['a2_mean_error'])

    def test_insufficient_count_and_distance_coverage(self):
        self.assertIsNone(summarize(rows(99))[1])
        data = rows()
        for row in data:
            row['depth'] = 10
        self.assertIsNone(summarize(data)[1])

    def test_rank_uses_improved_count(self):
        data = rows()
        for row in data[:10]:
            row['teacher']['depth'] = 2.
        self.assertEqual(summarize(data)[1], 'center')

    def test_identity_error_reports_gpu_difference(self):
        require_identity({'gpu': 'A100'}, {'gpu': 'A100'})
        with self.assertRaisesRegex(RuntimeError, 'A100'):
            require_identity({'gpu': 'A100'}, {'gpu': 'Blackwell'})

    def test_io_error_retry_preserves_file(self):
        with patch('diagnose_m62_r0_a2.npz_read', side_effect=[OSError(5, 'IO'), {'ok': 1}]) as read:
            with patch('diagnose_m62_r0_a2.time.sleep'):
                self.assertEqual(retry_read(Path('/tmp/example.npz')), {'ok': 1})
            self.assertEqual(read.call_count, 2)

    def test_persistent_io_error_names_file(self):
        with patch('diagnose_m62_r0_a2.npz_read', side_effect=OSError(5, 'IO')):
            with patch('diagnose_m62_r0_a2.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'example.npz'):
                    retry_read(Path('/tmp/example.npz'))

    def test_non_io_corruption_is_not_hidden(self):
        with patch('diagnose_m62_r0_a2.npz_read', side_effect=ValueError('bad archive')) as read:
            with self.assertRaises(ValueError):
                retry_read(Path('/tmp/example.npz'))
            self.assertEqual(read.call_count, 1)

    def test_notebook_is_self_contained_and_diagnostic_only(self):
        nb = json.loads((ROOT / 'notebooks/MonoDETR_M62_R0_A2_Diagnostic_Colab.ipynb').read_text())
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 8)
        for cell in cells:
            ast.parse(cell)
        self.assertIn('def diagnostic', cells[0])
        self.assertIn('M62-2026-09-28-r1', cells[0])
        joined = '\n'.join(cells)
        self.assertNotIn('MonoDGP.git', joined)
        self.assertNotIn('reset', joined)
        self.assertNotIn('train_val.py', joined)
        self.assertIn("diagnostic('analyze')", cells[-1])

    def test_no_optimizer_or_authorization_path(self):
        src = (ROOT / 'scripts/diagnose_m62_r0_a2.py').read_text()
        ast.parse(src)
        self.assertNotIn('optimizer.step', src)
        self.assertNotIn('training_authorized=True', src)
        self.assertIn('coreml_conversion_tested=False', src)
        self.assertIn('iphone_latency_estimate_ms=None', src)


if __name__ == '__main__':
    unittest.main()
