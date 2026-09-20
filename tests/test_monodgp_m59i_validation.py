import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import collect_monodgp_m59i_validation as collect
from scripts import evaluate_monodgp_m59i_coreml as evaluate


class M59iPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = evaluate.load_policy()
        self.historical = json.loads((evaluate.ROOT / 'artifacts/m59h_geometry_diagnostic_20260919.json').read_text())
        self.sample = copy.deepcopy(self.historical['samples'][0])

    def test_approved_policy_passes_all_16_without_rewriting_history(self):
        report = evaluate.approved_diagnostic_report(self.policy)
        self.assertTrue(report['approved_diagnostic_gate_passed'])
        self.assertEqual(len(report['sample_gate_results']), 16)
        self.assertFalse(report['historical_m59g_gate_passed'])
        self.assertFalse(report['full_validation_performed'])
        self.assertFalse(self.historical['all_parity_gates_passed'])

    def test_every_physical_limit_is_active(self):
        for name, limit in self.policy['geometry_limits'].items():
            row = copy.deepcopy(self.sample)
            row['continuous_geometry'][name]['max'] = limit + 1e-8
            self.assertFalse(evaluate.numerical_gate(row, self.policy)[name], name)
            row['continuous_geometry'][name]['max'] = limit
            self.assertTrue(evaluate.numerical_gate(row, self.policy)[name], name)

    def test_nonfinite_and_missing_measurements_fail(self):
        for value in (None, float('nan'), float('inf'), -1):
            row = copy.deepcopy(self.sample)
            row['continuous_geometry']['depth_m_delta']['max'] = value
            self.assertFalse(evaluate.numerical_gate(row, self.policy)['depth_m_delta'])
        self.assertFalse(any(evaluate.numerical_gate({}, self.policy).values()))

    def test_discrete_and_raw_checks_not_relaxed(self):
        row = copy.deepcopy(self.sample)
        row['selection']['rank_positions_changed'] = 1
        row['heading_bin_changes'] = 1
        row['filter_decisions']['native_score']['changed_count'] = 1
        row['unchanged_m59g_checks']['raw_output_parity']['pred_boxes']['passed'] = False
        row['invalid_geometry']['actual'] = 1
        row['decoder_port_max_abs_delta'] = 1e-8
        gates = evaluate.numerical_gate(row, self.policy)
        for key in ('same_identity_order', 'heading_bins', 'filters', 'raw_outputs', 'valid_geometry', 'native_decoder_exact'):
            self.assertFalse(gates[key], key)

    def test_policy_edits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            altered = copy.deepcopy(self.policy)
            altered['geometry_limits']['depth_m_delta'] = 1
            path.write_text(json.dumps(altered))
            with patch.object(evaluate, 'POLICY_PATH', path):
                with self.assertRaisesRegex(RuntimeError, 'policy changed'):
                    evaluate.load_policy()

    def test_frozen_ap_and_nearby_gates_retained(self):
        self.assertEqual(self.policy['preservation_gates'], evaluate.PRESERVATION_GATES)
        self.assertEqual(self.policy['raw_limits'], evaluate.PARITY_LIMITS)
        metrics = {key: value for key, value in evaluate.PRESERVATION_GATES.items() if key not in ('prediction_files', 'pedestrian_localization_failure_rate_max')}
        metrics['pedestrian_localization_failure_rate'] = evaluate.PRESERVATION_GATES['pedestrian_localization_failure_rate_max']
        self.assertTrue(all(evaluate.preservation_gate_results(metrics, 3769).values()))
        metrics['pedestrian_near_recall'] -= .001
        self.assertFalse(evaluate.preservation_gate_results(metrics, 3769)['pedestrian_near_recall'])


class M59iBundleAndResumeTests(unittest.TestCase):
    def test_collector_cli_and_zip_resume_without_inference(self):
        import numpy as np
        from PIL import Image
        from scripts.collect_monodgp_m59g_inputs import preprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('images', 'calib', 'labels'): (root/folder).mkdir()
            ids = ['000001', '000003']
            for item in ids:
                Image.new('RGB', (1240, 360), color=(40, 80, 100)).save(root/'images'/(item+'.png'))
                (root/'calib'/(item+'.txt')).write_text('P2: 700 0 620 0 0 700 180 0 0 0 1 0\n')
                (root/'labels'/(item+'.txt')).write_text('original label\n')
            split = root/'val.txt'; split.write_text('\n'.join(ids)+'\n')
            anchor = root/'anchor.npz'
            np.savez(anchor, **preprocess(root/'images/000001.png', root/'calib/000001.txt'))
            label_digest = collect.label_hash(root/'labels', ids)
            args = ['collect', '--image-dir', str(root/'images'), '--calibration-dir', str(root/'calib'),
                    '--label-dir', str(root/'labels'), '--split-file', str(split), '--anchor-npz', str(anchor), '--output-dir', str(root/'bundle')]
            # Small I/O fixture only: all production hashes/counts remain strict.
            with patch.object(collect.sys, 'argv', args), patch.object(collect, 'full_ids', return_value=ids), \
                 patch.object(collect, 'ANCHOR_SHA256', collect.sha256_file(anchor)), \
                 patch.object(collect, 'LABEL_TREE_SHA256', label_digest), \
                 patch.object(collect, 'reviewed_samples', return_value={}), \
                 patch.object(collect, 'verify_bundle', return_value={}):
                collect.main()
                first = (root/'bundle/training/image_2/000001.png').stat().st_mtime_ns
                collect.main()
                self.assertEqual((root/'bundle/training/image_2/000001.png').stat().st_mtime_ns, first)
            manifest = json.loads((root/'bundle/m59i_dataset_manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertFalse(manifest['model_execution_performed'])
            self.assertEqual(manifest['sample_ids'], ids)
            with zipfile.ZipFile(root/'bundle.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(len(archive.namelist()), 8)

    def test_copy_resume_checks_content_and_preserves_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest = root/'source', root/'nested/dest'
            source.write_text('original')
            first = collect.copy_exact(source, dest)
            self.assertEqual(collect.copy_exact(source, dest), first)
            dest.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                collect.copy_exact(source, dest)
            self.assertEqual(dest.read_text(), 'changed')

    def test_wrong_split_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'val.txt'
            path.write_text('000001\n')
            with self.assertRaisesRegex(RuntimeError, 'frozen Chen'):
                collect.full_ids(path)

    def test_label_digest_binds_ids_order_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['000001', '000003']
            for item in ids: (root/(item+'.txt')).write_text('Car original label\n')
            original = collect.label_hash(root, ids)
            self.assertNotEqual(original, collect.label_hash(root, ids[::-1]))
            (root/'000003.txt').write_text('Vehicle remapped label\n')
            self.assertNotEqual(original, collect.label_hash(root, ids))

    def test_bundle_checks_every_file_not_only_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['000001', '000003']
            manifest = {'schema_version': 1, 'complete': True, 'sample_count': 3769, 'sample_ids': ids,
                        'val_split_sha256': collect.VAL_SHA256, 'm59g_manifest_sha256': collect.M59G_MANIFEST_SHA256,
                        'anchor_preprocessing_bit_exact': True, 'samples': []}
            for item in ids:
                row = {'sample_id': item}
                for kind, (folder, suffix) in collect.FOLDERS.items():
                    path = root/folder/(item+suffix); path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(item+' '+kind)
                    row[kind+'_sha256'] = collect.sha256_file(path)
                manifest['samples'].append(row)
            digest = collect.label_hash(root/'training/label_2', ids)
            manifest['label_tree_sha256'] = digest
            collect.write_json(root/'m59i_dataset_manifest.json', manifest)
            with patch.object(collect, 'full_ids', return_value=ids), patch.object(collect, 'LABEL_TREE_SHA256', digest), patch.object(collect, 'reviewed_samples', return_value={}):
                self.assertEqual(collect.verify_bundle(root)['sample_ids'], ids)
                (root/'training/image_2/000003.png').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    collect.verify_bundle(root)

    def test_resume_binds_model_policy_software_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); item = '000001'
            binding = {'model': 'frozen', 'policy': 'v1', 'software': 'fixed'}
            files = {}
            for name in (f'pytorch/data/{item}.txt', f'coreml/data/{item}.txt', f'candidates/{item}.npz'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('data')
                files[name] = collect.sha256_file(path)
            row = {'complete': True, 'sample_id': item, 'binding': binding, 'files': files}
            report = root/'sample.json'; collect.write_json(report, row)
            self.assertEqual(evaluate.verify_cached_sample(report, binding, root, item), row)
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                evaluate.verify_cached_sample(report, {**binding, 'policy': 'v2'}, root, item)
            (root/f'coreml/data/{item}.txt').write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                evaluate.verify_cached_sample(report, binding, root, item)

    def test_prediction_id_set_exact_not_just_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'000001.txt').touch()
            evaluate.exact_prediction_set(root, ['000001'])
            with self.assertRaisesRegex(RuntimeError, 'missing or extra'):
                evaluate.exact_prediction_set(root, ['000003'])

    def test_notebook_three_self_contained_code_cells_compile(self):
        notebook = json.loads((evaluate.ROOT/'notebooks/MonoDGP_M59i_Full_Validation_Bundle_Colab.ipynb').read_text())
        cells = [''.join(x['source']) for x in notebook['cells'] if x['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        for number, source in enumerate(cells): compile(source, f'cell{number}', 'exec')
        self.assertIn('def run_logged', cells[0])
        self.assertIn('M59i-2026-09-19-r1', cells[0])
        self.assertIn('original_labels=True', cells[2])
        self.assertNotIn('git reset', '\n'.join(cells))
        self.assertNotIn('MonoDETR.git', '\n'.join(cells))


if __name__ == '__main__':
    unittest.main()
