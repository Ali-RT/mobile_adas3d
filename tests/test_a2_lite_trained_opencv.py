"""Coherent OpenCV setup and unchanged signed-input preprocessing guards."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import export_a2_lite_trained_onnx as export
import setup_a2_lite_trained_opencv as setup


class OpenCVSetupTests(unittest.TestCase):
    def coherent(self):
        return dict(cv2='4.14.0', distributions={'opencv-python-headless':'4.14.0.94'})

    def frozen(self, root):
        return dict(output=str(root/'training'), cpu_environment={
            'opencv':'4.14.0', 'opencv-python-headless':'5.0.0.93'})

    def test_coherent_requires_one_distribution_and_matching_import(self):
        self.assertTrue(setup.coherent(self.coherent()))
        for value in (dict(cv2='5.0.0', distributions=self.coherent()['distributions']),
                      dict(cv2='4.14.0', distributions={'opencv-python-headless':'5.0.0.93'}),
                      dict(cv2='4.14.0', distributions={**self.coherent()['distributions'], 'opencv-python':'4.14.0.94'}),
                      dict(import_error='missing')):
            with self.subTest(value=value): self.assertFalse(setup.coherent(value))

    def test_clean_setup_is_idempotent_and_historical_data_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); training=root/'training'; training.mkdir()
            record=training/'full_recovery_manifest.json'; record.write_bytes(b'frozen bytes')
            receipt=root/'export/colab_logs/setup.json'; frozen=self.frozen(root)
            original=copy.deepcopy(frozen)
            with patch.object(export,'reviewed',return_value=(frozen,{},{})), \
                 patch.object(setup,'probe',return_value=self.coherent()), patch.object(setup,'run') as run:
                setup.repair(training,receipt); setup.repair(training,receipt); run.assert_not_called()
            value=json.loads(receipt.read_text()); export.checked_signature(value,'report_sha256')
            self.assertTrue(value['complete']); self.assertFalse(value['frozen_manifest_modified'])
            self.assertFalse(value['torch_numpy_or_cuda_modified']); self.assertIsNone(value['downloaded_wheel'])
            self.assertEqual(record.read_bytes(),b'frozen bytes'); self.assertEqual(frozen,original)

    def test_download_precedes_uninstall_and_install_is_no_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); commands=[]
            def run(command):
                commands.append(command)
                if 'download' in command:
                    destination=Path(command[command.index('--dest')+1])
                    (destination/'opencv_python_headless-4.14.0.94-test.whl').write_bytes(b'test wheel')
            with patch.object(export,'reviewed',return_value=(self.frozen(root),{},{})), \
                 patch.object(setup,'probe',side_effect=[dict(cv2='5.0.0'),self.coherent()]), \
                 patch.object(setup,'run',side_effect=run):
                setup.repair(root/'training',root/'receipt.json')
            self.assertEqual([c[3] for c in commands],['download','uninstall','install'])
            self.assertIn('--only-binary=:all:',commands[0]); self.assertIn('--no-deps',commands[0])
            self.assertIn('--no-deps',commands[2])
            self.assertEqual(set(commands[1][5:]),set(setup.PACKAGES))
            value=json.loads((root/'receipt.json').read_text())
            self.assertEqual(value['downloaded_wheel']['sha256'],export.digest(b'test wheel'))

    def test_failed_or_missing_wheel_does_not_uninstall(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for error in (None,subprocess.CalledProcessError(1,['pip','download'])):
                with patch.object(export,'reviewed',return_value=(self.frozen(root),{},{})), \
                     patch.object(setup,'probe',return_value=dict(cv2='5.0.0')), \
                     patch.object(setup,'run',side_effect=error) as run:
                    with self.assertRaises((RuntimeError,subprocess.CalledProcessError)):
                        setup.repair(root/'training',root/'receipt.json')
                    self.assertEqual(run.call_count,1)
                    self.assertEqual(run.call_args.args[0][3],'download')
            self.assertFalse((root/'receipt.json').exists())

    def test_post_probe_failure_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(export,'reviewed',return_value=(self.frozen(root),{},{})), \
                 patch.object(setup,'probe',side_effect=[self.coherent(),dict(cv2='5.0.0')]), \
                 patch.object(setup,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'restoration failed'):
                    setup.repair(root/'training',root/'receipt.json')
                run.assert_not_called()
            self.assertFalse((root/'receipt.json').exists())

    def test_protected_receipt_or_unreviewed_metadata_stops_before_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); training=root/'training'; training.mkdir()
            with patch.object(export,'reviewed',return_value=(self.frozen(root),{},{})), \
                 patch.object(setup,'probe') as probe:
                with self.assertRaisesRegex(RuntimeError,'outside completed'):
                    setup.repair(training,training/'receipt.json')
                probe.assert_not_called()
            value=self.frozen(root); value['cpu_environment']['opencv']='4.13.0'
            with patch.object(export,'reviewed',return_value=(value,{},{})), \
                 patch.object(setup,'probe') as probe:
                with self.assertRaisesRegex(RuntimeError,'only covers'):
                    setup.repair(training,root/'receipt.json')
                probe.assert_not_called()

    def test_environment_allows_only_reviewed_metadata_repair(self):
        saved=dict(opencv='4.14.0', **{'opencv-python-headless':'5.0.0.93'},numpy='2.1.3',torch='2.11.0')
        before=copy.deepcopy(saved)
        current={**saved,'opencv-python-headless':'4.14.0.94',
                 'opencv_distributions':self.coherent()['distributions']}
        result=export.check_environment(saved,current)
        self.assertTrue(result['historical_opencv_distribution_metadata_repaired'])
        self.assertEqual(saved,before)
        for changes in (dict(opencv='5.0.0'),dict(numpy='2.4.0'),dict(torch='2.12.0'),
                        dict(opencv_distributions={'opencv-python-headless':'4.14.0.94','opencv-python':'4.14.0.94'}),
                        {'opencv-python-headless':'5.0.0.93','opencv_distributions':{'opencv-python-headless':'5.0.0.93'}}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                export.check_environment(saved,{**current,**changes})

    def test_signed_resize_keeps_strict_shapes_dtype_finite_and_tolerance(self):
        import numpy as np
        expected={n:np.zeros((1,2),np.float32) for n in ('image','calibration','image_size')}
        self.assertTrue(export.preprocessing_check(expected,expected)['passed'])
        actual={n:v.copy() for n,v in expected.items()}; actual['image'][0,0]=2e-6
        self.assertFalse(export.preprocessing_check(actual,expected)['passed'])
        for bad in (np.zeros((1,3),np.float32),np.zeros((1,2),np.float64),np.full((1,2),np.nan,np.float32)):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                export.preprocessing_check({**expected,'image':bad},expected)
        with self.assertRaises(RuntimeError): export.preprocessing_check({'image':expected['image']},expected)

    def test_failed_preprocessing_stops_before_full_validation_staging(self):
        import numpy as np
        import compact_a2_teacher as teacher
        import m68_onnx_common as common
        import recover_a2_lite_640 as engine
        import stage_a2_lite_data as staging
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); parent=root/'export/parent_inputs'; parent.mkdir(parents=True)
            (parent/'input_manifest.json').write_text('{}')
            tensors={n:np.zeros((1,2),np.float32) for n in common.INPUT_NAMES}
            fixture=dict(sample_id='000001',inputs={n:n for n in common.INPUT_NAMES})
            m=dict(data_file_sha256={},val_ids=['000001','000002'])
            args=SimpleNamespace(full_run=root/'training',output=root/'export',dataset=root/'local',candidate=[])
            with patch.object(export,'reviewed',return_value=(m,{},{})), \
                 patch.object(export,'load_export',return_value=({},dict(fixtures=[fixture]))), \
                 patch.object(staging,'stage') as stage, \
                 patch.object(engine.accuracy,'validate_package',return_value=dict(fixtures=[fixture])), \
                 patch.object(engine.accuracy,'parse_kitti_calib_file',return_value=dict(P2=np.zeros((3,4),np.float32))), \
                 patch.object(engine.resolution,'cpu_session'), \
                 patch.object(engine.resolution,'resize_inputs',return_value=tensors), \
                 patch.object(teacher,'teacher_inputs',return_value=tensors), \
                 patch.object(teacher,'preprocessing_comparison',side_effect=RuntimeError('bad reconstructed image')), \
                 patch.object(common,'read_f32',side_effect=lambda _,name:tensors[name]):
                with self.assertRaisesRegex(RuntimeError,'Signed input preprocessing failed'):
                    export.validate(args)
            stage.assert_called_once_with(args.dataset,[],['000001'],{},workers=4)
            result=json.loads((args.output/'input_preprocessing_check.json').read_text())
            export.checked_signature(result,'report_sha256')
            self.assertFalse(result['passed']); self.assertEqual(result['optimizer_steps'],0)
            self.assertIn('bad reconstructed image',result['checks'][0]['error'])

    def test_notebook_restores_opencv_before_export_and_requires_both_packages(self):
        notebook=json.loads((ROOT/'notebooks/MonoDETR_A2_Lite_Trained_ONNX_Colab.ipynb').read_text())
        text='\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn(setup.WHEEL_VERSION,text); self.assertNotIn('opencv-python-headless==5.0.0.93',text)
        self.assertLess(text.index('run_logged([PYTHON,OPENCV_SETUP'),text.index("stage('export'"))
        self.assertIn("'--parent-input-package',PARENT_INPUT_PACKAGE",text)
        self.assertIn("PARENT_INPUT_PACKAGE/'manifest.json'",text)
        self.assertIn("os.environ['A2_LITE_REVIEWED_RUN']=str(FULL_RUN)",text)
        self.assertIn('sparse one-RGB-level rounding limits',text)


if __name__=='__main__': unittest.main()
