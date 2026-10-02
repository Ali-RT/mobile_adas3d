import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from restore_m63_data import restore

class StandaloneTests(unittest.TestCase):
    def test_missing_inputs_reported_together_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(RuntimeError) as e:
                restore(root/'dataset',root/'splits',[])
            for name in ('image_2','label_2','calib','train.txt','val.txt'):
                self.assertIn(name,str(e.exception))
            self.assertFalse((root/'dataset').exists())

    def test_links_splits_and_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; splits=root/'splits'; splits.mkdir()
            for name in ('image_2','label_2','calib'):
                (source/'training'/name).mkdir(parents=True)
            for name,count in (('train',3712),('val',3769)):
                (splits/(name+'.txt')).write_text('\n'.join(f'{i:06d}' for i in range(count)))
            # Fixture checks restoration mechanics without creating 22k image files.
            with patch.object(Path,'is_file',return_value=True):
                restore(root/'dataset',splits,[source])
                restore(root/'dataset',splits,[source])
            self.assertTrue((root/'dataset/training/image_2').is_symlink())
            self.assertEqual((root/'dataset/ImageSets/train.txt').read_bytes(),(splits/'train.txt').read_bytes())

    def test_four_cell_flow(self):
        nb=json.loads((ROOT/'notebooks/MonoDETR_M63h_Reproducibility_Colab.ipynb').read_text())
        cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),4)
        for c in cells: ast.parse(c)
        self.assertIn("os.environ['CUDA_HOME']",cells[0])
        self.assertIn('cuda-toolkit-',cells[0])
        self.assertLess(cells[1].index('restore_data'),cells[1].index('load_manifest'))
        self.assertIn('PREFLIGHT_READY',cells[2])
        self.assertIn('diagnose_m63_reproducibility.py',cells[2])
        self.assertIn('m63h_results.zip',cells[3])
        self.assertNotIn("'--train'",'\n'.join(cells))

if __name__=='__main__': unittest.main()
