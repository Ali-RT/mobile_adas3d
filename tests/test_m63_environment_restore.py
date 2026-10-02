import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import restore_m63_environment as restore

class EnvironmentRestoreTests(unittest.TestCase):
    def test_cuda_wheel_index(self):
        self.assertEqual(restore.torch_index(dict(cuda='12.8', torch='2.11.0+cu128', torchvision='0.26.0+cu128')),
                         'https://download.pytorch.org/whl/cu128')
        with self.assertRaises(RuntimeError):
            restore.torch_index(dict(cuda='12.8', torch='2.11.0+cu130', torchvision='0.26.0+cu128'))

    def test_missing_wheel_stops_before_install(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d)/'manifest.json'
            manifest.write_text(json.dumps({'environment': dict(python=sys.version.split()[0], cuda='12.8',
                torch='2.11.0+cu128', torchvision='0.26.0+cu128', opencv='5.0.0')}))
            calls=[]
            def fail(args):
                calls.append(args)
                raise restore.subprocess.CalledProcessError(1,args)
            with patch.object(sys,'argv',['restore','--manifest',str(manifest)]), patch.object(restore,'cv_version',return_value='4.14.0'), patch.object(restore,'call',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'No package changes'):
                    restore.main()
            self.assertEqual(len(calls),1)
            self.assertIn('download',calls[0])

if __name__ == '__main__': unittest.main()
