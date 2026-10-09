import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import export_rtm3d_phone_metadata as metadata


class PhoneMetadataTests(unittest.TestCase):
    def test_png_dimensions_and_invalid_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"image.png"
            header=b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+struct.pack(">II",1242,375)
            path.write_bytes(header)
            self.assertEqual(metadata.png_size(path),(1242,375))
            path.write_bytes(b"not a PNG")
            with self.assertRaises(RuntimeError): metadata.png_size(path)

    def test_exact_provenance_and_no_output_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); dataset=root/"dataset"
            (dataset/"training/image_2").mkdir(parents=True)
            (dataset/"training/calib").mkdir(parents=True)
            image=b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+struct.pack(">II",1242,375)
            calib=b"P2: 700 0 600 0 0 700 180 0 0 0 1 0\n"
            ids=[f"{i:06d}" for i in range(16)]
            records=[]
            for sample_id in ids:
                (dataset/f"training/image_2/{sample_id}.png").write_bytes(image)
                (dataset/f"training/calib/{sample_id}.txt").write_bytes(calib)
                records.append(dict(sample_id=sample_id,image_sha256=hashlib.sha256(image).hexdigest(),
                                    calibration_sha256=hashlib.sha256(calib).hexdigest()))
            export=dict(export_complete=True,onnx_sha256=metadata.MODEL_SHA,
                        checkpoint_sha256=metadata.CHECKPOINT_SHA,
                        input=dict(fixture_sample_ids=ids,fixture_samples=records))
            bundle=root/"export.zip"
            with zipfile.ZipFile(bundle,"w") as z: z.writestr("rtm3d_onnx_export.json",json.dumps(export))
            with mock.patch.object(metadata,"BUNDLE_SHA",metadata.sha256(bundle)):
                output=root/"metadata.zip"
                result=metadata.prepare(bundle,dataset,output)
                self.assertEqual(len(result["records"]),16)
                self.assertFalse(result["original_images_included"])
                with zipfile.ZipFile(output) as z:
                    self.assertEqual(len(z.namelist()),17)
                    self.assertFalse(any(name.endswith(".png") for name in z.namelist()))
                with self.assertRaises(RuntimeError): metadata.prepare(bundle,dataset,output)
                (dataset/"training/calib/000000.txt").write_bytes(b"changed calibration")
                with self.assertRaises(RuntimeError): metadata.prepare(bundle,dataset,root/"new.zip")
                self.assertFalse((root/"new.zip").exists())

    def test_colab_helper_is_single_cell_and_self_contained(self):
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/"notebooks/RTM3D_iPhone_Decode_Metadata_Colab.ipynb").read_text())
        cells=[c for c in notebook["cells"] if c["cell_type"]=="code"]
        self.assertEqual(len(cells),1)
        source="".join(cells[0]["source"])
        compile(source,"phone-metadata-colab","exec")
        self.assertIn("def prepare(",source)
        self.assertIn("files.download(str(OUTPUT))",source)
        self.assertIn("sha256(path) == BUNDLE_SHA",source)
        self.assertNotIn("pip install",source)
        self.assertNotIn('if __name__ == "__main__":',source)

    def test_colab_embeds_the_current_standalone_exporter(self):
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/"notebooks/RTM3D_iPhone_Decode_Metadata_Colab.ipynb").read_text())
        source="".join(next(c for c in notebook["cells"] if c["cell_type"]=="code")["source"])
        exporter=(root/"scripts/export_rtm3d_phone_metadata.py").read_text()
        embedded=exporter.split('if __name__ == "__main__":')[0].rstrip()
        self.assertTrue(source.startswith(embedded))


if __name__=="__main__":
    unittest.main()
