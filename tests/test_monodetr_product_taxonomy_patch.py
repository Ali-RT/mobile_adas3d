from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/patch_monodetr_product_taxonomy.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_patch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MonoDETRProductTaxonomyPatchTests(unittest.TestCase):
    def test_patch_maps_before_filter_and_target_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            target = repo / "lib/datasets/kitti/kitti_dataset.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "        self.writelist = cfg.get('writelist', ['Car'])\n"
                "        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT\n"
                "        for i in range(object_num):\n"
                "            # filter objects by writelist\n"
                "            if objects[i].cls_type not in self.writelist:\n"
                "                continue\n"
                "            cls_id = self.cls2id[objects[i].cls_type]\n"
                "            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]\n",
                encoding="utf-8",
            )
            MODULE.patch_dataset(repo)
            patched = target.read_text(encoding="utf-8")
            self.assertIn("mapped_class = self.class_mapping.get", patched)
            self.assertIn("if mapped_class not in self.writelist", patched)
            self.assertIn("cls_id = self.cls2id[mapped_class]", patched)
            self.assertIn(
                "mean_size = self.cls_mean_size[self.cls2id[mapped_class]]",
                patched,
            )
            self.assertNotIn("self.cls2id[objects[i].cls_type]", patched)
            self.assertLess(
                patched.index("mapped_class ="),
                patched.index("if mapped_class not in self.writelist"),
            )
            MODULE.patch_dataset(repo)


if __name__ == "__main__":
    unittest.main()
