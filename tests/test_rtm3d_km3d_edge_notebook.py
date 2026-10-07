from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/RTM3D_KM3D_ResNet18_Edge_Screen_Colab.ipynb"
SCRIPT = ROOT / "scripts/run_rtm3d_km3d_res18_smoke.py"


class RTM3DEdgeNotebookTests(unittest.TestCase):
    def test_notebook_code_cells_compile_and_use_plain_official_url(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = notebook["cells"]
        self.assertTrue(cells)
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"rtm3d-notebook-cell-{index}", "exec")
        code = "\n".join("".join(c.get("source", [])) for c in cells if c["cell_type"] == "code")
        self.assertIn("https://github.com/Banconxuan/RTM3D.git", code)
        self.assertNotIn("[https://github.com/Banconxuan/RTM3D.git]", code)

    def test_notebook_is_an_inference_only_candidate_screen(self):
        notebook = json.loads(NOTEBOOK.read_text())
        all_text = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
        self.assertIn("run_rtm3d_km3d_res18_smoke.py", all_text)
        self.assertIn("no training", all_text.lower())
        self.assertIn("iphone timing", all_text.lower())
        self.assertIn("888c379e79d8a6d134f06a9b7d669118679e06dc", all_text)

    def test_setup_updates_clean_mobile_repo_before_script_check(self):
        notebook = json.loads(NOTEBOOK.read_text())
        setup = "".join(notebook["cells"][1].get("source", []))
        self.assertIn("https://github.com/Ali-RT/mobile_adas3d.git", setup)
        self.assertIn("pull', '--ff-only', 'origin', 'main", setup)
        self.assertLess(setup.index("pull', '--ff-only'"), setup.index("if not SCRIPT.is_file()"))
        self.assertIn("status', '--porcelain', '--untracked-files=no", setup)
        self.assertNotIn("reset', '--hard", setup)
        self.assertIn("2026-10-07-r2", notebook["metadata"]["rtm3d_revision"])

    def test_smoke_requires_safe_and_strict_checkpoint_loading(self):
        source = SCRIPT.read_text()
        self.assertIn("weights_only=True", source)
        self.assertIn("strict=True", source)
        self.assertIn("SAMPLES = 16", source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"iphone_performance_measured": False', source)


if __name__ == "__main__":
    unittest.main()
