from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A2_M67_CoreML_Feasibility_Colab.ipynb"


class MonoDETRM67NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.setup = next(cell for cell in cls.cells if "m67_build_attention" in cell)
        cls.prepare = next(cell for cell in cls.cells if "'prepare'" in cell and "--a2-manifest" in cell)
        cls.audit = (ROOT / "scripts/audit_m67_a2_coreml.py").read_text()

    def test_coreml_patch_is_applied_only_after_native_source_check(self):
        self.assertNotIn("patch_monodetr_coreml_export.py", self.setup)
        self.assertIn("NATIVE_SOURCE_SHA", self.setup)
        self.assertLess(self.setup.index("source_hash(A2_REPO)"), self.setup.index("m67_build_attention"))
        self.assertIn("SCRIPT,'prepare'", self.prepare)
        self.assertIn("patch_monodetr(repo)", self.audit)
        self.assertLess(self.audit.index("elif source != NATIVE_SOURCE_SHA:"), self.audit.index("patch_monodetr(repo)"))


if __name__ == "__main__":
    unittest.main()
