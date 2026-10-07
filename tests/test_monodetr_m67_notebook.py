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
        cls.audit_cell = next(cell for cell in cls.cells if "SCRIPT,'audit'" in cell)
        cls.diagnostic = next(cell for cell in cls.cells if "diagnose_m67_a2_cpu_trace.py" in cell)
        cls.bundle = next(cell for cell in cls.cells if "m67_cpu_trace_diagnostic_results.zip" in cell)
        cls.audit = (ROOT / "scripts/audit_m67_a2_coreml.py").read_text()

    def test_coreml_patch_is_applied_only_after_native_source_check(self):
        self.assertNotIn("patch_monodetr_coreml_export.py", self.setup)
        self.assertIn("NATIVE_SOURCE_SHA", self.setup)
        self.assertLess(self.setup.index("source_hash(A2_REPO)"), self.setup.index("m67_build_attention"))
        self.assertIn("SCRIPT,'prepare'", self.prepare)
        self.assertIn("patch_monodetr(repo)", self.audit)
        self.assertLess(self.audit.index("elif source != NATIVE_SOURCE_SHA:"), self.audit.index("patch_monodetr(repo)"))

    def test_cpu_trace_diagnostic_is_conditional_and_bundled_separately(self):
        self.assertIn("if rc == 0", self.diagnostic)
        self.assertIn("CPU export and traced graph versus native", self.diagnostic)
        self.assertIn("m67_parity_failure.json", self.diagnostic)
        self.assertIn("m67_cpu_trace_diagnostic.json", self.diagnostic)
        self.assertIn("'m67_cpu_trace_diagnostic_results.zip'", self.bundle)
        self.assertIn("Preserve existing diagnostic ZIP", self.bundle)
        self.assertLess(self.cells.index(self.audit_cell), self.cells.index(self.diagnostic))

    def test_notebook_marks_deployment_first_revision(self):
        notebook = json.loads(NOTEBOOK.read_text())
        intro = "".join(notebook["cells"][0].get("source", []))
        self.assertIn("M67-A2-EXPORT-RUNTIME-AUDIT-2026-10-07-r2", intro)
        self.assertIn("defer M66b", intro)
        self.assertEqual(
            notebook["metadata"]["m67_revision"],
            "M67-A2-EXPORT-RUNTIME-AUDIT-2026-10-07-r2",
        )

    def test_all_notebook_code_cells_are_syntactically_valid(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
