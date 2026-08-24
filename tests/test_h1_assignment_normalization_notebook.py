import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MobileADAS3D_H1_V2_Assignment_Normalization_Diagnostic_Colab.ipynb"


class H1AssignmentNormalizationNotebookTests(unittest.TestCase):
    def test_notebook_is_read_only_bounded_and_parses(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")
        self.assertIn("diagnose_h1_assignment_normalization.py", source)
        self.assertIn("epoch_100.pt", source)
        self.assertIn("epoch_500.pt", source)
        self.assertIn("h1_v2_assignment_normalization_diagnostic.json", source)
        self.assertNotIn("train_mobile_adas3d.py", source)
        self.assertNotIn("--resume", source)


if __name__ == "__main__":
    unittest.main()
