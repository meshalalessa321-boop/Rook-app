"""Checks the Colab GPU notebook is valid and follows Rook's rules (no torch needed)."""
import ast
import json
import re
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "rook_colab_gpu.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = ["".join(c["source"]) for c in cls.nb["cells"] if c["cell_type"] == "code"]

    def test_python_cells_parse(self):
        parsed = 0
        for src in self.code:
            if src.startswith("%%bash"):
                continue
            ast.parse(src)
            parsed += 1
        self.assertGreaterEqual(parsed, 6)

    def test_publishing_is_off_by_default_and_no_token_is_embedded(self):
        settings = next(src for src in self.code if "TOTAL_STEPS" in src)
        self.assertRegex(settings, r"PUBLISH\s*=\s*False")
        everything = "\n".join(self.code)
        self.assertNotRegex(everything, r"gh[pousr]_[A-Za-z0-9]{20,}")
        self.assertIn('userdata.get("GITHUB_TOKEN")', everything)

    def test_notebook_uses_the_full_sequence_pipeline_and_neutral_labels(self):
        everything = "\n".join(self.code)
        self.assertIn("training/rook_brain.py", everything)
        self.assertNotIn("finetune_rook_conversation", everything)
        self.assertNotIn("prepare_conversations", everything)
        self.assertIn("prepare_dialogue.py", everything)
        self.assertIn("Person A:", everything)
        for pattern in (r"\bAssistant:", r"\bUser:", r"you are (an? )?(ai|assistant|chatbot|machine)",
                        r"i am (an? )?(ai|assistant|chatbot|machine)"):
            self.assertIsNone(re.search(pattern, everything, re.I), pattern)


if __name__ == "__main__":
    unittest.main()
