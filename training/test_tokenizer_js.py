"""Checks that the browser tokenizer (rook-tokenizer.js) matches the Python one exactly."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rook_tokenizer import pretokens, train_bpe
from test_tokenizer import SAMPLE, random_bytes_cases

ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class JsParityTests(unittest.TestCase):
    def test_javascript_matches_python(self):
        tok = train_bpe(SAMPLE, vocab_size=259 + 300)
        cases = random_bytes_cases(seed=11) + [s.encode("utf-8") for s in SAMPLE[:4]]
        job = {"tokenizer": tok.to_json(), "cases_hex": [c.hex() for c in cases]}
        with tempfile.TemporaryDirectory() as tmp:
            job_path = Path(tmp) / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            run = subprocess.run(["node", str(ROOT / "tokenizer_parity.js"), str(job_path)],
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        results = json.loads(run.stdout)
        self.assertEqual(len(results), len(cases))
        for case, got in zip(cases, results):
            self.assertEqual(got["pieces"], [p.hex() for p in pretokens(case)], f"split differs for {case!r}")
            self.assertEqual(got["ids"], tok.encode(case), f"ids differ for {case!r}")
            self.assertEqual(got["back"], case.hex(), f"decode differs for {case!r}")


if __name__ == "__main__":
    unittest.main()
