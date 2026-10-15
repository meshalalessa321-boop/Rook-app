"""Persistence tests for the autonomous Rook worker."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rook_worker import atomic_json_save, atomic_torch_save


class WorkerPersistenceTests(unittest.TestCase):
    def test_checkpoint_is_written_atomically_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "resume.pt"
            expected = {"step": 42, "weights": torch.tensor([1.0, 2.0])}
            atomic_torch_save(expected, target)
            actual = torch.load(target, map_location="cpu", weights_only=False)
            self.assertEqual(actual["step"], 42)
            self.assertTrue(torch.equal(actual["weights"], expected["weights"]))
            self.assertFalse(target.with_suffix(".pt.tmp").exists())

    def test_json_report_is_valid_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            atomic_json_save({"status": "checkpointed", "step": 7}, target)
            self.assertEqual(json.loads(target.read_text())["step"], 7)
            atomic_json_save({"status": "checkpointed", "step": 8}, target)
            self.assertEqual(json.loads(target.read_text())["step"], 8)
            self.assertFalse(target.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
