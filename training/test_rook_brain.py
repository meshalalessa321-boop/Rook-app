import unittest

from rook_brain import examples, split_docs
from train_rook_next import BOS, EOS, PAD
from prepare_conversations import build_transcripts

class RookBrainTests(unittest.TestCase):
    def test_full_sequence_targets_include_text_and_end_token(self):
        doc = {"text": "Language grows through patterns."}
        rows = examples([doc], ctx=16)
        targets = []
        for _x, y in rows:
            targets.extend(token for token in y if token != PAD)
        expected = list(doc["text"].encode("utf-8")) + [EOS]
        self.assertEqual(targets, expected)
        self.assertNotIn(BOS, targets)

    def test_padding_is_only_padding_not_a_target(self):
        rows = examples([{"text": "abc"}], ctx=16)
        self.assertEqual(len(rows), 1)
        x, y = rows[0]
        self.assertEqual(len(x), 16)
        self.assertEqual(len(y), 16)
        self.assertTrue(all(token == PAD for token in y[4:]))

    def test_document_splits_do_not_overlap(self):
        docs = [{"path": f"doc-{i}.txt", "text": f"document number {i}"} for i in range(40)]
        train, valid, test = split_docs(docs, 1701)
        train_ids = {d["path"] for d in train}
        valid_ids = {d["path"] for d in valid}
        test_ids = {d["path"] for d in test}
        self.assertFalse(train_ids & valid_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(valid_ids & test_ids)
        self.assertEqual(len(train) + len(valid) + len(test), len(docs))

    def test_workflow_trains_conversation_responses_and_user_corrections(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent
        workflow = (root.parent / ".github" / "workflows" / "rook-training.yml").read_text(encoding="utf-8")
        self.assertIn("python training/finetune_rook_conversation.py --steps 3000 --batch 8", workflow)
        self.assertIn('"training/user_corrections.jsonl"', workflow)

    def test_conversation_corpus_keeps_both_sides_in_full_sequence(self):
        rows = [
            {
                "message_id": "user-1",
                "parent_id": None,
                "role": "prompter",
                "text": "What is language? " * 20,
                "lang": "en",
                "deleted": False,
            },
            {
                "message_id": "assistant-1",
                "parent_id": "user-1",
                "role": "assistant",
                "text": "Language is a structured system. " * 20,
                "lang": "en",
                "deleted": False,
            },
        ]
        docs = build_transcripts(rows)
        self.assertEqual(len(docs), 1)
        transcript = docs[0]["text"]
        self.assertIn("User: What is language?", transcript)
        self.assertIn("Assistant: Language is a structured system.", transcript)

    def test_conversation_candidate_must_beat_base_test_loss(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parent / "finetune_rook_conversation.py").read_text(encoding="utf-8")
        self.assertIn("accepted = test_loss < base_test_loss", source)
        self.assertIn("candidate_rejected_no_test_improvement", source)
        self.assertIn("CANDIDATE.replace(CONVERSATION)", source)
        self.assertIn("CONVERSATION.unlink(missing_ok=True)", source)

    def test_exporter_uses_full_sequence_checkpoint(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent
        source = (root / "export_rook_brain_onnx.py").read_text(encoding="utf-8")
        self.assertIn('"best.pt"', source)
        self.assertIn('"conversation-best.pt"', source)
        self.assertIn('CONVERSATION_CHECKPOINT if CONVERSATION_CHECKPOINT.is_file() else BASE_CHECKPOINT', source)

if __name__ == "__main__":
    unittest.main()
