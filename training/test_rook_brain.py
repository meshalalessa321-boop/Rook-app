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

    def test_workflow_never_launches_answer_only_finetuning(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent
        workflow = (root.parent / ".github" / "workflows" / "rook-training.yml").read_text(encoding="utf-8")
        self.assertNotIn("finetune_rook_conversation", workflow)
        self.assertIn("python training/prepare_tinystories.py", workflow)

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
        self.assertIn("Person A: What is language?", transcript)
        self.assertIn("Person B: Language is a structured system.", transcript)
        self.assertNotIn("Assistant", transcript)
        self.assertNotIn("User:", transcript)

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
        self.assertNotIn("conversation-best.pt", source)

    def test_live_chat_page_uses_neutral_speaker_labels(self):
        from pathlib import Path
        page = (Path(__file__).resolve().parent.parent / "rook-talk.html").read_text(encoding="utf-8")
        self.assertIn("Person A: ", page)
        self.assertNotIn('"Assistant: "', page)
        self.assertNotIn("Assistant:\"", page)

    def test_lr_schedule_warms_up_then_decays(self):
        from rook_brain import lr_at
        base, warmup, total, floor = 1e-3, 100, 1000, 0.1
        self.assertAlmostEqual(lr_at(50, base, warmup, total, floor), base * 0.5)
        self.assertAlmostEqual(lr_at(100, base, warmup, total, floor), base)
        self.assertLess(lr_at(600, base, warmup, total, floor), base)
        self.assertAlmostEqual(lr_at(1000, base, warmup, total, floor), base * floor)
        # After a resume the global step is already past warmup, so no warmup restarts.
        self.assertGreater(lr_at(5000, base, warmup, 100000, floor), base * 0.9)

    def test_dropout_is_off_in_eval_and_keeps_checkpoint_keys(self):
        import torch
        from train_rook_next import Rook
        plain = Rook(32, 2, 2, 16)
        dropped = Rook(32, 2, 2, 16, dropout=0.5)
        self.assertEqual(list(plain.state_dict().keys()), list(dropped.state_dict().keys()))
        dropped.load_state_dict(plain.state_dict())
        dropped.eval()
        plain.eval()
        x = torch.randint(0, 256, (1, 16))
        self.assertTrue(torch.allclose(plain(x), dropped(x), atol=1e-6))

    def test_tinystories_shards_keep_every_story_once(self):
        from prepare_tinystories import shard_stories
        stories = [(f"Story number {i}. " + "Tom had a red ball. " * 5).strip() for i in range(250)]
        shards = shard_stories(stories, shard_size=100)
        self.assertEqual(len(shards), 3)
        rejoined = "\n\n".join(shards).split("\n\n")
        self.assertEqual(rejoined, stories)

if __name__ == "__main__":
    unittest.main()
