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

    def test_bpe_windows_keep_every_token_as_a_target(self):
        import random
        from rook_tokenizer import train_bpe
        from rook_brain import Windows
        text = "The cat sat on the mat. The dog sat on the log. " * 30
        tok = train_bpe([text], vocab_size=259 + 40)
        rows = examples([{"text": text}], ctx=16, tokenizer=tok)
        self.assertIsInstance(rows, Windows)
        targets = [t for _x, y in rows for t in y if t != tok.pad]
        self.assertEqual(targets, tok.encode(text) + [tok.eos])
        self.assertEqual(len(rows[0][0]), 16)
        self.assertEqual(len(rows[:3]), 3)
        self.assertEqual(len(random.choices(rows, k=4)), 4)
        self.assertTrue(all(0 <= t < tok.vocab_size for x, y in rows for t in x + y))

    def test_evaluate_samples_windows_across_the_whole_set(self):
        import torch
        from torch import nn
        from rook_brain import evaluate

        class Probe(nn.Module):
            vocab = 300

            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, x):
                self.seen.extend(x[:, 0].tolist())
                return torch.zeros(x.size(0), x.size(1), 300)

        windows = [([i] + [5] * 3, [7] * 4) for i in range(1000)]
        probe = Probe()
        result = evaluate(probe, windows, "cpu", limit=10, pad=299)
        self.assertEqual(result["windows"], 10)
        self.assertEqual(probe.seen, list(range(0, 1000, 100)))

    def test_model_vocabulary_is_configurable(self):
        import torch
        from train_rook_next import Rook, V
        small = Rook(d=32, l=2, h=2, c=16, vocab=300)
        self.assertEqual(tuple(small(torch.randint(0, 300, (2, 16))).shape), (2, 16, 300))
        self.assertEqual(Rook(d=32, l=1, h=2, c=16).vocab, V)

    def test_pipeline_end_to_end_resume_and_onnx_export(self):
        import json, random, subprocess, sys, tempfile
        from pathlib import Path
        import torch
        root = Path(__file__).resolve().parent
        words = ("river stone cloud garden window yellow quiet morning bridge lantern pocket orange "
                 "whisper mountain basket silver thunder meadow candle puzzle harbor feather marble").split()
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus"
            corpus.mkdir()
            rows = []
            for i in range(12):
                (corpus / f"d{i}.txt").write_text(" ".join(rng.choice(words) for _ in range(140)), encoding="utf-8")
                rows.append({"path": f"d{i}.txt", "source": "synthetic test text", "license": "CC0-1.0",
                             "url": "https://example.invalid", "language": "en"})
            (corpus / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            out = tmp / "out"
            base = [sys.executable, str(root / "rook_brain.py"), "--corpus-dir", str(corpus), "--out-dir", str(out),
                    "--d", "48", "--layers", "2", "--heads", "2", "--ctx", "32", "--batch", "2", "--accum", "1",
                    "--steps", "2", "--warmup", "1", "--vocab-size", "330"]
            first = subprocess.run(base, capture_output=True, text=True, cwd=root, timeout=300)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            self.assertIn("tokenizer", checkpoint)
            self.assertEqual(checkpoint["config"]["vocab"], checkpoint["tokenizer"]["vocab_size"])
            self.assertGreater(checkpoint["tokenizer"]["vocab_size"], 259)
            self.assertTrue((out / "tokenizer.json").is_file())
            digest = checkpoint["tokenizer"]["sha256"]
            # Second run resumes the same weights and keeps the checkpoint's tokenizer.
            second = subprocess.run(base, capture_output=True, text=True, cwd=root, timeout=300)
            self.assertEqual(second.returncode, 0, second.stderr[-2000:])
            self.assertIn('"event": "resumed"', second.stdout)
            resumed = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["step"], 4)
            self.assertEqual(resumed["tokenizer"]["sha256"], digest)
            # Export to ONNX and check the browser files, when the export libraries are installed.
            try:
                import onnx, onnxruntime  # noqa: F401
            except ImportError:
                import os
                if os.environ.get("CI"):
                    raise  # On GitHub the export libraries are installed, so a skip would hide a problem.
                self.skipTest("onnx/onnxruntime not installed; training and resume checks passed")
            export = subprocess.run([sys.executable, str(root / "export_rook_brain_onnx.py"),
                                     "--checkpoint", str(out / "best.pt"), "--model-out", str(tmp / "m.onnx"),
                                     "--manifest-out", str(tmp / "brain.json"), "--tokenizer-out", str(tmp / "tokenizer.json")],
                                    capture_output=True, text=True, cwd=root, timeout=300)
            self.assertEqual(export.returncode, 0, export.stderr[-2000:])
            brain = json.loads((tmp / "brain.json").read_text(encoding="utf-8"))
            self.assertEqual(brain["onnx_matches_pytorch"], "passed")
            self.assertEqual(brain["tokenizer"]["sha256"], digest)
            self.assertEqual(json.loads((tmp / "tokenizer.json").read_text(encoding="utf-8"))["sha256"], digest)
            self.assertFalse(brain["pretrained_weights"])

    def test_no_identity_or_assistant_text_in_training_and_chat_code(self):
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parent
        files = ["rook_brain.py", "rook_tokenizer.py", "export_rook_brain_onnx.py", "prepare_conversations.py",
                 "prepare_tinystories.py", "prepare_wikitext.py", "../rook-talk.html", "../rook-tokenizer.js"]
        banned = [r"\bAssistant:", r"\bUser:", r"you are (an? )?(ai|assistant|chatbot|machine|language model|robot)",
                  r"i am (an? )?(ai|assistant|chatbot|machine|language model|robot)", r"as an ai\b"]
        for name in files:
            text = (root / name).read_text(encoding="utf-8")
            for pattern in banned:
                self.assertIsNone(re.search(pattern, text, re.I), f"{name} matches banned pattern {pattern!r}")

    def test_chat_page_loads_the_tokenizer_and_uses_neutral_labels(self):
        from pathlib import Path
        page = (Path(__file__).resolve().parent.parent / "rook-talk.html").read_text(encoding="utf-8")
        self.assertIn('src="./rook-tokenizer.js"', page)
        self.assertIn("generateBpe", page)
        self.assertIn("Person B:", page)

if __name__ == "__main__":
    unittest.main()
