import unittest

from rook_brain import examples, split_docs
from train_rook_next import BOS, EOS, PAD

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

if __name__ == "__main__":
    unittest.main()
