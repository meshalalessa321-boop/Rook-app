import unittest

from train_rook_tokenizer import train_bpe


class RookTokenizerTests(unittest.TestCase):
    def test_learns_merges_from_repeated_text(self):
        tokenizer = train_bpe(["hello hello hello", "hello there hello", "hello hello"], target_vocab=270)
        self.assertGreater(len(tokenizer["merges"]), 0)
        self.assertEqual(tokenizer["vocab_size"], 256 + len(tokenizer["merges"]) + 3)
        self.assertFalse(tokenizer["pretrained_tokenizer"])

    def test_special_tokens_follow_learned_tokens(self):
        tokenizer = train_bpe(["repeated words repeated words", "repeated words again", "words repeated"], target_vocab=265)
        self.assertEqual(tokenizer["pad_id"], 256 + len(tokenizer["merges"]))
        self.assertEqual(tokenizer["bos_id"], tokenizer["pad_id"] + 1)
        self.assertEqual(tokenizer["eos_id"], tokenizer["bos_id"] + 1)


if __name__ == "__main__":
    unittest.main()
