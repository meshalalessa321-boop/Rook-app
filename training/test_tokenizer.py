"""Tests for Rook's byte-level BPE tokenizer (pure Python, no torch needed)."""
import json
import random
import unittest

from rook_tokenizer import ByteBPE, byte_tokenizer, pretokens, train_bpe, train_bpe_reference

SAMPLE = [
    "Tom had a red ball. He liked to play with the ball in the park every day.\n\n"
    "One day, Tom lost the ball! \"Where is my ball?\" he asked.  His mom said: look under the tree.",
    "Language grows through patterns, and patterns repeat. The cat sat on the mat; the dog sat on the log.",
    "Person A: What do you like to do?\nPerson B: I like to read stories and play outside with my friends.",
    "مرحبا بالعالم. 你好，世界！ Ça va très bien, merci. naïve café",
] * 40


def random_bytes_cases(seed=5, count=300):
    rng = random.Random(seed)
    cases = [b"", b" ", b"  ", b"\n", b" \n ", b"a", b" a", b"  a", b"\n\na", b"a  ", b"a\n", b"\t x",
             b"...", b" ...!", b"x!!  y", b"\r\n\r\n", b"\x00\x01 \xff\xfe", b"caf\xc3\xa9 \xe2\x82\xac"]
    alphabet = b" \n\t\r.,!?:;'\"()-_abcXYZ0189\x80\xc3\xa9\xff\x00\x0b\x0c"
    for _ in range(count):
        cases.append(bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 40))))
    for _ in range(60):
        cases.append(bytes(rng.randrange(256) for _ in range(rng.randint(0, 60))))
    return cases


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = train_bpe(SAMPLE, vocab_size=259 + 300)

    def test_pretokens_always_rebuild_the_input(self):
        for data in random_bytes_cases():
            self.assertEqual(b"".join(pretokens(data)), data)

    def test_round_trip_is_exact_for_any_bytes(self):
        for data in random_bytes_cases():
            self.assertEqual(self.tok.decode_bytes(self.tok.encode(data)), data)

    def test_text_round_trip(self):
        text = "Person A: Hello there!\nPerson B: Hi. How are you today?"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_merges_shorten_text_and_never_emit_special_ids(self):
        text = SAMPLE[0]
        ids = self.tok.encode(text)
        self.assertLess(len(ids), len(text.encode("utf-8")) * 0.7)
        self.assertTrue(all(0 <= i < 256 + self.tok.n for i in ids))
        self.assertEqual(self.tok.vocab_size, 259 + self.tok.n)

    def test_special_ids_follow_the_merges(self):
        self.assertEqual((self.tok.pad, self.tok.bos, self.tok.eos),
                         (256 + self.tok.n, 257 + self.tok.n, 258 + self.tok.n))
        byte = byte_tokenizer()
        self.assertEqual((byte.pad, byte.bos, byte.eos, byte.vocab_size), (256, 257, 258, 259))
        self.assertEqual(byte.encode("hi"), [104, 105])

    def test_training_is_deterministic(self):
        again = train_bpe(SAMPLE, vocab_size=259 + 300)
        self.assertEqual(self.tok.merges, again.merges)
        self.assertEqual(self.tok.digest(), again.digest())

    def test_fast_trainer_matches_slow_reference(self):
        small = SAMPLE[:3]
        fast = train_bpe(small, vocab_size=259 + 80)
        slow = train_bpe_reference(small, vocab_size=259 + 80)
        self.assertEqual(fast.merges, slow.merges)

    def test_json_round_trip_and_checksum(self):
        spec = json.loads(json.dumps(self.tok.to_json()))
        loaded = ByteBPE.from_json(spec)
        text = SAMPLE[3]
        self.assertEqual(loaded.encode(text), self.tok.encode(text))
        spec["merges"][0] = [spec["merges"][0][1], spec["merges"][0][0]]
        with self.assertRaises(ValueError):
            ByteBPE.from_json(spec)

    def test_training_uses_only_the_given_text(self):
        # Merges can only join bytes that actually appear next to each other in the text.
        tok = train_bpe(["abababab abab"], vocab_size=259 + 5)
        self.assertEqual(tok.merges[0], (ord("a"), ord("b")))
        for a, b in tok.merges:
            self.assertTrue(a < 256 + tok.n and b < 256 + tok.n)


if __name__ == "__main__":
    unittest.main()
