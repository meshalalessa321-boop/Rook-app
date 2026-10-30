"""Tests for the DailyDialog preparation (pure Python; no network, no torch)."""
import unittest

from prepare_dialogue import clean_utterance, format_dialogue, shard_dialogues


class DialogueTests(unittest.TestCase):
    def test_tokenized_spacing_is_cleaned(self):
        self.assertEqual(clean_utterance("Say , Jim , how about going for a few beers after dinner ?"),
                         "Say, Jim, how about going for a few beers after dinner?")
        self.assertEqual(clean_utterance("I don ' t think it ' s right ."), "I don't think it's right.")
        self.assertEqual(clean_utterance("We ’ ll see ! ( maybe ) "), "We'll see! (maybe)")
        self.assertEqual(clean_utterance("  hello   there "), "hello there")
        self.assertEqual(clean_utterance(None), "")

    def test_speakers_alternate_with_neutral_labels(self):
        text = format_dialogue(["Hello !", "Hi , how are you ?", "Fine , thanks ."])
        self.assertEqual(text, "Person A: Hello!\nPerson B: Hi, how are you?\nPerson A: Fine, thanks.")
        for banned in ("Assistant", "User", "System", "AI"):
            self.assertNotIn(banned, text)

    def test_one_line_dialogues_are_dropped(self):
        self.assertIsNone(format_dialogue(["Hello !"]))
        self.assertIsNone(format_dialogue(["", "  "]))

    def test_shards_keep_every_dialogue_once_and_in_order(self):
        dialogues = [format_dialogue([f"Line {i} one .", f"Line {i} two ."]) for i in range(45)]
        shards = shard_dialogues(dialogues, shard_size=20)
        self.assertEqual(len(shards), 3)
        self.assertEqual("\n\n".join(shards).split("\n\n"), dialogues)

    def test_tiny_remainder_is_not_a_shard(self):
        dialogues = [format_dialogue(["a b .", "c d ."]) for _ in range(41)]
        self.assertEqual(len(shard_dialogues(dialogues, shard_size=20)), 2)


if __name__ == "__main__":
    unittest.main()
