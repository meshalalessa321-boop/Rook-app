"""Static checks on the chat page (the browser behaviour was verified with a scripted model)."""
import re
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "rook-talk.html"


class ChatPageTests(unittest.TestCase):
    def test_half_written_speaker_labels_are_hidden_while_streaming(self):
        page = PAGE.read_text(encoding="utf-8")
        self.assertIn("onUpdate(clean(hold(full)))", page)
        self.assertNotIn("onUpdate(clean(full))", page)
        # The hold-back pattern must cover every partial form: "\n", "\nP" ... "\nPerson A".
        pattern = re.search(r"hold=t=>t\.replace\(/(.*?)/,\"\"\)", page).group(1)
        regex = re.compile(pattern.replace("\\s", r"\s"))
        for partial in ["a\n", "a\nP", "a\nPe", "a\nPers", "a\nPerson", "a\nPerson ", "a\nPerson A", "a\n\nPerson B"]:
            self.assertIsNotNone(regex.search(partial), partial)
            self.assertEqual(regex.sub("", partial), "a")
        self.assertEqual(regex.sub("", "Person A is here"), "Person A is here")
        self.assertEqual(regex.sub("", "a\nPlaying is fun."), "a\nPlaying is fun.")


if __name__ == "__main__":
    unittest.main()
