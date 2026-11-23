import unittest

from medagent.prompts import build_search_prompt


class BuildSearchPromptTest(unittest.TestCase):
    def test_matches_training_format_without_a_hint(self):
        for hint in ("", " \t"):
            with self.subTest(hint=hint):
                self.assertEqual(
                    build_search_prompt(" 原题？ ", "Search instruction.", search_hint=hint),
                    "Search instruction.\nQuestion: 原题？",
                )

    def test_keeps_the_question_and_adds_a_separate_hint(self):
        self.assertEqual(
            build_search_prompt("原题？", "Search instruction.", search_hint=" focused query "),
            "Search instruction.\nQuestion: 原题？\nSearch hint: focused query",
        )
