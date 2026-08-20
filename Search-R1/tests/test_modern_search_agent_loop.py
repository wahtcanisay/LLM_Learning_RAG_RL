import unittest

from search_r1_modern.protocol import (
    append_environment_tokens,
    format_passages,
    parse_action,
    truncate_to_first_action,
)


class SearchAgentLoopProtocolTest(unittest.TestCase):
    def test_parse_action_uses_first_complete_xml_action(self):
        self.assertEqual(parse_action("reason<search> Jeff Burr birthplace </search>tail"), ("search", "Jeff Burr birthplace"))
        self.assertEqual(parse_action("<answer>Aurora, Ohio</answer>"), ("answer", "Aurora, Ohio"))
        self.assertEqual(parse_action("unfinished <search>query"), (None, ""))

    def test_truncate_keeps_only_the_executed_action_prefix(self):
        text = "think <search>Jeff Burr</search> ignored <answer>wrong</answer>"
        self.assertEqual(truncate_to_first_action(text), "think <search>Jeff Burr</search>")

    def test_format_passages_preserves_search_r1_prompt_protocol(self):
        result = [
            {"document": {"contents": "Jeff Burr\nJeff Burr was born in Aurora, Ohio."}},
            {"document": {"contents": "Aurora\nAurora is a city in Ohio."}},
        ]
        self.assertEqual(
            format_passages(result),
            "Doc 1(Title: Jeff Burr) Jeff Burr was born in Aurora, Ohio.\n"
            "Doc 2(Title: Aurora) Aurora is a city in Ohio.\n",
        )

    def test_environment_tokens_participate_in_attention_but_not_policy_loss(self):
        runtime_ids, response_mask = append_environment_tokens(
            runtime_ids=[10, 11, 20, 21],
            response_mask=[1, 1],
            environment_ids=[30, 31, 32],
            response_length=8,
        )
        self.assertEqual(runtime_ids, [10, 11, 20, 21, 30, 31, 32])
        self.assertEqual(response_mask, [1, 1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
