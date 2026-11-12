import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from search_r1_modern.search_agent_loop import SearchR1AgentLoop


class _Tokenizer:
    def __init__(self):
        self.generated = {
            (101,): "<search>Jeff Burr birthplace</search>",
            (102,): "<answer>Aurora, Ohio</answer>",
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return self.generated[tuple(token_ids)]

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]


class ModernAgentLoopIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_observation_is_masked_and_answer_is_trainable(self):
        self.assertTrue(issubclass(SearchR1AgentLoop, AgentLoopBase))
        loop = SearchR1AgentLoop.__new__(SearchR1AgentLoop)
        loop.search_url = "http://127.0.0.1:8000/retrieve"
        loop.topk = 3
        loop.max_turns = 1
        loop.max_generation_length = 96
        loop.max_observation_length = 192
        loop.request_timeout = 1.0
        loop.response_length = 512
        loop.tokenizer = _Tokenizer()
        loop.server_manager = SimpleNamespace(
            generate=AsyncMock(
                side_effect=[
                    SimpleNamespace(token_ids=[101], num_preempted=0),
                    SimpleNamespace(token_ids=[102], num_preempted=0),
                ]
            )
        )
        loop.ct_build_initial_tokens = AsyncMock(return_value=[10, 11])

        async def merge(runtime_ids, assistant_ids, response_mask, *args, **kwargs):
            return SimpleNamespace(token_ids=runtime_ids + assistant_ids), response_mask + [1] * len(assistant_ids), None

        loop.ct_merge_assistant_token = merge

        with patch(
            "search_r1_modern.search_agent_loop.retrieve",
            return_value="Doc 1(Title: Jeff Burr) Born in Aurora, Ohio.\n",
        ):
            output = await loop.run({}, raw_prompt=[{"role": "user", "content": "question"}])

        first_zero = output.response_mask.index(0)
        self.assertTrue(all(value == 1 for value in output.response_mask[:first_zero]))
        self.assertIn(1, output.response_mask[first_zero:])
        self.assertEqual(output.extra_fields["valid_search_stats"], 1)
        self.assertEqual(output.extra_fields["valid_action_stats"], 2)


if __name__ == "__main__":
    unittest.main()
