import unittest
from typing import Iterable

from medagent.contracts import Usage
from medagent.search_agent import ActionResponse


class ScriptedActionRequester:
    """按顺序返回预设动作，仅供测试 SearchAgent 编排使用。"""

    def __init__(self, replies: Iterable[str]):
        self._replies = iter(replies)

    def __call__(self, context: str) -> ActionResponse:
        try:
            text = next(self._replies)
        except StopIteration:
            raise RuntimeError("scripted action responses exhausted") from None
        return ActionResponse(text=text, usage=Usage(None, None, None, False))


class ScriptedActionRequesterTest(unittest.TestCase):
    def test_returns_medsearch_actions_in_order_without_claiming_model_usage(self):
        requester = ScriptedActionRequester((
            "<search>aspirin evidence</search>",
            "<answer>Example answer</answer>",
        ))

        first = requester("question")
        second = requester("question<information>evidence</information>")

        self.assertEqual(first.text, "<search>aspirin evidence</search>")
        self.assertEqual(second.text, "<answer>Example answer</answer>")
        self.assertFalse(first.usage.known)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            requester("question")


if __name__ == "__main__":
    unittest.main()
