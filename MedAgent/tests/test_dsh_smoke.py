import json
import threading
import time
import unittest
from types import SimpleNamespace

from dsh_smoke.role_smoke import RoleDeadlineExceeded, extract_usage, run_planner, validate_planner


def completed_result(response: str, usage: dict | None = None):
    data = {"message": {"content": [{"type": "text", "text": response}]}}
    if usage is not None:
        data["usage"] = usage
    return SimpleNamespace(
        session_id="planner-1",
        final_response=response,
        finish_reason="completed",
        events=[
            {"type": "turn/start", "data": {}},
            {"type": "assistant/message", "data": data},
            {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
        ],
    )


class FakeHarness:
    def __init__(self, result=None, block=False):
        self.result = result
        self.block = block
        self.closed = threading.Event()

    def run(self, _prompt, *, session_id):
        if self.block:
            self.closed.wait(2)
            raise RuntimeError("runtime closed")
        self.result.session_id = session_id
        return self.result

    def close(self):
        self.closed.set()


class DshBoundaryTest(unittest.TestCase):
    def test_completed_planner_and_usage(self):
        response = json.dumps({"route": "SEARCH", "search_query": "aspirin"})
        harness = FakeHarness(completed_result(response, {"inputTokens": 17, "outputTokens": 9, "totalTokens": 26}))
        result = run_planner(harness, "question", "planner-1", 1)
        self.assertEqual(result.payload["route"], "SEARCH")
        self.assertEqual(result.usage.total_tokens, 26)
        self.assertEqual(result.event_types[-1], "turn/end")

    def test_missing_usage_is_unknown(self):
        self.assertIsNone(extract_usage(completed_result('{"route":"DIRECT"}').events))

    def test_invalid_planner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_planner({"route": "SEARCH", "search_query": " "})

    def test_deadline_closes_runtime_and_rejects_late_result(self):
        harness = FakeHarness(block=True)
        started = time.monotonic()
        with self.assertRaises(RoleDeadlineExceeded):
            run_planner(harness, "question", "planner-timeout", 0.05)
        self.assertTrue(harness.closed.is_set())
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
