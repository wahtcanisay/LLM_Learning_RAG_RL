from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace

from medagent.prompts import PLANNER_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT
from medagent.role_runner import RoleRunner


def completed_result(session_id: str, payload: dict[str, object]):
    return SimpleNamespace(
        session_id=session_id,
        final_response=json.dumps(payload),
        finish_reason="completed",
        events=[
            {
                "type": "assistant/message",
                "data": {
                    "usage": {
                        "inputTokens": 17,
                        "outputTokens": 9,
                        "totalTokens": 26,
                    }
                },
            },
            {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
        ],
    )


class FakeBackend:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.closed = False
        self.calls: list[tuple[str, str]] = []

    def run(self, prompt: str, *, session_id: str):
        self.calls.append((prompt, session_id))
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self, backend):
        self.backend = backend
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.backend


class BlockingBackend:
    """Cooperative fake; real DSH shutdown behavior is an integration concern."""

    def __init__(self):
        self.release = threading.Event()
        self.finished = threading.Event()
        self.closed = False

    def run(self, _prompt: str, *, session_id: str):
        self.release.wait()
        self.finished.set()
        return completed_result(
            session_id,
            {"route": "SEARCH", "search_query": "late result"},
        )

    def close(self) -> None:
        self.closed = True
        self.release.set()


class RoleRunnerTest(unittest.TestCase):
    def make_runner(self, backend):
        factory = Factory(backend)
        return RoleRunner(config={"name": "test"}, backend_factory=factory), factory

    def test_planner_success(self):
        backend = FakeBackend(
            completed_result(
                "planner-1",
                {"route": "SEARCH", "search_query": "aspirin evidence"},
            )
        )
        runner, factory = self.make_runner(backend)

        result = runner.run("planner", "plan this", "planner-1", 1.0)

        self.assertEqual(result.payload["route"], "SEARCH")
        self.assertEqual(result.event_types, ("assistant/message", "turn/end"))
        self.assertEqual(result.usage.total_tokens, 26)
        self.assertTrue(result.usage.known)
        self.assertEqual(factory.calls[0]["system_prompt"], PLANNER_SYSTEM_PROMPT)
        self.assertEqual(backend.calls, [("plan this", "planner-1")])
        self.assertTrue(backend.closed)

    def test_verifier_success_uses_its_own_system_prompt(self):
        backend = FakeBackend(
            completed_result(
                "verifier-1",
                {"verdict": "PASS", "failure_codes": []},
            )
        )
        runner, factory = self.make_runner(backend)

        result = runner.run("verifier", "verify this", "verifier-1", 1.0)

        self.assertEqual(result.payload["verdict"], "PASS")
        self.assertEqual(factory.calls[0]["system_prompt"], VERIFIER_SYSTEM_PROMPT)
        self.assertNotEqual(PLANNER_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT)
        self.assertTrue(backend.closed)

    def test_unknown_role_is_rejected_before_backend_creation(self):
        backend = FakeBackend()
        runner, factory = self.make_runner(backend)

        with self.assertRaisesRegex(ValueError, "unknown role"):
            runner.run("generator", "prompt", "generator-1", 1.0)

        self.assertEqual(factory.calls, [])
        self.assertFalse(backend.closed)

    def test_invalid_json_is_rejected_and_backend_is_closed(self):
        result = completed_result("planner-1", {})
        result.final_response = "not-json"
        backend = FakeBackend(result)
        runner, _factory = self.make_runner(backend)

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            runner.run("planner", "prompt", "planner-1", 1.0)

        self.assertTrue(backend.closed)

    def test_wrong_session_is_rejected_and_backend_is_closed(self):
        backend = FakeBackend(
            completed_result(
                "other-session",
                {"route": "SEARCH", "search_query": "evidence"},
            )
        )
        runner, _factory = self.make_runner(backend)

        with self.assertRaisesRegex(ValueError, "session_id"):
            runner.run("planner", "prompt", "planner-1", 1.0)

        self.assertTrue(backend.closed)

    def test_non_completed_result_is_rejected_and_backend_is_closed(self):
        result = completed_result(
            "planner-1",
            {"route": "SEARCH", "search_query": "evidence"},
        )
        result.finish_reason = "error"
        backend = FakeBackend(result)
        runner, _factory = self.make_runner(backend)

        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            runner.run("planner", "prompt", "planner-1", 1.0)

        self.assertTrue(backend.closed)

    def test_backend_error_is_propagated_and_backend_is_closed(self):
        backend = FakeBackend(error=OSError("synthetic transport failure"))
        runner, _factory = self.make_runner(backend)

        with self.assertRaisesRegex(OSError, "synthetic transport failure"):
            runner.run("planner", "prompt", "planner-1", 1.0)

        self.assertTrue(backend.closed)

    def test_missing_usage_is_unknown(self):
        result = completed_result(
            "planner-1",
            {"route": "SEARCH", "search_query": "evidence"},
        )
        result.events[0]["data"].pop("usage")
        backend = FakeBackend(result)
        runner, _factory = self.make_runner(backend)

        role_result = runner.run("planner", "prompt", "planner-1", 1.0)

        self.assertFalse(role_result.usage.known)
        self.assertIsNone(role_result.usage.total_tokens)

    def test_timeout_closes_cooperative_fake_and_never_returns_late_result(self):
        backend = BlockingBackend()
        runner, _factory = self.make_runner(backend)

        with self.assertRaises(TimeoutError):
            runner.run("planner", "prompt", "planner-1", 0.01)

        self.assertTrue(backend.closed)
        self.assertTrue(backend.finished.wait(1.0))


if __name__ == "__main__":
    unittest.main()
