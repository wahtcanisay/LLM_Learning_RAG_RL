import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from medagent.contracts import CycleStatus, Evidence, SearchResult, VerificationResult, VerificationVerdict
from medagent.role_runner import RoleRunner
from medagent.verification import EvidenceVerifier, check_evidence
from medagent.workflow import run_search_cycle
from test_role_runner import FakeBackend, completed_result


class CheckEvidenceTest(unittest.TestCase):
    def test_empty_answers_use_the_stable_failure_code(self):
        for answer in ("", "  \t\n"):
            with self.subTest(answer=answer):
                self.assertEqual(check_evidence(SearchResult(0, answer, [], None)), ("EMPTY_ANSWER",))

    def test_nonempty_answer_clears_the_empty_answer_check(self):
        result = SearchResult(0, "candidate answer", [], None)
        self.assertEqual(check_evidence(result), ())


class EvidenceVerifierTest(unittest.TestCase):
    def test_empty_answer_abstains_without_a_role_call(self):
        runner = Mock(spec=RoleRunner)
        verifier = EvidenceVerifier(question="original question", runner=runner,
                                    session_id="verifier-test", timeout_s=1.0)

        result = verifier.verify(SearchResult(0, " \t", [], None))

        self.assertEqual(result, VerificationResult(VerificationVerdict.ABSTAIN, ("EMPTY_ANSWER",), None))
        runner.run.assert_not_called()

    def test_nonempty_answer_sends_one_structured_request(self):
        for verdict in ("PASS", "ABSTAIN"):
            with self.subTest(verdict=verdict):
                codes = [] if verdict == "PASS" else ["UNSUPPORTED_ANSWER"]
                payload = {"verdict": verdict, "failure_codes": codes, "retry_query_hint": None}
                backend = FakeBackend(completed_result("verifier-test", payload))
                factory = Mock(return_value=backend)
                runner = RoleRunner(config={}, backend_factory=factory)
                verifier = EvidenceVerifier(question="original question", runner=runner,
                                            session_id="verifier-test", timeout_s=1.0)
                search = SearchResult(0, "first answer", [Evidence("first passage"), Evidence("second passage")],
                                      "<think>private context</think>")

                result = verifier.verify(search)

                self.assertEqual(result, VerificationResult(VerificationVerdict(verdict), tuple(codes), None))
                factory.assert_called_once()
                [(prompt, session)] = backend.calls
                self.assertEqual(session, "verifier-test")
                self.assertEqual(json.loads(prompt), {"question": "original question", "candidate_answer": "first answer",
                                                     "evidence": ["first passage", "second passage"]})
                self.assertTrue(backend.closed)

    def test_retry_hint_and_feedback_reach_the_next_search(self):
        retry = FakeBackend(completed_result("verifier-test", {
            "verdict": "RETRY", "failure_codes": ["MISSING_EVIDENCE"], "retry_query_hint": "targeted search",
        }))
        accept = FakeBackend(completed_result("verifier-test", {
            "verdict": "PASS", "failure_codes": [], "retry_query_hint": None,
        }))
        runner = RoleRunner(config={}, backend_factory=Mock(side_effect=[retry, accept]))
        verifier = EvidenceVerifier(question="original question", runner=runner,
                                    session_id="verifier-test", timeout_s=1.0)
        searchagent = SimpleNamespace(run=Mock(side_effect=[
            SearchResult(0, "first answer", [Evidence("first passage")], None),
            SearchResult(1, "second answer", [Evidence("second passage")], None),
        ]))

        result = run_search_cycle("original question", "initial query", searchagent, verifier)

        self.assertEqual((result.status, result.attempts), (CycleStatus.VERIFIED, 2))
        calls = searchagent.run.call_args_list
        self.assertEqual(len(calls), 2)
        second_task = calls[1].kwargs["task"]
        self.assertEqual(second_task.query, "targeted search")
        self.assertEqual(second_task.verifier_feedback, ("MISSING_EVIDENCE",))
        [(prompt, _)] = accept.calls
        self.assertEqual(json.loads(prompt)["evidence"], ["second passage"])
