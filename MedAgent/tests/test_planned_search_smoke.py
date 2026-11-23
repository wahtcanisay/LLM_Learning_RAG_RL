import json
import unittest
from unittest.mock import Mock

from medagent.contracts import CycleStatus, Evidence, RiskLevel, Route, SafetyResult, Usage
from medagent.planning import plan_request
from medagent.role_runner import RoleRunner
from medagent.search_agent import ActionResponse, SearchAgent
from medagent.verification import EvidenceVerifier
from medagent.workflow import run_search_cycle
from MedSearch.tools.build_medical_verifiable_rl import INSTRUCTION
from test_role_runner import FakeBackend, completed_result


class PlannedSearchSmokeTest(unittest.TestCase):
    def test_retry_rebuilds_model_input_and_verifies_only_the_new_evidence(self):
        question = "What is the sample identifier?"
        hint = "find the sample label"
        action_query = "sample label in the reference passage"
        retry_hint = "look up the exact identifier"
        retry_action = "identifier in the sample register"
        candidate = "ALPHA-7"
        first_candidate = "ALPHA"
        passage = "The sample identifier is ALPHA-7."
        first_passage = "The entry is listed under the ALPHA series."
        planner = FakeBackend(completed_result("planner-smoke", {
            "route": "SEARCH", "risk_level": "LOW", "search_query": hint,
            "constraints": [], "reason_codes": ["EVIDENCE_NEEDED"],
        }))
        retry = FakeBackend(completed_result("verifier-smoke", {
            "verdict": "RETRY", "failure_codes": ["MISSING_IDENTIFIER"], "retry_query_hint": retry_hint,
        }))
        accept = FakeBackend(completed_result("verifier-smoke", {
            "verdict": "PASS", "failure_codes": [], "retry_query_hint": None,
        }))
        runner = RoleRunner(config={}, backend_factory=Mock(side_effect=[planner, retry, accept]))
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = (
            lambda messages, **_: messages[0]["content"] + "\n[assistant]"
        )
        usage = Usage(None, None, None, False)
        request = Mock(side_effect=[
            ActionResponse(f"<search>{action_query}</search>", usage),
            ActionResponse(f"<answer>{first_candidate}</answer>", usage),
            ActionResponse(f"<search>{retry_action}</search>", usage),
            ActionResponse(f"<answer>{candidate}</answer>", usage),
        ])
        retrieve = Mock(side_effect=[[Evidence(first_passage)], [Evidence(passage)]])
        search_agent = SearchAgent(request, retrieve, {}, 1, tokenizer=tokenizer)

        safety = SafetyResult(RiskLevel.LOW, (), "synthetic-test")
        plan = plan_request(question, safety, runner, "planner-smoke", 1.0)
        self.assertEqual(plan.route, Route.SEARCH)
        result = run_search_cycle(
            question=question, instruction=INSTRUCTION, searchagent=search_agent,
            verifier=EvidenceVerifier(question=question, runner=runner, session_id="verifier-smoke", timeout_s=1.0),
            initial_search_hint=plan.search_query,
        )

        [first_search, first_answer, second_search, second_answer] = request.call_args_list
        for model_call, answer_call, direction, search_query, evidence in (
            (first_search, first_answer, hint, action_query, first_passage),
            (second_search, second_answer, retry_hint, retry_action, passage),
        ):
            expected = f"{INSTRUCTION}\nQuestion: {question}\nSearch hint: {direction}\n[assistant]"
            self.assertEqual(model_call.args[0], expected)
            self.assertEqual(answer_call.args[0], expected + (
                f"<search>{search_query}</search><information>{evidence}</information>"
            ))
        self.assertEqual([entry.args[0] for entry in retrieve.call_args_list], [action_query, retry_action])
        for backend, answer, evidence in ((retry, first_candidate, first_passage), (accept, candidate, passage)):
            [(audit_prompt, session_id)] = backend.calls
            self.assertEqual(session_id, "verifier-smoke")
            self.assertEqual(json.loads(audit_prompt), {
                "question": question, "candidate_answer": answer, "evidence": [evidence],
            })
        self.assertEqual((result.status, result.attempts), (CycleStatus.VERIFIED, 2))
        self.assertEqual(result.search.candidate_answer, candidate)
