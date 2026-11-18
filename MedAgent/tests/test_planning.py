import json
import unittest

from medagent.contracts import Plan, RiskLevel, Route, SafetyResult
from medagent.planning import plan_request
from medagent.prompts import PLANNER_SYSTEM_PROMPT
from medagent.role_runner import RoleRunner
from test_role_runner import Factory, FakeBackend, completed_result


class PlanRequestTest(unittest.TestCase):
    question = '这道题需要查找什么证据？'
    session_id = 'planner-test'
    safety = SafetyResult(RiskLevel.HIGH, ('TEST_RULE',), 'test')

    def make_runner(self, risk_level):
        payload = {
            'route': 'SEARCH',
            'risk_level': risk_level,
            'search_query': 'evidence query',
            'constraints': ['use supplied evidence'],
            'reason_codes': ['EVIDENCE_NEEDED'],
        }
        backend = FakeBackend(completed_result(self.session_id, payload))
        factory = Factory(backend)
        return RoleRunner(config={}, backend_factory=factory), backend, factory

    def test_valid_search_sends_question_and_floor_once_and_returns_plan(self):
        runner, backend, factory = self.make_runner('HIGH')

        plan = plan_request(self.question, self.safety, runner, self.session_id, 1.0)

        self.assertEqual(plan, Plan(Route.SEARCH, RiskLevel.HIGH, 'evidence query',
                                    ('use supplied evidence',), ('EVIDENCE_NEEDED',)))
        [(prompt, session_id)] = backend.calls
        self.assertEqual(session_id, self.session_id)
        self.assertEqual(json.loads(prompt), {'question': self.question, 'risk_floor': 'HIGH'})
        [request] = factory.calls
        self.assertEqual(request['system_prompt'], PLANNER_SYSTEM_PROMPT)

    def test_plan_below_supplied_risk_floor_is_rejected(self):
        runner, _, _ = self.make_runner('LOW')

        with self.assertRaisesRegex(ValueError, 'below the safety floor'):
            plan_request(self.question, self.safety, runner, self.session_id, 1.0)
