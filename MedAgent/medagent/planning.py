import json

from .contracts import Plan, SafetyResult, validate_plan
from .role_runner import RoleRunner


def plan_request(
    question: str,
    safety: SafetyResult,
    runner: RoleRunner,
    session_id: str,
    timeout_s: float,
) -> Plan:
    """请求 Planner 并校验返回的计划。"""
    planner_input = {
        "question": question,
        "risk_floor": safety.risk_floor.value,
    }

    prompt = json.dumps(planner_input, ensure_ascii=False)

    result = runner.run(
        "planner",
        prompt,
        session_id,
        timeout_s,
    )

    return validate_plan(result.payload, safety)
