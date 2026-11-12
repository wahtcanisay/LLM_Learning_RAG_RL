from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError

from medagent.contracts import Plan, RiskLevel, Route, SafetyResult, validate_plan


BASE_PAYLOAD = {
    "route": "SEARCH",
    "risk_level": "MEDIUM",
    "search_query": "aspirin contraindications",
    "constraints": ["use medical evidence"],
    "reason_codes": ["MEDICAL_QUESTION"],
}


def safety(floor: RiskLevel = RiskLevel.LOW) -> SafetyResult:
    return SafetyResult(
        risk_floor=floor,
        rule_ids=("RULE-1",),
        rules_version="v1",
    )


class ValidatePlanTest(unittest.TestCase):
    def test_valid_search_is_normalized_without_mutating_input(self):
        payload = copy.deepcopy(BASE_PAYLOAD)
        original = copy.deepcopy(payload)

        result = validate_plan(payload, safety())

        self.assertEqual(payload, original)
        self.assertEqual(
            result,
            Plan(
                route=Route.SEARCH,
                risk_level=RiskLevel.MEDIUM,
                search_query="aspirin contraindications",
                constraints=("use medical evidence",),
                reason_codes=("MEDICAL_QUESTION",),
            ),
        )
        self.assertIsInstance(result.constraints, tuple)
        self.assertIsInstance(result.reason_codes, tuple)

    def test_valid_direct_and_emergency_routes(self):
        cases = [
            (
                {
                    **BASE_PAYLOAD,
                    "route": "DIRECT",
                    "risk_level": "LOW",
                    "search_query": "",
                },
                safety(RiskLevel.LOW),
                Route.DIRECT,
            ),
            (
                {
                    **BASE_PAYLOAD,
                    "route": "EMERGENCY",
                    "risk_level": "EMERGENCY",
                    "search_query": "",
                },
                safety(RiskLevel.EMERGENCY),
                Route.EMERGENCY,
            ),
        ]
        for payload, safety_result, expected_route in cases:
            with self.subTest(route=expected_route):
                self.assertIs(
                    validate_plan(payload, safety_result).route,
                    expected_route,
                )

    def test_payload_and_safety_types_are_checked(self):
        with self.assertRaisesRegex(TypeError, "payload"):
            validate_plan([], safety())
        with self.assertRaisesRegex(TypeError, "safety"):
            validate_plan(BASE_PAYLOAD, object())

    def test_every_required_field_is_required(self):
        for field in BASE_PAYLOAD:
            payload = copy.deepcopy(BASE_PAYLOAD)
            payload.pop(field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_plan(payload, safety())

    def test_unknown_route_and_risk_are_rejected(self):
        for field, value in (("route", "TOOL"), ("risk_level", "CRITICAL")):
            payload = {**BASE_PAYLOAD, field: value}
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_plan(payload, safety())

    def test_route_risk_and_query_must_be_strings(self):
        for field, value in (
            ("route", 1),
            ("risk_level", None),
            ("search_query", ["query"]),
        ):
            payload = {**BASE_PAYLOAD, field: value}
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    validate_plan(payload, safety())

    def test_constraints_and_reason_codes_must_be_string_lists(self):
        cases = [
            ("constraints", ("tuple",)),
            ("reason_codes", "not-a-list"),
            ("constraints", ["ok", 1]),
            ("reason_codes", [None]),
        ]
        for field, value in cases:
            payload = {**BASE_PAYLOAD, field: value}
            with self.subTest(field=field, value=value):
                with self.assertRaises(TypeError):
                    validate_plan(payload, safety())

    def test_search_requires_non_empty_query(self):
        for query in ("", "   "):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "SEARCH"):
                    validate_plan(
                        {**BASE_PAYLOAD, "search_query": query},
                        safety(),
                    )

    def test_non_search_routes_require_exactly_empty_query(self):
        for route, risk in (("DIRECT", "LOW"), ("EMERGENCY", "EMERGENCY")):
            with self.subTest(route=route):
                with self.assertRaisesRegex(ValueError, "non-SEARCH"):
                    validate_plan(
                        {
                            **BASE_PAYLOAD,
                            "route": route,
                            "risk_level": risk,
                            "search_query": "unexpected query",
                        },
                        safety(),
                    )

    def test_plan_risk_cannot_be_below_safety_floor(self):
        downgrade_cases = [
            (RiskLevel.MEDIUM, "LOW"),
            (RiskLevel.HIGH, "MEDIUM"),
            (RiskLevel.EMERGENCY, "HIGH"),
        ]
        for floor, proposed in downgrade_cases:
            with self.subTest(floor=floor, proposed=proposed):
                with self.assertRaisesRegex(ValueError, "below the safety floor"):
                    validate_plan(
                        {**BASE_PAYLOAD, "risk_level": proposed},
                        safety(floor),
                    )

    def test_emergency_route_and_floor_cannot_be_downgraded(self):
        with self.assertRaisesRegex(ValueError, "EMERGENCY route"):
            validate_plan(
                {
                    **BASE_PAYLOAD,
                    "route": "EMERGENCY",
                    "risk_level": "HIGH",
                    "search_query": "",
                },
                safety(RiskLevel.LOW),
            )

        with self.assertRaisesRegex(ValueError, "cannot be downgraded"):
            validate_plan(
                {
                    **BASE_PAYLOAD,
                    "route": "SEARCH",
                    "risk_level": "EMERGENCY",
                },
                safety(RiskLevel.EMERGENCY),
            )

    def test_invalid_risk_floor_type_is_rejected(self):
        malformed = SafetyResult(  # type: ignore[arg-type]
            risk_floor="HIGH",
            rule_ids=(),
            rules_version="v1",
        )
        with self.assertRaisesRegex(TypeError, "risk_floor"):
            validate_plan(BASE_PAYLOAD, malformed)

    def test_plan_is_frozen(self):
        plan = validate_plan(BASE_PAYLOAD, safety())
        with self.assertRaises(FrozenInstanceError):
            plan.route = Route.DIRECT  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
