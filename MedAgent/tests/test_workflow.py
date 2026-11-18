from __future__ import annotations

import unittest

from medagent.contracts import (
    CycleStatus,
    SearchResult,
    SearchTask,
    VerificationResult,
    VerificationVerdict,
)
from medagent.workflow import run_search_cycle


def verdict(
    value: VerificationVerdict,
    *,
    failure_codes: tuple[str, ...] = (),
    retry_query_hint: str | None = None,
) -> VerificationResult:
    return VerificationResult(value, failure_codes, retry_query_hint)


class FakeSearchAgent:
    def __init__(self) -> None:
        self.tasks: list[SearchTask] = []

    def run(self, task: SearchTask) -> SearchResult:
        self.tasks.append(task)
        return SearchResult(
            attempt=task.attempt,
            candidate_answer=f"answer-{task.attempt}",
            evidence=[],
            context=None,
        )


class FakeVerifier:
    def __init__(self, *results: VerificationResult) -> None:
        self.results = list(results)
        self.calls: list[SearchResult] = []

    def verify(self, result: SearchResult) -> VerificationResult:
        self.calls.append(result)
        return self.results[len(self.calls) - 1]


class RunSearchCycleTest(unittest.TestCase):
    def run_cycle(
        self,
        *verifications: VerificationResult,
        max_retries: int = 2,
    ):
        search = FakeSearchAgent()
        verifier = FakeVerifier(*verifications)
        result = run_search_cycle(
            "What is the evidence?",
            "initial query",
            search,
            verifier,
            max_retries=max_retries,
        )
        return result, search, verifier

    def test_public_api_accepts_searchagent_keyword(self):
        search = FakeSearchAgent()
        verifier = FakeVerifier(verdict(VerificationVerdict.PASS))

        result = run_search_cycle(
            question="question",
            initial_query="query",
            searchagent=search,
            verifier=verifier,
        )

        self.assertIs(result.status, CycleStatus.VERIFIED)

    def test_first_pass_verifies_after_one_search(self):
        result, search, verifier = self.run_cycle(
            verdict(VerificationVerdict.PASS),
        )

        self.assertIs(result.status, CycleStatus.VERIFIED)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(search.tasks), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(search.tasks[0].attempt, 0)
        self.assertEqual(search.tasks[0].verifier_feedback, None)

    def test_first_abstain_ends_after_one_search(self):
        result, search, verifier = self.run_cycle(
            verdict(
                VerificationVerdict.ABSTAIN,
                failure_codes=("INSUFFICIENT_EVIDENCE",),
            ),
        )

        self.assertIs(result.status, CycleStatus.ABSTAINED)
        self.assertEqual(result.reason, "verifier_abstained")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(search.tasks), 1)
        self.assertEqual(len(verifier.calls), 1)

    def test_retry_then_pass_uses_hint_and_feedback(self):
        result, search, verifier = self.run_cycle(
            verdict(
                VerificationVerdict.RETRY,
                failure_codes=("MISSING_SOURCE",),
                retry_query_hint="  refined query  ",
            ),
            verdict(VerificationVerdict.PASS),
        )

        self.assertIs(result.status, CycleStatus.VERIFIED)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(search.tasks), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(search.tasks[1].attempt, 1)
        self.assertEqual(search.tasks[1].query, "refined query")
        self.assertEqual(search.tasks[1].verifier_feedback, ("MISSING_SOURCE",))

    def test_second_retry_abstains_without_third_search(self):
        result, search, verifier = self.run_cycle(
            verdict(
                VerificationVerdict.RETRY,
                failure_codes=("FIRST_FAILURE",),
                retry_query_hint="refined query",
            ),
            verdict(
                VerificationVerdict.RETRY,
                failure_codes=("STILL_BAD",),
                retry_query_hint="unused query",
            ),
        )

        self.assertIs(result.status, CycleStatus.ABSTAINED)
        self.assertEqual(result.reason, "retry_exhausted")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(search.tasks), 2)
        self.assertEqual(len(verifier.calls), 2)

    def test_retry_then_abstain_ends_after_two_searches(self):
        result, search, verifier = self.run_cycle(
            verdict(
                VerificationVerdict.RETRY,
                failure_codes=("NEEDS_REFINEMENT",),
                retry_query_hint="refined query",
            ),
            verdict(
                VerificationVerdict.ABSTAIN,
                failure_codes=("NO_SUPPORT",),
            ),
        )

        self.assertIs(result.status, CycleStatus.ABSTAINED)
        self.assertEqual(result.reason, "verifier_abstained")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(search.tasks), 2)
        self.assertEqual(len(verifier.calls), 2)

    def test_one_attempt_allows_initial_search_but_no_retry(self):
        result, search, verifier = self.run_cycle(
            verdict(
                VerificationVerdict.RETRY,
                failure_codes=("RETRY_REQUESTED",),
                retry_query_hint="unused query",
            ),
            max_retries=1,
        )

        self.assertIs(result.status, CycleStatus.ABSTAINED)
        self.assertEqual(result.reason, "retry_exhausted")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(search.tasks), 1)
        self.assertEqual(len(verifier.calls), 1)

    def test_retry_requires_non_empty_query_hint(self):
        for hint in (None, "", "   "):
            with self.subTest(hint=hint):
                with self.assertRaisesRegex(ValueError, "retry_query_hint"):
                    self.run_cycle(
                        verdict(
                            VerificationVerdict.RETRY,
                            failure_codes=("BAD_QUERY",),
                            retry_query_hint=hint,
                        ),
                    )

    def test_retry_bound_rejects_values_outside_one_or_two(self):
        for max_retries in (0, 3):
            with self.subTest(max_retries=max_retries):
                with self.assertRaisesRegex(ValueError, "max_retries"):
                    self.run_cycle(
                        verdict(VerificationVerdict.PASS),
                        max_retries=max_retries,
                    )


if __name__ == "__main__":
    unittest.main()
