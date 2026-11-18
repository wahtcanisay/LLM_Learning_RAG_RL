import unittest

from MedSearch.tools.review_distillation_v2_candidates import build_ledger


class ReviewDistillationV2CandidatesTest(unittest.TestCase):
    def test_manual_test_overlap_and_neighbor_queue_do_not_approve_any_candidate(self):
        candidates = [
            {"question_id": "a", "source_id": 1, "question": "Question A?", "references": ["Gold A"], "split": "unassigned"},
            {"question_id": "b", "source_id": 2, "question": "Question B?", "references": ["Gold B"], "split": "unassigned"},
            {"question_id": "c", "source_id": 3, "question": "Question C?", "references": ["Gold C"], "split": "unassigned"},
        ]
        neighbors = [
            {"kind": "history", "candidate": "a", "match": "final_test:3", "matched_pool": "final_test", "cosine": 0.97},
            {"kind": "internal", "candidate": "b", "match": "c", "matched_pool": "v2", "cosine": 0.94},
        ]
        manual = [{"question_id": "a", "decision": "reject", "reason": "same_question_as_frozen_test", "reviewer": "Luna", "evidence": "paired text comparison"}]

        ledger = build_ledger(candidates, neighbors, manual)

        self.assertEqual([row["decision"] for row in ledger], ["reject", "quarantine", "quarantine"])
        self.assertEqual([row["review_basis"] for row in ledger], ["manual_pair_text", "embedding_queue", "embedding_queue"])
        self.assertEqual(ledger[0]["references"], ["Gold A"])
        self.assertEqual(ledger[0]["split"], "unassigned")
        self.assertEqual(len(ledger[1]["neighbor_evidence"]), 1)
        self.assertEqual(len(ledger[2]["neighbor_evidence"]), 1)

    def test_unflagged_record_remains_pending(self):
        candidates = [{"question_id": "a", "source_id": 1, "question": "Question?", "references": ["Gold"], "split": "unassigned"}]
        ledger = build_ledger(candidates, [], [])
        self.assertEqual(ledger[0]["decision"], "pending")
        self.assertEqual(ledger[0]["reason"], "medical_review_not_done")

    def test_lexical_only_neighbor_is_quarantined_and_traced(self):
        candidates = [{"question_id": "a", "source_id": 1, "question": "Question?", "references": ["Gold"], "split": "unassigned"}]
        lexical = [{"kind": "history", "candidate": "a", "match": "old:3", "matched_pool": "rl_train", "cosine": 0.82}]
        ledger = build_ledger(candidates, [], [], lexical_neighbors=lexical)
        self.assertEqual(ledger[0]["decision"], "quarantine")
        self.assertEqual(ledger[0]["lexical_neighbor_evidence"][0]["match"], "old:3")

    def test_unknown_manual_id_or_neighbor_id_is_rejected(self):
        candidates = [{"question_id": "a", "source_id": 1, "question": "Question?", "references": ["Gold"], "split": "unassigned"}]
        with self.assertRaises(ValueError):
            build_ledger(candidates, [], [{"question_id": "missing", "decision": "reject", "reason": "x", "reviewer": "Luna", "evidence": "x"}])
        with self.assertRaises(ValueError):
            build_ledger(candidates, [{"kind": "history", "candidate": "missing", "match": "other", "cosine": 0.95}], [])


if __name__ == "__main__":
    unittest.main()
