import unittest

from MedSearch.tools.build_distillation_v2_batch2_review import merge_ledgers


class CombinedV2LedgerTest(unittest.TestCase):
    def test_all_rows_retained_but_reject_and_quarantine_never_eligible(self):
        first = [
            {"question_id": "a", "decision": "reject", "references": ["A"], "split": "unassigned"},
            {"question_id": "b", "decision": "pending", "references": ["B"], "split": "unassigned"},
        ]
        second = [
            {"question_id": "c", "decision": "quarantine", "references": ["C"], "split": "unassigned"},
            {"question_id": "d", "decision": "pending", "references": ["D"], "split": "unassigned"},
        ]
        merged = merge_ledgers(first, second)
        self.assertEqual([row["batch"] for row in merged], [1, 1, 2, 2])
        self.assertEqual([row["screening_eligible"] for row in merged], [False, True, False, True])
        self.assertTrue(all(not row["training_accepted"] for row in merged))
        self.assertTrue(all(row["split"] == "unassigned" for row in merged))
        self.assertEqual([row["references"] for row in merged], [["A"], ["B"], ["C"], ["D"]])

    def test_duplicate_ids_across_batches_fail_closed(self):
        with self.assertRaises(ValueError):
            merge_ledgers([{"question_id": "a", "decision": "pending", "split": "unassigned"}], [{"question_id": "a", "decision": "pending", "split": "unassigned"}])


if __name__ == "__main__":
    unittest.main()
