import unittest

from MedSearch.tools.build_distillation_v2_batch2 import select_reserve_batch


class BuildV2Batch2Test(unittest.TestCase):
    def test_fixed_seed_is_stable_and_keeps_source_content(self):
        raw = [
            {"Open-ended Verifiable Question": f"Question {i}?", "Ground-True Answer": f"Gold {i}"}
            for i in range(8)
        ]
        reserve = [
            {"question_id": f"medical-o1-verifiable:{i}", "source": "FreedomIntelligence/medical-o1-verifiable-problem", "source_id": i, "question": raw[i]["Open-ended Verifiable Question"], "references": [raw[i]["Ground-True Answer"]], "status": "reserve"}
            for i in range(8)
        ]
        first = select_reserve_batch(reserve, raw, 4, 20260919)
        second = select_reserve_batch(list(reversed(reserve)), raw, 4, 20260919)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(row["split"] == "unassigned" for row in first))
        self.assertEqual([row["source_id"] for row in first], sorted(row["source_id"] for row in first))
        for row in first:
            self.assertEqual(row["question"], raw[row["source_id"]]["Open-ended Verifiable Question"])
            self.assertEqual(row["references"], [raw[row["source_id"]]["Ground-True Answer"]])

    def test_rejects_nonreserve_and_tampered_source(self):
        raw = [{"Open-ended Verifiable Question": "Q?", "Ground-True Answer": "A"}]
        with self.assertRaises(ValueError):
            select_reserve_batch([{"status": "quarantine", "source_id": 0, "question_id": "medical-o1-verifiable:0", "question": "Q?", "references": ["A"]}], raw, 1, 1)
        with self.assertRaises(ValueError):
            select_reserve_batch([{"status": "reserve", "source_id": 0, "question_id": "medical-o1-verifiable:0", "question": "wrong", "references": ["A"]}], raw, 1, 1)


if __name__ == "__main__":
    unittest.main()
