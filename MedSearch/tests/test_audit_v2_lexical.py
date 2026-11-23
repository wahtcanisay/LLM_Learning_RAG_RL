import unittest

from MedSearch.tools.audit_v2_lexical import find_nearest_pairs


class AuditV2LexicalTest(unittest.TestCase):
    def test_flags_textual_variants_but_not_unrelated_questions(self):
        candidate_questions = [
            "A child has fever and a red rash for two days. Which diagnosis is most likely?",
            "What is the function of the optic nerve?",
        ]
        history_questions = [
            "A child has fever and a red rash for two days. What diagnosis is most likely?",
            "Which protein is secreted by the liver?",
        ]
        pairs = find_nearest_pairs(candidate_questions, history_questions, threshold=0.75)
        self.assertEqual([(item["candidate_index"], item["match_index"]) for item in pairs], [(0, 0)])

    def test_internal_duplicate_keeps_one_pair(self):
        questions = [
            "What is the best treatment for a patient with condition A?",
            "What is the best treatment for a patient with condition A?",
            "How is condition B diagnosed?",
        ]
        pairs = find_nearest_pairs(questions, questions, threshold=0.95, same_pool=True)
        self.assertEqual([(item["candidate_index"], item["match_index"]) for item in pairs], [(0, 1)])


if __name__ == "__main__":
    unittest.main()
