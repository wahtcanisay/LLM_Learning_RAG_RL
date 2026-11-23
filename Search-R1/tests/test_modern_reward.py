import unittest

from search_r1_modern.reward import compute_score


class ModernRewardTest(unittest.TestCase):
    def test_single_response_answer_matches_gold_alias(self):
        score = compute_score(
            data_source="nq",
            solution_str="<think>evidence</think><answer>The Aurora, Ohio.</answer>",
            ground_truth={"target": ["Aurora Ohio", "Aurora, Ohio"]},
        )
        self.assertEqual(score, 1)

    def test_last_answer_wins_and_invalid_format_scores_zero(self):
        self.assertEqual(
            compute_score(
                data_source="hotpotqa",
                solution_str="<answer>wrong</answer><answer>right</answer>",
                ground_truth={"target": "right"},
            ),
            1,
        )
        self.assertEqual(
            compute_score("nq", "answer without tags", {"target": "answer"}),
            0,
        )


if __name__ == "__main__":
    unittest.main()
