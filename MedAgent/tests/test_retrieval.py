import unittest

from medagent.contracts import Evidence
from medagent.retrieval import normalize_evidence


class NormalizeEvidenceTest(unittest.TestCase):
    def test_preserves_rank_and_text_without_requiring_document_id(self):
        response = {"result": [[
            {"document": {"contents": "Title\nBody", "title": "Title", "id": 7}, "score": 1.2},
            {"document": {"contents": "Other body", "title": "Title"}},
        ]]}
        self.assertEqual(normalize_evidence(response), [
            Evidence(text="Title\nBody", title="Title", score=1.2),
            Evidence(text="Other body", title="Title", score=None),
        ])

    def test_empty_recall_and_invalid_response_are_distinct(self):
        self.assertEqual(normalize_evidence({"result": [[]]}), [])
        for response in ({"result": []}, {"result": [[{"document": {"title": "Missing contents"}}]]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                normalize_evidence(response)

if __name__ == "__main__":
    unittest.main()
