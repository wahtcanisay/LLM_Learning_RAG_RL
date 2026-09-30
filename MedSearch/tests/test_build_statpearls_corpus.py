import json
import tempfile
import unittest
from pathlib import Path

from MedSearch.tools.build_statpearls_corpus import build_combined_corpus


class CombinedCorpusTest(unittest.TestCase):
    def test_ids_match_line_order_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            textbooks = root / "textbooks.jsonl"
            textbooks.write_text(
                json.dumps({"id": "0", "title": "Book", "content": "A", "contents": "Book. A"}) + "\n",
                encoding="utf-8",
            )
            statpearls = root / "statpearls"
            statpearls.mkdir()
            (statpearls / "b.jsonl").write_text(
                json.dumps({"id": "article-b_0", "title": "B", "content": "B text", "contents": "B. B text"}) + "\n",
                encoding="utf-8",
            )
            (statpearls / "a.jsonl").write_text(
                json.dumps({"id": "article-a_0", "title": "A", "content": "A text", "contents": "A. A text"}) + "\n",
                encoding="utf-8",
            )
            output = root / "source" / "corpus.jsonl"
            self.assertEqual(build_combined_corpus(textbooks, statpearls, output), 3)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], ["0", "1", "2"])
            self.assertEqual([row["title"] for row in rows], ["Book", "A", "B"])
            with self.assertRaises(FileExistsError):
                build_combined_corpus(textbooks, statpearls, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
