from pathlib import Path

import pytest

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl, write_qrels
from medical_graphrag.data.schemas import Document, PubMedQARecord, Qrel


def test_pubmedqa_record_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="answer"):
        PubMedQARecord(
            pmid="1",
            question="Question?",
            contexts=("Evidence",),
            answer="unknown",
            long_answer="Conclusion",
            year="2020",
            split="dev",
        )


def test_document_requires_content() -> None:
    with pytest.raises(ValueError, match="content"):
        Document("PMID:1", "Title", "", "pubmedqa", "2020")


def test_qrel_requires_binary_relevance() -> None:
    with pytest.raises(ValueError, match="relevance"):
        Qrel("1", "PMID:1", 2)


def test_atomic_writers_create_reproducible_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "value.json"
    jsonl_path = tmp_path / "rows.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    write_json(json_path, {"answer": "yes"})
    write_jsonl(jsonl_path, [{"b": 2, "a": 1}])
    write_qrels(qrels_path, [("q1", "d1", 1)])

    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert jsonl_path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'
    assert qrels_path.read_text(encoding="utf-8") == "query_id\tdoc_id\trelevance\nq1\td1\t1\n"
    assert len(sha256_file(json_path)) == 64
