"""Tests for the BM25 search/build library functions (extracted to src)."""
import json
from pathlib import Path

import pytest

from medical_graphrag.data.io import sha256_directory, sha256_file
from medical_graphrag.retrieval.search_bm25 import (
    build_lucene_index,
    package_version,
    search_one,
    summarize_hit_counts,
    validate_index,
    _read_jsonl,
)


def test_scripts_read_pyserini_version_from_distribution_metadata() -> None:
    requested = []

    def resolver(package: str) -> str:
        requested.append(package)
        return "0.22.1"

    assert package_version("pyserini", resolver=resolver) == "0.22.1"
    assert requested == ["pyserini"]


def test_search_one_query_records_rank_score_and_doc_id() -> None:
    class Hit:
        def __init__(self, docid: str, score: float) -> None:
            self.docid = docid
            self.score = score

    class Searcher:
        def search(self, query: str, k: int):
            assert query == "alpha?"
            assert k == 1
            return [Hit("c1", 2.5)]

    result = search_one(
        Searcher(),
        {"query_id": "q1", "question": "alpha?", "split": "dev"},
        {"c1": "d1"},
        top_k=1,
        clock=iter([1.0, 1.012]).__next__,
    )

    assert result["query_id"] == "q1"
    assert result["split"] == "dev"
    assert result["latency_ms"] == 12.0
    assert result["hits"] == [
        {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "score": 2.5}
    ]


def test_search_jsonl_reader_preserves_unicode_paragraph_separator(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"chunk_id":"c1","title":"alpha beta"}\n', encoding="utf-8")

    rows = _read_jsonl(path)

    assert rows == [{"chunk_id": "c1", "title": "alpha beta"}]


def test_search_summary_records_rankings_shorter_than_requested_top_k() -> None:
    rows = [
        {"hits": [{"chunk_id": "c1"}] * 100},
        {"hits": [{"chunk_id": "c2"}] * 3},
    ]

    summary = summarize_hit_counts(rows, requested_top_k=100)

    assert summary == {
        "requested_top_k": 100,
        "min_hits": 3,
        "max_hits": 100,
        "short_ranking_count": 1,
        "hit_count_histogram": {"3": 1, "100": 1},
    }


def test_build_index_validation_rejects_collection_from_another_export(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    chunks = collection / "chunks.jsonl"
    chunks.write_text('{"id":"c1","contents":"alpha"}\n', encoding="utf-8")
    report = tmp_path / "export_report.json"
    report.write_text(
        json.dumps(
            {
                "text_mode": "abstract_only",
                "chunk_count": 1,
                "collection_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collection SHA-256 mismatch"):
        build_lucene_index(
            collection=collection,
            index=tmp_path / "index",
            report=tmp_path / "index_build.json",
            export_report=report,
        )


def test_search_input_validation_rejects_modified_index(tmp_path: Path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    (index / "segments_1").write_bytes(b"index")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"c1","doc_id":"d1"}\n', encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"query_id":"q1","question":"alpha"}\n', encoding="utf-8")
    report = tmp_path / "index_report.json"
    report.write_text(
        json.dumps(
            {
                "index_sha256": sha256_directory(index),
                "metadata_sha256": sha256_file(metadata),
                "dataset_artifact_hashes": {"questions.jsonl": sha256_file(questions)},
                "text_mode": "abstract_only",
            }
        ),
        encoding="utf-8",
    )
    (index / "segments_1").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        validate_index(index, metadata, questions, report)


def test_search_input_validation_rejects_questions_from_another_dataset(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index"
    index.mkdir()
    (index / "segments_1").write_bytes(b"index")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"c1","doc_id":"d1"}\n', encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"query_id":"q1","question":"changed"}\n', encoding="utf-8")
    report = tmp_path / "index_report.json"
    report.write_text(
        json.dumps(
            {
                "index_sha256": sha256_directory(index),
                "metadata_sha256": sha256_file(metadata),
                "text_mode": "abstract_only",
                "dataset_artifact_hashes": {"questions.jsonl": "0" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="questions SHA-256 mismatch"):
        validate_index(index, metadata, questions, report)
