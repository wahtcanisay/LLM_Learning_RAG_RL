import pytest

from medical_graphrag.retrieval.bm25 import collapse_chunk_hits


def test_collapse_uses_max_score_and_stable_ties() -> None:
    metadata = {
        "c1": {"doc_id": "d1"},
        "c2": {"doc_id": "d1"},
        "c3": {"doc_id": "d2"},
        "c4": {"doc_id": "d0"},
    }
    hits = [
        {"chunk_id": "c1", "chunk_rank": 1, "score": 4.0},
        {"chunk_id": "c3", "chunk_rank": 2, "score": 5.0},
        {"chunk_id": "c4", "chunk_rank": 3, "score": 5.0},
        {"chunk_id": "c2", "chunk_rank": 4, "score": 6.0},
    ]

    ranking = collapse_chunk_hits(hits, metadata, min_unique_docs=2)

    assert [item["doc_id"] for item in ranking] == ["d1", "d2", "d0"]
    assert ranking[0]["score"] == 6.0
    assert ranking[0]["best_chunk_id"] == "c2"
    assert ranking[1]["best_chunk_rank"] == 2


def test_collapse_rejects_unknown_chunk() -> None:
    with pytest.raises(ValueError, match="unknown chunk_id: missing"):
        collapse_chunk_hits(
            [{"chunk_id": "missing", "chunk_rank": 1, "score": 1.0}],
            {},
            min_unique_docs=1,
        )


def test_collapse_rejects_short_document_ranking() -> None:
    with pytest.raises(ValueError, match="expected at least 2 unique documents, got 1"):
        collapse_chunk_hits(
            [{"chunk_id": "c1", "chunk_rank": 1, "score": 1.0}],
            {"c1": {"doc_id": "d1"}},
            min_unique_docs=2,
        )
