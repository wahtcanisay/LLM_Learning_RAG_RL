import numpy as np
import pytest

from medical_graphrag.data.retrieval_passages import RetrievalPassage
from medical_graphrag.retrieval.graph_edges import (
    build_adjacent_edges,
    build_similarity_edges,
)


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _p(passage_id, doc_id, order):
    return RetrievalPassage(passage_id=passage_id, doc_id=doc_id, order=order,
                            title="t", content="c", source="s")


def test_similarity_edges_projects_pairs_for_membership():
    emb = np.array([
        _unit([1.0, 0.0]),
        _unit([0.95, 0.312]),
        _unit([0.0, 1.0]),
    ], dtype=np.float32)
    edges = build_similarity_edges(["d0", "d1", "d2"], emb, k=5, min_cosine=0.90)
    pairs = {(a, b) for a, b, _w in edges}
    assert ("d0", "d1") in pairs or ("d1", "d0") in pairs
    for a, b, w in edges:
        assert a != b
        assert 0.90 <= w <= 1.0 + 1e-6
    assert not any("d2" in (a, b) for a, b, _ in edges)


def test_similarity_edges_union_knn():
    emb = np.array([
        _unit([1.0, 0.0, 0.0]),
        _unit([0.0, 1.0, 0.0]),
        _unit([0.0, 0.99, 0.1]),
        _unit([0.0, 0.1, 0.99]),
    ], dtype=np.float32)
    edges = build_similarity_edges(["d0", "d1", "d2", "d3"], emb, k=1, min_cosine=0.5)
    pairs = {(a, b) for a, b, _ in edges}
    assert ("d1", "d2") in pairs or ("d2", "d1") in pairs


def test_similarity_edges_never_selects_more_than_k_per_source_when_self_is_absent():
    embeddings = np.ones((4, 2), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    edges = build_similarity_edges(
        ["a", "b", "c", "d"], embeddings, k=1, min_cosine=0.5
    )
    # union-kNN contains at most n*k undirected pairs because every source
    # contributes at most k non-self candidates, even when FAISS tie ordering
    # omits that source's own row from the first k+1 results.
    assert len(edges) <= 4


def test_similarity_edges_no_self_loop_no_duplicate_pairs():
    emb = np.array([
        _unit([1.0, 0.0, 0.0]),
        _unit([0.9, 0.1, 0.0]),
        _unit([0.9, -0.1, 0.0]),
        _unit([0.0, 1.0, 0.0]),
    ], dtype=np.float32)
    edges = build_similarity_edges(["a", "b", "c", "d"], emb, k=5, min_cosine=0.5)
    assert len(edges) > 0
    pairs = [(a, b) for a, b, _ in edges]
    assert len(set(pairs)) == len(pairs)  # 无重复无向边
    for a, b, w in edges:
        assert a != b                     # 无自环
        assert 0.5 <= w <= 1.0 + 1e-6     # 权重域


def test_similarity_edges_validates_inputs():
    with pytest.raises(ValueError, match="unique"):
        build_similarity_edges(["a", "a"], np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        build_similarity_edges(["a", "b"], np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="row count"):
        build_similarity_edges(["a", "b"], np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        build_similarity_edges(["a", "b"], np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="L2-normalized"):
        build_similarity_edges(["a", "b"], np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))


def test_similarity_edges_rejects_bad_params():
    emb = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="k must be"):
        build_similarity_edges(["a", "b"], emb, k=0)
    with pytest.raises(ValueError, match="min_cosine"):
        build_similarity_edges(["a", "b"], emb, min_cosine=1.5)
    with pytest.raises(ValueError, match="scale"):
        build_similarity_edges(["a", "b"], emb, scale=0.0)


def test_adjacent_edges_two_docs_three_chunks_each():
    passages = [_p("A0", "A", 0), _p("A1", "A", 1), _p("A2", "A", 2),
                _p("B0", "B", 0), _p("B1", "B", 1), _p("B2", "B", 2)]
    edges, gaps = build_adjacent_edges(passages)
    assert len(edges) == 4
    assert all(w == 1.0 for _, _, w in edges)
    assert not any(a.startswith("A") and b.startswith("B") for a, b, _ in edges)
    assert gaps == []


def test_adjacent_edges_order_gap_never_crosses():
    passages = [_p("A0", "A", 0), _p("A1", "A", 2), _p("A2", "A", 3)]
    edges, gaps = build_adjacent_edges(passages)
    assert edges == [("A1", "A2", 1.0)]
    assert gaps == [("A", 0, 2)]


def test_adjacent_edges_duplicate_order_fails():
    passages = [_p("A0", "A", 0), _p("A1", "A", 0)]
    with pytest.raises(ValueError, match="duplicate order"):
        build_adjacent_edges(passages)


def test_adjacent_edges_negative_order_fails():
    passages = [_p("A0", "A", -1)]
    with pytest.raises(ValueError, match="negative"):
        build_adjacent_edges(passages)


def test_adjacent_edges_missing_order_fails_with_contract_error():
    passages = [_p("A0", "A", None), _p("A1", "A", 1)]
    with pytest.raises(ValueError, match="negative/missing order"):
        build_adjacent_edges(passages)
