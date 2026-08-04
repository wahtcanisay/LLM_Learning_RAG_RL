import pytest

from medical_graphrag.evaluation.retrieval import evaluate_rankings, read_qrels


def test_single_gold_metrics_are_exact() -> None:
    qrels = {"q1": "d1", "q2": "d2"}
    rankings = {
        "q1": ["d1", "x", "y"],
        "q2": ["x", "d2", "y"],
    }
    result = evaluate_rankings(qrels, rankings, ks=(1, 2, 10))
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["recall@10"] == 1.0
    assert result["mrr@10"] == 0.75
    assert round(result["ndcg@10"], 6) == round((1.0 + 1.0 / 1.584962500721156) / 2, 6)


def test_rankings_deduplicate_document_ids_before_scoring() -> None:
    result = evaluate_rankings({"q1": "d1"}, {"q1": ["x", "x", "d1"]}, ks=(1, 2))
    assert result["recall@2"] == 1.0
    assert result["mrr@10"] == 0.5


def test_empty_qrels_are_rejected() -> None:
    with pytest.raises(ValueError, match="qrels"):
        evaluate_rankings({}, {})


def test_multi_gold_recall_is_fraction_of_relevant_found() -> None:
    qrels = {"q1": ["d1", "d2"]}
    rankings = {"q1": ["d1", "d3", "d2"]}
    result = evaluate_rankings(qrels, rankings, ks=(1, 5, 10))
    assert result["recall@1"] == 0.5  # 1 of 2 gold in top-1
    assert result["recall@5"] == 1.0  # both gold in top-5
    assert result["mrr@10"] == 1.0  # first gold at rank 1
    dcg = 1.0 / 1.0 + 1.0 / 2.0  # d1@1, d2@3 -> 1/log2(2) + 1/log2(4)
    idcg = 1.0 + 1.0 / 1.584962500721156  # two gold -> 1/log2(2) + 1/log2(3)
    assert abs(result["ndcg@10"] - dcg / idcg) < 1e-9


def test_multi_gold_none_in_top_k_zeros() -> None:
    result = evaluate_rankings({"q1": ["d1", "d2"]}, {"q1": ["x", "y"]}, ks=(10,))
    assert result["recall@10"] == 0.0
    assert result["mrr@10"] == 0.0
    assert result["ndcg@10"] == 0.0


def test_read_qrels_returns_lists_for_multi_gold(tmp_path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text("query_id\tdoc_id\trelevance\nq1\td1\t1\nq1\td2\t1\nq2\td3\t1\n", encoding="utf-8")
    qrels = read_qrels(path)
    assert qrels == {"q1": ["d1", "d2"], "q2": ["d3"]}
