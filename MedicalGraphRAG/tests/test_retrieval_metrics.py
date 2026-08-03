import pytest

from medical_graphrag.evaluation.retrieval import evaluate_rankings


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
