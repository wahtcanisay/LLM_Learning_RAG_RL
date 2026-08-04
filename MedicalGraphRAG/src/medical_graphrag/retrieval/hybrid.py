"""RRF (Reciprocal Rank Fusion) helpers for combining document rankings.

Fuses two document-level rankings (the folded BM25 and Dense rankings) on rank
position rather than raw score, so no score calibration is needed between a
lexical and a semantic retriever.
"""
from collections.abc import Sequence

DEFAULT_RRF_K = 60


def fuse_rrf(
    first: Sequence[str],
    second: Sequence[str],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[str]:
    """Fuse two document rankings with RRF.

    ``rrf_score(doc) = 1/(k + rank_first(doc)) + 1/(k + rank_second(doc))``.
    Documents absent from a ranking contribute 0 from that retriever. Returns
    ``doc_id`` values sorted by score descending, then by ``doc_id`` ascending
    so ties are deterministic.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in (first, second):
        for rank, doc_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
