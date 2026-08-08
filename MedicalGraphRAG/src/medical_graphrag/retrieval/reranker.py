"""Hybrid2: Qwen3-Reranker cross-encoder reranking.

Scores (query, document) pairs with the Qwen3-Reranker model and returns a
document ranking. It re-ranks the candidate union produced by the BM25 / Dense
/ Graph retrievers, replacing pure rank-based fusion (RRF) with a learned,
joint query-document scorer.

The model is loaded through the sentence-transformers ``CrossEncoder`` API
(``predict`` returns a relevance score per pair); sentence-transformers >= 2.7
and a transformers build that recognises the ``qwen3`` architecture are
required.
"""
from collections.abc import Sequence
from typing import Any

DEFAULT_RERANKER_MODEL = "models/Qwen3-Reranker-0.6B"


class QwenReranker:
    """Thin wrapper over the sentence-transformers CrossEncoder Qwen3Reranker."""

    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 128,
    ):
        from sentence_transformers.cross_encoder import CrossEncoder

        self.model = CrossEncoder(model_path)
        self.batch_size = batch_size

    def rerank(
        self, query: str, candidates: Sequence[dict[str, Any]]
    ) -> list[tuple[str, float]]:
        """Score ``(query, candidate)`` pairs; return ``(doc_id, score)`` desc.

        ``candidates`` is a sequence of ``{"doc_id": str, "content": str}``.
        Higher score = more relevant.
        """
        if not candidates:
            return []
        pairs = [[query, str(c["content"])] for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(
            (
                (str(c["doc_id"]), float(score))
                for c, score in zip(candidates, scores)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked
