"""Passage–Passage edge builders (pure functions, no graph side effects).

Similarity and Adjacent are mutually exclusive: build_graph_index must call
exactly one of them per index (spec §6.3).

Similarity edges expect L2-normalized passage embeddings (the artifact produced
by ``embed_documents_full``), so IndexFlatIP inner product equals cosine. The
builder validates input shape/finiteness/normalization and fails closed.
"""
import faiss
import numpy as np

from medical_graphrag.data.retrieval_passages import RetrievalPassage

_NORM_TOLERANCE = 1e-3


def _validate_embeddings(
    passage_ids: list[str],
    embeddings: np.ndarray,
) -> None:
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("passage_ids must be unique")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    if embeddings.shape[0] != len(passage_ids):
        raise ValueError("embedding row count must equal passage count")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
        raise ValueError("embeddings must be L2-normalized for cosine similarity")


def build_similarity_edges(
    passage_ids: list[str],
    embeddings: np.ndarray,
    *,
    k: int = 5,
    min_cosine: float = 0.50,
    scale: float = 1.0,
) -> list[tuple[str, str, float]]:
    """kNN similarity soft edges over L2-normalized passage embeddings.

    - validate unique passage IDs, 2-D finite unit-norm embeddings;
    - request k+1 neighbours, drop the self hit;
    - drop candidates below ``min_cosine``;
    - union-kNN: keep undirected pair (lo, hi) if EITHER direction is in its
      top-k;
    - dedup identical pairs (tie-break deterministically by passage id);
    - edge weight = cosine (dot product on unit vectors) × ``scale``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (0 <= min_cosine <= 1):
        raise ValueError("min_cosine must be in [0, 1]")
    if scale <= 0:
        raise ValueError("similarity_edge_scale must be > 0")
    _validate_embeddings(passage_ids, embeddings)
    n = len(passage_ids)
    if n == 0:
        return []
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.asarray(embeddings, dtype=np.float32))
    k_eff = min(k + 1, n)
    scores, idx = index.search(np.asarray(embeddings, dtype=np.float32), k_eff)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        # FAISS returns neighbours in descending score order; ties are broken
        # by index, which is stable for a fixed input order.
        selected_non_self = 0
        for pos in range(k_eff):
            j = int(idx[i][pos])
            if j == i:
                continue
            if selected_non_self >= k:
                break
            selected_non_self += 1
            s = float(scores[i][pos])
            if s < min_cosine:
                continue
            pairs.add((min(i, j), max(i, j)))

    # Membership comes from union-kNN, but the frozen embeddings are the source
    # of truth for edge weights (spec section 6.5).
    edges = []
    for i, j in pairs:
        a, b = passage_ids[i], passage_ids[j]
        lo, hi = (a, b) if a <= b else (b, a)
        weight = float(np.dot(embeddings[i], embeddings[j])) * scale
        edges.append((lo, hi, weight))
    return sorted(edges, key=lambda edge: (edge[0], edge[1]))


def build_adjacent_edges(
    passages: list[RetrievalPassage],
) -> tuple[list[tuple[str, str, float]], list[tuple[str, int, int]]]:
    """Same-document adjacent edges, weight strictly 1.0.

    Group by doc_id, sort by numeric order, only connect ``next.order ==
    order + 1``. Order gaps are recorded and never crossed; duplicate or
    negative order raises. Returns ``(edges, gaps)`` where gaps is
    ``[(doc_id, from_order, to_order)]``.
    """
    from collections import defaultdict

    by_doc: dict[str, list[RetrievalPassage]] = defaultdict(list)
    for passage in passages:
        by_doc[passage.doc_id].append(passage)

    edges: list[tuple[str, str, float]] = []
    gaps: list[tuple[str, int, int]] = []
    for doc_id in sorted(by_doc):
        raw_group = by_doc[doc_id]
        if any(type(p.order) is not int or p.order < 0 for p in raw_group):
            raise ValueError(f"doc {doc_id} has negative/missing order")
        group = sorted(raw_group, key=lambda p: p.order)
        orders = [p.order for p in group]
        if len(set(orders)) != len(orders):
            raise ValueError(f"doc {doc_id} has duplicate order values")
        for prev, current in zip(group, group[1:]):
            if current.order == prev.order + 1:
                a, b = prev.passage_id, current.passage_id
                lo, hi = (a, b) if a <= b else (b, a)
                edges.append((lo, hi, 1.0))
            else:
                gaps.append((doc_id, prev.order, current.order))
    edges.sort(key=lambda edge: (edge[0], edge[1]))
    return edges, gaps
