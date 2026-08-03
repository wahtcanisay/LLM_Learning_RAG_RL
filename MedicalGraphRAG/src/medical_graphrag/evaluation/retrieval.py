import math
from collections.abc import Mapping, Sequence


def evaluate_rankings(
    qrels: Mapping[str, str],
    rankings: Mapping[str, Sequence[str]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if not qrels:
        raise ValueError("qrels must not be empty")
    totals = {f"recall@{k}": 0.0 for k in ks}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id, gold_doc_id in qrels.items():
        ranking = list(dict.fromkeys(rankings.get(query_id, ())))
        rank = ranking.index(gold_doc_id) + 1 if gold_doc_id in ranking else None
        for k in ks:
            totals[f"recall@{k}"] += float(rank is not None and rank <= k)
        reciprocal_ranks.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0)
    count = len(qrels)
    result = {name: value / count for name, value in totals.items()}
    result["mrr@10"] = sum(reciprocal_ranks) / count
    result["ndcg@10"] = sum(ndcgs) / count
    return result
