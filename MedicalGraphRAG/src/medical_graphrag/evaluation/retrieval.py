import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_qrels(path: Path) -> dict[str, str]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    result: dict[str, str] = {}
    for row in rows:
        query_id, doc_id, relevance = row.split("\t")
        if relevance != "1" or query_id in result:
            raise ValueError("qrels must contain one relevance=1 row per query")
        result[query_id] = doc_id
    return result


def validate_hit_rows(raw_rows: list[dict[str, Any]]) -> None:
    """Shared ranking-shape checks: contiguous one-based ranks and finite scores."""
    for row in raw_rows:
        for expected_rank, hit in enumerate(row["hits"], start=1):
            score = float(hit["score"])
            if int(hit["chunk_rank"]) != expected_rank:
                raise ValueError("hit ranks must be contiguous and one-based")
            if not math.isfinite(score):
                raise ValueError("hit score must be finite")
