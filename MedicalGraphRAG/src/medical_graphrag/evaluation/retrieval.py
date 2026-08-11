import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _as_gold_set(gold: str | Sequence[str]) -> set[str]:
    """Accept a single doc id (our own qrels) or a sequence of ids (BEIR-style)."""
    if isinstance(gold, str):
        return {gold}
    return set(gold)


def first_gold_rank(ranking: Sequence[str], gold_docs: Sequence[str]) -> int | None:
    """Rank (1-based) of the first relevant document in a ranking, or None."""
    gold_set = set(gold_docs)
    for i, doc in enumerate(ranking, start=1):
        if doc in gold_set:
            return i
    return None


def evaluate_rankings(
    qrels: Mapping[str, str | Sequence[str]],
    rankings: Mapping[str, Sequence[str]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if not qrels:
        raise ValueError("qrels must not be empty")
    totals = {f"recall@{k}": 0.0 for k in ks}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id, gold in qrels.items():
        gold_set = _as_gold_set(gold)
        ranking = list(dict.fromkeys(rankings.get(query_id, ())))
        first_rank = next(
            (i for i, doc in enumerate(ranking, start=1) if doc in gold_set), None
        )
        for k in ks:
            if gold_set:
                totals[f"recall@{k}"] += len(gold_set & set(ranking[:k])) / len(
                    gold_set
                )
        reciprocal_ranks.append(
            1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
        )
        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, doc in enumerate(ranking[:10], start=1)
            if doc in gold_set
        )
        idcg = sum(
            1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), 10) + 1)
        )
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
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


def read_qrels(path: Path) -> dict[str, list[str]]:
    """Read qrels as ``{query_id: [doc_id, ...]}`` (multiple relevant docs allowed).

    Single-gold datasets (one row per query) become single-element lists, so this
    is backward compatible with the pubmedqa_hard_v1 contract.
    """
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        query_id, doc_id, relevance = (part.strip() for part in line.split("\t"))
        if relevance != "1" or not query_id or not doc_id:
            raise ValueError("qrels must contain non-empty IDs and relevance=1")
        result.setdefault(query_id, []).append(doc_id)
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
