import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import evaluate_rankings
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _qrels(path: Path) -> dict[str, str]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    result: dict[str, str] = {}
    for row in rows:
        query_id, doc_id, relevance = row.split("\t")
        if relevance != "1" or query_id in result:
            raise ValueError("qrels must contain one relevance=1 row per query")
        result[query_id] = doc_id
    return result


def evaluate_bm25_run(
    dataset_dir: Path,
    metadata_path: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    min_unique_docs: int = 0,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    questions = {
        str(row["query_id"]): row for row in _jsonl(dataset_dir / "questions.jsonl")
    }
    documents = {
        str(row["doc_id"]): row for row in _jsonl(dataset_dir / "documents.jsonl")
    }
    chunks = _jsonl(dataset_dir / "chunks.jsonl")
    metadata_rows = _jsonl(metadata_path)
    metadata = {str(row["chunk_id"]): row for row in metadata_rows}
    raw_rows = _jsonl(rankings_path)
    if len(raw_rows) != len(questions) or {
        str(row["query_id"]) for row in raw_rows
    } != set(questions):
        raise ValueError("ranking query set does not match questions")

    qrels = _qrels(dataset_dir / "qrels.tsv")
    if set(qrels) != set(questions) or any(
        doc_id not in documents for doc_id in qrels.values()
    ):
        raise ValueError("qrels do not resolve one existing document for every question")

    collapsed: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    detailed: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        query_id = str(row["query_id"])
        if row["split"] != questions[query_id]["split"]:
            raise ValueError(f"split mismatch for {query_id}")
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"invalid latency for {query_id}")
        ranking = collapse_chunk_hits(
            row["hits"], metadata, min_unique_docs=min_unique_docs
        )
        collapsed[query_id] = [str(item["doc_id"]) for item in ranking]
        detailed[query_id] = [
            {**item, "title": documents[str(item["doc_id"])]["title"]}
            for item in ranking
        ]
        latencies[query_id] = latency

    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        ids = [
            query_id
            for query_id, row in questions.items()
            if row["split"] == split
        ]
        split_qrels = {query_id: qrels[query_id] for query_id in ids}
        split_rankings = {query_id: collapsed[query_id] for query_id in ids}
        values = [latencies[query_id] for query_id in ids]
        split_metrics = evaluate_rankings(
            split_qrels, split_rankings, ks=(1, 5, 10)
        )
        metrics[split] = {
            "sample_count": len(ids),
            **split_metrics,
            "latency_ms": {
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            },
        }

    test_ids = [
        query_id for query_id, row in questions.items() if row["split"] == "test"
    ]
    gold_ranks = {
        query_id: (
            collapsed[query_id].index(qrels[query_id]) + 1
            if qrels[query_id] in collapsed[query_id]
            else None
        )
        for query_id in test_ids
    }
    success_all = [
        query_id
        for query_id in test_ids
        if gold_ranks[query_id] is not None and gold_ranks[query_id] <= 10
    ]
    failure_all = [
        query_id
        for query_id in test_ids
        if gold_ranks[query_id] is None or gold_ranks[query_id] > 10
    ]
    cases: dict[str, Any] = {}
    for label, ids in (("success", success_all[:5]), ("failure", failure_all[:5])):
        cases[label] = [
            {
                "query_id": query_id,
                "question": questions[query_id]["question"],
                "gold_doc_id": qrels[query_id],
                "gold_title": documents[qrels[query_id]]["title"],
                "gold_rank": gold_ranks[query_id],
                "gold_chunk_excerpt": next(
                    str(chunk["content"])[:500]
                    for chunk in chunks
                    if chunk["doc_id"] == qrels[query_id]
                ),
                "top_documents": detailed[query_id][:10],
            }
            for query_id in ids
        ]
    cases["summary"] = {
        "success_at_10_available": len(success_all),
        "failure_at_10_available": len(failure_all),
        "saved_per_group_max": 5,
    }

    run_manifest = {
        **run_context,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rankings_sha256": sha256_file(rankings_path),
        "question_count": len(questions),
        "document_count": len(documents),
        "chunk_metadata_count": len(metadata),
        "split_counts": {
            name: value["sample_count"] for name, value in metrics.items()
        },
        "chunk_top_k": 100,
        "aggregation": "max_chunk_score",
        "bm25": {"k1": 0.9, "b": 0.4},
        "text_mode": "abstract_only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics
