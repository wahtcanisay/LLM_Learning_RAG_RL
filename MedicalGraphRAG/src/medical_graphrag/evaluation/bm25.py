import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import (
    evaluate_rankings,
    first_gold_rank,
    percentile,
    read_jsonl,
    read_qrels,
    validate_hit_rows,
)
from medical_graphrag.evaluation.document import evaluate_document_run
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits, validate_frozen_dataset


def _validate_run_context(
    run_context: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    metadata_path: Path,
    rankings_path: Path,
    dataset_manifest_sha256: str,
    dataset_artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    try:
        index_report = run_context["index"]
        search_report = run_context["search"]
    except KeyError as error:
        raise ValueError(f"missing run report: {error.args[0]}") from error
    if search_report.get("index_sha256") != index_report.get("index_sha256"):
        raise ValueError("search report does not match index report")
    if search_report.get("index_report_sha256") != run_context.get(
        "index_report_sha256"
    ):
        raise ValueError("search report does not match the supplied index report file")
    for report in (index_report, search_report):
        if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("run report does not match frozen dataset manifest")
    if index_report.get("dataset_artifact_hashes") != dataset_artifact_hashes:
        raise ValueError("index report does not match frozen dataset artifacts")
    if search_report.get("metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("search report metadata SHA-256 mismatch")
    if search_report.get("questions_sha256") != dataset_artifact_hashes.get(
        "questions.jsonl"
    ):
        raise ValueError("search report questions SHA-256 mismatch")
    if search_report.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError("search report rankings SHA-256 mismatch")
    if search_report.get("text_mode") != "abstract_only":
        raise ValueError("search report must use abstract_only text mode")
    requested_top_k = int(search_report["requested_top_k"])
    if requested_top_k <= 0:
        raise ValueError("requested_top_k must be positive")
    counts = [len(row["hits"]) for row in raw_rows]
    histogram = Counter(counts)
    actual_summary = {
        "query_count": len(raw_rows),
        "requested_top_k": requested_top_k,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(count < requested_top_k for count in counts),
        "hit_count_histogram": {
            str(count): frequency for count, frequency in sorted(histogram.items())
        },
    }
    reported_summary = {key: search_report.get(key) for key in actual_summary}
    if actual_summary != reported_summary:
        raise ValueError("search report hit summary mismatch")
    validate_hit_rows(raw_rows)
    return search_report


def evaluate_bm25_run(
    dataset_dir: Path,
    metadata_path: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    min_unique_docs: int = 0,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    dataset_manifest = validate_frozen_dataset(dataset_dir)
    dataset_manifest_sha256 = sha256_file(dataset_dir / "manifest.json")
    questions = {
        str(row["query_id"]): row for row in read_jsonl(dataset_dir / "questions.jsonl")
    }
    documents = {
        str(row["doc_id"]): row for row in read_jsonl(dataset_dir / "documents.jsonl")
    }
    chunks = read_jsonl(dataset_dir / "chunks.jsonl")
    metadata_rows = read_jsonl(metadata_path)
    metadata = {str(row["chunk_id"]): row for row in metadata_rows}
    raw_rows = read_jsonl(rankings_path)
    if len(raw_rows) != len(questions) or {
        str(row["query_id"]) for row in raw_rows
    } != set(questions):
        raise ValueError("ranking query set does not match questions")
    search_report = _validate_run_context(
        run_context,
        raw_rows,
        metadata_path,
        rankings_path,
        dataset_manifest_sha256,
        dataset_manifest["artifact_hashes"],
    )

    qrels = read_qrels(dataset_dir / "qrels.tsv")
    if set(qrels) != set(questions) or any(
        doc_id not in documents for doc_ids in qrels.values() for doc_id in doc_ids
    ):
        raise ValueError("qrels do not resolve an existing document for every question")

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
        for hit in row["hits"]:
            chunk_id = str(hit["chunk_id"])
            if chunk_id in metadata and str(hit["doc_id"]) != str(
                metadata[chunk_id]["doc_id"]
            ):
                raise ValueError(f"raw hit doc_id mismatch for {chunk_id}")
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
        if not ids:
            continue  # an empty split contributes no metrics
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
                "mean": statistics.fmean(values) if values else 0.0,
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
            },
        }

    test_ids = [
        query_id for query_id, row in questions.items() if row["split"] == "test"
    ]
    gold_ranks = {
        query_id: first_gold_rank(collapsed[query_id], qrels[query_id])
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
                "gold_doc_id": qrels[query_id][0],
                "gold_title": documents[qrels[query_id][0]]["title"],
                "gold_rank": gold_ranks[query_id],
                "gold_chunk_excerpt": next(
                    (
                        str(chunk["content"])[:500]
                        for chunk in chunks
                        if chunk["doc_id"] == qrels[query_id][0]
                    ),
                    "",
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
        "chunk_top_k": int(search_report["requested_top_k"]),
        "aggregation": "max_chunk_score",
        "bm25": {"k1": float(search_report["k1"]), "b": float(search_report["b"])},
        "text_mode": search_report["text_mode"],
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics


def evaluate_bm25_document_run(
    dataset_dir: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a BM25-document raw ranking (no chunk collapse)."""
    return evaluate_document_run(
        dataset_dir, rankings_path, output_dir, run_context=run_context
    )
