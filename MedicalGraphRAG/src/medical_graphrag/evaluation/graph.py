"""LinearGraphRetriever evaluation.

The graph search emits chunk-level rankings exactly like BM25/Dense, so the
evaluation reuses the same folding (max chunk score per document), the same
`evaluate_rankings` metrics, and the same split discipline. The report
validation is graph-specific: it binds the search report to the graph index
report and to the actual rankings file.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import (
    evaluate_rankings,
    first_gold_rank,
    read_jsonl,
    read_qrels,
    validate_hit_rows,
)
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits, validate_frozen_dataset


def _validate_run_context(
    run_context: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    rankings_path: Path,
    dataset_manifest_sha256: str,
    dataset_artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    try:
        index_report = run_context["index"]
        search_report = run_context["search"]
    except KeyError as error:
        raise ValueError(f"missing run report: {error.args[0]}") from error
    if search_report.get("graph_sha256") != index_report.get("graph_sha256"):
        raise ValueError("search report does not match index report")
    if search_report.get("graph_build_report_sha256") != run_context.get(
        "index_report_sha256"
    ):
        raise ValueError("search report does not match the supplied index report file")
    for report in (index_report, search_report):
        if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("run report does not match frozen dataset manifest")
    if index_report.get("dataset_artifact_hashes") != dataset_artifact_hashes:
        raise ValueError("index report does not match frozen dataset artifacts")
    if search_report.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError("search report rankings SHA-256 mismatch")
    if search_report.get("questions_sha256") != dataset_artifact_hashes.get(
        "questions.jsonl"
    ):
        raise ValueError("search report questions SHA-256 mismatch")
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


def evaluate_graph_run(
    dataset_dir: Path,
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
    metadata = {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "order": int(row["order"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
        }
        for row in chunks
    }
    raw_rows = read_jsonl(rankings_path)
    if len(raw_rows) != len(questions) or {
        str(row["query_id"]) for row in raw_rows
    } != set(questions):
        raise ValueError("ranking query set does not match questions")
    search_report = _validate_run_context(
        run_context,
        raw_rows,
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
    detailed: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        query_id = str(row["query_id"])
        if row["split"] != questions[query_id]["split"]:
            raise ValueError(f"split mismatch for {query_id}")
        ranking = collapse_chunk_hits(
            row["hits"], metadata, min_unique_docs=min_unique_docs
        )
        collapsed[query_id] = [str(item["doc_id"]) for item in ranking]
        detailed[query_id] = [
            {**item, "title": documents[str(item["doc_id"])]["title"]}
            for item in ranking
        ]

    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        ids = [
            query_id for query_id, row in questions.items() if row["split"] == split
        ]
        if not ids:
            continue
        split_qrels = {query_id: qrels[query_id] for query_id in ids}
        split_rankings = {query_id: collapsed[query_id] for query_id in ids}
        metrics[split] = {
            "sample_count": len(ids),
            **evaluate_rankings(split_qrels, split_rankings, ks=(1, 5, 10)),
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
        "graph": {
            "ner_model": search_report["ner_model"],
            "embedding_model": search_report["embedding_model"],
            "config": search_report["config"],
            "adjacent_passage_edges": False,
        },
        "text_mode": "abstract_only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics
