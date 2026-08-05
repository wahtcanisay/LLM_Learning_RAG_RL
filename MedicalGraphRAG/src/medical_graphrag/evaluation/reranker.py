"""Hybrid2 reranker evaluation.

The reranker output is a document-level ranking (no chunk folding needed), so
this evaluation computes metrics directly from ``doc_ids`` per query with the
shared multi-gold `evaluate_rankings`. The run report is bound to the actual
ranking file, the questions file, and the frozen dataset manifest.
"""
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import (
    evaluate_rankings,
    first_gold_rank,
    read_jsonl,
    read_qrels,
)
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset


def _validate_report(
    run_context: dict[str, Any],
    rankings_path: Path,
    questions_path: Path,
    dataset_manifest_sha256: str,
) -> None:
    try:
        report = run_context["reranker"]
    except KeyError as error:
        raise ValueError(f"missing reranker run report: {error.args[0]}") from error
    if report.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError("reranker rankings SHA-256 mismatch")
    if report.get("questions_sha256") != sha256_file(questions_path):
        raise ValueError("reranker questions SHA-256 mismatch")
    if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("reranker report does not match frozen dataset manifest")


def evaluate_reranker_run(
    dataset_dir: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    dataset_manifest = validate_frozen_dataset(dataset_dir)
    dataset_manifest_sha256 = sha256_file(dataset_dir / "manifest.json")
    questions = {
        str(row["query_id"]): row for row in read_jsonl(dataset_dir / "questions.jsonl")
    }
    qrels = read_qrels(dataset_dir / "qrels.tsv")
    rows = read_jsonl(rankings_path)
    if len(rows) != len(questions) or {str(r["query_id"]) for r in rows} != set(
        questions
    ):
        raise ValueError("ranking query set does not match questions")
    _validate_report(
        run_context, rankings_path, dataset_dir / "questions.jsonl", dataset_manifest_sha256
    )
    documents = {
        str(row["doc_id"]): row for row in read_jsonl(dataset_dir / "documents.jsonl")
    }

    doc_rankings: dict[str, list[str]] = {}
    scores_by_query: dict[str, list[float]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if row["split"] != questions[query_id]["split"]:
            raise ValueError(f"split mismatch for {query_id}")
        doc_ids = [str(d) for d in row["doc_ids"]]
        scores = [float(s) for s in row["scores"]]
        if len(doc_ids) != len(scores):
            raise ValueError(f"doc_ids/scores length mismatch for {query_id}")
        if not all(math.isfinite(s) for s in scores):
            raise ValueError(f"non-finite reranker score for {query_id}")
        for doc_id in doc_ids:
            if doc_id not in documents:
                raise ValueError(f"unknown doc_id {doc_id} in reranked ranking")
        doc_rankings[query_id] = doc_ids
        scores_by_query[query_id] = scores

    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        ids = [qid for qid, row in questions.items() if row["split"] == split]
        if not ids:
            continue
        split_qrels = {qid: qrels[qid] for qid in ids}
        split_rankings = {qid: doc_rankings[qid] for qid in ids}
        metrics[split] = {
            "sample_count": len(ids),
            **evaluate_rankings(split_qrels, split_rankings, ks=(1, 5, 10)),
        }

    test_ids = [qid for qid, row in questions.items() if row["split"] == "test"]
    gold_ranks = {
        qid: first_gold_rank(doc_rankings[qid], qrels[qid]) for qid in test_ids
    }
    success_all = [
        qid
        for qid in test_ids
        if gold_ranks[qid] is not None and gold_ranks[qid] <= 10
    ]
    failure_all = [
        qid for qid in test_ids if gold_ranks[qid] is None or gold_ranks[qid] > 10
    ]
    cases: dict[str, Any] = {}
    for label, ids in (("success", success_all[:5]), ("failure", failure_all[:5])):
        cases[label] = [
            {
                "query_id": qid,
                "question": questions[qid]["question"],
                "gold_doc_id": qrels[qid][0],
                "gold_rank": gold_ranks[qid],
                "top_documents": [
                    {"doc_id": doc_id, "score": scores_by_query[qid][i]}
                    for i, doc_id in enumerate(doc_rankings[qid][:10])
                ],
            }
            for qid in ids
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
        "split_counts": {
            name: value["sample_count"] for name, value in metrics.items()
        },
        "reranker": {
            "model": run_context["reranker"].get("model"),
            "top_n": run_context["reranker"].get("top_n"),
            "bm25_rankings_sha256": run_context["reranker"].get(
                "bm25_rankings_sha256"
            ),
            "dense_rankings_sha256": run_context["reranker"].get(
                "dense_rankings_sha256"
            ),
            "graph_rankings_sha256": run_context["reranker"].get(
                "graph_rankings_sha256"
            ),
        },
        "text_mode": "abstract_only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics
