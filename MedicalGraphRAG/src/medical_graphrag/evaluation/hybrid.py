"""Hybrid RRF evaluation: fuse the audited BM25 and Dense document rankings.

Both source retrievers already produced audited chunk-level raw rankings. This
module collapses each to a document ranking (max chunk score, the same rule the
single-retriever evaluations use), fuses them with RRF, and evaluates the fused
document ranking with the exact same metric contract. Latency is not reported:
fusion is an offline post-processing step, not a retrieval call.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import (
    evaluate_rankings,
    read_jsonl,
    read_qrels,
    validate_hit_rows,
)
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits, validate_frozen_dataset
from medical_graphrag.retrieval.hybrid import DEFAULT_RRF_K, fuse_rrf

KNOWN_FAILURE_QUERIES = ("11570976", "18359123")


def _validate_source_reports(
    run_context: dict[str, Any],
    bm25_rankings_path: Path,
    dense_rankings_path: Path,
    metadata_path: Path,
    dataset_manifest_sha256: str,
    dataset_artifact_hashes: dict[str, str],
) -> None:
    """Bind each leg's search report to its own index report and rankings file.

    Mirrors the single-retriever validation so a swapped rankings file plus a
    hand-edited search report cannot pass silently: the search report must agree
    with its own index report, the supplied index-report file hash, the frozen
    dataset manifest, and the actual metadata/rankings files on disk.
    """
    for label, rankings_path in (("bm25", bm25_rankings_path), ("dense", dense_rankings_path)):
        try:
            leg = run_context[label]
            index_report = leg["index"]
            search_report = leg["search"]
        except KeyError as error:
            raise ValueError(f"missing hybrid source report: {error.args[0]}") from error
        if search_report.get("index_sha256") != index_report.get("index_sha256"):
            raise ValueError(f"{label} search report does not match its index report")
        if search_report.get("index_report_sha256") != leg.get("index_report_sha256"):
            raise ValueError(
                f"{label} search report does not match the supplied index report file"
            )
        for report in (index_report, search_report):
            if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
                raise ValueError(f"{label} report does not match frozen dataset manifest")
        if index_report.get("dataset_artifact_hashes") != dataset_artifact_hashes:
            raise ValueError(f"{label} index report does not match frozen dataset artifacts")
        if search_report.get("rankings_sha256") != sha256_file(rankings_path):
            raise ValueError(f"{label} rankings SHA-256 mismatch")
        if search_report.get("metadata_sha256") != sha256_file(metadata_path):
            raise ValueError(f"{label} metadata SHA-256 mismatch")
        if search_report.get("text_mode") != "abstract_only":
            raise ValueError(f"{label} search report must use abstract_only text mode")


def evaluate_hybrid_run(
    dataset_dir: Path,
    bm25_rankings_path: Path,
    dense_rankings_path: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    k: int = DEFAULT_RRF_K,
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
    metadata = {str(row["chunk_id"]): row for row in read_jsonl(metadata_path)}

    bm25_rows = read_jsonl(bm25_rankings_path)
    dense_rows = read_jsonl(dense_rankings_path)
    validate_hit_rows(bm25_rows)
    validate_hit_rows(dense_rows)
    for label, rows in (("bm25", bm25_rows), ("dense", dense_rows)):
        if len(rows) != len(questions) or {str(r["query_id"]) for r in rows} != set(
            questions
        ):
            raise ValueError(f"{label} ranking query set does not match questions")
    _validate_source_reports(
        run_context,
        bm25_rankings_path,
        dense_rankings_path,
        metadata_path,
        dataset_manifest_sha256,
        dataset_manifest["artifact_hashes"],
    )

    qrels = read_qrels(dataset_dir / "qrels.tsv")
    if set(qrels) != set(questions) or any(
        doc_id not in documents for doc_id in qrels.values()
    ):
        raise ValueError("qrels do not resolve one existing document for every question")

    bm25_by_qid = {str(r["query_id"]): r for r in bm25_rows}
    dense_by_qid = {str(r["query_id"]): r for r in dense_rows}

    fused_rankings: dict[str, list[str]] = {}
    for query_id, row in questions.items():
        for label, source_row in (
            ("bm25", bm25_by_qid[query_id]),
            ("dense", dense_by_qid[query_id]),
        ):
            if source_row["split"] != row["split"]:
                raise ValueError(f"split mismatch for {query_id} in {label} rankings")
        bm25_doc_ids = [
            str(item["doc_id"])
            for item in collapse_chunk_hits(
                bm25_by_qid[query_id]["hits"],
                metadata,
                min_unique_docs=min_unique_docs,
            )
        ]
        dense_doc_ids = [
            str(item["doc_id"])
            for item in collapse_chunk_hits(
                dense_by_qid[query_id]["hits"],
                metadata,
                min_unique_docs=min_unique_docs,
            )
        ]
        fused = fuse_rrf(bm25_doc_ids, dense_doc_ids, k=k)
        if not fused:
            raise ValueError(f"empty fused ranking for {query_id}")
        fused_rankings[query_id] = fused

    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        ids = [query_id for query_id, row in questions.items() if row["split"] == split]
        if not ids:
            continue  # an empty split contributes no metrics
        split_qrels = {query_id: qrels[query_id] for query_id in ids}
        split_rankings = {query_id: fused_rankings[query_id] for query_id in ids}
        metrics[split] = {
            "sample_count": len(ids),
            **evaluate_rankings(split_qrels, split_rankings, ks=(1, 5, 10)),
        }

    test_ids = [
        query_id for query_id, row in questions.items() if row["split"] == "test"
    ]
    gold_ranks = {
        query_id: (
            fused_rankings[query_id].index(qrels[query_id]) + 1
            if qrels[query_id] in fused_rankings[query_id]
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
                "top_documents": [
                    {
                        "doc_id": doc_id,
                        "title": documents[doc_id]["title"],
                        "fused_rank": rank,
                    }
                    for rank, doc_id in enumerate(fused_rankings[query_id][:10], start=1)
                ],
            }
            for query_id in ids
        ]
    cases["summary"] = {
        "success_at_10_available": len(success_all),
        "failure_at_10_available": len(failure_all),
        "saved_per_group_max": 5,
    }
    known: dict[str, Any] = {}
    for query_id in KNOWN_FAILURE_QUERIES:
        if query_id not in questions:
            continue
        gold = qrels[query_id]
        bm25_doc_ids = [
            str(item["doc_id"])
            for item in collapse_chunk_hits(
                bm25_by_qid[query_id]["hits"],
                metadata,
                min_unique_docs=min_unique_docs,
            )
        ]
        dense_doc_ids = [
            str(item["doc_id"])
            for item in collapse_chunk_hits(
                dense_by_qid[query_id]["hits"],
                metadata,
                min_unique_docs=min_unique_docs,
            )
        ]
        known[query_id] = {
            "question": questions[query_id]["question"],
            "gold_rank_bm25": (
                bm25_doc_ids.index(gold) + 1 if gold in bm25_doc_ids else None
            ),
            "gold_rank_dense": (
                dense_doc_ids.index(gold) + 1 if gold in dense_doc_ids else None
            ),
            "gold_rank_hybrid": (
                fused_rankings[query_id].index(gold) + 1
                if gold in fused_rankings[query_id]
                else None
            ),
        }
    cases["known_failures"] = known

    run_manifest = {
        **run_context,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rrf_k": k,
        "question_count": len(questions),
        "document_count": len(documents),
        "split_counts": {
            name: value["sample_count"] for name, value in metrics.items()
        },
        "aggregation": "max_chunk_score",
        "text_mode": "abstract_only",
        "bm25_rankings_sha256": sha256_file(bm25_rankings_path),
        "dense_rankings_sha256": sha256_file(dense_rankings_path),
        "bm25_config": {
            "k1": run_context["bm25"]["search"].get("k1"),
            "b": run_context["bm25"]["search"].get("b"),
            "requested_top_k": run_context["bm25"]["search"].get("requested_top_k"),
        },
        "dense_config": {
            "embedding_model": run_context["dense"]["search"].get("embedding_model"),
            "dim": run_context["dense"]["search"].get("dim"),
            "normalized": run_context["dense"]["search"].get("normalized"),
            "index_type": run_context["dense"]["search"].get("index_type"),
            "requested_top_k": run_context["dense"]["search"].get("requested_top_k"),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics
