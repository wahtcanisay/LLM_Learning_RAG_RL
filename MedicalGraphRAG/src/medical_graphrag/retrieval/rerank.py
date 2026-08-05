"""Hybrid2: rerank the BM25 / Dense / Graph candidate union with Qwen3-Reranker.

Extracted from the old ``scripts/rerank_candidates.py`` into a library
function. Validates the three source rankings against their search reports,
folds each to a document ranking (max chunk score), takes the top-N union as
candidates, scores every (query, doc) pair with Qwen3-Reranker, and writes a
document-level ranking file plus a hashed report.
"""
import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.bm25 import (
    collapse_chunk_hits,
    validate_frozen_dataset,
)
from medical_graphrag.retrieval.reranker import DEFAULT_RERANKER_MODEL, QwenReranker


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_source(
    rankings: Path,
    search_report: Path,
    label: str,
    dataset_manifest_sha256: str,
) -> None:
    report = json.loads(search_report.read_text(encoding="utf-8"))
    if report.get("rankings_sha256") != sha256_file(rankings):
        raise ValueError(f"{label} rankings SHA-256 mismatch")
    if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError(f"{label} search report does not match frozen dataset manifest")
    if report.get("text_mode") != "abstract_only":
        raise ValueError(f"{label} search report must use abstract_only text mode")


def _fold_doc_ranking(
    hits: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[str]:
    return [
        str(item["doc_id"])
        for item in collapse_chunk_hits(hits, metadata, min_unique_docs=0)
    ]


def run_rerank(
    *,
    bm25_rankings: Path,
    dense_rankings: Path,
    graph_rankings: Path,
    bm25_search_report: Path,
    dense_search_report: Path,
    graph_search_report: Path,
    questions: Path,
    documents: Path,
    chunks: Path,
    dataset_manifest: Path,
    output: Path,
    report: Path,
    top_n: int = 50,
    model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, Any]:
    """Rerank the three-way candidate union with Qwen3-Reranker."""
    dataset_manifest_data = validate_frozen_dataset(dataset_manifest.parent)
    dataset_manifest_sha256 = sha256_file(dataset_manifest)
    if sha256_file(documents) != dataset_manifest_data["artifact_hashes"]["documents.jsonl"]:
        raise ValueError("documents do not match frozen dataset manifest")
    if sha256_file(chunks) != dataset_manifest_data["artifact_hashes"]["chunks.jsonl"]:
        raise ValueError("chunks do not match frozen dataset manifest")
    _validate_source(
        bm25_rankings, bm25_search_report, "bm25", dataset_manifest_sha256
    )
    _validate_source(
        dense_rankings, dense_search_report, "dense", dataset_manifest_sha256
    )
    _validate_source(
        graph_rankings, graph_search_report, "graph", dataset_manifest_sha256
    )

    sources = {
        "bm25": {str(r["query_id"]): r for r in _read_jsonl(bm25_rankings)},
        "dense": {str(r["query_id"]): r for r in _read_jsonl(dense_rankings)},
        "graph": {str(r["query_id"]): r for r in _read_jsonl(graph_rankings)},
    }
    questions_rows = {str(r["query_id"]): r for r in _read_jsonl(questions)}
    documents_rows = {str(r["doc_id"]): r for r in _read_jsonl(documents)}
    metadata = {str(r["chunk_id"]): r for r in _read_jsonl(chunks)}
    for label, src in sources.items():
        if set(src) != set(questions_rows):
            raise ValueError(f"{label} query set does not match questions")

    reranker = QwenReranker(model)
    rows: list[dict[str, Any]] = []
    for query_id, question in questions_rows.items():
        candidate_ids: set[str] = set()
        for src in sources.values():
            candidate_ids.update(
                _fold_doc_ranking(src[query_id]["hits"], metadata)[: top_n]
            )
        candidates = []
        for doc_id in sorted(candidate_ids):
            if doc_id not in documents_rows:
                raise ValueError(f"candidate doc {doc_id} missing from documents")
            candidates.append(
                {"doc_id": doc_id, "content": str(documents_rows[doc_id]["content"])}
            )
        ranked = reranker.rerank(str(question["question"]), candidates)
        rows.append(
            {
                "query_id": query_id,
                "split": str(question["split"]),
                "doc_ids": [doc_id for doc_id, _ in ranked],
                "scores": [score for _, score in ranked],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "command": ["rerank.run_rerank", str(questions), str(output)],
        "query_count": len(rows),
        "top_n": top_n,
        "model": model,
        "bm25_rankings_sha256": sha256_file(bm25_rankings),
        "dense_rankings_sha256": sha256_file(dense_rankings),
        "graph_rankings_sha256": sha256_file(graph_rankings),
        "rankings_sha256": sha256_file(output),
        "questions_sha256": sha256_file(questions),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_artifact_hashes": dataset_manifest_data["artifact_hashes"],
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
