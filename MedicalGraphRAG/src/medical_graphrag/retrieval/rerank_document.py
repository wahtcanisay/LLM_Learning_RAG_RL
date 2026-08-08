"""Document-level reranker (Hybrid2 for the document retrieval unit).

The document baselines already produce document-level rankings
(``{doc_id, rank, score}``), so no chunk→doc collapse is needed. The candidate
union of a configurable subset of sources (bm25 / dense / graph-ep /
graph-sim) is reranked with Qwen3-Reranker and written as a document ranking
file (``doc_ids`` + ``scores``), the same schema ``evaluate_reranker_run``
consumes.

``reranker`` is injectable for unit tests; when None the real
``QwenReranker`` is loaded (integration/smoke only).
"""
import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset
from medical_graphrag.retrieval.reranker import DEFAULT_RERANKER_MODEL, QwenReranker


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_document_source(
    rankings: Path,
    search_report: Path,
    label: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    report = json.loads(search_report.read_text(encoding="utf-8"))
    if report.get("rankings_sha256") != sha256_file(rankings):
        raise ValueError(f"{label} rankings SHA-256 mismatch")
    if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError(f"{label} search report does not match frozen dataset manifest")
    if str(report.get("retrieval_unit", "chunk")) != "document":
        raise ValueError(f"{label} search report must be a document retrieval unit")
    return report


def run_rerank_document(
    *,
    sources: dict[str, Path],
    source_reports: dict[str, Path],
    questions: Path,
    documents: Path,
    dataset_manifest: Path,
    output: Path,
    report: Path,
    top_n: int = 50,
    model: str = DEFAULT_RERANKER_MODEL,
    reranker: Any | None = None,
) -> dict[str, Any]:
    """Rerank the document-level candidate union with Qwen3-Reranker."""
    dataset_manifest_data = validate_frozen_dataset(dataset_manifest.parent)
    dataset_manifest_sha256 = sha256_file(dataset_manifest)
    if sha256_file(documents) != dataset_manifest_data["artifact_hashes"]["documents.jsonl"]:
        raise ValueError("documents do not match frozen dataset manifest")
    for label in sources:
        _validate_document_source(
            sources[label], source_reports[label], label, dataset_manifest_sha256
        )

    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in sources.items():
        source_rows[label] = {str(r["query_id"]): r for r in _read_jsonl(path)}
    questions_rows = {str(r["query_id"]): r for r in _read_jsonl(questions)}
    documents_rows = {str(r["doc_id"]): r for r in _read_jsonl(documents)}
    for label, src in source_rows.items():
        if set(src) != set(questions_rows):
            raise ValueError(f"{label} query set does not match questions")

    if reranker is None:
        reranker = QwenReranker(model)
    rows: list[dict[str, Any]] = []
    for query_id, question in questions_rows.items():
        candidate_ids: set[str] = set()
        for label, src in source_rows.items():
            candidate_ids.update(
                str(hit["doc_id"]) for hit in src[query_id]["hits"][: top_n]
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
    report_data: dict[str, Any] = {
        "command": ["rerank_document.run_rerank_document", str(questions), str(output)],
        "query_count": len(rows),
        "top_n": top_n,
        "model": model,
        "sources": sorted(sources),
        "source_rankings_sha256": {
            label: sha256_file(path) for label, path in sorted(sources.items())
        },
        "rankings_sha256": sha256_file(output),
        "questions_sha256": sha256_file(questions),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_artifact_hashes": dataset_manifest_data["artifact_hashes"],
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
