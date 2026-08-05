"""Hybrid2: rerank the BM25 / Dense / Graph candidate union with Qwen3-Reranker.

Run with the project venv python (needs sentence-transformers >= 2.7 for the
Qwen3Reranker class). Standalone script: validates the three source rankings
against their search reports, folds each to a document ranking (max chunk
score), takes the top-N union as candidates, scores every (query, doc) pair with
Qwen3-Reranker, and writes a document-level ranking file plus a hashed report.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.data.io import sha256_file  # noqa: E402
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset  # noqa: E402
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits  # noqa: E402
from medical_graphrag.retrieval.reranker import DEFAULT_RERANKER_MODEL, QwenReranker  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_source(
    rankings: Path, search_report: Path, label: str, dataset_manifest_sha256: str
) -> None:
    report = json.loads(search_report.read_text(encoding="utf-8"))
    if report.get("rankings_sha256") != sha256_file(rankings):
        raise ValueError(f"{label} rankings SHA-256 mismatch")
    if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError(f"{label} search report does not match frozen dataset manifest")
    if report.get("text_mode") != "abstract_only":
        raise ValueError(f"{label} search report must use abstract_only text mode")


def _fold_doc_ranking(hits: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[str]:
    return [str(item["doc_id"]) for item in collapse_chunk_hits(hits, metadata, min_unique_docs=0)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-rankings", type=Path, required=True)
    parser.add_argument("--dense-rankings", type=Path, required=True)
    parser.add_argument("--graph-rankings", type=Path, required=True)
    parser.add_argument("--bm25-search-report", type=Path, required=True)
    parser.add_argument("--dense-search-report", type=Path, required=True)
    parser.add_argument("--graph-search-report", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args()

    dataset_manifest = validate_frozen_dataset(args.dataset_manifest.parent)
    dataset_manifest_sha256 = sha256_file(args.dataset_manifest)
    if sha256_file(args.documents) != dataset_manifest["artifact_hashes"]["documents.jsonl"]:
        raise ValueError("documents do not match frozen dataset manifest")
    if sha256_file(args.chunks) != dataset_manifest["artifact_hashes"]["chunks.jsonl"]:
        raise ValueError("chunks do not match frozen dataset manifest")
    _validate_source(
        args.bm25_rankings, args.bm25_search_report, "bm25", dataset_manifest_sha256
    )
    _validate_source(
        args.dense_rankings, args.dense_search_report, "dense", dataset_manifest_sha256
    )
    _validate_source(
        args.graph_rankings, args.graph_search_report, "graph", dataset_manifest_sha256
    )

    sources = {
        "bm25": {str(r["query_id"]): r for r in _read_jsonl(args.bm25_rankings)},
        "dense": {str(r["query_id"]): r for r in _read_jsonl(args.dense_rankings)},
        "graph": {str(r["query_id"]): r for r in _read_jsonl(args.graph_rankings)},
    }
    questions = {str(r["query_id"]): r for r in _read_jsonl(args.questions)}
    documents = {str(r["doc_id"]): r for r in _read_jsonl(args.documents)}
    metadata = {str(r["chunk_id"]): r for r in _read_jsonl(args.chunks)}
    for label, src in sources.items():
        if set(src) != set(questions):
            raise ValueError(f"{label} query set does not match questions")

    reranker = QwenReranker(args.model)
    rows: list[dict[str, Any]] = []
    for query_id, question in questions.items():
        candidate_ids: set[str] = set()
        for src in sources.values():
            candidate_ids.update(
                _fold_doc_ranking(src[query_id]["hits"], metadata)[: args.top_n]
            )
        candidates = []
        for doc_id in sorted(candidate_ids):
            if doc_id not in documents:
                raise ValueError(f"candidate doc {doc_id} missing from documents")
            candidates.append({"doc_id": doc_id, "content": str(documents[doc_id]["content"])})
        ranked = reranker.rerank(str(question["question"]), candidates)
        rows.append(
            {
                "query_id": query_id,
                "split": str(question["split"]),
                "doc_ids": [doc_id for doc_id, _ in ranked],
                "scores": [score for _, score in ranked],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "command": sys.argv,
                "query_count": len(rows),
                "top_n": args.top_n,
                "model": args.model,
                "bm25_rankings_sha256": sha256_file(args.bm25_rankings),
                "dense_rankings_sha256": sha256_file(args.dense_rankings),
                "graph_rankings_sha256": sha256_file(args.graph_rankings),
                "rankings_sha256": sha256_file(args.output),
                "questions_sha256": sha256_file(args.questions),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "dataset_artifact_hashes": dataset_manifest["artifact_hashes"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
