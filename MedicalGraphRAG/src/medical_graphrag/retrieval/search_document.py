"""Document-level BM25/Dense search over the same retrieval passages.

Adapts the chunk-level search contract to the document unit: hits carry
``doc_id`` directly (no chunk→doc collapse, P0-5). Dense queries are encoded
with the same embedding model used by the frozen document embedding artifact;
BM25 searches a Lucene index whose docid is the doc_id. Both write
``raw_rankings.jsonl`` (document schema) plus a hashed ``search_run.json``.
"""
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_directory, sha256_file
from medical_graphrag.retrieval.dense import DEFAULT_EMBEDDING_MODEL


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_hit_counts(
    rows: list[dict[str, object]],
    *,
    requested_top_k: int,
) -> dict[str, object]:
    counts = [len(row["hits"]) for row in rows]
    histogram = Counter(counts)
    return {
        "requested_top_k": requested_top_k,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(count < requested_top_k for count in counts),
        "hit_count_histogram": {
            str(count): frequency for count, frequency in sorted(histogram.items())
        },
    }


def _write_output(
    rows: list[dict[str, object]],
    output: Path,
    report: Path,
    report_data: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rankings_sha256 = sha256_file(output)
    report_data["rankings_sha256"] = rankings_sha256
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")


def _validate_dense_index(index, embeddings_metadata, questions, report_path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("retrieval_unit") != "document":
        raise ValueError("index must be a document retrieval unit")
    if sha256_file(index) != report.get("index_sha256"):
        raise ValueError("index SHA-256 mismatch")
    if sha256_file(embeddings_metadata) != report.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 mismatch")
    if sha256_file(questions) != report.get("dataset_artifact_hashes", {}).get(
        "questions.jsonl"
    ):
        raise ValueError("questions SHA-256 mismatch")
    return report


def run_dense_document_search(
    *,
    index: Path,
    metadata: Path,
    index_report: Path,
    questions: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, object]:
    """Encode every question, retrieve with FAISS, write document rankings."""
    import faiss
    import numpy as np

    from sentence_transformers import SentenceTransformer

    index_report_data = _validate_dense_index(index, metadata, questions, index_report)
    if embedding_model != index_report_data["embedding_model"]:
        raise ValueError("requested embedding model does not match index report")
    faiss_index = faiss.read_index(str(index))
    embedder = SentenceTransformer(embedding_model)
    questions_rows = _read_jsonl(questions)
    metadata_rows = _read_jsonl(metadata)
    doc_ids = [str(row["doc_id"]) for row in metadata_rows]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate doc_id in metadata")
    if index_report_data["document_count"] != len(doc_ids):
        raise ValueError("metadata count does not match index report")

    rows: list[dict[str, object]] = []
    for question in questions_rows:
        started = time.perf_counter()
        vector = np.asarray(
            embedder.encode(
                str(question["question"]), normalize_embeddings=True, show_progress_bar=False
            ),
            dtype="float32",
        ).reshape(1, -1)
        scores, indices = faiss_index.search(vector, top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 6)
        hits = []
        for rank in range(top_k):
            doc_index = int(indices[0, rank])
            if doc_index < 0:
                break
            hits.append(
                {"doc_id": doc_ids[doc_index], "rank": rank + 1, "score": float(scores[0, rank])}
            )
        rows.append(
            {
                "query_id": str(question["query_id"]),
                "split": str(question["split"]),
                "latency_ms": latency_ms,
                "hits": hits,
            }
        )
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=top_k)
    report_data: dict[str, object] = {
        "command": ["search_document.run_dense_document_search", str(questions), str(output)],
        "query_count": len(rows),
        **hit_summary,
        "retrieval_unit": "document",
        "embedding_model": index_report_data["embedding_model"],
        "dim": index_report_data["dim"],
        "index_type": index_report_data["index_type"],
        "embedding_report_sha256": index_report_data.get("embedding_report_sha256"),
        "index_sha256": index_report_data["index_sha256"],
        "index_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "text_mode": "abstract_only",
    }
    _write_output(rows, output, report, report_data)
    return report_data


def run_bm25_document_search(
    *,
    index: Path,
    metadata: Path,
    index_report: Path,
    questions: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, object]:
    """Search every question with Pyserini BM25 over the document collection."""
    index_report_data = json.loads(index_report.read_text(encoding="utf-8"))
    if index_report_data.get("retrieval_unit") != "document":
        raise ValueError("index must be a document retrieval unit")
    if sha256_directory(index) != index_report_data.get("index_sha256"):
        raise ValueError("index SHA-256 mismatch")
    if sha256_file(metadata) != index_report_data.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 mismatch")
    if sha256_file(questions) != index_report_data.get(
        "dataset_artifact_hashes", {}
    ).get("questions.jsonl"):
        raise ValueError("questions SHA-256 mismatch")

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index))
    searcher.set_bm25(k1=k1, b=b)
    questions_rows = _read_jsonl(questions)
    metadata_rows = _read_jsonl(metadata)
    doc_ids = {str(row["doc_id"]) for row in metadata_rows}

    rows: list[dict[str, object]] = []
    for question in questions_rows:
        started = time.perf_counter()
        raw_hits = searcher.search(str(question["question"]), k=top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 6)
        hits = []
        for rank, hit in enumerate(raw_hits, start=1):
            if hit.docid not in doc_ids:
                raise ValueError(f"unknown doc_id from Pyserini: {hit.docid}")
            hits.append({"doc_id": hit.docid, "rank": rank, "score": float(hit.score)})
        rows.append(
            {
                "query_id": str(question["query_id"]),
                "split": str(question["split"]),
                "latency_ms": latency_ms,
                "hits": hits,
            }
        )
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=top_k)
    report_data: dict[str, object] = {
        "command": ["search_document.run_bm25_document_search", str(questions), str(output)],
        "query_count": len(rows),
        **hit_summary,
        "retrieval_unit": "document",
        "k1": k1,
        "b": b,
        "index_sha256": index_report_data["index_sha256"],
        "index_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "text_mode": "abstract_only",
    }
    _write_output(rows, output, report, report_data)
    return report_data
