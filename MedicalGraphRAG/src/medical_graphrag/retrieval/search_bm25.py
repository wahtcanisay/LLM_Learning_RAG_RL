"""BM25 retrieval over a Pyserini/Lucene index, extracted from the old
``scripts/search_pyserini_bm25.py`` into a library function.

``run_search`` validates the index report hash bindings, searches every
question (timed), and writes ``raw_rankings.jsonl`` plus ``search_run.json``
with every artifact SHA-256 — the same audit contract the standalone script
produced, so ``evaluate-bm25`` keeps verifying nothing was swapped.
"""
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_directory, sha256_file


def package_version(
    package: str,
    resolver: Callable[[str], str] = distribution_version,
) -> str:
    return resolver(package)


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


def validate_index(
    index: Path, metadata: Path, questions: Path, report_path: Path
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("text_mode") != "abstract_only":
        raise ValueError("index must use abstract_only text mode")
    if sha256_directory(index) != report.get("index_sha256"):
        raise ValueError("index SHA-256 mismatch")
    if sha256_file(metadata) != report.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 mismatch")
    expected_questions = report.get("dataset_artifact_hashes", {}).get(
        "questions.jsonl"
    )
    if sha256_file(questions) != expected_questions:
        raise ValueError("questions SHA-256 mismatch")
    return report


def search_one(
    searcher: Any,
    question: Mapping[str, object],
    chunk_to_doc: Mapping[str, str],
    top_k: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    raw_hits = searcher.search(str(question["question"]), k=top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = []
    for rank, hit in enumerate(raw_hits, start=1):
        if hit.docid not in chunk_to_doc:
            raise ValueError(f"unknown chunk_id from Pyserini: {hit.docid}")
        hits.append(
            {
                "chunk_id": hit.docid,
                "doc_id": chunk_to_doc[hit.docid],
                "chunk_rank": rank,
                "score": float(hit.score),
            }
        )
    return {
        "query_id": str(question["query_id"]),
        "split": str(question["split"]),
        "latency_ms": latency_ms,
        "hits": hits,
    }


def run_search(
    *,
    index: Path,
    index_report: Path,
    questions: Path,
    metadata: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, object]:
    """Search every question with Pyserini BM25 and write rankings + report."""
    index_report_data = validate_index(index, metadata, questions, index_report)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index))
    searcher.set_bm25(k1=k1, b=b)
    questions_rows = _read_jsonl(questions)
    metadata_rows = _read_jsonl(metadata)
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in metadata_rows}
    rows = [search_one(searcher, row, chunk_to_doc, top_k) for row in questions_rows]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=top_k)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rankings_sha256 = sha256_file(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "command": ["search_bm25.run_search", str(questions), str(output)],
        "query_count": len(rows),
        **hit_summary,
        "k1": k1,
        "b": b,
        "pyserini_version": package_version("pyserini"),
        "text_mode": index_report_data["text_mode"],
        "metadata_sha256": index_report_data["metadata_sha256"],
        "index_sha256": index_report_data["index_sha256"],
        "index_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "rankings_sha256": rankings_sha256,
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data


def build_lucene_index(
    *,
    collection: Path,
    index: Path,
    report: Path,
    export_report: Path,
    threads: int = 8,
) -> dict[str, object]:
    """Build a Pyserini/Lucene index over an exported JSON collection.

    Extracted from the old ``scripts/build_pyserini_index.py``. Validates the
    export report hashes, runs the Lucene indexer, and writes a hashed
    ``index_build.json`` report.
    """
    export = json.loads(export_report.read_text(encoding="utf-8"))
    if export.get("text_mode") != "abstract_only":
        raise ValueError("export must use abstract_only text mode")
    collection_file = collection / "chunks.jsonl"
    actual = sha256_file(collection_file)
    if actual != export.get("collection_sha256"):
        raise ValueError("collection SHA-256 mismatch")
    count = sum(1 for line in collection_file.open(encoding="utf-8") if line.strip())
    if count != export.get("chunk_count"):
        raise ValueError("collection count does not match export report")

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--stemmer",
        "porter",
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in index.rglob("*") if path.is_file())

    java_version = subprocess.run(
        ["java", "-version"],
        text=True,
        capture_output=True,
        check=True,
    ).stderr.splitlines()[0]
    report_data = {
        "elapsed_seconds": elapsed,
        "index_bytes": size,
        "pyserini_version": package_version("pyserini"),
        "python_version": platform.python_version(),
        "java_version": java_version,
        "threads": threads,
        "stemmer": "porter",
        "command": command,
        "document_count": export["chunk_count"],
        "text_mode": export["text_mode"],
        "collection_sha256": export["collection_sha256"],
        "metadata_sha256": export["metadata_sha256"],
        "dataset_manifest_sha256": export["dataset_manifest_sha256"],
        "dataset_artifact_hashes": export["dataset_artifact_hashes"],
        "export_report_sha256": sha256_file(export_report),
        "index_sha256": sha256_directory(index),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data


def build_lucene_document_index(
    *,
    collection: Path,
    index: Path,
    report: Path,
    export_report: Path,
    threads: int = 8,
) -> dict[str, object]:
    """Build a Pyserini/Lucene index over the document collection
    (``collection/documents.jsonl``, document retrieval unit)."""
    export = json.loads(export_report.read_text(encoding="utf-8"))
    if export.get("retrieval_unit") != "document":
        raise ValueError("export must be a document retrieval unit")
    collection_file = collection / "documents.jsonl"
    actual = sha256_file(collection_file)
    if actual != export.get("collection_sha256"):
        raise ValueError("collection SHA-256 mismatch")
    count = sum(1 for line in collection_file.open(encoding="utf-8") if line.strip())
    if count != export.get("document_count"):
        raise ValueError("collection count does not match export report")

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--stemmer",
        "porter",
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in index.rglob("*") if path.is_file())
    report_data = {
        "retrieval_unit": "document",
        "elapsed_seconds": elapsed,
        "index_bytes": size,
        "pyserini_version": package_version("pyserini"),
        "python_version": platform.python_version(),
        "threads": threads,
        "stemmer": "porter",
        "command": command,
        "document_count": export["document_count"],
        "source_artifact": export["source_artifact"],
        "source_artifact_sha256": export["source_artifact_sha256"],
        "collection_sha256": export["collection_sha256"],
        "metadata_sha256": export["metadata_sha256"],
        "dataset_manifest_sha256": export["dataset_manifest_sha256"],
        "dataset_artifact_hashes": export["dataset_artifact_hashes"],
        "export_report_sha256": sha256_file(export_report),
        "index_sha256": sha256_directory(index),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
