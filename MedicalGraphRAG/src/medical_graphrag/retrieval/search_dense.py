"""Dense retrieval over a FAISS IndexFlatIP index, extracted from the old
``scripts/search_faiss_dense.py`` into a library function.

``run_search`` validates the index report hash bindings, encodes each query
with the same embedding model used at index build time, retrieves up to
``top_k`` chunks, and writes ``raw_rankings.jsonl`` plus ``search_run.json``.
The search report binds every artifact SHA-256 so ``evaluate-dense`` can verify
nothing was swapped.
"""
import json
import time
from collections import Counter
from collections.abc import Mapping
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.dense import DEFAULT_EMBEDDING_MODEL


def package_version(
    package: str,
    resolver: Any = distribution_version,
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
    index: Path,
    embeddings: Path,
    metadata: Path,
    questions: Path,
    report_path: Path,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("text_mode") != "abstract_only":
        raise ValueError("index must use abstract_only text mode")
    if sha256_file(index) != report.get("index_sha256"):
        raise ValueError("index SHA-256 mismatch")
    if sha256_file(embeddings) != report.get("embeddings_sha256"):
        raise ValueError("embeddings SHA-256 mismatch")
    if sha256_file(metadata) != report.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 mismatch")
    expected_questions = report.get("dataset_artifact_hashes", {}).get(
        "questions.jsonl"
    )
    if sha256_file(questions) != expected_questions:
        raise ValueError("questions SHA-256 mismatch")
    return report


def search_one(
    embedder: Any,
    index: Any,
    question: Mapping[str, object],
    chunk_id_by_index: list[str],
    chunk_to_doc: Mapping[str, str],
    top_k: int,
    *,
    normalize: bool = True,
    clock: Any = time.perf_counter,
) -> dict[str, object]:
    import numpy as np

    started = clock()
    vector = np.asarray(
        embedder.encode(
            str(question["question"]),
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ),
        dtype="float32",
    ).reshape(1, -1)
    scores, indices = index.search(vector, top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = []
    for rank in range(top_k):
        chunk_index = int(indices[0, rank])
        if chunk_index < 0:
            # FAISS pads missing neighbours with -1 when top_k exceeds index size.
            break
        chunk_id = chunk_id_by_index[chunk_index]
        hits.append(
            {
                "chunk_id": chunk_id,
                "doc_id": chunk_to_doc[chunk_id],
                "chunk_rank": rank + 1,
                "score": float(scores[0, rank]),
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
    embeddings: Path,
    index_report: Path,
    questions: Path,
    metadata: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, object]:
    """Encode every question, retrieve with FAISS, write rankings + report."""
    index_report_data = validate_index(
        index, embeddings, metadata, questions, index_report
    )

    import faiss

    from sentence_transformers import SentenceTransformer

    if embedding_model != index_report_data["embedding_model"]:
        raise ValueError(
            "requested embedding model does not match index report: "
            f"{embedding_model!r} != {index_report_data['embedding_model']!r}"
        )
    if top_k > int(index_report_data["chunk_count"]):
        raise ValueError(
            f"requested top_k {top_k} exceeds index chunk count "
            f"{index_report_data['chunk_count']}"
        )
    faiss_index = faiss.read_index(str(index))
    embedder = SentenceTransformer(embedding_model)
    questions_rows = _read_jsonl(questions)
    metadata_rows = _read_jsonl(metadata)
    chunk_id_by_index = [str(row["chunk_id"]) for row in metadata_rows]
    if len(set(chunk_id_by_index)) != len(chunk_id_by_index):
        raise ValueError("duplicate chunk_id in metadata")
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in metadata_rows}
    if index_report_data["chunk_count"] != len(chunk_id_by_index):
        raise ValueError("metadata count does not match index report")

    rows = [
        search_one(
            embedder,
            faiss_index,
            row,
            chunk_id_by_index,
            chunk_to_doc,
            top_k,
            normalize=bool(index_report_data["normalized"]),
        )
        for row in questions_rows
    ]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=top_k)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    rankings_sha256 = sha256_file(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "command": ["search_dense.run_search", str(questions), str(output)],
        "query_count": len(rows),
        **hit_summary,
        "embedding_model": index_report_data["embedding_model"],
        "dim": index_report_data["dim"],
        "normalized": index_report_data["normalized"],
        "index_type": index_report_data["index_type"],
        "sentence_transformers_version": package_version("sentence-transformers"),
        "faiss_version": package_version("faiss-cpu"),
        "text_mode": index_report_data["text_mode"],
        "metadata_sha256": index_report_data["metadata_sha256"],
        "embeddings_sha256": index_report_data["embeddings_sha256"],
        "index_sha256": index_report_data["index_sha256"],
        "index_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "rankings_sha256": rankings_sha256,
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
