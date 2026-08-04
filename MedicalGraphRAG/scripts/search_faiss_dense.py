"""Search the FAISS IndexFlatIP index for every question inside the container.

Run inside the `llm-pytorch` container. Standalone script: adds `src/` to
``sys.path``. Encodes each query with the same embedding model used at index
build time, retrieves up to ``--top-k`` chunks, and writes ``raw_rankings.jsonl``
plus ``search_run.json``. The search report binds every artifact SHA-256 so the
local ``evaluate-dense`` step can verify nothing was swapped.
"""
import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.data.io import sha256_file  # noqa: E402
from medical_graphrag.retrieval.dense import DEFAULT_EMBEDDING_MODEL  # noqa: E402


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
    clock: Any = time.perf_counter,
) -> dict[str, object]:
    import numpy as np

    started = clock()
    vector = np.asarray(
        embedder.encode(
            str(question["question"]),
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype="float32",
    ).reshape(1, -1)
    scores, indices = index.search(vector, top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = []
    for rank in range(top_k):
        chunk_id = chunk_id_by_index[int(indices[0, rank])]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    index_path = Path(args.index)
    embeddings_path = Path(args.embeddings)
    index_report = validate_index(
        index_path,
        embeddings_path,
        args.metadata,
        args.questions,
        args.index_report,
    )

    import faiss

    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(index_path))
    embedder = SentenceTransformer(args.embedding_model)
    questions = _read_jsonl(args.questions)
    metadata = _read_jsonl(args.metadata)
    chunk_id_by_index = [str(row["chunk_id"]) for row in metadata]
    if len(set(chunk_id_by_index)) != len(chunk_id_by_index):
        raise ValueError("duplicate chunk_id in metadata")
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in metadata}
    if index_report["chunk_count"] != len(chunk_id_by_index):
        raise ValueError("metadata count does not match index report")

    rows = [
        search_one(
            embedder,
            index,
            row,
            chunk_id_by_index,
            chunk_to_doc,
            args.top_k,
        )
        for row in questions
    ]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    rankings_sha256 = sha256_file(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "command": sys.argv,
                "query_count": len(rows),
                **hit_summary,
                "embedding_model": index_report["embedding_model"],
                "dim": index_report["dim"],
                "normalized": index_report["normalized"],
                "index_type": index_report["index_type"],
                "sentence_transformers_version": package_version(
                    "sentence-transformers"
                ),
                "faiss_version": package_version("faiss-cpu"),
                "text_mode": index_report["text_mode"],
                "metadata_sha256": index_report["metadata_sha256"],
                "embeddings_sha256": index_report["embeddings_sha256"],
                "index_sha256": index_report["index_sha256"],
                "index_report_sha256": sha256_file(args.index_report),
                "dataset_manifest_sha256": index_report["dataset_manifest_sha256"],
                "questions_sha256": sha256_file(args.questions),
                "rankings_sha256": rankings_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
