"""Run LinearGraphRetriever.search over every question inside the container.

Run with the project venv python: `/opt/venv/bin/python`. Standalone script.
Validates the graph index report hash bindings, retrieves top-k chunks per
question (timed), and writes ``raw_rankings.jsonl`` plus ``search_run.json``
with every artifact SHA-256.
"""
import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.data.io import sha256_file  # noqa: E402
from medical_graphrag.retrieval.graph import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NER_MODEL,
    GraphConfig,
    LinearGraphRetriever,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_hit_counts(
    rows: list[dict[str, object]], *, requested_top_k: int
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


def validate_index(index_dir: Path, questions: Path, report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("text_mode") != "abstract_only":
        raise ValueError("index must use abstract_only text mode")
    expected = {
        "graph.graphml": "graph_sha256",
        "sentence_embeddings.npy": "sentence_embeddings_sha256",
        "entity_embeddings.npy": "entity_embeddings_sha256",
        "passage_embeddings.npy": "passage_embeddings_sha256",
        "entities.jsonl": "entities_sha256",
        "entity_to_sentences.jsonl": "entity_to_sentences_sha256",
        "sentence_to_entities.jsonl": "sentence_to_entities_sha256",
    }
    for filename, field in expected.items():
        if sha256_file(index_dir / filename) != report.get(field):
            raise ValueError(f"{filename} SHA-256 mismatch")
    expected_questions = report.get("dataset_artifact_hashes", {}).get("questions.jsonl")
    if sha256_file(questions) != expected_questions:
        raise ValueError("questions SHA-256 mismatch")
    return report


def search_one(
    retriever: LinearGraphRetriever,
    question: Mapping[str, object],
    chunk_to_doc: Mapping[str, str],
    top_k: int,
    clock: Any = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    passage_ids, scores = retriever.search(str(question["question"]), top_k=top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = [
        {
            "chunk_id": chunk_id,
            "doc_id": chunk_to_doc[chunk_id],
            "chunk_rank": rank,
            "score": float(score),
        }
        for rank, (chunk_id, score) in enumerate(zip(passage_ids, scores), start=1)
    ]
    return {
        "query_id": str(question["query_id"]),
        "split": str(question["split"]),
        "latency_ms": latency_ms,
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ner-model", default=DEFAULT_NER_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    index_report = validate_index(args.index, args.questions, args.index_report)
    if args.ner_model != index_report["ner_model"]:
        raise ValueError("requested ner model does not match index report")
    if args.embedding_model != index_report["embedding_model"]:
        raise ValueError("requested embedding model does not match index report")
    index_config = index_report["config"]
    config = GraphConfig(
        damping=index_config["damping"],
        passage_ratio=index_config["passage_ratio"],
        passage_node_weight=index_config["passage_node_weight"],
        iteration_threshold=index_config["iteration_threshold"],
        top_k_sentence=index_config["top_k_sentence"],
        max_iterations=index_config["max_iterations"],
        ner_model=args.ner_model,
        embedding_model=args.embedding_model,
    )
    retriever = LinearGraphRetriever(args.index, config=config)

    questions = _read_jsonl(args.questions)
    chunks = _read_jsonl(args.chunks)
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in chunks}
    missing = [p for p in retriever.passage_ids if p not in chunk_to_doc]
    if missing:
        raise ValueError(f"{len(missing)} index passages missing from --chunks")
    rows = [
        search_one(retriever, row, chunk_to_doc, args.top_k) for row in questions
    ]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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
                "text_mode": index_report["text_mode"],
                "ner_model": index_report["ner_model"],
                "embedding_model": index_report["embedding_model"],
                "graph_sha256": index_report["graph_sha256"],
                "graph_build_report_sha256": sha256_file(args.index_report),
                "dataset_manifest_sha256": index_report["dataset_manifest_sha256"],
                "questions_sha256": sha256_file(args.questions),
                "rankings_sha256": rankings_sha256,
                "config": index_report["config"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
