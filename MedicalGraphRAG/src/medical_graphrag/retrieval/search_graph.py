"""Graph retrieval over a built LinearGraphRetriever index, extracted from the
old ``scripts/search_graph.py`` into a library function.

``run_search`` validates the graph index report hash bindings, retrieves
top-k chunks per question (timed), and writes ``raw_rankings.jsonl`` plus
``search_run.json`` with every artifact SHA-256.
"""
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.graph import (
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


def run_search(
    *,
    index: Path,
    index_report: Path,
    questions: Path,
    chunks: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    ner_model: str = DEFAULT_NER_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, object]:
    """Retrieve top-k passages per question via LinearGraphRetriever."""
    index_report_data = validate_index(index, questions, index_report)
    if ner_model != index_report_data["ner_model"]:
        raise ValueError("requested ner model does not match index report")
    if embedding_model != index_report_data["embedding_model"]:
        raise ValueError("requested embedding model does not match index report")
    index_config = index_report_data["config"]
    config = GraphConfig(
        damping=index_config["damping"],
        passage_ratio=index_config["passage_ratio"],
        passage_node_weight=index_config["passage_node_weight"],
        iteration_threshold=index_config["iteration_threshold"],
        top_k_sentence=index_config["top_k_sentence"],
        max_iterations=index_config["max_iterations"],
        ner_model=ner_model,
        embedding_model=embedding_model,
    )
    retriever = LinearGraphRetriever(index, config=config)

    questions_rows = _read_jsonl(questions)
    chunks_rows = _read_jsonl(chunks)
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in chunks_rows}
    missing = [p for p in retriever.passage_ids if p not in chunk_to_doc]
    if missing:
        raise ValueError(f"{len(missing)} index passages missing from --chunks")
    rows = [search_one(retriever, row, chunk_to_doc, top_k) for row in questions_rows]
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
        "command": ["search_graph.run_search", str(questions), str(output)],
        "query_count": len(rows),
        **hit_summary,
        "text_mode": index_report_data["text_mode"],
        "ner_model": index_report_data["ner_model"],
        "embedding_model": index_report_data["embedding_model"],
        "graph_sha256": index_report_data["graph_sha256"],
        "graph_build_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "rankings_sha256": rankings_sha256,
        "config": index_report_data["config"],
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
