import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl


ARTIFACT_NAMES = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")


def collapse_chunk_hits(
    hits: Sequence[Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    min_unique_docs: int = 10,
) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for hit in hits:
        chunk_id = str(hit["chunk_id"])
        if chunk_id not in metadata:
            raise ValueError(f"unknown chunk_id: {chunk_id}")
        doc_id = str(metadata[chunk_id]["doc_id"])
        candidate: dict[str, object] = {
            "doc_id": doc_id,
            "score": float(hit["score"]),
            "best_chunk_id": chunk_id,
            "best_chunk_rank": int(hit["chunk_rank"]),
        }
        current = best.get(doc_id)
        if current is None or (
            -float(candidate["score"]), int(candidate["best_chunk_rank"]), chunk_id
        ) < (
            -float(current["score"]),
            int(current["best_chunk_rank"]),
            str(current["best_chunk_id"]),
        ):
            best[doc_id] = candidate

    ranking = sorted(
        best.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["best_chunk_rank"]),
            str(item["doc_id"]),
        ),
    )
    if len(ranking) < min_unique_docs:
        raise ValueError(
            f"expected at least {min_unique_docs} unique documents, got {len(ranking)}"
        )
    return ranking


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _unique_ids(
    rows: list[dict[str, Any]],
    field: str,
    label: str,
) -> set[str]:
    values: set[str] = set()
    for row in rows:
        if row.get(field) is None:
            raise ValueError(f"{label} must not be empty")
        value = str(row[field]).strip()
        if not value:
            raise ValueError(f"{label} must not be empty")
        if value in values:
            raise ValueError(f"duplicate {label}: {value}")
        values.add(value)
    return values


def _read_qrels(path: Path) -> list[tuple[str, str, int]]:
    with path.open(encoding="utf-8") as handle:
        header = next(handle, "").rstrip("\r\n")
        if header != "query_id\tdoc_id\trelevance":
            raise ValueError("invalid qrels header")
        rows = []
        for line in handle:
            if not line.strip():
                continue
            query_id, doc_id, relevance = line.rstrip("\r\n").split("\t")
            rows.append((query_id.strip(), doc_id.strip(), int(relevance)))
        return rows


def validate_frozen_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ARTIFACT_NAMES:
        actual = sha256_file(dataset_dir / name)
        expected = manifest["artifact_hashes"][name]
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")

    questions = _read_jsonl(dataset_dir / "questions.jsonl")
    documents = _read_jsonl(dataset_dir / "documents.jsonl")
    chunks = _read_jsonl(dataset_dir / "chunks.jsonl")
    qrels = _read_qrels(dataset_dir / "qrels.tsv")
    counts = manifest["counts"]
    for label, rows, key in (
        ("question", questions, "questions"),
        ("document", documents, "documents"),
        ("chunk", chunks, "chunks"),
        ("qrel", qrels, "qrels"),
    ):
        if len(rows) != counts[key]:
            raise ValueError(f"{label} count does not match manifest")

    query_ids = _unique_ids(questions, "query_id", "query_id")
    document_ids = _unique_ids(documents, "doc_id", "doc_id")
    _unique_ids(chunks, "chunk_id", "chunk_id")
    for question in questions:
        if not str(question["question"]).strip():
            raise ValueError("question text must not be empty")
        if question["split"] not in {"dev", "test"}:
            raise ValueError(f"invalid split: {question['split']}")
    if any(not str(document["content"]).strip() for document in documents):
        raise ValueError("document content must not be empty")
    if any(
        not str(chunk["content"]).strip() or str(chunk["doc_id"]) not in document_ids
        for chunk in chunks
    ):
        raise ValueError("chunk must contain content and reference an existing document")

    qrel_query_ids: set[str] = set()
    for query_id, doc_id, relevance in qrels:
        if not query_id or not doc_id or relevance != 1:
            raise ValueError("qrels must contain non-empty IDs and relevance=1")
        if query_id in qrel_query_ids:
            raise ValueError(f"duplicate qrel query_id: {query_id}")
        if query_id not in query_ids or doc_id not in document_ids:
            raise ValueError("qrel must reference an existing query and document")
        qrel_query_ids.add(query_id)
    if qrel_query_ids != query_ids:
        raise ValueError("every question must have exactly one qrel")
    return manifest


def export_pyserini_collection(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = validate_frozen_dataset(dataset_dir)
    chunks = _read_jsonl(dataset_dir / "chunks.jsonl")
    if len(chunks) != manifest["counts"]["chunks"]:
        raise ValueError("chunk count does not match manifest")

    seen: set[str] = set()
    collection: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for row in chunks:
        chunk_id = str(row["chunk_id"])
        content = str(row["content"]).strip()
        doc_id = str(row["doc_id"]).strip()
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        if not chunk_id or not content or not doc_id:
            raise ValueError("chunk_id, doc_id, and content must not be empty")
        seen.add(chunk_id)
        collection.append({"id": chunk_id, "contents": content})
        metadata.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "order": int(row["order"]),
                "title": str(row["title"]),
                "source": str(row["source"]),
            }
        )

    collection_path = output_dir / "collection/chunks.jsonl"
    metadata_path = output_dir / "chunk_metadata.jsonl"
    write_jsonl(collection_path, collection)
    write_jsonl(metadata_path, metadata)
    report = {
        "text_mode": "abstract_only",
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_artifact_hashes": manifest["artifact_hashes"],
        "chunk_count": len(collection),
        "collection_sha256": sha256_file(collection_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(output_dir / "export_report.json", report)
    return report
