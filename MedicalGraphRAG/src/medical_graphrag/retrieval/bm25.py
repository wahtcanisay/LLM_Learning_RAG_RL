import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_jsonl


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


def validate_frozen_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ARTIFACT_NAMES:
        actual = sha256_file(dataset_dir / name)
        expected = manifest["artifact_hashes"][name]
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
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
    return {
        "chunk_count": len(collection),
        "collection_sha256": sha256_file(collection_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
