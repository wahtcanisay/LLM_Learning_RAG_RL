import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_jsonl


ARTIFACT_NAMES = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")


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
