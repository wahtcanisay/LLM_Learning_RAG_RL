"""Unified retrieval-passage loading for document / chunk retrieval units.

document unit → one Passage per frozen ``documents.jsonl`` row, passage_id = doc_id.
chunk unit    → one Passage per frozen ``chunks.jsonl`` row, doc/order retained
                (the only inputs allowed to build Adjacent edges).

Both units fail closed on manifest hash mismatch, empty/missing content and
duplicate IDs, so every downstream index binds to exactly the frozen inputs.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from medical_graphrag.retrieval.bm25 import validate_frozen_dataset


@dataclass(frozen=True)
class RetrievalPassage:
    passage_id: str
    doc_id: str
    order: int | None
    title: str
    content: str
    source: str


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name} must contain JSON objects")
                rows.append(value)
    return rows


def load_retrieval_passages(
    dataset_dir: Path,
    retrieval_unit: Literal["document", "chunk"],
) -> list[RetrievalPassage]:
    """Load retrieval passages in frozen file order.

    document: passage_id = doc_id, content = full document.content, order = None.
    chunk:    passage_id = chunk_id, doc_id + numeric order retained.
    """
    validate_frozen_dataset(dataset_dir)
    if retrieval_unit == "document":
        rows = _read_jsonl(dataset_dir / "documents.jsonl")
        seen: set[str] = set()
        passages: list[RetrievalPassage] = []
        for row in rows:
            doc_id = str(row["doc_id"]).strip()
            content = str(row["content"]).strip()
            if not doc_id or doc_id in seen:
                raise ValueError(f"doc_id must be non-empty and unique: {doc_id!r}")
            if not content:
                raise ValueError(f"document {doc_id} content must not be empty")
            seen.add(doc_id)
            passages.append(RetrievalPassage(
                passage_id=doc_id,
                doc_id=doc_id,
                order=None,
                title=str(row.get("title", "")),
                content=content,
                source=str(row.get("source", "")),
            ))
        return passages
    if retrieval_unit != "chunk":
        raise ValueError("retrieval_unit must be 'document' or 'chunk'")
    rows = _read_jsonl(dataset_dir / "chunks.jsonl")
    passages: list[RetrievalPassage] = []
    seen_chunks: set[str] = set()
    for row in rows:
        chunk_id = str(row["chunk_id"]).strip()
        doc_id = str(row["doc_id"]).strip()
        order = row.get("order")
        if order is None:
            raise ValueError(f"chunk {chunk_id} missing order")
        if not chunk_id or chunk_id in seen_chunks:
            raise ValueError(f"chunk_id must be non-empty and unique: {chunk_id!r}")
        if not doc_id or not str(row.get("content", "")).strip():
            raise ValueError(f"chunk {chunk_id} must have doc_id and content")
        seen_chunks.add(chunk_id)
        passages.append(RetrievalPassage(
            passage_id=chunk_id,
            doc_id=doc_id,
            order=int(order),
            title=str(row.get("title", "")),
            content=str(row["content"]),
            source=str(row.get("source", "")),
        ))
    return passages
