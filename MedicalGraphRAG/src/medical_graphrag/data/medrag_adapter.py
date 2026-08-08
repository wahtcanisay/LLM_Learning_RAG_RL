"""MedRAG corpus → RetrievalPassage adapter (spec §5.2, P1-3).

Rules: the file stem is the document ID (textbook stem = doc, StatPearls
article stem = doc); the trailing integer in the original ``id`` is the chunk
order. Any prefix mismatch, empty content, unparseable/negative order,
duplicate passage id or duplicate order fails closed — nothing is silently
filtered or skipped.
"""
import json
from pathlib import Path

from medical_graphrag.data.retrieval_passages import RetrievalPassage


def adapt_medrag_chunks(
    rows: list[dict[str, object]],
    *,
    doc_id: str,
) -> list[RetrievalPassage]:
    prefix = doc_id + "_"
    passages: list[RetrievalPassage] = []
    seen_orders: set[int] = set()
    seen_ids: set[str] = set()
    for row in rows:
        raw_id = str(row["id"]).strip()
        if not raw_id.startswith(prefix):
            raise ValueError(
                f"id {raw_id!r} does not match doc_id prefix {prefix!r}"
            )
        suffix = raw_id[len(prefix):]
        try:
            order = int(suffix)
        except ValueError as error:
            raise ValueError(
                f"id {raw_id!r} has unparseable order {suffix!r}"
            ) from error
        if order < 0:
            raise ValueError(f"id {raw_id!r} has negative order {order}")
        if raw_id in seen_ids:
            raise ValueError(f"duplicate passage id {raw_id!r}")
        if order in seen_orders:
            raise ValueError(f"doc {doc_id} has duplicate order {order}")
        content = str(row.get("content", "")).strip()
        if not content:
            raise ValueError(f"id {raw_id!r} has empty content")
        seen_ids.add(raw_id)
        seen_orders.add(order)
        passages.append(RetrievalPassage(
            passage_id=raw_id,
            doc_id=doc_id,
            order=order,
            title=str(row.get("title", "")),
            content=content,
            source="medrag",
        ))
    return passages


def adapt_medrag_file(path: Path) -> list[RetrievalPassage]:
    """Adapt one MedRAG JSONL chunk file; doc_id = file stem."""
    rows = [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    return adapt_medrag_chunks(rows, doc_id=path.stem)
