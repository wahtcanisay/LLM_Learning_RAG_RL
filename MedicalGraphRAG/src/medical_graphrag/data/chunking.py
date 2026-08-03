from collections.abc import Sequence
from typing import Protocol

from medical_graphrag.data.schemas import Chunk


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


def chunk_sections(
    *,
    doc_id: str,
    title: str,
    sections: tuple[str, ...],
    source: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap: int,
) -> list[Chunk]:
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("require max_tokens > overlap >= 0")
    chunks: list[Chunk] = []
    order = 0
    stride = max_tokens - overlap
    for section in sections:
        token_ids = list(tokenizer.encode(section, add_special_tokens=False))
        if not token_ids:
            continue
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + max_tokens]
            content = tokenizer.decode(window, skip_special_tokens=True).strip()
            if content:
                chunks.append(Chunk(f"{doc_id}#{order}", doc_id, order, title, content, source))
                order += 1
            if start + max_tokens >= len(token_ids):
                break
            start += stride
    if not chunks:
        raise ValueError(f"document {doc_id} produced no chunks")
    return chunks
