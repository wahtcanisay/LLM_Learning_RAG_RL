import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, text)


def write_qrels(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    lines = ["query_id\tdoc_id\trelevance\n"]
    lines.extend(f"{query_id}\t{doc_id}\t{relevance}\n" for query_id, doc_id, relevance in rows)
    _atomic_text(path, "".join(lines))
