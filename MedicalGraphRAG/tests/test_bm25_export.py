import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.retrieval.bm25 import export_pyserini_collection


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path) -> None:
    root.mkdir()
    questions = root / "questions.jsonl"
    documents = root / "documents.jsonl"
    chunks = root / "chunks.jsonl"
    qrels = root / "qrels.tsv"
    questions.write_text(
        '{"query_id":"q1","question":"alpha?","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8",
    )
    documents.write_text(
        '{"doc_id":"d1","title":"LEAK TITLE","content":"alpha body","source":"pubmedqa","year":null}\n',
        encoding="utf-8",
    )
    chunks.write_text(
        '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"LEAK TITLE","content":"alpha body","source":"pubmedqa"}\n',
        encoding="utf-8",
    )
    qrels.write_text("query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8")
    artifacts = {path.name: _sha(path) for path in (questions, documents, chunks, qrels)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"questions": 1, "documents": 1, "chunks": 1, "qrels": 1},
                "artifact_hashes": artifacts,
            }
        ),
        encoding="utf-8",
    )


def test_export_uses_content_only_and_keeps_metadata_separate(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_dataset(dataset)

    summary = export_pyserini_collection(dataset, output)

    collection = json.loads((output / "collection/chunks.jsonl").read_text(encoding="utf-8"))
    metadata = json.loads((output / "chunk_metadata.jsonl").read_text(encoding="utf-8"))
    assert collection == {"contents": "alpha body", "id": "c1"}
    assert "LEAK TITLE" not in collection["contents"]
    assert metadata["chunk_id"] == collection["id"]
    assert metadata["doc_id"] == "d1"
    assert summary["chunk_count"] == 1


def test_export_rejects_tampered_artifact(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    with (dataset / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch.*chunks.jsonl"):
        export_pyserini_collection(dataset, tmp_path / "output")


def test_export_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    row = (dataset / "chunks.jsonl").read_text(encoding="utf-8")
    (dataset / "chunks.jsonl").write_text(row + row, encoding="utf-8")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["chunks"] = 2
    manifest["artifact_hashes"]["chunks.jsonl"] = _sha(dataset / "chunks.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate chunk_id: c1"):
        export_pyserini_collection(dataset, tmp_path / "output")
