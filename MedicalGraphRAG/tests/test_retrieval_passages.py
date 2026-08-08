import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.data.retrieval_passages import (
    RetrievalPassage,
    load_retrieval_passages,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text(
        "\n".join([
            json.dumps({"doc_id": "PMID:1", "title": "t1",
                        "content": "full abstract one", "source": "pubmedqa", "year": "1992"}),
            json.dumps({"doc_id": "PMID:2", "title": "t2",
                        "content": "full abstract two", "source": "pubmedqa", "year": "1993"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join([
            json.dumps({"chunk_id": "PMID:1#0", "doc_id": "PMID:1", "order": 0,
                        "title": "t1", "content": "full abstract one", "source": "pubmedqa"}),
            json.dumps({"chunk_id": "PMID:1#1", "doc_id": "PMID:1", "order": 1,
                        "title": "t1", "content": "more text", "source": "pubmedqa"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (dataset / "questions.jsonl").write_text("", encoding="utf-8")
    (dataset / "qrels.tsv").write_text("query_id\tdoc_id\trelevance\n", encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 2, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )
    return dataset


def test_load_documents_uses_full_content_as_single_passage(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    passages = load_retrieval_passages(dataset, "document")
    assert len(passages) == 2
    assert passages[0].passage_id == "PMID:1"
    assert passages[0].doc_id == "PMID:1"
    assert passages[0].content == "full abstract one"
    assert passages[0].order is None
    assert passages[1].passage_id == "PMID:2"


def test_load_chunks_keeps_doc_and_order(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    passages = load_retrieval_passages(dataset, "chunk")
    assert [p.passage_id for p in passages] == ["PMID:1#0", "PMID:1#1"]
    assert [p.order for p in passages] == [0, 1]
    assert passages[1].doc_id == "PMID:1"


def test_document_loader_rejects_manifest_mismatch(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    (dataset / "documents.jsonl").write_text(
        '{"doc_id":"PMID:1","title":"t","content":"c","source":"s","year":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_retrieval_passages(dataset, "document")


def test_document_loader_rejects_duplicate_doc_id(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    (dataset / "documents.jsonl").write_text(
        "\n".join([
            '{"doc_id":"PMID:1","title":"t","content":"a","source":"s","year":"y"}',
            '{"doc_id":"PMID:1","title":"t","content":"b","source":"s","year":"y"}',
        ]) + "\n",
        encoding="utf-8",
    )
    # 改文件必须同步改 manifest 的 artifact hash,否则先撞 SHA 校验
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 2, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )
    # validate_frozen_dataset 先拦截重复 doc_id
    with pytest.raises(ValueError, match="duplicate"):
        load_retrieval_passages(dataset, "document")


def test_chunk_loader_rejects_missing_order(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    row = {"chunk_id": "PMID:1#9", "doc_id": "PMID:1", "title": "t",
           "content": "c", "source": "s"}  # 缺 order,但引用已存在 doc → 通过冻结校验
    (dataset / "chunks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 1, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="order"):
        load_retrieval_passages(dataset, "chunk")


@pytest.mark.parametrize("bad_order", [1.5, True, -1])
def test_chunk_loader_rejects_non_integer_or_negative_order(
    tmp_path: Path, bad_order
) -> None:
    dataset = _write_dataset(tmp_path)
    chunks_path = dataset / "chunks.jsonl"
    row = {
        "chunk_id": "PMID:1#9",
        "doc_id": "PMID:1",
        "order": bad_order,
        "title": "t",
        "content": "c",
        "source": "s",
    }
    chunks_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 1, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        load_retrieval_passages(dataset, "chunk")
