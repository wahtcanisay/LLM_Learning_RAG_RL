import json
from pathlib import Path

import pytest

from medical_graphrag.data.medrag_adapter import (
    adapt_medrag_chunks,
    adapt_medrag_file,
)


def test_textbook_adapter_groups_by_file_stem(tmp_path: Path):
    rows = [
        {"id": "Anatomy_Gray_0", "title": "t", "content": "a"},
        {"id": "Anatomy_Gray_1", "title": "t", "content": "b"},
        {"id": "Anatomy_Gray_2", "title": "t", "content": "c"},
    ]
    passages = adapt_medrag_chunks(rows, doc_id="Anatomy_Gray")
    assert [p.order for p in passages] == [0, 1, 2]
    assert all(p.doc_id == "Anatomy_Gray" for p in passages)
    assert [p.passage_id for p in passages] == [
        "Anatomy_Gray_0", "Anatomy_Gray_1", "Anatomy_Gray_2",
    ]


def test_adapter_rejects_prefix_mismatch():
    rows = [{"id": "OtherDoc_0", "title": "t", "content": "c"}]
    with pytest.raises(ValueError, match="prefix"):
        adapt_medrag_chunks(rows, doc_id="Anatomy_Gray")


def test_adapter_rejects_empty_content():
    rows = [{"id": "Anatomy_Gray_0", "title": "t", "content": "  "}]
    with pytest.raises(ValueError, match="empty content"):
        adapt_medrag_chunks(rows, doc_id="Anatomy_Gray")


def test_adapter_rejects_unparseable_or_negative_order():
    with pytest.raises(ValueError, match="unparseable"):
        adapt_medrag_chunks(
            [{"id": "Anatomy_Gray_x", "title": "t", "content": "c"}],
            doc_id="Anatomy_Gray",
        )
    with pytest.raises(ValueError, match="negative"):
        adapt_medrag_chunks(
            [{"id": "Anatomy_Gray_-1", "title": "t", "content": "c"}],
            doc_id="Anatomy_Gray",
        )


def test_adapter_rejects_duplicate_id():
    # MedRAG 的 id = doc_<order>,重复 id 即重复 order,先撞 duplicate id
    with pytest.raises(ValueError, match="duplicate"):
        adapt_medrag_chunks(
            [{"id": "Anatomy_Gray_1", "title": "t", "content": "a"},
             {"id": "Anatomy_Gray_1", "title": "t", "content": "b"}],
            doc_id="Anatomy_Gray",
        )


def test_adapt_medrag_file_uses_file_stem(tmp_path: Path):
    path = tmp_path / "Immunology_Janeway.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"Immunology_Janeway_{i}", "title": "t",
                        "content": f"chunk {i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    passages = adapt_medrag_file(path)
    assert len(passages) == 3
    assert all(p.doc_id == "Immunology_Janeway" for p in passages)
    assert [p.order for p in passages] == [0, 1, 2]
