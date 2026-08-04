import hashlib
import json
from pathlib import Path

from medical_graphrag.retrieval.dense import export_chunk_metadata


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"q1","question":"a","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8",
    )
    (dataset / "documents.jsonl").write_text(
        '{"doc_id":"d1","title":"one","content":"a","source":"pubmedqa","year":null}\n',
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","content":"first half","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d1","order":1,"title":"one","content":"second half","source":"pubmedqa"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8"
    )
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    hashes = {name: _sha(dataset / name) for name in names}
    counts = {
        "questions": 1,
        "documents": 1,
        "chunks": 2,
        "qrels": 1,
    }
    (dataset / "manifest.json").write_text(
        json.dumps({"counts": counts, "artifact_hashes": hashes}), encoding="utf-8"
    )
    return dataset


def test_export_chunk_metadata_preserves_frozen_order_and_binds_hashes(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "out"

    report = export_chunk_metadata(dataset, output)

    assert report["chunk_count"] == 2
    assert report["text_mode"] == "abstract_only"
    assert report["dataset_manifest_sha256"] == _sha(dataset / "manifest.json")
    metadata_path = output / "chunk_metadata.jsonl"
    assert report["metadata_sha256"] == _sha(metadata_path)
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["chunk_id"] for row in rows] == ["c1", "c2"]
    assert [row["doc_id"] for row in rows] == ["d1", "d1"]
    assert [row["order"] for row in rows] == [0, 1]
    assert (output / "export_report.json").exists()
