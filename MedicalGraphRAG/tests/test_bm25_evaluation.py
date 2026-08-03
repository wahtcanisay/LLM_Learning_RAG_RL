import json
import hashlib
from pathlib import Path

import pytest

from medical_graphrag.evaluation.bm25 import evaluate_bm25_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(dataset: Path) -> dict[str, str]:
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    hashes = {name: _sha(dataset / name) for name in names}
    counts = {
        "questions": sum(1 for line in (dataset / "questions.jsonl").read_text(encoding="utf-8").split("\n") if line),
        "documents": sum(1 for line in (dataset / "documents.jsonl").read_text(encoding="utf-8").split("\n") if line),
        "chunks": sum(1 for line in (dataset / "chunks.jsonl").read_text(encoding="utf-8").split("\n") if line),
        "qrels": sum(1 for line in (dataset / "qrels.tsv").read_text(encoding="utf-8").split("\n")[1:] if line),
    }
    (dataset / "manifest.json").write_text(
        json.dumps({"counts": counts, "artifact_hashes": hashes}), encoding="utf-8"
    )
    return hashes


def test_evaluation_reports_splits_separately(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        "\n".join(
            [
                '{"query_id":"q1","question":"a","answer":"yes","long_answer":"x","split":"dev"}',
                '{"query_id":"q2","question":"b","answer":"no","long_answer":"y","split":"test"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "documents.jsonl").write_text(
        "\n".join(
            [
                '{"doc_id":"d1","title":"one","content":"a","source":"pubmedqa","year":null}',
                '{"doc_id":"d2","title":"two\u2029line","content":"b","source":"pubmedqa","year":null}',
                '{"doc_id":"x","title":"other","content":"z","source":"medrag_pubmed","year":null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","content":"gold alpha text","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two\u2029line","content":"gold beta text","source":"pubmedqa"}',
                '{"chunk_id":"cx","doc_id":"x","order":0,"title":"other","content":"other text","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\nq2\td2\t1\n",
        encoding="utf-8",
    )
    artifact_hashes = _write_manifest(dataset)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two\u2029line","source":"pubmedqa"}',
                '{"chunk_id":"cx","doc_id":"x","order":0,"title":"other","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","latency_ms":10,"hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":3},{"chunk_id":"cx","doc_id":"x","chunk_rank":2,"score":2}]}',
                '{"query_id":"q2","split":"test","latency_ms":20,"hits":[{"chunk_id":"cx","doc_id":"x","chunk_rank":1,"score":3},{"chunk_id":"c2","doc_id":"d2","chunk_rank":2,"score":2}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "experiment"

    index_report = {
        "elapsed_seconds": 1.0,
        "index_sha256": "a" * 64,
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        "dataset_artifact_hashes": artifact_hashes,
    }
    search_report = {
        "query_count": 2,
        "requested_top_k": 2,
        "min_hits": 2,
        "max_hits": 2,
        "short_ranking_count": 0,
        "hit_count_histogram": {"2": 2},
        "k1": 0.9,
        "b": 0.4,
        "metadata_sha256": _sha(metadata),
        "text_mode": "abstract_only",
        "index_sha256": "a" * 64,
        "index_report_sha256": "b" * 64,
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
    }
    result = evaluate_bm25_run(
        dataset,
        metadata,
        rankings,
        output,
        run_context={
            "git_commit": "abc",
            "index": index_report,
            "search": search_report,
            "index_report_sha256": "b" * 64,
        },
    )

    assert set(result) == {"dev", "test"}
    assert result["dev"]["sample_count"] == 1
    assert result["dev"]["recall@1"] == 1.0
    assert result["test"]["recall@1"] == 0.0
    assert result["test"]["recall@5"] == 1.0
    assert result["test"]["latency_ms"]["p50"] == 20.0
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == result
    assert (output / "run_manifest.json").exists()
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert cases["success"][0]["gold_chunk_excerpt"] == "gold beta text"
    assert cases["success"][0]["gold_title"] == "two\u2029line"
    assert cases["success"][0]["top_documents"][0]["title"] == "other"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_top_k"] == 2
    assert manifest["bm25"] == {"k1": 0.9, "b": 0.4}


def test_evaluation_rejects_search_report_that_mislabels_raw_rankings(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"q1","question":"a","split":"dev"}\n', encoding="utf-8"
    )
    (dataset / "documents.jsonl").write_text(
        '{"doc_id":"d1","title":"one","content":"a"}\n', encoding="utf-8"
    )
    (dataset / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","doc_id":"d1","content":"a"}\n', encoding="utf-8"
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8"
    )
    artifact_hashes = _write_manifest(dataset)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"c1","doc_id":"d1"}\n', encoding="utf-8")
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        '{"query_id":"q1","split":"dev","latency_ms":1,"hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":1}]}\n',
        encoding="utf-8",
    )
    context = {
        "index_report_sha256": "b" * 64,
        "index": {
            "index_sha256": "a" * 64,
            "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            "dataset_artifact_hashes": artifact_hashes,
        },
        "search": {
            "query_count": 1,
            "requested_top_k": 100,
            "min_hits": 100,
            "max_hits": 100,
            "short_ranking_count": 0,
            "hit_count_histogram": {"100": 1},
            "k1": 0.9,
            "b": 0.4,
            "metadata_sha256": _sha(metadata),
            "text_mode": "abstract_only",
            "index_sha256": "a" * 64,
            "index_report_sha256": "b" * 64,
            "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        },
    }

    with pytest.raises(ValueError, match="search report hit summary mismatch"):
        evaluate_bm25_run(dataset, metadata, rankings, tmp_path / "out", run_context=context)
