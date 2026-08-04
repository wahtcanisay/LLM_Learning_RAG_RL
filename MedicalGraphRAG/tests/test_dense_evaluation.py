import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.evaluation.dense import evaluate_dense_run

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(dataset: Path) -> dict[str, str]:
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    hashes = {name: _sha(dataset / name) for name in names}
    counts = {
        "questions": sum(
            1
            for line in (dataset / "questions.jsonl")
            .read_text(encoding="utf-8")
            .split("\n")
            if line
        ),
        "documents": sum(
            1
            for line in (dataset / "documents.jsonl")
            .read_text(encoding="utf-8")
            .split("\n")
            if line
        ),
        "chunks": sum(
            1
            for line in (dataset / "chunks.jsonl").read_text(encoding="utf-8").split("\n")
            if line
        ),
        "qrels": sum(
            1
            for line in (dataset / "qrels.tsv").read_text(encoding="utf-8").split("\n")[1:]
            if line
        ),
    }
    (dataset / "manifest.json").write_text(
        json.dumps({"counts": counts, "artifact_hashes": hashes}), encoding="utf-8"
    )
    return hashes


def _make_dataset(tmp_path: Path) -> tuple[Path, dict[str, str]]:
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
                '{"doc_id":"d2","title":"two","content":"b","source":"pubmedqa","year":null}',
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
                '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two","content":"gold beta text","source":"pubmedqa"}',
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
    return dataset, artifact_hashes


def _metadata(tmp_path: Path) -> Path:
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two","source":"pubmedqa"}',
                '{"chunk_id":"cx","doc_id":"x","order":0,"title":"other","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _rankings(tmp_path: Path) -> Path:
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","latency_ms":10,"hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":0.9},{"chunk_id":"cx","doc_id":"x","chunk_rank":2,"score":0.6}]}',
                '{"query_id":"q2","split":"test","latency_ms":20,"hits":[{"chunk_id":"cx","doc_id":"x","chunk_rank":1,"score":0.9},{"chunk_id":"c2","doc_id":"d2","chunk_rank":2,"score":0.6}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rankings


def _run_context(
    dataset: Path,
    metadata: Path,
    rankings: Path,
    artifact_hashes: dict[str, str],
    *,
    embedding_model: str = EMBEDDING_MODEL,
    search_embedding_model: str | None = None,
    dim: int = 768,
    index_sha: str = "a" * 64,
    index_report_sha: str = "b" * 64,
) -> dict[str, object]:
    return {
        "git_commit": "abc",
        "index_report_sha256": index_report_sha,
        "index": {
            "embedding_model": embedding_model,
            "dim": dim,
            "normalized": True,
            "index_type": "IndexFlatIP",
            "index_sha256": index_sha,
            "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            "dataset_artifact_hashes": artifact_hashes,
        },
        "search": {
            "query_count": 2,
            "requested_top_k": 2,
            "min_hits": 2,
            "max_hits": 2,
            "short_ranking_count": 0,
            "hit_count_histogram": {"2": 2},
            "embedding_model": (
                search_embedding_model
                if search_embedding_model is not None
                else embedding_model
            ),
            "dim": dim,
            "normalized": True,
            "index_type": "IndexFlatIP",
            "metadata_sha256": _sha(metadata),
            "text_mode": "abstract_only",
            "index_sha256": index_sha,
            "index_report_sha256": index_report_sha,
            "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            "questions_sha256": artifact_hashes["questions.jsonl"],
            "rankings_sha256": _sha(rankings),
        },
    }


def test_evaluation_reports_splits_separately_and_dense_manifest(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    rankings = _rankings(tmp_path)
    output = tmp_path / "experiment"

    result = evaluate_dense_run(
        dataset,
        metadata,
        rankings,
        output,
        run_context=_run_context(dataset, metadata, rankings, artifact_hashes),
    )

    assert set(result) == {"dev", "test"}
    assert result["dev"]["sample_count"] == 1
    assert result["dev"]["recall@1"] == 1.0
    assert result["test"]["recall@1"] == 0.0
    assert result["test"]["recall@5"] == 1.0
    assert result["test"]["latency_ms"]["p50"] == 20.0
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == result
    assert (output / "run_manifest.json").exists()
    assert (output / "cases.json").exists()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_top_k"] == 2
    assert manifest["dense"] == {
        "embedding_model": EMBEDDING_MODEL,
        "dim": 768,
        "normalized": True,
        "index_type": "IndexFlatIP",
    }


def test_evaluation_rejects_search_report_with_different_embedding_model(
    tmp_path: Path,
) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    rankings = _rankings(tmp_path)

    with pytest.raises(ValueError, match="embedding_model does not match index report"):
        evaluate_dense_run(
            dataset,
            metadata,
            rankings,
            tmp_path / "out",
            run_context=_run_context(
                dataset,
                metadata,
                rankings,
                artifact_hashes,
                search_embedding_model="some/other-model",
            ),
        )


def test_evaluation_rejects_rankings_replaced_after_search_report(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    rankings = _rankings(tmp_path)
    original_sha = _sha(rankings)
    context = _run_context(dataset, metadata, rankings, artifact_hashes)
    rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","latency_ms":1,"hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":1}]}',
                '{"query_id":"q2","split":"test","latency_ms":1,"hits":[{"chunk_id":"cx","doc_id":"x","chunk_rank":1,"score":1}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rankings SHA-256 mismatch"):
        evaluate_dense_run(
            dataset, metadata, rankings, tmp_path / "out", run_context=context
        )
    assert context["search"]["rankings_sha256"] == original_sha
