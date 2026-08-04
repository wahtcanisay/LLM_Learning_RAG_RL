import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.evaluation.hybrid import evaluate_hybrid_run
from medical_graphrag.retrieval.hybrid import fuse_rrf


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
                '{"doc_id":"d3","title":"three","content":"c","source":"medrag_pubmed","year":null}',
                '{"doc_id":"d4","title":"four","content":"d","source":"medrag_pubmed","year":null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","content":"gold alpha","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two","content":"gold beta","source":"pubmedqa"}',
                '{"chunk_id":"c3","doc_id":"d3","order":0,"title":"three","content":"other","source":"medrag_pubmed"}',
                '{"chunk_id":"c4","doc_id":"d4","order":0,"title":"four","content":"other","source":"medrag_pubmed"}',
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
                '{"chunk_id":"c3","doc_id":"d3","order":0,"title":"three","source":"medrag_pubmed"}',
                '{"chunk_id":"c4","doc_id":"d4","order":0,"title":"four","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _rankings(tmp_path: Path) -> tuple[Path, Path]:
    bm25 = tmp_path / "bm25_rankings.jsonl"
    bm25.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":3},{"chunk_id":"c3","doc_id":"d3","chunk_rank":2,"score":2}]}',
                '{"query_id":"q2","split":"test","hits":[{"chunk_id":"c3","doc_id":"d3","chunk_rank":1,"score":3},{"chunk_id":"c2","doc_id":"d2","chunk_rank":2,"score":2}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dense = tmp_path / "dense_rankings.jsonl"
    dense.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","hits":[{"chunk_id":"c4","doc_id":"d4","chunk_rank":1,"score":0.9},{"chunk_id":"c1","doc_id":"d1","chunk_rank":2,"score":0.8}]}',
                '{"query_id":"q2","split":"test","hits":[{"chunk_id":"c2","doc_id":"d2","chunk_rank":1,"score":0.9},{"chunk_id":"c3","doc_id":"d3","chunk_rank":2,"score":0.8}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bm25, dense


def _run_context(
    dataset: Path,
    metadata: Path,
    bm25_rankings: Path,
    dense_rankings: Path,
    artifact_hashes: dict[str, str],
    *,
    rrf_k: int = 60,
) -> dict[str, object]:
    manifest_sha = _sha(dataset / "manifest.json")

    def leg(
        index_sha: str,
        index_report_sha: str,
        rankings: Path,
        search_extra: dict[str, object],
    ) -> dict[str, object]:
        return {
            "index": {
                "index_sha256": index_sha,
                "dataset_manifest_sha256": manifest_sha,
                "dataset_artifact_hashes": artifact_hashes,
            },
            "search": {
                "dataset_manifest_sha256": manifest_sha,
                "index_sha256": index_sha,
                "index_report_sha256": index_report_sha,
                "metadata_sha256": _sha(metadata),
                "rankings_sha256": _sha(rankings),
                "text_mode": "abstract_only",
                **search_extra,
            },
            "index_report_sha256": index_report_sha,
        }

    return {
        "git_commit": "abc",
        "rrf_k": rrf_k,
        "bm25": leg(
            "a" * 64,
            "b" * 64,
            bm25_rankings,
            {"k1": 0.9, "b": 0.4, "requested_top_k": 2},
        ),
        "dense": leg(
            "c" * 64,
            "d" * 64,
            dense_rankings,
            {
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "dim": 768,
                "normalized": True,
                "index_type": "IndexFlatIP",
                "requested_top_k": 2,
            },
        ),
    }


# ---- RRF fusion unit tests ----


def test_fuse_rrf_hand_computed_ranking() -> None:
    fused = fuse_rrf(["d1", "d2", "d3"], ["d2", "d1", "d4"], k=60)
    # d1: 1/61 + 1/62, d2: 1/62 + 1/61 (tie, d1 < d2), d3/d4: 1/63 each (d3 < d4)
    assert fused == ["d1", "d2", "d3", "d4"]


def test_fuse_rrf_doc_missing_from_one_ranking_gets_one_contribution() -> None:
    fused = fuse_rrf(["d1", "d2"], ["d2", "d3"], k=1)
    # d1: 1/2, d2: 1/3+1/2=5/6, d3: 1/2
    assert fused[0] == "d2"
    assert set(fused[1:]) == {"d1", "d3"}


def test_fuse_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="RRF k must be positive"):
        fuse_rrf(["d1"], ["d1"], k=0)


def test_fuse_rrf_deterministic_tie_break() -> None:
    first = ["b", "a"]
    second = ["a", "b"]
    assert fuse_rrf(first, second, k=1) == ["a", "b"]
    assert fuse_rrf(first, second, k=1) == fuse_rrf(second, first, k=1)


# ---- Hybrid evaluation tests ----


def test_hybrid_evaluation_reports_splits_and_manifest(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    bm25_rankings, dense_rankings = _rankings(tmp_path)
    output = tmp_path / "experiment"

    result = evaluate_hybrid_run(
        dataset,
        bm25_rankings,
        dense_rankings,
        metadata,
        output,
        run_context=_run_context(
            dataset, metadata, bm25_rankings, dense_rankings, artifact_hashes
        ),
    )

    assert set(result) == {"dev", "test"}
    assert result["dev"]["sample_count"] == 1
    assert result["dev"]["recall@1"] == 1.0
    assert result["test"]["sample_count"] == 1
    assert result["test"]["recall@1"] == 1.0  # d2 is rank 1 in dense
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == result
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rrf_k"] == 60
    assert manifest["bm25_config"] == {"k1": 0.9, "b": 0.4, "requested_top_k": 2}
    assert manifest["dense_config"]["embedding_model"] == (
        "sentence-transformers/all-mpnet-base-v2"
    )
    assert manifest["dense_config"]["normalized"] is True
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert cases["known_failures"] == {}


def test_hybrid_evaluation_rejects_replaced_bm25_rankings(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    bm25_rankings, dense_rankings = _rankings(tmp_path)
    context = _run_context(
        dataset, metadata, bm25_rankings, dense_rankings, artifact_hashes
    )
    bm25_rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"dev","hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":9}]}',
                '{"query_id":"q2","split":"test","hits":[{"chunk_id":"c3","doc_id":"d3","chunk_rank":1,"score":9}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bm25 rankings SHA-256 mismatch"):
        evaluate_hybrid_run(
            dataset,
            bm25_rankings,
            dense_rankings,
            metadata,
            tmp_path / "out",
            run_context=context,
        )


def test_hybrid_evaluation_records_known_failure_query(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"11570976","question":"Is it Crohn\'s disease?","answer":"yes","long_answer":"x","split":"test"}\n',
        encoding="utf-8",
    )
    (dataset / "documents.jsonl").write_text(
        "\n".join(
            [
                '{"doc_id":"gold","title":"gold","content":"g","source":"pubmedqa","year":null}',
                '{"doc_id":"x1","title":"x1","content":"a","source":"medrag_pubmed","year":null}',
                '{"doc_id":"x2","title":"x2","content":"b","source":"medrag_pubmed","year":null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"cg","doc_id":"gold","order":0,"title":"gold","content":"g","source":"pubmedqa"}',
                '{"chunk_id":"cx1","doc_id":"x1","order":0,"title":"x1","content":"a","source":"medrag_pubmed"}',
                '{"chunk_id":"cx2","doc_id":"x2","order":0,"title":"x2","content":"b","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\n11570976\tgold\t1\n", encoding="utf-8"
    )
    artifact_hashes = _write_manifest(dataset)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                '{"chunk_id":"cg","doc_id":"gold","order":0,"title":"gold","source":"pubmedqa"}',
                '{"chunk_id":"cx1","doc_id":"x1","order":0,"title":"x1","source":"medrag_pubmed"}',
                '{"chunk_id":"cx2","doc_id":"x2","order":0,"title":"x2","source":"medrag_pubmed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bm25 = tmp_path / "bm25.jsonl"
    bm25.write_text(
        '{"query_id":"11570976","split":"test","hits":[{"chunk_id":"cx1","doc_id":"x1","chunk_rank":1,"score":3},{"chunk_id":"cx2","doc_id":"x2","chunk_rank":2,"score":2}]}\n',
        encoding="utf-8",
    )
    dense = tmp_path / "dense.jsonl"
    dense.write_text(
        '{"query_id":"11570976","split":"test","hits":[{"chunk_id":"cx2","doc_id":"x2","chunk_rank":1,"score":0.9},{"chunk_id":"cx1","doc_id":"x1","chunk_rank":2,"score":0.8}]}\n',
        encoding="utf-8",
    )
    context = _run_context(dataset, metadata, bm25, dense, artifact_hashes)

    evaluate_hybrid_run(
        dataset, bm25, dense, metadata, tmp_path / "out", run_context=context
    )
    cases = json.loads((tmp_path / "out" / "cases.json").read_text(encoding="utf-8"))
    known = cases["known_failures"]["11570976"]
    assert known["gold_rank_bm25"] is None  # gold not in bm25 top hits
    assert known["gold_rank_dense"] is None
    assert known["gold_rank_hybrid"] is None
    assert cases["failure"][0]["query_id"] == "11570976"


def test_hybrid_evaluation_rejects_tampered_metadata(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    bm25_rankings, dense_rankings = _rankings(tmp_path)
    context = _run_context(
        dataset, metadata, bm25_rankings, dense_rankings, artifact_hashes
    )
    metadata.write_text(
        '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"tampered","source":"pubmedqa"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        evaluate_hybrid_run(
            dataset,
            bm25_rankings,
            dense_rankings,
            metadata,
            tmp_path / "out",
            run_context=context,
        )


def test_hybrid_evaluation_errors_on_empty_fused_ranking(tmp_path: Path) -> None:
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
        '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","content":"gold","source":"pubmedqa"}\n',
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8"
    )
    artifact_hashes = _write_manifest(dataset)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","source":"pubmedqa"}\n',
        encoding="utf-8",
    )
    bm25 = tmp_path / "bm25.jsonl"
    bm25.write_text(
        '{"query_id":"q1","split":"dev","hits":[]}\n', encoding="utf-8"
    )
    dense = tmp_path / "dense.jsonl"
    dense.write_text(
        '{"query_id":"q1","split":"dev","hits":[]}\n', encoding="utf-8"
    )
    context = _run_context(dataset, metadata, bm25, dense, artifact_hashes)

    with pytest.raises(ValueError, match="empty fused ranking for q1"):
        evaluate_hybrid_run(
            dataset, bm25, dense, metadata, tmp_path / "out", run_context=context
        )


def test_hybrid_evaluation_records_custom_k_in_manifest(tmp_path: Path) -> None:
    dataset, artifact_hashes = _make_dataset(tmp_path)
    metadata = _metadata(tmp_path)
    bm25_rankings, dense_rankings = _rankings(tmp_path)
    output = tmp_path / "experiment"

    evaluate_hybrid_run(
        dataset,
        bm25_rankings,
        dense_rankings,
        metadata,
        output,
        k=10,
        run_context=_run_context(
            dataset, metadata, bm25_rankings, dense_rankings, artifact_hashes, rrf_k=60
        ),
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rrf_k"] == 10
