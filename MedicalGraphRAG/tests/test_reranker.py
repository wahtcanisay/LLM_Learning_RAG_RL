import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.evaluation.reranker import evaluate_reranker_run
from medical_graphrag.retrieval.reranker import QwenReranker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_dataset(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        "\n".join(
            [
                '{"query_id":"q1","question":"a","answer":"","long_answer":"","split":"test"}',
                '{"query_id":"q2","question":"b","answer":"","long_answer":"","split":"test"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "documents.jsonl").write_text(
        "\n".join(
            [
                '{"doc_id":"d1","title":"one","content":"alpha","source":"nfcorpus","year":null}',
                '{"doc_id":"d2","title":"two","content":"beta","source":"nfcorpus","year":null}',
                '{"doc_id":"d3","title":"three","content":"gamma","source":"nfcorpus","year":null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"d1","doc_id":"d1","order":0,"title":"one","content":"alpha","source":"nfcorpus"}',
                '{"chunk_id":"d2","doc_id":"d2","order":0,"title":"two","content":"beta","source":"nfcorpus"}',
                '{"chunk_id":"d3","doc_id":"d3","order":0,"title":"three","content":"gamma","source":"nfcorpus"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\nq1\td2\t1\nq2\td2\t1\n",
        encoding="utf-8",
    )
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    hashes = {name: _sha(dataset / name) for name in names}
    counts = {"questions": 2, "documents": 3, "chunks": 3, "qrels": 3}
    (dataset / "manifest.json").write_text(
        json.dumps({"counts": counts, "artifact_hashes": hashes}), encoding="utf-8"
    )
    return dataset, hashes


def test_reranker_scores_and_sorts_desc() -> None:
    reranker = QwenReranker.__new__(QwenReranker)

    class FakeModel:
        def predict(self, pairs):
            return [1.0 if "Paris" in p[1] else 0.0 for p in pairs]

    reranker.model = FakeModel()
    ranked = reranker.rerank(
        "capital",
        [{"doc_id": "a", "content": "some text"}, {"doc_id": "b", "content": "Paris is here"}],
    )
    assert ranked == [("b", 1.0), ("a", 0.0)]


def test_reranker_empty_candidates_returns_empty() -> None:
    reranker = QwenReranker.__new__(QwenReranker)
    reranker.model = object()
    assert reranker.rerank("q", []) == []


def test_reranker_evaluation_metrics_and_bindings(tmp_path: Path) -> None:
    dataset, _ = _make_dataset(tmp_path)
    rankings = tmp_path / "reranked.jsonl"
    rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"test","doc_ids":["d1","d2","d3"],"scores":[2.0,1.0,0.0]}',
                '{"query_id":"q2","split":"test","doc_ids":["d3","d2","d1"],"scores":[1.0,0.5,0.0]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions_path = dataset / "questions.jsonl"
    report = tmp_path / "reranker_report.json"
    report.write_text(
        json.dumps(
            {
                "model": "models/Qwen3-Reranker-0.6B",
                "top_n": 50,
                "rankings_sha256": _sha(rankings),
                "questions_sha256": _sha(questions_path),
                "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = evaluate_reranker_run(
        dataset,
        rankings,
        output,
        run_context={
            "git_commit": "abc",
            "reranker": json.loads(report.read_text(encoding="utf-8")),
            "questions_path": str(questions_path),
            "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        },
    )

    # q1: both gold d1,d2 in top-2 -> recall@10 = 1.0, mrr = 1.0
    # q2: gold d2 at rank 2 -> recall@10 = 1.0, mrr = 0.5
    assert result["test"]["sample_count"] == 2
    assert result["test"]["recall@10"] == 1.0
    assert result["test"]["mrr@10"] == 0.75
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reranker"] == {"model": "models/Qwen3-Reranker-0.6B", "top_n": 50}


def test_reranker_evaluation_rejects_tampered_rankings(tmp_path: Path) -> None:
    dataset, _ = _make_dataset(tmp_path)
    rankings = tmp_path / "reranked.jsonl"
    rankings.write_text(
        "\n".join(
            [
                '{"query_id":"q1","split":"test","doc_ids":["d1","d2","d3"],"scores":[2.0,1.0,0.0]}',
                '{"query_id":"q2","split":"test","doc_ids":["d3","d2","d1"],"scores":[1.0,0.5,0.0]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions_path = dataset / "questions.jsonl"
    report = tmp_path / "reranker_report.json"
    report.write_text(
        json.dumps(
            {
                "model": "models/Qwen3-Reranker-0.6B",
                "top_n": 50,
                "rankings_sha256": "0" * 64,
                "questions_sha256": _sha(questions_path),
                "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reranker rankings SHA-256 mismatch"):
        evaluate_reranker_run(
            dataset,
            rankings,
            tmp_path / "out",
            run_context={
                "git_commit": "abc",
                "reranker": json.loads(report.read_text(encoding="utf-8")),
                "questions_path": str(questions_path),
                "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
            },
        )
