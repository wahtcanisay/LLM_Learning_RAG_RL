import json
from pathlib import Path

from medical_graphrag.evaluation.bm25 import evaluate_bm25_run


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

    result = evaluate_bm25_run(
        dataset,
        metadata,
        rankings,
        output,
        run_context={"git_commit": "abc", "index": {"elapsed_seconds": 1.0}},
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
