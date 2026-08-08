import hashlib
import json
from pathlib import Path

from medical_graphrag.retrieval.rerank_document import run_rerank_document

from tests.test_graph_document import _write_pubmedqa_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _MockReranker:
    def rerank(self, query, candidates):
        # 确定性:按 doc_id 排序后给分数,验证候选 union 传入正确
        return [(c["doc_id"], 1.0 - i * 0.01)
                for i, c in enumerate(sorted(candidates, key=lambda c: c["doc_id"]))]


def _write_source_rankings(path: Path, search_report: Path, dataset: Path) -> None:
    path.write_text(
        json.dumps({"query_id": "q1", "split": "dev", "latency_ms": 1.0,
                    "hits": [{"doc_id": "PMID:1", "rank": 1, "score": 0.9}]}) + "\n",
        encoding="utf-8")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    search_report.write_text(json.dumps({
        "rankings_sha256": _sha(path),
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        "retrieval_unit": "document",
        "questions_sha256": manifest["artifact_hashes"]["questions.jsonl"],
    }, indent=2) + "\n", encoding="utf-8")


def test_run_rerank_document_union_and_output(tmp_path: Path):
    dataset = _write_pubmedqa_dataset(tmp_path)
    bm25 = tmp_path / "bm25.jsonl"
    bm25_report = tmp_path / "bm25_report.json"
    dense = tmp_path / "dense.jsonl"
    dense_report = tmp_path / "dense_report.json"
    _write_source_rankings(bm25, bm25_report, dataset)
    # dense 提供一个 bm25 没有的候选(PMID:2),验证 union 并集
    dense.write_text(
        json.dumps({"query_id": "q1", "split": "dev", "latency_ms": 1.0,
                    "hits": [{"doc_id": "PMID:2", "rank": 1, "score": 0.8}]}) + "\n",
        encoding="utf-8")
    dense_report.write_text(json.dumps({
        "rankings_sha256": _sha(dense),
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        "retrieval_unit": "document",
    }, indent=2) + "\n", encoding="utf-8")

    out = tmp_path / "reranked.jsonl"
    rep = tmp_path / "rerank_report.json"
    report = run_rerank_document(
        sources={"bm25": bm25, "dense": dense},
        source_reports={"bm25": bm25_report, "dense": dense_report},
        questions=dataset / "questions.jsonl",
        documents=dataset / "documents.jsonl",
        dataset_manifest=dataset / "manifest.json",
        output=out,
        report=rep,
        top_n=50,
        reranker=_MockReranker(),
    )
    assert report["sources"] == ["bm25", "dense"]
    assert report["query_count"] == 1
    assert "source_rankings_sha256" in report
    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    assert rows[0]["query_id"] == "q1"
    # 两个候选(PMID:1 + PMID:2)都被 union 进 reranker
    assert set(rows[0]["doc_ids"]) == {"PMID:1", "PMID:2"}
    assert len(rows[0]["scores"]) == 2
    assert report["rankings_sha256"] == _sha(out)


def test_run_rerank_document_rejects_chunk_source(tmp_path: Path):
    dataset = _write_pubmedqa_dataset(tmp_path)
    bm25 = tmp_path / "bm25.jsonl"
    bm25_report = tmp_path / "bm25_report.json"
    _write_source_rankings(bm25, bm25_report, dataset)
    # 篡改报告为 chunk unit → 必须拒绝
    bm25_report.write_text(json.dumps({
        "rankings_sha256": _sha(bm25),
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        "retrieval_unit": "chunk",
    }, indent=2) + "\n", encoding="utf-8")
    dense = tmp_path / "dense.jsonl"
    dense_report = tmp_path / "dense_report.json"
    _write_source_rankings(dense, dense_report, dataset)
    try:
        run_rerank_document(
            sources={"bm25": bm25, "dense": dense},
            source_reports={"bm25": bm25_report, "dense": dense_report},
            questions=dataset / "questions.jsonl",
            documents=dataset / "documents.jsonl",
            dataset_manifest=dataset / "manifest.json",
            output=tmp_path / "out.jsonl",
            report=tmp_path / "r.json",
            top_n=50,
            reranker=_MockReranker(),
        )
        raise AssertionError("expected ValueError for chunk source")
    except ValueError as error:
        assert "document retrieval unit" in str(error)
