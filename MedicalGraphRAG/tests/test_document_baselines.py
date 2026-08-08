import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.data.io import sha256_file
from medical_graphrag.evaluation.document import evaluate_document_run
from medical_graphrag.evaluation.graph import write_graph_pair_cases
from medical_graphrag.evaluation.retrieval import validate_hit_rows
from medical_graphrag.retrieval.dense import build_dense_document_index
from medical_graphrag.retrieval.document_embeddings import build_document_embeddings
from medical_graphrag.retrieval.bm25 import export_document_collection
from tests.mocks import MockEmbedder

from tests.test_graph_document import _write_pubmedqa_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_hit_rows_document_schema():
    ok = [{"hits": [{"doc_id": "d1", "rank": 1, "score": 0.9},
                    {"doc_id": "d2", "rank": 2, "score": 0.8}]}]
    validate_hit_rows(ok, retrieval_unit="document")

    missing_rank = [{"hits": [{"doc_id": "d1", "score": 0.9}]}]
    with pytest.raises(ValueError, match="missing rank"):
        validate_hit_rows(missing_rank, retrieval_unit="document")

    duplicate_doc = [{"hits": [{"doc_id": "d1", "rank": 1, "score": 0.9},
                               {"doc_id": "d1", "rank": 2, "score": 0.8}]}]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        validate_hit_rows(duplicate_doc, retrieval_unit="document")

    with_chunk_id = [{"hits": [{"doc_id": "d1", "rank": 1, "score": 0.9, "chunk_id": "c1"}]}]
    with pytest.raises(ValueError, match="chunk_id"):
        validate_hit_rows(with_chunk_id, retrieval_unit="document")

    chunk_ok = [{"hits": [{"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "score": 0.9}]}]
    validate_hit_rows(chunk_ok, retrieval_unit="chunk")


def _build_dense_report(dataset: Path, tmp_path: Path, embedder) -> tuple[Path, dict]:
    artifact = tmp_path / "artifact"
    build_document_embeddings(dataset, artifact, model_name="mock", embedder=embedder)
    dense_out = tmp_path / "dense"
    report = build_dense_document_index(
        dataset, dense_out, document_embeddings_dir=artifact, batch_size=4
    )
    return artifact, dense_out, report


def test_p0_8_three_consumers_same_embedding_report_sha(tmp_path: Path):
    """Dense index 与 Graph build 消费同一 embedding artifact 的 report hash."""
    embedder = MockEmbedder()
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact, dense_out, dense_report = _build_dense_report(dataset, tmp_path, embedder)

    expected = sha256_file(artifact / "document_embedding_report.json")
    assert dense_report["embedding_report_sha256"] == expected

    from medical_graphrag.retrieval.graph import GraphBuildConfig, build_graph_index
    from tests.mocks import MockNlp

    index_dir = tmp_path / "graph"
    graph_config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="similarity",
        embedding_model="mock", ner_model="mock_ner",
        similarity_k=5, similarity_min_cosine=0.0,
    )
    graph_report = build_graph_index(
        dataset, index_dir, build_config=graph_config, batch_size=4,
        embedder=embedder, nlp=MockNlp(), document_embeddings_dir=artifact,
    )
    assert graph_report["embedding_report_sha256"] == expected
    assert graph_report["embedding_embeddings_sha256"] == dense_report[
        "embedding_embeddings_sha256"
    ]


def test_dense_document_rejects_tampered_embedding_metadata(tmp_path: Path):
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = tmp_path / "artifact"
    build_document_embeddings(
        dataset, artifact, model_name="mock", embedder=MockEmbedder()
    )
    metadata = artifact / "document_embedding_metadata.jsonl"
    rows = metadata.read_text(encoding="utf-8").splitlines()
    metadata.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata SHA-256"):
        build_dense_document_index(
            dataset,
            tmp_path / "dense",
            document_embeddings_dir=artifact,
        )


def test_export_document_collection_preserves_frozen_document_order(tmp_path: Path):
    dataset = tmp_path / "ordered_dataset"
    dataset.mkdir()
    doc_ids = [f"doc-{index}" for index in range(20)]
    (dataset / "documents.jsonl").write_text(
        "".join(
            json.dumps({
                "doc_id": doc_id,
                "title": doc_id,
                "content": f"content {doc_id}",
                "source": "test",
            }) + "\n"
            for doc_id in doc_ids
        ),
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text("", encoding="utf-8")
    (dataset / "questions.jsonl").write_text("", encoding="utf-8")
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\n", encoding="utf-8"
    )
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 20, "chunks": 0, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )

    output = tmp_path / "bm25"
    export_document_collection(dataset, output)
    metadata_ids = [
        json.loads(line)["doc_id"]
        for line in (output / "document_metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert metadata_ids == doc_ids


def test_evaluate_document_run_happy_path(tmp_path: Path):
    embedder = MockEmbedder()
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact, dense_out, dense_report = _build_dense_report(dataset, tmp_path, embedder)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))

    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        '{"query_id":"q1","split":"dev","latency_ms":1.0,"hits":[{"doc_id":"PMID:1","rank":1,"score":0.9}]}\n',
        encoding="utf-8")
    search_report = {
        "query_count": 1, "requested_top_k": 10, "min_hits": 1, "max_hits": 1,
        "short_ranking_count": 1, "hit_count_histogram": {"1": 1},
        "retrieval_unit": "document",
        "embedding_model": dense_report["embedding_model"],
        "dim": dense_report["dim"], "index_type": dense_report["index_type"],
        "embedding_report_sha256": dense_report["embedding_report_sha256"],
        "index_sha256": dense_report["index_sha256"],
        "index_report_sha256": sha256_file(dense_out / "index_build.json"),
        "dataset_manifest_sha256": dense_report["dataset_manifest_sha256"],
        "questions_sha256": manifest["artifact_hashes"]["questions.jsonl"],
        "rankings_sha256": sha256_file(rankings),
        "text_mode": "abstract_only",
    }
    run_context = {
        "index": dense_report,
        "search": search_report,
        "index_report_sha256": sha256_file(dense_out / "index_build.json"),
        "dataset_manifest_sha256": dense_report["dataset_manifest_sha256"],
        "git_commit": "test", "host_platform": "test",
        "host_python_version": "test", "docker_image": "test",
        "evaluation_command": ["test"],
    }
    exp = tmp_path / "exp"
    metrics = evaluate_document_run(dataset, rankings, exp, run_context=run_context)
    assert metrics["dev"]["recall@1"] == 1.0
    assert metrics["dev"]["mrr@10"] == 1.0
    assert (exp / "metrics.json").exists()


def _write_doc_rankings(path: Path, query_id: str, split: str, hits) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"query_id": query_id, "split": split, "latency_ms": 1.0,
                        "hits": hits}) + "\n")


def test_write_graph_pair_cases(tmp_path: Path):
    dataset = _write_pubmedqa_dataset(tmp_path)
    # 增加一个 test split 问题,让 paired cases 有内容
    questions_path = dataset / "questions.jsonl"
    questions_path.write_text(
        '{"query_id":"q1","question":"Do vaccines need cold storage?","answer":"yes","long_answer":"x","split":"dev"}\n'
        '{"query_id":"q2","question":"Do patients like first names?","answer":"yes","long_answer":"x","split":"test"}\n',
        encoding="utf-8")
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\tPMID:1\t1\nq2\tPMID:2\t1\n", encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(json.dumps({
        "counts": {"questions": 2, "documents": 2, "chunks": 4, "qrels": 2},
        "artifact_hashes": {name: _sha(dataset / name) for name in names},
    }), encoding="utf-8")

    ep = tmp_path / "ep.jsonl"
    sim = tmp_path / "sim.jsonl"
    _write_doc_rankings(ep, "q1", "dev", [{"doc_id": "PMID:1", "rank": 1, "score": 0.9}])
    _write_doc_rankings(ep, "q2", "test", [{"doc_id": "PMID:2", "rank": 1, "score": 0.9}])
    _write_doc_rankings(sim, "q1", "dev", [{"doc_id": "PMID:1", "rank": 1, "score": 0.9}])
    _write_doc_rankings(sim, "q2", "test",
                        [{"doc_id": "PMID:2", "rank": 1, "score": 0.9},
                         {"doc_id": "PMID:1", "rank": 2, "score": 0.5}])
    out = tmp_path / "paired.jsonl"
    summary = write_graph_pair_cases(dataset, ep, sim, out)
    assert summary["total_test_questions"] == 1
    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    assert rows[0]["query_id"] == "q2"
    assert rows[0]["gold_rank_ep"] == 1
    assert rows[0]["gold_rank_sim"] == 1
    assert rows[0]["label"] == "no_change"
    assert rows[0]["new_similarity_neighbours"][0]["doc_id"] == "PMID:1"
