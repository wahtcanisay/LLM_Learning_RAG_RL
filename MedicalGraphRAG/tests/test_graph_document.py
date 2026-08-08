import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from medical_graphrag.data.io import sha256_file
from medical_graphrag.evaluation.graph import evaluate_graph_run
from medical_graphrag.retrieval.document_embeddings import build_document_embeddings
from medical_graphrag.retrieval.graph import (
    GraphBuildConfig,
    GraphConfig,
    LinearGraphRetriever,
    build_graph_index,
)
from tests.mocks import MockEmbedder, MockNlp


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pubmedqa_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    docs = [
        {"doc_id": "PMID:1", "title": "vaccine storage", "content": (
            "To assess quality of storage of vaccines in the community.\n"
            "Vaccines were exposed to subzero temperatures in three fridges.\n"
            "Aspirin relieves pain but can cause fever."),
         "source": "pubmedqa", "year": "1992"},
        {"doc_id": "PMID:2", "title": "first names", "content": (
            "To assess the acceptability to patients of the use of patients' first names.\n"
            "Most patients liked being called by their first names.\n"
            "Aspirin is used for pain."),
         "source": "pubmedqa", "year": "1990"},
    ]
    (dataset / "documents.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    chunks = [
        {"chunk_id": "PMID:1#0", "doc_id": "PMID:1", "order": 0, "title": "t",
         "content": "To assess quality of storage of vaccines.", "source": "pubmedqa"},
        {"chunk_id": "PMID:1#1", "doc_id": "PMID:1", "order": 1, "title": "t",
         "content": "Vaccines exposed to subzero temperatures.", "source": "pubmedqa"},
        {"chunk_id": "PMID:2#0", "doc_id": "PMID:2", "order": 0, "title": "t",
         "content": "Acceptability of first names to patients.", "source": "pubmedqa"},
        {"chunk_id": "PMID:2#1", "doc_id": "PMID:2", "order": 1, "title": "t",
         "content": "Most patients liked first names.", "source": "pubmedqa"},
    ]
    (dataset / "chunks.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"q1","question":"Do vaccines need cold storage?","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8")
    (dataset / "qrels.tsv").write_text("query_id\tdoc_id\trelevance\nq1\tPMID:1\t1\n",
                                       encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(json.dumps({
        "counts": {"questions": 1, "documents": 2, "chunks": 4, "qrels": 1},
        "artifact_hashes": {name: _sha(dataset / name) for name in names},
    }), encoding="utf-8")
    return dataset


def _build_artifact(dataset: Path, tmp_path: Path, embedder) -> Path:
    artifact = tmp_path / "artifact"
    build_document_embeddings(dataset, artifact, model_name="mock", embedder=embedder)
    return artifact


@pytest.fixture
def mock_models():
    return MockEmbedder(), MockNlp()


def test_graph_build_config_illegal_combinations():
    with pytest.raises(ValueError, match="document"):
        GraphBuildConfig(retrieval_unit="document", passage_edge_mode="adjacent",
                         embedding_model="m", ner_model="n")
    with pytest.raises(ValueError, match="chunk"):
        GraphBuildConfig(retrieval_unit="chunk", passage_edge_mode="similarity",
                         embedding_model="m", ner_model="n")


def test_graph_build_config_profiles():
    ep = GraphBuildConfig(retrieval_unit="document", passage_edge_mode="none",
                          embedding_model="m", ner_model="n")
    sim = GraphBuildConfig(retrieval_unit="document", passage_edge_mode="similarity",
                           embedding_model="m", ner_model="n")
    assert ep.graph_profile == "document_ep_v1"
    assert sim.graph_profile == "document_similarity_v1"
    chunk_none = GraphBuildConfig(retrieval_unit="chunk", passage_edge_mode="none",
                                  embedding_model="m", ner_model="n")
    assert chunk_none.graph_profile == "chunk_entity_only_v1"


def test_build_document_similarity_index(tmp_path: Path, mock_models):
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = _build_artifact(dataset, tmp_path, embedder)
    index_dir = tmp_path / "graph_sim"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="similarity",
        embedding_model="mock", ner_model="mock_ner",
        similarity_k=5, similarity_min_cosine=0.0,
    )
    report = build_graph_index(
        dataset, index_dir, build_config=config, batch_size=4,
        embedder=embedder, nlp=nlp, document_embeddings_dir=artifact,
    )
    assert report["retrieval_unit"] == "document"
    assert report["passage_edge_mode"] == "similarity"
    assert report["graph_profile"] == "document_similarity_v1"
    assert report["passage_count"] == 2
    assert report["source_artifact"] == "documents.jsonl"
    assert report["entity_passage_edge_count"] >= 1
    assert report["edge_count_by_type"]["similarity"] >= 0
    assert report["embedding_report_sha256"] is not None
    assert report["embedding_embeddings_sha256"] == sha256_file(
        artifact / "document_embeddings.npy"
    )
    assert (index_dir / "graph.graphml").exists()


def test_build_document_ep_index_has_no_similarity_edges(tmp_path: Path, mock_models):
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = _build_artifact(dataset, tmp_path, embedder)
    index_dir = tmp_path / "graph_ep"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="none",
        embedding_model="mock", ner_model="mock_ner",
    )
    report = build_graph_index(
        dataset, index_dir, build_config=config, batch_size=4,
        embedder=embedder, nlp=nlp, document_embeddings_dir=artifact,
    )
    assert report["graph_profile"] == "document_ep_v1"
    assert report["edge_count_by_type"]["similarity"] == 0
    assert report["edge_count_by_type"]["adjacent"] == 0


def test_build_document_requires_artifact(tmp_path: Path, mock_models):
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="none",
        embedding_model="mock", ner_model="mock_ner",
    )
    with pytest.raises(ValueError, match="document_embeddings_dir"):
        build_graph_index(
            dataset, tmp_path / "g", build_config=config, batch_size=4,
            embedder=embedder, nlp=nlp,
        )


def test_document_retriever_returns_doc_ids(tmp_path: Path, mock_models):
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = _build_artifact(dataset, tmp_path, embedder)
    index_dir = tmp_path / "graph_sim"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="similarity",
        embedding_model="mock", ner_model="mock_ner",
        similarity_k=5, similarity_min_cosine=0.0,
    )
    build_graph_index(
        dataset, index_dir, build_config=config, batch_size=4,
        embedder=embedder, nlp=nlp, document_embeddings_dir=artifact,
    )
    retriever = LinearGraphRetriever(
        index_dir, config=GraphConfig(embedding_model="mock", ner_model="mock_ner"),
        embedder=embedder, nlp=nlp,
    )
    assert retriever.retrieval_unit == "document"
    ids, scores = retriever.search("Do vaccines need cold storage?", top_k=2)
    assert set(ids) <= {"PMID:1", "PMID:2"}
    assert len(scores) == len(ids)


def _build_valid_search_report(
    dataset: Path, index_dir: Path, rankings_path: Path,
) -> dict[str, object]:
    report = json.loads((index_dir / "graph_build.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rankings_path.open(encoding="utf-8") if line.strip()]
    counts = [len(row["hits"]) for row in rows]
    return {
        "query_count": len(rows),
        "requested_top_k": 10,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(c < 10 for c in counts),
        "hit_count_histogram": {str(c): counts.count(c) for c in sorted(set(counts))},
        "retrieval_unit": "document",
        "ner_model": report["ner_model"],
        "embedding_model": report["embedding_model"],
        "graph_sha256": report["graph_sha256"],
        # P0-6: 必须是 graph build report 文件本身的 SHA-256
        "graph_build_report_sha256": sha256_file(index_dir / "graph_build.json"),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "questions_sha256": manifest["artifact_hashes"]["questions.jsonl"],
        "rankings_sha256": sha256_file(rankings_path),
        "config": report["config"],
    }


def _write_rankings(dataset: Path, rankings_path: Path) -> None:
    rankings_path.write_text(
        '{"query_id":"q1","split":"dev","latency_ms":1.0,"hits":[{"doc_id":"PMID:1","rank":1,"score":0.9}]}\n',
        encoding="utf-8")


def test_evaluate_graph_run_document_hash_chain(tmp_path: Path, mock_models):
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = _build_artifact(dataset, tmp_path, embedder)
    index_dir = tmp_path / "graph_ep"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="none",
        embedding_model="mock", ner_model="mock_ner",
    )
    report = build_graph_index(
        dataset, index_dir, build_config=config, batch_size=4,
        embedder=embedder, nlp=nlp, document_embeddings_dir=artifact,
    )
    rankings = tmp_path / "rankings.jsonl"
    _write_rankings(dataset, rankings)
    search_report = _build_valid_search_report(dataset, index_dir, rankings)
    run_context = {
        "index": report,
        "search": search_report,
        "index_report_sha256": sha256_file(index_dir / "graph_build.json"),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "dataset_artifact_hashes": report["dataset_artifact_hashes"],
        "git_commit": "test",
        "host_platform": "test",
        "host_python_version": "test",
        "docker_image": "test",
        "evaluation_command": ["test"],
    }
    exp_dir = tmp_path / "exp"
    metrics = evaluate_graph_run(dataset, rankings, exp_dir, run_context=run_context)
    assert metrics["dev"]["recall@1"] == 1.0


def test_evaluate_graph_run_rejects_wrong_graph_build_report_sha(
    tmp_path: Path, mock_models
):
    """P0-6: graph_build_report_sha256 用 graph_sha256 代替必须被拒绝."""
    embedder, nlp = mock_models
    dataset = _write_pubmedqa_dataset(tmp_path)
    artifact = _build_artifact(dataset, tmp_path, embedder)
    index_dir = tmp_path / "graph_ep"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="none",
        embedding_model="mock", ner_model="mock_ner",
    )
    report = build_graph_index(
        dataset, index_dir, build_config=config, batch_size=4,
        embedder=embedder, nlp=nlp, document_embeddings_dir=artifact,
    )
    rankings = tmp_path / "rankings.jsonl"
    _write_rankings(dataset, rankings)
    search_report = _build_valid_search_report(dataset, index_dir, rankings)
    # 用 graph_sha256 冒充 graph_build_report_sha256 → 必须失败
    search_report["graph_build_report_sha256"] = report["graph_sha256"]
    run_context = {
        "index": report,
        "search": search_report,
        "index_report_sha256": sha256_file(index_dir / "graph_build.json"),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "dataset_artifact_hashes": report["dataset_artifact_hashes"],
        "git_commit": "test", "host_platform": "test",
        "host_python_version": "test", "docker_image": "test",
        "evaluation_command": ["test"],
    }
    with pytest.raises(ValueError, match="index report file"):
        evaluate_graph_run(dataset, rankings, tmp_path / "exp", run_context=run_context)
