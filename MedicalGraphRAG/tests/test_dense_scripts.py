import importlib.util
import json
from pathlib import Path

import pytest

from medical_graphrag.data.io import sha256_file


ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_standalone_scripts_add_project_src_to_import_path() -> None:
    for name in ("build_faiss_dense_index", "search_faiss_dense"):
        module = _load(name)
        assert str(ROOT / "src") in module.sys.path


def test_search_reads_embedding_version_from_distribution_metadata() -> None:
    module = _load("search_faiss_dense")
    requested: list[str] = []

    def resolver(package: str) -> str:
        requested.append(package)
        return "2.2.2"

    assert (
        module.package_version("sentence-transformers", resolver=resolver) == "2.2.2"
    )
    assert requested == ["sentence-transformers"]


def test_search_summary_reports_full_top_k_rankings() -> None:
    module = _load("search_faiss_dense")
    rows = [{"hits": [{} for _ in range(100)]} for _ in range(2)]

    summary = module.summarize_hit_counts(rows, requested_top_k=100)

    assert summary == {
        "requested_top_k": 100,
        "min_hits": 100,
        "max_hits": 100,
        "short_ranking_count": 0,
        "hit_count_histogram": {"100": 2},
    }


def test_search_one_maps_index_to_chunk_and_doc() -> None:
    module = _load("search_faiss_dense")

    class FakeEmbedder:
        def encode(self, text: str, normalize_embeddings: bool, show_progress_bar: bool):
            assert text == "alpha?"
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [0.1, 0.2, 0.3]

    class FakeIndex:
        def search(self, vector, k: int):
            assert k == 2
            import numpy as np

            return (np.array([[0.9, 0.8]]), np.array([[1, 0]]))

    result = module.search_one(
        FakeEmbedder(),
        FakeIndex(),
        {"query_id": "q1", "question": "alpha?", "split": "dev"},
        ["c1", "c2"],
        {"c1": "d1", "c2": "d2"},
        top_k=2,
        clock=iter([1.0, 1.012]).__next__,
    )

    assert result["query_id"] == "q1"
    assert result["split"] == "dev"
    assert result["latency_ms"] == 12.0
    assert result["hits"] == [
        {"chunk_id": "c2", "doc_id": "d2", "chunk_rank": 1, "score": 0.9},
        {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 2, "score": 0.8},
    ]


def test_search_input_validation_rejects_modified_index(tmp_path: Path) -> None:
    module = _load("search_faiss_dense")
    index = tmp_path / "index.faiss"
    index.write_bytes(b"index")
    embeddings = tmp_path / "chunk_embeddings.npy"
    embeddings.write_bytes(b"embeddings")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"c1","doc_id":"d1"}\n', encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"query_id":"q1","question":"alpha"}\n', encoding="utf-8")
    report = tmp_path / "index_report.json"
    report.write_text(
        json.dumps(
            {
                "index_sha256": sha256_file(index),
                "embeddings_sha256": sha256_file(embeddings),
                "metadata_sha256": sha256_file(metadata),
                "text_mode": "abstract_only",
                "dataset_artifact_hashes": {"questions.jsonl": sha256_file(questions)},
            }
        ),
        encoding="utf-8",
    )
    index.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        module.validate_index(index, embeddings, metadata, questions, report)


def test_search_input_validation_rejects_modified_embeddings(tmp_path: Path) -> None:
    module = _load("search_faiss_dense")
    index = tmp_path / "index.faiss"
    index.write_bytes(b"index")
    embeddings = tmp_path / "chunk_embeddings.npy"
    embeddings.write_bytes(b"embeddings")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"c1","doc_id":"d1"}\n', encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"query_id":"q1","question":"alpha"}\n', encoding="utf-8")
    report = tmp_path / "index_report.json"
    report.write_text(
        json.dumps(
            {
                "index_sha256": sha256_file(index),
                "embeddings_sha256": sha256_file(embeddings),
                "metadata_sha256": sha256_file(metadata),
                "text_mode": "abstract_only",
                "dataset_artifact_hashes": {"questions.jsonl": sha256_file(questions)},
            }
        ),
        encoding="utf-8",
    )
    embeddings.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="embeddings SHA-256 mismatch"):
        module.validate_index(index, embeddings, metadata, questions, report)
