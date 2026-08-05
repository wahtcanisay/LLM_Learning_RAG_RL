import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from medical_graphrag.retrieval.graph import (
    GraphConfig,
    build_entity_passage_edges,
    build_graph_index,
    extract_entities,
    LinearGraphRetriever,
)

ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"q1","question":"Does aspirin relieve pain?","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8",
    )
    (dataset / "documents.jsonl").write_text(
        '{"doc_id":"d1","title":"aspirin","content":"aspirin relieves pain","source":"pubmedqa","year":null}\n',
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"aspirin","content":"Aspirin reduces pain in patients with osteoarthritis.","source":"pubmedqa"}',
                '{"chunk_id":"c2","doc_id":"d1","order":1,"title":"aspirin","content":"Aspirin can cause bleeding in the stomach.","source":"pubmedqa"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8"
    )
    _write_manifest(dataset)
    return dataset


def test_build_entity_passage_edges_normalizes_counts() -> None:
    entities, edges = build_entity_passage_edges(
        ["c1", "c2"],
        ["Aspirin reduces pain with Aspirin", "Aspirin causes bleeding"],
        [["Aspirin", "pain"], ["Aspirin", "bleeding"]],
    )
    assert entities == ["Aspirin", "bleeding", "pain"]
    assert edges["c1"]["Aspirin"] == pytest.approx(2 / 3)
    assert edges["c1"]["pain"] == pytest.approx(1 / 3)
    assert edges["c2"]["Aspirin"] == pytest.approx(1 / 2)
    assert edges["c2"]["bleeding"] == pytest.approx(1 / 2)


def test_graph_config_defaults() -> None:
    config = GraphConfig()
    assert config.damping == 0.85
    assert config.passage_ratio == 1.5
    assert config.iteration_threshold == 0.5
    assert config.top_k_sentence == 1
    assert config.max_iterations == 3


def test_extract_entities_medical_ner() -> None:
    import spacy

    nlp = spacy.load("en_ner_bc5cdr_md")
    results = extract_entities(
        nlp, ["Aspirin reduces pain in patients with osteoarthritis."]
    )
    assert set(results[0]) >= {"Aspirin", "pain", "osteoarthritis"}


def test_graph_build_and_search_smoke(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    index_dir = tmp_path / "graph"
    report = build_graph_index(dataset, index_dir)
    assert report["entity_count"] >= 1
    assert report["passage_count"] == 2
    assert report["edge_count"] >= 1
    assert report["adjacent_passage_edges"] is False
    assert (index_dir / "graph.graphml").exists()

    retriever = LinearGraphRetriever(index_dir)
    passage_ids, scores = retriever.search("Does aspirin relieve pain?", top_k=2)
    assert len(passage_ids) == 2
    assert "c1" in passage_ids  # the passage that mentions aspirin and pain
    assert len(scores) == 2


def test_search_graph_script_standalone_and_text_mode_gate(tmp_path: Path) -> None:
    """search_graph 脚本可独立加载；validate_index 在哈希检查前先强制 abstract_only。

    该门与 rerank_candidates._validate_source 对 graph search report 的 text_mode
    要求配套：保证图检索链路与 BM25/Dense 一样固定 abstract_only 文本模式。
    """
    module = _load("search_graph")
    assert str(ROOT / "src") in module.sys.path

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    # validate_index 依次哈希这些产物；占位文件让第二次断言走到 SHA-256 mismatch 而非 FileNotFoundError。
    for filename in (
        "graph.graphml",
        "sentence_embeddings.npy",
        "entity_embeddings.npy",
        "passage_embeddings.npy",
        "entities.jsonl",
        "entity_to_sentences.jsonl",
        "sentence_to_entities.jsonl",
    ):
        (index_dir / filename).write_bytes(b"placeholder")
    index_report = tmp_path / "index_report.json"

    # 非 abstract_only：在触碰任何文件哈希前就应被拒绝。
    index_report.write_text(
        json.dumps({"text_mode": "title_abstract"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="abstract_only"):
        module.validate_index(index_dir, tmp_path / "questions.jsonl", index_report)

    # abstract_only：通过 text_mode 门，随后才做文件哈希绑定（缺文件→不同的错误）。
    index_report.write_text(
        json.dumps({"text_mode": "abstract_only"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.validate_index(index_dir, tmp_path / "questions.jsonl", index_report)
