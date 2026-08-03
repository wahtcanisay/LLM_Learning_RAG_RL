import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_index_command_is_fixed() -> None:
    module = _load("build_pyserini_index")

    command = module.build_command(Path("collection"), Path("index"), threads=8)

    assert command[-12:] == [
        "--collection",
        "JsonCollection",
        "--input",
        "collection",
        "--index",
        "index",
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "8",
        "--stemmer",
        "porter",
    ]


def test_scripts_read_pyserini_version_from_distribution_metadata() -> None:
    for name in ("build_pyserini_index", "search_pyserini_bm25"):
        module = _load(name)
        requested = []

        def resolver(package: str) -> str:
            requested.append(package)
            return "0.22.1"

        assert module.package_version("pyserini", resolver=resolver) == "0.22.1"
        assert requested == ["pyserini"]


def test_search_one_query_records_rank_score_and_doc_id() -> None:
    module = _load("search_pyserini_bm25")

    class Hit:
        def __init__(self, docid: str, score: float) -> None:
            self.docid = docid
            self.score = score

    class Searcher:
        def search(self, query: str, k: int):
            assert query == "alpha?"
            assert k == 1
            return [Hit("c1", 2.5)]

    result = module.search_one(
        Searcher(),
        {"query_id": "q1", "question": "alpha?", "split": "dev"},
        {"c1": "d1"},
        top_k=1,
        clock=iter([1.0, 1.012]).__next__,
    )

    assert result["query_id"] == "q1"
    assert result["split"] == "dev"
    assert result["latency_ms"] == 12.0
    assert result["hits"] == [
        {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "score": 2.5}
    ]


def test_search_jsonl_reader_preserves_unicode_paragraph_separator(tmp_path: Path) -> None:
    module = _load("search_pyserini_bm25")
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"chunk_id":"c1","title":"alpha\u2029beta"}\n', encoding="utf-8")

    rows = module._read_jsonl(path)

    assert rows == [{"chunk_id": "c1", "title": "alpha\u2029beta"}]


def test_search_summary_records_rankings_shorter_than_requested_top_k() -> None:
    module = _load("search_pyserini_bm25")
    rows = [
        {"hits": [{"chunk_id": "c1"}] * 100},
        {"hits": [{"chunk_id": "c2"}] * 3},
    ]

    summary = module.summarize_hit_counts(rows, requested_top_k=100)

    assert summary == {
        "requested_top_k": 100,
        "min_hits": 3,
        "max_hits": 100,
        "short_ranking_count": 1,
        "hit_count_histogram": {"3": 1, "100": 1},
    }
