import subprocess
from pathlib import Path

import pytest


def _is_ignored(path: str) -> bool:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=repository_root,
        check=False,
    )
    return result.returncode == 0


def test_bm25_experiment_json_is_trackable() -> None:
    assert not _is_ignored(
        "MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json"
    )


def test_bm25_large_outputs_stay_ignored() -> None:
    assert _is_ignored(
        "MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only/segments_1"
    )
    assert _is_ignored(
        "MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl"
    )


@pytest.mark.parametrize(
    "tracked_path",
    [
        Path("MedicalGraphRAG/src/medical_graphrag/data/new_module.py"),
        Path("MedicalGraphRAG/tests/fixtures/medrag_pubmed/new_fixture.jsonl"),
        Path("MedicalGraphRAG/data/processed/pubmedqa_hard_v1/manifest.json"),
    ],
)
def test_project_sources_and_fixtures_are_not_gitignored(tracked_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", str(tracked_path)],
        cwd=repository_root,
        check=False,
    )
    assert result.returncode == 1


@pytest.mark.parametrize(
    "ignored_path",
    [
        "MedRAG/src/data/generated.py",
        "MedicalGraphRAG/data/processed/pubmedqa_hard_v1/chunks.jsonl",
    ],
)
def test_large_and_unrelated_data_remain_gitignored(ignored_path: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", ignored_path],
        cwd=repository_root,
        check=False,
    )
    assert result.returncode == 0
