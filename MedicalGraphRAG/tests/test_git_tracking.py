import subprocess
from pathlib import Path

import pytest


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
