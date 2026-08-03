import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "tracked_path",
    [
        Path("MedicalGraphRAG/src/medical_graphrag/data/new_module.py"),
        Path("MedicalGraphRAG/tests/fixtures/medrag_pubmed/new_fixture.jsonl"),
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


def test_unrelated_data_directories_remain_gitignored() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "MedRAG/src/data/generated.py"],
        cwd=repository_root,
        check=False,
    )
    assert result.returncode == 0
