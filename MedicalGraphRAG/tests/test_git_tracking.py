import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "tracked_path",
    [
        Path("MedicalGraphRAG/src/medical_graphrag/data/__init__.py"),
        Path("MedicalGraphRAG/tests/fixtures/medrag_pubmed/pubmed23n0001.jsonl"),
    ],
)
def test_project_sources_and_fixtures_are_not_gitignored(tracked_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(tracked_path)],
        cwd=repository_root,
        check=False,
    )
    assert result.returncode == 1
