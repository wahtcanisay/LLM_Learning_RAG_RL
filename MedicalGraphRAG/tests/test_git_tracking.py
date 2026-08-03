import subprocess
from pathlib import Path


def test_source_data_package_is_not_gitignored() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_init = Path("MedicalGraphRAG/src/medical_graphrag/data/__init__.py")
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(source_init)],
        cwd=repository_root,
        check=False,
    )
    assert result.returncode == 1
