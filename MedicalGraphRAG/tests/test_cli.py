import subprocess
import sys


def test_cli_lists_pipeline_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "medical_graphrag.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for command in (
        "fetch-pubmedqa",
        "audit",
        "build",
        "export-pyserini",
        "evaluate-bm25",
        "run",
    ):
        assert command in result.stdout


def test_cli_run_rejects_unknown_retriever() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "medical_graphrag.cli",
            "run",
            "not-a-retriever",
            "--dataset",
            "nfcorpus_v1",
            "--git-commit",
            "abc",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


def test_cli_run_rejects_missing_git_commit() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "medical_graphrag.cli",
            "run",
            "dense",
            "--dataset",
            "nfcorpus_v1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
