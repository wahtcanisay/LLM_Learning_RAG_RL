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
    assert "fetch-pubmedqa" in result.stdout
    assert "audit" in result.stdout
    assert "build" in result.stdout
