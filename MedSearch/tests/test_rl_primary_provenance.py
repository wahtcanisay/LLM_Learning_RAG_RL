import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_primary_review_is_not_recorded_as_sol():
    root = Path(__file__).resolve().parents[2]
    clean = root / "MedSearch/data/cleaning_v2"
    with tempfile.TemporaryDirectory(prefix="rl_primary_test_", dir=clean) as tmp:
        run = Path(tmp)
        (run / "staging").mkdir()
        item = {"question_id": "test:1", "C2": "pass", "C3": "E2", "assignment": "rl",
                "c3_review": {"uncertain": False, "conflict": False}}
        (run / "c1_semantic_ledger.jsonl").write_text(json.dumps(item) + "\n", encoding="utf8")
        decisions = run / "primary.jsonl"
        decisions.write_text(json.dumps({"question_id": "test:1", "decision": "retain",
                                        "semantic_status": "correct", "rationale": "valid"}) + "\n", encoding="utf8")
        subprocess.run([sys.executable, "-m", "MedSearch.data.cleaning_v2.rl_to500_20261022.apply_sol_packet",
                        "--run", run.name, "--decisions", decisions.relative_to(root).as_posix(),
                        "--primary"], cwd=root, check=True)
        saved = json.loads((run / "c1_semantic_ledger.jsonl").read_text(encoding="utf8"))
        assert saved["primary_review_ref"]["reviewer"] == "GPT-6.1 Terra medium"
        assert "sol_review_ref" not in saved
        subprocess.run([sys.executable, "-m", "MedSearch.data.cleaning_v2.rl_to500_20261022.apply_sol_packet",
                        "--run", run.name, "--decisions", decisions.relative_to(root).as_posix(),
                        "--reviewer", "GPT-6.1-Sol-medium-subagent"], cwd=root, check=True)
        saved = json.loads((run / "c1_semantic_ledger.jsonl").read_text(encoding="utf8"))
        assert saved["primary_review_ref"]["reviewer"] == "GPT-6.1 Terra medium"
        assert saved["sol_review_ref"]["reviewer"] == "GPT-6.1-Sol-medium-subagent"
