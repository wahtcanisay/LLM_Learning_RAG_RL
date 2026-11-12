"""Build a blind second-review input for batch4 replay accepts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "MedSearch/data/medical_verifiable_v2_batch4_20261001/replay_trajectory_review"
QUEUE = REVIEW / "trajectory_review_batch_01.jsonl"
LEDGER = REVIEW / "trajectory_review_manual_01.jsonl"
OUT = REVIEW / "independent_accept_audit"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit package: {OUT}")
    queue_rows, ledger_rows = read_jsonl(QUEUE), read_jsonl(LEDGER)
    queue = {row["question_id"]: row for row in queue_rows}
    if len(queue) != len(queue_rows) or len({row["question_id"] for row in ledger_rows}) != len(ledger_rows):
        raise ValueError("duplicate question IDs in queue or first-pass ledger")
    if set(queue) != {row["question_id"] for row in ledger_rows}:
        raise ValueError("first-pass ledger does not cover the original review queue")
    selected = [queue[row["question_id"]] for row in ledger_rows if row.get("independent_decision") == "accept"]
    if len(selected) != 7:
        raise ValueError(f"expected 7 first-pass accepts, found {len(selected)}")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "batch4_replay_accepts_blind_second_review.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(selected, key=lambda item: item["question_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "independent_second_review_input_only",
        "training_ready": False,
        "review_rows": len(selected),
        "first_pass_decisions_removed": True,
        "verifier_fields_removed": True,
        "queue_sha256": sha256(QUEUE),
        "first_pass_ledger_sha256": sha256(LEDGER),
        "output_sha256": sha256(output),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"review_rows": len(selected), "output_sha256": manifest["output_sha256"], "training_ready": False}))


if __name__ == "__main__":
    main()
