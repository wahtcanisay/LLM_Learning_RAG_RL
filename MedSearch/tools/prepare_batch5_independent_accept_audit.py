"""Build a verifier-blind, decision-blind second-review input for batch5 accepts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch5_20260930"
REVIEW = BATCH / "trajectory_review"
OUT = REVIEW / "independent_accept_audit_final"
SLICES = [
    (REVIEW / "trajectory_review_batch_01.jsonl", REVIEW / "trajectory_review_manual_01.jsonl"),
    (REVIEW / "trajectory_review_batch_02.jsonl", REVIEW / "trajectory_review_manual_02.jsonl"),
    (REVIEW / "trajectory_review_batch_03.jsonl", REVIEW / "trajectory_review_manual_03.jsonl"),
]
FIELDS = (
    "trajectory_id", "question_id", "source", "source_id", "source_reference_overlap_flag",
    "question", "references", "final_answer", "search_count", "invalid_actions", "searches",
    "segments", "response", "response_ids_sha256", "response_mask_sha256",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit package: {OUT}")
    selected: dict[str, dict] = {}
    sources = []
    for input_path, ledger_path in SLICES:
        source_rows = {row["question_id"]: row for row in read_jsonl(input_path)}
        ledger_rows = read_jsonl(ledger_path)
        if len(source_rows) != len(read_jsonl(input_path)):
            raise ValueError(f"duplicate question IDs in {input_path.name}")
        if {row["question_id"] for row in ledger_rows} != set(source_rows):
            raise ValueError(f"review ledger does not cover the input slice: {ledger_path.name}")
        accepted_ids = [row["question_id"] for row in ledger_rows if row.get("sft_decision") == "accept"]
        for question_id in accepted_ids:
            row = source_rows[question_id]
            selected[question_id] = {key: row[key] for key in FIELDS if key in row}
        sources.append({
            "input": input_path.name,
            "input_sha256": sha256(input_path),
            "ledger": ledger_path.name,
            "ledger_sha256": sha256(ledger_path),
            "accepted_ids": len(accepted_ids),
        })
    if len(selected) != 106:
        raise ValueError(f"expected 106 first-pass accepts from batches 01-03; found {len(selected)}")
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "batch5_accepts_blind_second_review.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for question_id in sorted(selected):
            handle.write(json.dumps(selected[question_id], ensure_ascii=False) + "\n")
    manifest = {
        "status": "independent_second_review_input_only",
        "training_ready": False,
        "review_rows": len(selected),
        "first_pass_decisions_removed": True,
        "verifier_fields_removed": True,
        "sources": sources,
        "output_sha256": sha256(output),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"review_rows": len(selected), "output_sha256": manifest["output_sha256"], "training_ready": False}))


if __name__ == "__main__":
    main()
