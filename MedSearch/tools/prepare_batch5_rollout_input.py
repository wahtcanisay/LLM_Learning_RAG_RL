"""Create an isolated teacher-rollout input from the batch5 eligibility ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BATCH = Path(__file__).resolve().parents[1] / "data/medical_verifiable_v2_batch5_20260930"
SOURCE = BATCH / "candidate_batch5.jsonl"
ELIGIBILITY = BATCH / "candidate_semantic_eligibility_batch5.jsonl"
ELIGIBILITY_MANIFEST = BATCH / "candidate_semantic_eligibility_batch5.manifest.json"
OUTPUT = BATCH / "teacher_rollout_input.jsonl"
MANIFEST = BATCH / "teacher_rollout_input.manifest.json"
EXPECTED_ELIGIBLE = 491


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("refusing to overwrite existing batch5 rollout input")

    eligibility_manifest = json.loads(ELIGIBILITY_MANIFEST.read_text(encoding="utf-8"))
    if eligibility_manifest.get("sft_ready") is not False:
        raise ValueError("eligibility manifest must not claim SFT readiness")
    if sha256(ELIGIBILITY) != eligibility_manifest.get("candidate_jsonl_sha256"):
        raise ValueError("eligibility ledger hash does not match its manifest")
    expected_source_hash = eligibility_manifest["source_files"]["candidate_batch5.jsonl"]["sha256"]
    if sha256(SOURCE) != expected_source_hash:
        raise ValueError("raw candidate source hash does not match eligibility manifest")

    source_rows = read_jsonl(SOURCE)
    eligibility_rows = read_jsonl(ELIGIBILITY)
    if len(source_rows) != 1000 or len(eligibility_rows) != 1000:
        raise ValueError("unexpected batch5 row count")

    source_ids = [str(row.get("question_id")) for row in source_rows]
    eligibility_ids = [str(row.get("question_id")) for row in eligibility_rows]
    if len(set(source_ids)) != len(source_ids) or len(set(eligibility_ids)) != len(eligibility_ids):
        raise ValueError("duplicate candidate ID")
    if source_ids != eligibility_ids:
        raise ValueError("source and eligibility ledger IDs/order differ")

    status_keys = {
        key
        for row in eligibility_rows
        for key, value in row.items()
        if value == "eligible_for_teacher_trajectory_review"
    }
    if len(status_keys) != 1:
        raise ValueError(f"could not identify one eligibility field: {sorted(status_keys)}")
    status_key = status_keys.pop()
    eligible_ids = {
        str(row["question_id"])
        for row in eligibility_rows
        if row.get(status_key) == "eligible_for_teacher_trajectory_review"
    }
    if len(eligible_ids) != EXPECTED_ELIGIBLE:
        raise ValueError(f"expected {EXPECTED_ELIGIBLE} eligible rows, found {len(eligible_ids)}")

    output_rows = []
    for row in source_rows:
        question_id = str(row["question_id"])
        if question_id not in eligible_ids:
            continue
        if not row.get("question") or not row.get("references"):
            raise ValueError(f"missing question or verifier references: {question_id}")
        if "prompt" in row:
            raise ValueError(f"refusing candidate with prebuilt prompt field: {question_id}")
        output_rows.append({**row, "split": "cold_start"})

    if [str(row["question_id"]) for row in output_rows] != [
        str(row["question_id"]) for row in source_rows if str(row["question_id"]) in eligible_ids
    ]:
        raise ValueError("rollout input order differs from source order")

    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "artifact": "batch5 teacher rollout input only",
        "status": "teacher_rollout_input_only",
        "rows": len(output_rows),
        "unique_question_ids": len({str(row["question_id"]) for row in output_rows}),
        "eligibility_status_key": status_key,
        "eligible_candidate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(eligible_ids)).encode("utf-8")
        ).hexdigest(),
        "source_sha256": sha256(SOURCE),
        "eligibility_ledger_sha256": sha256(ELIGIBILITY),
        "output_sha256": sha256(OUTPUT),
        "split_label_reason": "evaluate_agent_rollout.py requires cold_start; this does not establish SFT or RL readiness",
        "gold_exposed_in_teacher_prompt": False,
        "question_and_semantic_gates": "passed candidate-level review only",
        "evidence_and_trajectory_adjudication": "pending after rollout",
        "training_ready": False,
        "rollout_started": False,
    }
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": manifest["rows"],
                "unique_question_ids": manifest["unique_question_ids"],
                "output_sha256": manifest["output_sha256"],
                "training_ready": manifest["training_ready"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
