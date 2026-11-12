"""Create the cold_start-shaped input required by the teacher rollout loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data/medical_verifiable_v2_batch3_20260930"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    gate_path = DATA / "manual_gate_manifest.json"
    source_path = DATA / "teacher_input.jsonl"
    output_path = DATA / "teacher_rollout_input.jsonl"
    manifest_path = DATA / "teacher_rollout_input.manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite batch3 rollout input")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "question_and_semantic_pair_review_complete_teacher_input_only":
        raise ValueError("manual question/semantic review gate is incomplete")
    if gate.get("training_ready") is not False:
        raise ValueError("unexpected gate state")

    with source_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != gate.get("teacher_input_rows") or not rows:
        raise ValueError("teacher input count differs from the gate manifest")
    ids = [row.get("question_id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("teacher input contains duplicate question IDs")

    rollout_rows = []
    for row in rows:
        if row.get("split") != "unassigned" or not row.get("question") or not row.get("references"):
            raise ValueError(f"unexpected candidate schema for {row.get('question_id')}")
        rollout_rows.append({**row, "split": "cold_start"})

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rollout_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "status": "teacher_rollout_input_only",
        "rows": len(rollout_rows),
        "unique_question_ids": len(set(ids)),
        "input_sha256": digest(source_path),
        "output_sha256": digest(output_path),
        "manual_gate_manifest_sha256": digest(gate_path),
        "split_label_reason": "evaluate_agent_rollout.py requires cold_start; this does not mark SFT or RL readiness",
        "gold_exposed_in_teacher_prompt": False,
        "evidence_and_gold_adjudication": "pending after rollout",
        "training_ready": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "rows", "unique_question_ids", "output_sha256", "training_ready"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
