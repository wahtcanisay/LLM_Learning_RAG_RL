"""Add the rollout tool's cold_start label without changing frozen candidates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch6_20260930"
PACKAGE = BATCH / "teacher_input_review"
SOURCE = PACKAGE / "teacher_rollout_input.jsonl"
DISPOSITIONS = PACKAGE / "batch6_semantic_dispositions.jsonl"
OUTPUT = PACKAGE / "teacher_rollout_input_cold_start.jsonl"
MANIFEST = PACKAGE / "teacher_rollout_input_cold_start.manifest.json"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("refusing to overwrite batch6 cold-start rollout input")
    source_rows = read_jsonl(SOURCE)
    dispositions = read_jsonl(DISPOSITIONS)
    eligible_ids = {row["question_id"] for row in dispositions if row["semantic_disposition"] == "eligible_after_semantic_review"}
    if len(source_rows) != 305 or len({row["question_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("expected 305 unique semantically eligible candidates")
    if {row["question_id"] for row in source_rows} != eligible_ids:
        raise ValueError("rollout source IDs do not match the manual semantic disposition")

    original_split_counts = Counter(row.get("split") for row in source_rows)
    output_rows = []
    for row in source_rows:
        if not row.get("question") or not row.get("references"):
            raise ValueError(f"missing question or references: {row['question_id']}")
        if "prompt" in row:
            raise ValueError(f"unexpected prebuilt prompt field: {row['question_id']}")
        output_rows.append({**row, "source_dataset_split": row.get("split"), "split": "cold_start"})
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "teacher_rollout_input_only",
        "rows": len(output_rows),
        "unique_question_ids": len({row["question_id"] for row in output_rows}),
        "source_sha256": sha256(SOURCE),
        "semantic_disposition_sha256": sha256(DISPOSITIONS),
        "output_sha256": sha256(OUTPUT),
        "original_source_split_counts": dict(original_split_counts),
        "rollout_split_label": "cold_start; required by evaluate_agent_rollout.py and not a train/dev/test assignment",
        "gold_exposed_in_teacher_prompt": False,
        "evidence_and_trajectory_review": "pending after rollout",
        "rollout_started": False,
        "training_ready": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"rows": manifest["rows"], "output_sha256": manifest["output_sha256"], "original_source_split_counts": manifest["original_source_split_counts"], "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
