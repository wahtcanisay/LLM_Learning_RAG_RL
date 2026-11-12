"""Freeze structurally eligible batch6 teacher traces for blind human review."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch6_20260930"
TEACHER = ROOT / "MedSearch/data/distillation/teacher_batch6_20260930"
ANSWERED = ROOT / "MedSearch/data/distillation/teacher_batch6_20260930_answered"
INPUT = BATCH / "teacher_input_review/teacher_rollout_input_cold_start.jsonl"
OUT = BATCH / "trajectory_review"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_query(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", text).casefold().split())


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite review package: {OUT}")
    teacher_manifest_path = TEACHER / "manifest.json"
    answered_manifest_path = ANSWERED / "manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    answered_manifest = json.loads(answered_manifest_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("status") != "complete" or answered_manifest.get("status") != "complete":
        raise ValueError("teacher export or answered extraction is incomplete")

    attempts_path = TEACHER / "teacher_attempts.jsonl"
    answered_path = ANSWERED / "accepted.jsonl"
    if sha256(attempts_path) != teacher_manifest["output_sha256"]["teacher_attempts.jsonl"]:
        raise ValueError("teacher attempts hash mismatch")
    if sha256(answered_path) != answered_manifest["output_sha256"]["accepted.jsonl"]:
        raise ValueError("answered candidate hash mismatch")
    if sha256(INPUT) != teacher_manifest["config"]["data_sha256"]:
        raise ValueError("cold-start rollout input hash mismatch")

    candidates = {row["question_id"]: row for row in read_jsonl(INPUT)}
    attempts = read_jsonl(attempts_path)
    answered = read_jsonl(answered_path)
    if len(attempts) != teacher_manifest["processed"] or len(attempts) != len(candidates):
        raise ValueError("teacher attempt count differs from frozen input")
    if len(answered) != answered_manifest["counts"]["accepted"]:
        raise ValueError("answered count differs from extraction manifest")
    if len({row["question_id"] for row in attempts}) != len(attempts):
        raise ValueError("teacher attempts contain duplicate question IDs")

    exclusions: list[dict] = []
    review_rows: list[dict] = []
    reason_counts: Counter[str] = Counter()
    response_limit = teacher_manifest["config"]["response_length"]
    for row in answered:
        reasons = []
        searches = row.get("searches") or []
        if row.get("search_count", 0) < 1:
            reasons.append("no_search")
        if row.get("invalid_actions", 0) != 0:
            reasons.append("invalid_action")
        if not searches or not any(item.get("document_hashes") for item in searches):
            reasons.append("no_visible_retrieved_document_hash")
        queries = [normalize_query(item.get("query", "")) for item in searches]
        queries = [query for query in queries if query]
        if len(queries) != len(set(queries)):
            reasons.append("exact_repeated_search")
        response_ids, response_mask = row.get("response_ids", []), row.get("response_mask", [])
        if not isinstance(response_ids, list) or not isinstance(response_mask, list) or len(response_ids) != len(response_mask):
            reasons.append("token_mask_mismatch")
        if len(response_ids) > response_limit:
            reasons.append("response_too_long")
        if reasons:
            exclusions.append({"question_id": row["question_id"], "trajectory_id": row["trajectory_id"], "reasons": reasons})
            reason_counts.update(reasons)
            continue

        candidate = candidates.get(row["question_id"])
        if candidate is None:
            raise ValueError(f"teacher output not found in frozen input: {row['question_id']}")
        review_rows.append({
            "trajectory_id": row["trajectory_id"],
            "question_id": row["question_id"],
            "source": candidate.get("source"),
            "source_id": candidate.get("source_id"),
            "source_dataset_split": candidate.get("source_dataset_split"),
            "question": candidate["question"],
            "references": candidate.get("references"),
            "final_answer": row["final_answer"],
            "search_count": row["search_count"],
            "invalid_actions": row["invalid_actions"],
            "searches": searches,
            "segments": row["segments"],
            "response": row["response"],
            "response_ids_sha256": hashlib.sha256(json.dumps(response_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "response_mask_sha256": hashlib.sha256(json.dumps(response_mask, separators=(",", ":")).encode("utf-8")).hexdigest(),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    exclusions_path = OUT / "structural_exclusions.jsonl"
    with exclusions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    batches = [[], [], []]
    for index, row in enumerate(review_rows):
        batches[index % len(batches)].append(row)
    batch_paths = []
    for index, rows in enumerate(batches, 1):
        path = OUT / f"trajectory_review_batch_{index:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        batch_paths.append(path)

    manifest = {
        "status": "human_evidence_review_queue_only",
        "training_ready": False,
        "verifier_used_as_filter": False,
        "teacher_attempt_rows": len(attempts),
        "legal_answer_rows": len(answered),
        "structural_review_rows": len(review_rows),
        "structural_exclusion_rows": len(exclusions),
        "structural_exclusion_reason_counts": dict(reason_counts),
        "review_batches": {path.name: {"rows": len(rows), "sha256": sha256(path)} for path, rows in zip(batch_paths, batches, strict=True)},
        "inputs_sha256": {
            "teacher_manifest.json": sha256(teacher_manifest_path),
            "teacher_attempts.jsonl": sha256(attempts_path),
            "answered_manifest.json": sha256(answered_manifest_path),
            "answered_accepted.jsonl": sha256(answered_path),
            "cold_start_input.jsonl": sha256(INPUT),
        },
        "exclusions_sha256": sha256(exclusions_path),
        "filter_rule": ["legal final answer", "search with document hash", "zero invalid actions", "no exact repeated query", "aligned token/mask", "within response budget"],
        "medical_evidence_gold_reasoning_and_action_review": "pending; structural pass does not imply SFT acceptance",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: manifest[key] for key in (
        "teacher_attempt_rows", "legal_answer_rows", "structural_review_rows",
        "structural_exclusion_rows", "structural_exclusion_reason_counts", "training_ready"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
