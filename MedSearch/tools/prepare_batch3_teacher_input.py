"""Gate batch3 candidates on blinded question and semantic-pair review before rollout."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MedSearch/data/medical_verifiable_v2_batch3_20260930"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pair_key(row: dict) -> tuple[str, str, str, float]:
    # The raw semantic queue records the retrieval origin as `matched_pool`;
    # split review rows additionally enrich it with `matched_source_pool`.
    # Match on the original field so raw and reviewed rows share one identity.
    return (row["candidate"], row["match"], row["matched_pool"], float(row["cosine"]))


def main() -> None:
    teacher_input = DATA / "teacher_input.jsonl"
    gate_manifest = DATA / "manual_gate_manifest.json"
    if teacher_input.exists() or gate_manifest.exists():
        raise FileExistsError("refusing to overwrite batch3 manual-gate outputs")

    candidate_path = DATA / "candidate_batch3.jsonl"
    queue_path = DATA / "semantic_neighbors.jsonl"
    candidate_rows = read_jsonl(candidate_path)
    queue_rows = read_jsonl(queue_path)
    question_reviews = [
        row for batch in range(1, 4)
        for row in read_jsonl(DATA / f"question_review_manual_{batch:02d}.jsonl")
    ]
    pair_reviews = [
        row for batch in range(1, 4)
        for row in read_jsonl(DATA / f"semantic_pair_review_manual_{batch:02d}.jsonl")
    ]

    candidate_ids = [row["question_id"] for row in candidate_rows]
    if len(candidate_ids) != 1000 or len(set(candidate_ids)) != 1000:
        raise ValueError("candidate input must contain 1,000 unique IDs")
    question_by_id = {row["question_id"]: row for row in question_reviews}
    if len(question_by_id) != len(question_reviews) or set(question_by_id) != set(candidate_ids):
        raise ValueError("blinded question review must cover each candidate exactly once")
    for row in question_reviews:
        if row.get("question_decision") not in {"keep", "exclude", "uncertain"} or not row.get("reason_zh"):
            raise ValueError(f"invalid blinded question review: {row.get('question_id')}")

    expected_pairs = Counter(pair_key(row) for row in queue_rows)
    observed_pairs = Counter(pair_key(row) for row in pair_reviews)
    if expected_pairs != observed_pairs:
        missing = expected_pairs - observed_pairs
        extra = observed_pairs - expected_pairs
        raise ValueError(f"semantic pair reviews do not exactly cover the queue; missing={sum(missing.values())}, extra={sum(extra.values())}")
    allowed_actions = {"exclude_for_duplicate", "keep_with_source_reference_flag", "keep", "hold_uncertain"}
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in pair_reviews:
        if row.get("pair_decision") not in {"same_task", "related_only", "uncertain"}:
            raise ValueError(f"invalid pair decision: {row.get('candidate')}")
        if row.get("candidate_action") not in allowed_actions or not row.get("reason_zh"):
            raise ValueError(f"invalid pair review/action: {row.get('candidate')}")
        by_candidate[row["candidate"]].append(row)

    final_rows, question_actions, pair_actions = [], Counter(), Counter()
    held_ids, excluded_ids, source_reference_ids = [], [], []
    for row in candidate_rows:
        item_id = row["question_id"]
        q_review = question_by_id[item_id]
        q_action = q_review["question_decision"]
        question_actions[q_action] += 1
        actions = [entry["candidate_action"] for entry in by_candidate.get(item_id, [])]
        pair_actions.update(actions)
        if q_action == "exclude" or "exclude_for_duplicate" in actions:
            excluded_ids.append(item_id)
            continue
        if q_action == "uncertain" or "hold_uncertain" in actions:
            held_ids.append(item_id)
            continue
        if "keep_with_source_reference_flag" in actions:
            source_reference_ids.append(item_id)
        final_rows.append(row)

    exact_report = json.loads((DATA / "exact_isolation.json").read_text(encoding="utf-8"))
    semantic_manifest = json.loads((DATA / "semantic_neighbors.manifest.json").read_text(encoding="utf-8"))
    if not exact_report.get("exact_check_passed"):
        raise ValueError("exact isolation report is not passing")
    if semantic_manifest.get("status") != "review_queue_not_automatic_exclusion":
        raise ValueError("unexpected semantic screening status")
    with teacher_input.open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "question_and_semantic_pair_review_complete_teacher_input_only",
        "candidate_rows": len(candidate_rows),
        "question_review_rows": len(question_reviews),
        "semantic_pair_rows": len(pair_reviews),
        "teacher_input_rows": len(final_rows),
        "question_decision_counts": dict(question_actions),
        "semantic_pair_action_counts": dict(pair_actions),
        "excluded_unique_candidate_count": len(excluded_ids),
        "held_unique_candidate_count": len(held_ids),
        "source_reference_flag_unique_candidate_count": len(source_reference_ids),
        "excluded_question_ids": sorted(excluded_ids),
        "held_question_ids": sorted(held_ids),
        "source_reference_overlap_question_ids": sorted(source_reference_ids),
        "input_sha256": {
            "candidate_batch3.jsonl": sha256(candidate_path),
            "semantic_neighbors.jsonl": sha256(queue_path),
            "exact_isolation.json": sha256(DATA / "exact_isolation.json"),
            **{f"question_review_manual_{batch:02d}.jsonl": sha256(DATA / f"question_review_manual_{batch:02d}.jsonl") for batch in range(1, 4)},
            **{f"semantic_pair_review_manual_{batch:02d}.jsonl": sha256(DATA / f"semantic_pair_review_manual_{batch:02d}.jsonl") for batch in range(1, 4)},
        },
        "teacher_input_sha256": sha256(teacher_input),
        "gold_exposed_during_question_review": False,
        "source_reference_overlap_is_training_leakage": False,
        "teacher_generation": "pending",
        "evidence_and_gold_adjudication": "pending after rollout",
        "training_ready": False,
    }
    gate_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "candidate_rows", "question_decision_counts", "semantic_pair_action_counts",
        "teacher_input_rows", "excluded_unique_candidate_count", "held_unique_candidate_count",
        "source_reference_flag_unique_candidate_count", "teacher_input_sha256", "training_ready",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
