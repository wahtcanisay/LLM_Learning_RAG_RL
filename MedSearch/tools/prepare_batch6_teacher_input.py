"""Apply batch6 manual semantic dispositions to the keep-only teacher queue."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch6_20260930"
OUT = BATCH / "teacher_input_review"
QUESTION_REVIEWS = [BATCH / f"question_review_manual_{index:02d}.jsonl" for index in range(1, 4)]
PAIR_GROUPS = [
    (BATCH / "semantic_pair_review_keep_batch_01.jsonl", BATCH / "semantic_pair_review_keep_manual_01.jsonl"),
    (BATCH / "semantic_pair_review_keep_batch_02.jsonl", BATCH / "semantic_pair_review_manual_02.jsonl"),
    (BATCH / "semantic_pair_review_keep_batch_03.jsonl", BATCH / "semantic_pair_review_keep_manual_03.jsonl"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def question_id(row: dict) -> str:
    return row.get("question_id") or row.get("candidate_question_id") or row.get("candidate")


def classification(row: dict) -> str:
    return row.get("classification", "")


def assignment_pair_key(row: dict) -> tuple:
    return (row.get("candidate"), row.get("kind"), row.get("matched_row"), row.get("match") or row.get("matched_question"))


def review_matches_assignment(review: dict, assignment: dict) -> bool:
    candidate = review.get("candidate") or review.get("candidate_question_id")
    if assignment.get("candidate") != candidate or assignment.get("matched_row") != review.get("matched_row"):
        return False
    if review.get("match") is not None:
        return assignment.get("match") == review.get("match")
    if review.get("historical_question") is not None:
        return assignment.get("matched_question") == review.get("historical_question")
    return assignment.get("kind") == (review.get("kind") or ("history" if review.get("matched_row") is not None else "internal"))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite teacher-input package: {OUT}")
    candidate_path = BATCH / "candidate_batch6.jsonl"
    candidates = read_jsonl(candidate_path)
    candidate_by_id = {row["question_id"]: row for row in candidates}
    if len(candidate_by_id) != len(candidates) or len(candidates) != 1000:
        raise ValueError("candidate batch must contain 1,000 unique question IDs")

    question_rows = [row for path in QUESTION_REVIEWS for row in read_jsonl(path)]
    keep_ids = {row["question_id"] for row in question_rows if row.get("question_decision") == "keep"}
    if len(question_rows) != 1000 or {row["question_id"] for row in question_rows} != set(candidate_by_id):
        raise ValueError("question-review ledgers do not cover the frozen 1,000 candidates")
    if len(keep_ids) != 506:
        raise ValueError(f"expected 506 gold-blind keep questions; found {len(keep_ids)}")

    pair_rows: list[dict] = []
    pair_sources = []
    for assignment_path, ledger_path in PAIR_GROUPS:
        assignments = read_jsonl(assignment_path)
        ledger = read_jsonl(ledger_path)
        if len(assignments) != len(ledger):
            raise ValueError(f"assignment/ledger row count mismatch: {ledger_path.name}")
        assignment_by_pair = {assignment_pair_key(row): row for row in assignments}
        if len(assignment_by_pair) != len(assignments):
            raise ValueError(f"duplicate pair keys in {assignment_path.name}")
        seen_review_pairs = set()
        for review in ledger:
            candidate = review.get("candidate") or review.get("candidate_question_id")
            possible_sources = [row for row in assignments if row.get("candidate") == candidate]
            source = next((row for row in possible_sources if review_matches_assignment(review, row)), None)
            if source is None:
                raise ValueError(f"review candidate not present in assignment: {candidate}")
            pair_key = assignment_pair_key(source)
            if pair_key in seen_review_pairs:
                raise ValueError(f"duplicate reviewed pair: {candidate}")
            seen_review_pairs.add(pair_key)
            if review.get("review_scope") == "question_only" and review.get("gold_or_answer_inspected") is not False:
                raise ValueError("semantic review was not gold-blind")
            if candidate not in keep_ids:
                raise ValueError(f"semantic review includes non-keep candidate: {candidate}")
            pair_rows.append({"candidate": candidate, "match": source.get("match"), "kind": source.get("kind"), "matched_row": source.get("matched_row"), "classification": classification(review), "pair_id": review.get("pair_id"), "review_file": ledger_path.name})
        if len(seen_review_pairs) != len(assignment_by_pair):
            raise ValueError(f"review ledger does not cover every pair in {assignment_path.name}")
        pair_sources.append({"assignment": assignment_path.name, "assignment_sha256": sha256(assignment_path), "ledger": ledger_path.name, "ledger_sha256": sha256(ledger_path), "rows": len(ledger)})

    if len(pair_rows) != 202:
        raise ValueError(f"expected 202 manual semantic decisions; found {len(pair_rows)}")

    dispositions: dict[str, dict] = {qid: {"question_id": qid, "semantic_disposition": "eligible_after_semantic_review", "matched_ids": [], "review_classes": []} for qid in keep_ids}
    history_duplicate_ids: set[str] = set()
    unclear_ids: set[str] = set()
    internal_adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pair_rows:
        candidate, match, kind, label = pair["candidate"], pair["match"], pair["kind"], pair["classification"]
        record = dispositions[candidate]
        record["review_classes"].append(label)
        if match:
            record["matched_ids"].append(match)
        if label in {"same_task", "exact/near_duplicate"}:
            if kind == "history":
                history_duplicate_ids.add(candidate)
            elif kind == "internal":
                if match not in keep_ids:
                    raise ValueError(f"internal match not in the keep pool: {match}")
                internal_adjacency[candidate].add(match)
                internal_adjacency[match].add(candidate)
            else:
                raise ValueError(f"unknown pair kind: {kind}")
        elif label == "unclear":
            unclear_ids.add(candidate)
        elif label == "related_distinct":
            pass
        else:
            raise ValueError(f"unknown semantic classification: {label}")

    # Resolve connected internal duplicate clusters before admitting any candidate.
    visited: set[str] = set()
    for start in internal_adjacency:
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(internal_adjacency[node] - component)
        visited |= component
        if component & history_duplicate_ids:
            for qid in component:
                history_duplicate_ids.add(qid)
        elif component & unclear_ids:
            unclear_ids |= component
        else:
            keep_id = min(component, key=lambda qid: candidate_by_id[qid]["source_id"])
            for qid in component - {keep_id}:
                dispositions[qid]["semantic_disposition"] = "exclude_internal_duplicate"
                dispositions[qid]["matched_ids"].append(keep_id)
            dispositions[keep_id]["review_classes"].append("internal_duplicate_keep_lower_source_id")

    for qid in history_duplicate_ids:
        dispositions[qid]["semantic_disposition"] = "exclude_history_duplicate"
    for qid in unclear_ids - history_duplicate_ids:
        if dispositions[qid]["semantic_disposition"] == "eligible_after_semantic_review":
            dispositions[qid]["semantic_disposition"] = "hold_semantic_unclear"
    for record in dispositions.values():
        record["matched_ids"] = sorted(set(record["matched_ids"]))
        record["review_classes"] = sorted(set(record["review_classes"]))

    rollout_ids = {qid for qid, row in dispositions.items() if row["semantic_disposition"] == "eligible_after_semantic_review"}
    disposition_counts = Counter(row["semantic_disposition"] for row in dispositions.values())
    if disposition_counts != Counter({"eligible_after_semantic_review": 305, "exclude_history_duplicate": 196, "exclude_internal_duplicate": 1, "hold_semantic_unclear": 4}):
        raise ValueError(f"unexpected candidate dispositions: {dict(disposition_counts)}")
    OUT.mkdir(parents=True, exist_ok=True)
    disposition_path = OUT / "batch6_semantic_dispositions.jsonl"
    rollout_path = OUT / "teacher_rollout_input.jsonl"
    with disposition_path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(dispositions):
            handle.write(json.dumps(dispositions[qid], ensure_ascii=False) + "\n")
    with rollout_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidates:
            if row["question_id"] in rollout_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "teacher_rollout_input_ready_only",
        "training_ready": False,
        "teacher_rollout_started": False,
        "candidate_rows": len(candidates),
        "question_keep_rows": len(keep_ids),
        "semantic_pair_rows": len(pair_rows),
        "semantic_pair_class_counts": dict(Counter(row["classification"] for row in pair_rows)),
        "semantic_pair_kind_class_counts": {f"{kind}:{label}": count for (kind, label), count in Counter((row["kind"], row["classification"]) for row in pair_rows).items()},
        "candidate_disposition_counts": dict(disposition_counts),
        "teacher_rollout_rows": len(rollout_ids),
        "candidate_sha256": sha256(candidate_path),
        "question_review_sha256": {path.name: sha256(path) for path in QUESTION_REVIEWS},
        "semantic_pair_sources": pair_sources,
        "outputs_sha256": {disposition_path.name: sha256(disposition_path), rollout_path.name: sha256(rollout_path)},
        "rule": "gold-blind question keep; manually reviewed history duplicates excluded; internal same-task pair keeps lower source_id; unclear pairs held; related-distinct remains eligible for teacher trajectory review",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"teacher_rollout_rows": len(rollout_ids), "candidate_disposition_counts": dict(disposition_counts), "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
