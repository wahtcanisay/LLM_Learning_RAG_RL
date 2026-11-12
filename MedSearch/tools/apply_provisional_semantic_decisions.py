"""Materialize the manually reviewed semantic exclusions for the provisional SFT pool."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "MedSearch/data/distillation/sft_provisional_pool_audit_20260930"
INPUT = AUDIT / "confirmed_inventory_exact_unique.jsonl"
QUEUE = AUDIT / "semantic_near_duplicate_review_queue.jsonl"
REVIEW = AUDIT / "semantic_near_duplicate_manual_review_3pairs.jsonl"
REVIEW_MANIFEST = AUDIT / "semantic_near_duplicate_manual_review_3pairs_manifest.json"
OUTPUT_DIR = AUDIT / "semantic_dedup_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source_rows = read_jsonl(INPUT)
    queue_rows = read_jsonl(QUEUE)
    review_rows = read_jsonl(REVIEW)
    review_manifest = json.loads(REVIEW_MANIFEST.read_text(encoding="utf-8"))
    source_by_id = {row["pool_item_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows) or len(source_rows) != 393:
        raise ValueError("provisional exact-unique inventory must contain 393 unique pool_item_id rows")
    if len(queue_rows) != 3 or len(review_rows) != 3 or review_manifest.get("automatic_exclusion") is not False:
        raise ValueError("expected the complete, human-reviewed three-pair queue")
    if review_manifest.get("input_sha256") != file_hash(QUEUE) or review_manifest.get("output_sha256") != file_hash(REVIEW):
        raise ValueError("semantic review manifest does not match its queue or decisions")

    decisions = {}
    exclusions = []
    for index, (pair, reviewed) in enumerate(zip(queue_rows, review_rows), start=1):
        pair_id = f"queue_line_{index:04d}"
        candidate_id = pair.get("candidate_id")
        candidate = source_by_id.get(candidate_id)
        if pair.get("kind") != "history" or reviewed.get("pair_id") != pair_id or reviewed.get("queue_line") != index:
            raise ValueError(f"unexpected semantic pair order or scope at {pair_id}")
        if candidate is None or candidate.get("question") != pair.get("candidate_question"):
            raise ValueError(f"candidate question does not map to the provisional inventory: {candidate_id}")
        if reviewed.get("candidate_question_sha256") != candidate.get("raw_question_sha256"):
            raise ValueError(f"candidate question hash mismatch: {candidate_id}")
        if reviewed.get("relationship_label") not in {"same_task", "related_distinct"}:
            raise ValueError(f"unresolved semantic label at {pair_id}")
        decisions[candidate_id] = reviewed
        if reviewed["relationship_label"] == "same_task":
            exclusions.append({
                "pool_item_id": candidate_id,
                "question_id": candidate["question_id"],
                "source_set": candidate["source_set"],
                "relationship_label": reviewed["relationship_label"],
                "reason_zh": reviewed["reason_zh"],
                "matched_question": pair["matched_question"],
                "matched_sources": pair.get("matched_sources", []),
                "pair_id": pair_id,
            })

    excluded_ids = {row["pool_item_id"] for row in exclusions}
    if len(decisions) != 3 or len(excluded_ids) != 2:
        raise ValueError("expected two historical same-task exclusions and one related-distinct retention")
    kept_rows = [row for row in source_rows if row["pool_item_id"] not in excluded_ids]
    if len(kept_rows) != 391 or {row["pool_item_id"] for row in kept_rows} | excluded_ids != set(source_by_id):
        raise ValueError("derived inventory coverage/count mismatch")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "confirmed_inventory_semantic_dedup.jsonl"
    exclusions_path = OUTPUT_DIR / "semantic_exclusions.jsonl"
    report_path = OUTPUT_DIR / "report.md"
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows), encoding="utf-8")
    exclusions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exclusions), encoding="utf-8")

    reread = read_jsonl(output_path)
    if len(reread) != 391 or any(row.get("training_ready") is not False for row in reread):
        raise ValueError("output count or provisional training_ready invariant failed")
    source_counts = dict(Counter(row["source_set"] for row in reread))
    report = (
        "# Provisional SFT semantic-dedup inventory\n\n"
        "This is a derived candidate inventory only. It is not a training export and does not pass historical exposure or split gates.\n\n"
        f"- Input rows: {len(source_rows)}; retained: {len(reread)}; excluded as historical same-task: {len(exclusions)}.\n"
        "- The remaining `related_distinct` pair was retained.\n"
        "- Every retained row is copied from the exact-unique source inventory; original source, trajectory, references, and review provenance fields remain intact.\n"
        "- `training_ready=false`; complete Stage A/B exposure inventory and final train/dev/test split are still required.\n\n"
        "## Retained rows by source\n\n"
        + "\n".join(f"- {source}: {count}" for source, count in sorted(source_counts.items()))
        + "\n"
    )
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "status": "provisional_semantic_dedup_only",
        "training_ready": False,
        "final_split_created": False,
        "input_rows": len(source_rows),
        "retained_rows": len(reread),
        "excluded_historical_same_task_rows": len(exclusions),
        "retained_related_distinct_pairs": 1,
        "source_counts": source_counts,
        "inputs": {
            "inventory_path": INPUT.relative_to(ROOT).as_posix(),
            "inventory_sha256": file_hash(INPUT),
            "semantic_queue_path": QUEUE.relative_to(ROOT).as_posix(),
            "semantic_queue_sha256": file_hash(QUEUE),
            "human_review_path": REVIEW.relative_to(ROOT).as_posix(),
            "human_review_sha256": file_hash(REVIEW),
            "human_review_manifest_sha256": file_hash(REVIEW_MANIFEST),
        },
        "outputs": {
            output_path.name: {"path": output_path.relative_to(ROOT).as_posix(), "rows": len(reread), "sha256": file_hash(output_path)},
            exclusions_path.name: {"path": exclusions_path.relative_to(ROOT).as_posix(), "rows": len(exclusions), "sha256": file_hash(exclusions_path)},
            report_path.name: {"path": report_path.relative_to(ROOT).as_posix(), "sha256": file_hash(report_path)},
        },
        "excluded_items": exclusions,
        "limitations": [
            "Stage A/B historical assets are unavailable in the current workspace.",
            "Other large raw SFT/MedQA/MedMCQA corpora have not been covered by this semantic screen.",
            "No final training split or model-training export is created here.",
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    write_json(manifest_path, manifest)
    check = json.loads(manifest_path.read_text(encoding="utf-8"))
    if check["outputs"][output_path.name]["sha256"] != file_hash(output_path):
        raise ValueError("manifest hash verification failed")
    print(json.dumps({"status": check["status"], "retained": check["retained_rows"], "excluded": check["excluded_historical_same_task_rows"], "training_ready": check["training_ready"], "manifest": manifest_path.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
