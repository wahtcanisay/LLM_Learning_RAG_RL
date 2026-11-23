"""Build a separate batch2 review ledger and a non-training combined view."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from MedSearch.tools.review_distillation_v2_candidates import build_ledger


ROOT = Path(__file__).resolve().parents[2]
FIRST = ROOT / "MedSearch/data/medical_verifiable_v2_20260919"
SECOND = ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926"


def merge_ledgers(first: list[dict], second: list[dict]) -> list[dict]:
    ids = [row["question_id"] for row in first + second]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ID across V2 batches")
    result = []
    for batch_number, rows in ((1, first), (2, second)):
        for row in rows:
            if row["decision"] not in {"reject", "quarantine", "pending"} or row["split"] != "unassigned":
                raise ValueError(f"unreviewed decision/split state: {row['question_id']}")
            result.append({
                **row,
                "batch": batch_number,
                "screening_eligible": row["decision"] == "pending",
                "training_accepted": False,
            })
    return result


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    first_manifest = json.loads((FIRST / "review_ledger.manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((SECOND / "manifest.json").read_text(encoding="utf-8"))
    first_path = FIRST / "review_ledger.jsonl"
    candidate_path = SECOND / "candidate_batch2.jsonl"
    semantic_path = SECOND / "semantic_neighbors.jsonl"
    lexical_path = SECOND / "lexical_neighbors.jsonl"
    manual_path = SECOND / "manual_review_notes.jsonl"
    exact_path = SECOND / "exact_isolation.json"
    if _sha256(first_path) != first_manifest["ledger_sha256"] or _sha256(candidate_path) != second_manifest["candidate_sha256"]:
        raise ValueError("frozen V2 input changed")
    semantic_manifest = json.loads((SECOND / "semantic_neighbors.manifest.json").read_text(encoding="utf-8"))
    lexical_manifest = json.loads((SECOND / "lexical_neighbors.manifest.json").read_text(encoding="utf-8"))
    if _sha256(semantic_path) != semantic_manifest["output_sha256"] or _sha256(lexical_path) != lexical_manifest["output_sha256"]:
        raise ValueError("near-duplicate queue changed")
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    if not exact["exact_check_passed"] or exact["conflicts"] or exact["missing"] or exact["unreviewed"]:
        raise ValueError("exact split isolation did not pass")
    first = _read_jsonl(first_path)
    second = build_ledger(
        _read_jsonl(candidate_path), _read_jsonl(semantic_path), _read_jsonl(manual_path),
        lexical_neighbors=_read_jsonl(lexical_path),
    )
    if len(first) != 3000 or len(second) != 1000:
        raise ValueError("V2 batch sizes differ from frozen candidates")
    combined = merge_ledgers(first, second)
    second_path = SECOND / "review_ledger.jsonl"
    combined_path = SECOND / "combined_review_ledger.jsonl"
    _write_jsonl(second_path, second)
    _write_jsonl(combined_path, combined)
    report = {
        "status": "batch2_review_partial_combined_view_not_training_ready",
        "first_ledger_sha256": _sha256(first_path),
        "batch2_candidate_sha256": _sha256(candidate_path),
        "semantic_queue_sha256": _sha256(semantic_path),
        "lexical_queue_sha256": _sha256(lexical_path),
        "manual_notes_sha256": _sha256(manual_path),
        "exact_isolation_sha256": _sha256(exact_path),
        "batch2_ledger_sha256": _sha256(second_path),
        "combined_ledger_sha256": _sha256(combined_path),
        "batch1_rows": len(first),
        "batch2_rows": len(second),
        "combined_rows": len(combined),
        "batch1_decisions": dict(Counter(row["decision"] for row in first)),
        "batch2_decisions": dict(Counter(row["decision"] for row in second)),
        "combined_decisions": dict(Counter(row["decision"] for row in combined)),
        "screening_eligible_rows": sum(row["screening_eligible"] for row in combined),
        "training_accepted_rows": 0,
        "split_assignment": "not_done",
        "no_auto_replenishment": True,
        "training_ready": False,
    }
    (SECOND / "review_ledger.manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
