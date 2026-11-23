"""Verify frozen V2 batch2 provenance, review evidence, and combined safety gates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIRST = ROOT / "MedSearch/data/medical_verifiable_v2_20260919"
SECOND = ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    raw = json.loads((ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json").read_text(encoding="utf-8"))
    first = read_jsonl(FIRST / "review_ledger.jsonl")
    candidates = read_jsonl(SECOND / "candidate_batch2.jsonl")
    second = read_jsonl(SECOND / "review_ledger.jsonl")
    combined = read_jsonl(SECOND / "combined_review_ledger.jsonl")
    semantic = read_jsonl(SECOND / "semantic_neighbors.jsonl")
    pairs = read_jsonl(SECOND / "semantic_pair_review.jsonl")
    lexical = read_jsonl(SECOND / "lexical_neighbors.jsonl")
    lexical_pairs = read_jsonl(SECOND / "lexical_pair_review.jsonl")
    manual = read_jsonl(SECOND / "manual_review_notes.jsonl")
    candidate_manifest = json.loads((SECOND / "manifest.json").read_text(encoding="utf-8"))
    pair_manifest = json.loads((SECOND / "pair_review.manifest.json").read_text(encoding="utf-8"))
    ledger_manifest = json.loads((SECOND / "review_ledger.manifest.json").read_text(encoding="utf-8"))
    exact = json.loads((SECOND / "exact_isolation.json").read_text(encoding="utf-8"))
    expected_hashes = {
        SECOND / "candidate_batch2.jsonl": candidate_manifest["candidate_sha256"],
        SECOND / "semantic_neighbors.jsonl": pair_manifest["semantic_queue_sha256"],
        SECOND / "lexical_neighbors.jsonl": pair_manifest["lexical_queue_sha256"],
        SECOND / "semantic_pair_review.jsonl": pair_manifest["semantic_pair_review_sha256"],
        SECOND / "lexical_pair_review.jsonl": pair_manifest["lexical_pair_review_sha256"],
        SECOND / "manual_review_notes.jsonl": ledger_manifest["manual_notes_sha256"],
        SECOND / "review_ledger.jsonl": ledger_manifest["batch2_ledger_sha256"],
        SECOND / "combined_review_ledger.jsonl": ledger_manifest["combined_ledger_sha256"],
        FIRST / "review_ledger.jsonl": ledger_manifest["first_ledger_sha256"],
    }
    for path, expected in expected_hashes.items():
        assert sha256(path) == expected, f"SHA256 mismatch: {path}"
    assert pair_manifest["manual_notes_sha256"] == ledger_manifest["manual_notes_sha256"]
    assert exact["exact_check_passed"] and not any(exact[key] for key in ("conflicts", "invalid_questions", "missing", "unreviewed"))
    assert exact["inputs"]["batch2"]["sha256"] == candidate_manifest["candidate_sha256"]
    assert len(raw) == 40644 and len(first) == 3000 and len(candidates) == len(second) == 1000 and len(combined) == 4000
    assert len({row["question_id"] for row in combined}) == 4000
    assert len(manual) == len({row["question_id"] for row in manual}) == pair_manifest["manual_candidate_rows"]
    for candidate, reviewed in zip(candidates, second, strict=True):
        source = raw[candidate["source_id"]]
        assert candidate["question"] == source["Open-ended Verifiable Question"]
        assert candidate["references"] == [source["Ground-True Answer"]]
        assert all(reviewed[key] == value for key, value in candidate.items())
    assert combined[:3000] == [{**row, "batch": 1, "screening_eligible": row["decision"] == "pending", "training_accepted": False} for row in first]
    assert combined[3000:] == [{**row, "batch": 2, "screening_eligible": row["decision"] == "pending", "training_accepted": False} for row in second]
    assert len(semantic) == len(pairs) == 82 and len(lexical) == 50 and len(lexical_pairs) == 44
    assert Counter(row["pair_verdict"] for row in pairs) == {"same_question": 76, "uncertain": 6}
    assert Counter(row["pair_verdict"] for row in lexical_pairs) == {"not_separately_adjudicated": 40, "same_question": 4}
    assert Counter(row["decision"] for row in second) == {"reject": 80, "quarantine": 20, "pending": 900}
    assert Counter(row["decision"] for row in combined) == {"reject": 374, "quarantine": 140, "pending": 3486}
    assert all(not row["training_accepted"] and row["split"] == "unassigned" for row in combined)
    assert all(not row["screening_eligible"] for row in combined if row["decision"] != "pending")
    assert not ledger_manifest["training_ready"] and ledger_manifest["training_accepted_rows"] == 0
    print("validated: batch2 raw question/gold 1000; exact conflicts 0; pair queues 82/50; batch2 80/20/900; combined 374/140/3486; training accepted 0")


if __name__ == "__main__":
    main()
