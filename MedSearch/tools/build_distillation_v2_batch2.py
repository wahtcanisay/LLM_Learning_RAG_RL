"""Select exactly 1,000 traceable V2 batch2 candidates from the frozen reserve."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "MedSearch/data/medical_verifiable_v2_20260919"
OUT = ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926"
SEED = 20260919
COUNT = 1000


def select_reserve_batch(reserve: list[dict], raw: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0 or len(reserve) < count:
        raise ValueError("positive count cannot exceed reserve size")
    seen = set()
    for row in reserve:
        index = row.get("source_id")
        if row.get("status") != "reserve" or not isinstance(index, int) or not 0 <= index < len(raw):
            raise ValueError("invalid reserve row")
        if index in seen or row.get("question_id") != f"medical-o1-verifiable:{index}":
            raise ValueError("repeated or mismatched source ID")
        seen.add(index)
        source = raw[index]
        if row.get("question") != source.get("Open-ended Verifiable Question") or row.get("references") != [source.get("Ground-True Answer")]:
            raise ValueError(f"reserve row differs from raw source: {index}")
    ranked = sorted(
        reserve,
        key=lambda row: (hashlib.sha256(f"{seed}:{row['source_id']}".encode("ascii")).digest(), row["source_id"]),
    )[:count]
    return [
        {"question_id": row["question_id"], "source": row["source"], "source_id": row["source_id"],
         "question": row["question"], "references": row["references"], "split": "unassigned"}
        for row in sorted(ranked, key=lambda row: row["source_id"])
    ]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    first_manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    raw_path = ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    other_path = V2 / "candidate_other.jsonl"
    first_path = V2 / "candidate_keep.jsonl"
    for path, expected in (
        (raw_path, first_manifest["source_sha256"]),
        (other_path, first_manifest["outputs_sha256"]["candidate_other.jsonl"]),
        (first_path, first_manifest["outputs_sha256"]["candidate_keep.jsonl"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen input SHA256 changed: {path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    other = _read_jsonl(other_path)
    if Counter(row["status"] for row in other) != first_manifest["other_status_counts"]:
        raise ValueError("reserve status counts differ from frozen manifest")
    reserve = [row for row in other if row["status"] == "reserve"]
    selected = select_reserve_batch(reserve, raw, COUNT, SEED)
    first_ids = {row["question_id"] for row in _read_jsonl(first_path)}
    if first_ids.intersection(row["question_id"] for row in selected):
        raise ValueError("batch2 overlaps first candidate pool by ID")
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "candidate_batch2.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "candidate_only_pending_isolation_and_medical_review",
        "source": first_manifest["source"],
        "source_revision": first_manifest["source_revision"],
        "source_sha256": _sha256(raw_path),
        "reserve_input_sha256": _sha256(other_path),
        "first_candidate_sha256": _sha256(first_path),
        "reserve_rows": len(reserve),
        "candidate_rows": len(selected),
        "selection_seed": SEED,
        "selection_rule": "SHA256 ASCII(seed:source_id), lexicographically smallest 1000; output ascending source_id",
        "candidate_sha256": _sha256(output),
        "split_assignment": "not_done",
        "medical_review": "not_done",
        "training_ready": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
