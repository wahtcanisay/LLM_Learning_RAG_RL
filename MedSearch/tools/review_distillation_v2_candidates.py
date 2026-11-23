"""Record review state for V2 candidates without treating screening as medical approval."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MedSearch/data/medical_verifiable_v2_20260919"


def build_ledger(candidates: list[dict], neighbors: list[dict], manual: list[dict], lexical_neighbors: list[dict] | None = None) -> list[dict]:
    ids = [row["question_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs are not unique")
    known = set(ids)
    neighbor_by_id = defaultdict(list)
    lexical_by_id = defaultdict(list)
    for item, target_map in [(item, neighbor_by_id) for item in neighbors] + [(item, lexical_by_id) for item in (lexical_neighbors or [])]:
        targets = [item["candidate"]]
        if item["kind"] == "internal":
            targets.append(item["match"])
        for target in targets:
            if target not in known:
                raise ValueError(f"unknown neighbor candidate: {target}")
            target_map[target].append({
                "kind": item["kind"],
                "match": item["match"] if target == item["candidate"] else item["candidate"],
                "matched_pool": item["matched_pool"],
                "cosine": item["cosine"],
            })
    manual_by_id = {}
    for item in manual:
        item_id = item["question_id"]
        if item_id not in known or item_id in manual_by_id:
            raise ValueError(f"unknown or repeated manual candidate: {item_id}")
        if item["decision"] not in {"reject", "quarantine"} or not all(item.get(key) for key in ("reason", "reviewer", "evidence")):
            raise ValueError(f"invalid manual decision: {item_id}")
        manual_by_id[item_id] = item
    ledger = []
    for row in candidates:
        item_id = row["question_id"]
        found = neighbor_by_id[item_id]
        lexical_found = lexical_by_id[item_id]
        decision = manual_by_id.get(item_id)
        if decision:
            review_state = decision["decision"]
            basis = decision.get("review_basis", "manual_pair_text")
            reason = decision["reason"]
        elif found or lexical_found:
            review_state, basis, reason = "quarantine", "embedding_queue", "near_duplicate_pending_pair_review"
        else:
            review_state, basis, reason = "pending", "rule_screen_only", "medical_review_not_done"
        ledger.append({
            **row,
            "decision": review_state,
            "reason": reason,
            "review_basis": basis,
            "reviewer": decision["reviewer"] if decision else None,
            "manual_evidence": decision["evidence"] if decision else None,
            "neighbor_evidence": found,
            "lexical_neighbor_evidence": lexical_found,
        })
    return ledger


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    candidate_path = DATA / "candidate_keep.jsonl"
    neighbor_path = DATA / "semantic_neighbors.jsonl"
    lexical_path = DATA / "lexical_neighbors.jsonl"
    manual_path = DATA / "manual_review_notes.jsonl"
    candidates = _read_jsonl(candidate_path)
    neighbors = _read_jsonl(neighbor_path)
    lexical = _read_jsonl(lexical_path)
    manual = _read_jsonl(manual_path)
    candidate_manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    neighbor_manifest = json.loads((DATA / "semantic_neighbors.manifest.json").read_text(encoding="utf-8"))
    lexical_manifest = json.loads((DATA / "lexical_neighbors.manifest.json").read_text(encoding="utf-8"))
    if _sha256(candidate_path) != candidate_manifest["outputs_sha256"]["candidate_keep.jsonl"]:
        raise ValueError("candidate SHA256 differs from frozen manifest")
    if _sha256(candidate_path) != neighbor_manifest["candidate_sha256"] or _sha256(neighbor_path) != neighbor_manifest["output_sha256"]:
        raise ValueError("semantic queue input SHA256 differs from frozen manifest")
    if _sha256(candidate_path) != lexical_manifest["candidate_sha256"] or _sha256(lexical_path) != lexical_manifest["output_sha256"]:
        raise ValueError("lexical queue input SHA256 differs from frozen manifest")
    ledger = build_ledger(candidates, neighbors, manual, lexical_neighbors=lexical)
    output = DATA / "review_ledger.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["decision"] for row in ledger)
    report = {
        "status": "partial_review_not_training_ready",
        "candidate_sha256": _sha256(candidate_path),
        "neighbor_sha256": _sha256(neighbor_path),
        "lexical_neighbor_sha256": _sha256(lexical_path),
        "manual_notes_sha256": _sha256(manual_path),
        "ledger_sha256": _sha256(output),
        "candidate_rows": len(candidates),
        "neighbor_queue_rows": len(neighbors),
        "neighbor_flagged_candidate_ids": sum(bool(row["neighbor_evidence"]) for row in ledger),
        "lexical_queue_rows": len(lexical),
        "lexical_flagged_candidate_ids": sum(bool(row["lexical_neighbor_evidence"]) for row in ledger),
        "manual_review_rows": len(manual),
        "decision_counts": dict(counts),
        "accepted_rows": 0,
        "split_assignment": "not_done",
        "training_ready": False,
        "limitations": [
            "Embedding Top-1 at cosine 0.92 is a screening queue, not exhaustive duplicate adjudication.",
            "Lexical Top-1 at cosine 0.8 is a second screening queue, not exhaustive duplicate adjudication.",
            "Manual pair-text comparison does not verify clinical correctness of gold answers.",
            "Pending candidates remain unapproved for medical question/gold quality and training.",
        ],
    }
    (DATA / "review_ledger.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
