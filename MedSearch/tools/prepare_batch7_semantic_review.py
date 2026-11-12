"""Prepare a manual semantic duplicate queue from batch7 keep questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch7_20261001"
OUT = BATCH / "semantic_pair_review"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite review package: {OUT}")
    candidate_path = BATCH / "candidate_batch7.jsonl"
    candidates = {row["question_id"]: row for row in read_jsonl(candidate_path)}
    if len(candidates) != 1000:
        raise ValueError(f"expected 1,000 source candidates, found {len(candidates)}")

    review_rows = []
    for index in range(1, 4):
        ledger = read_jsonl(BATCH / f"question_review_manual_0{index}.jsonl")
        if len({row["question_id"] for row in ledger}) != len(ledger):
            raise ValueError(f"duplicate IDs in question review {index}")
        for row in ledger:
            candidate = candidates.get(row["question_id"])
            evidence_quote = row.get("evidence_quote") or ""
            if candidate is None or not evidence_quote or evidence_quote not in candidate["question"]:
                raise ValueError(f"question review does not match candidate: {row['question_id']}")
            expected_hash = hashlib.sha256(candidate["question"].encode("utf-8")).hexdigest()
            if expected_hash != row.get("question_sha256"):
                raise ValueError(f"question hash mismatch: {row['question_id']}")
            review_rows.append(row)
    if len(review_rows) != 1000 or {row["question_id"] for row in review_rows} != set(candidates):
        raise ValueError("question-review ledgers do not cover all 1,000 candidates")
    keep = {row["question_id"] for row in review_rows if row["question_decision"] == "keep"}

    neighbors_path = BATCH / "semantic_neighbors.jsonl"
    neighbors_manifest_path = BATCH / "semantic_neighbors.manifest.json"
    neighbors_manifest = json.loads(neighbors_manifest_path.read_text(encoding="utf-8"))
    if sha256(neighbors_path) != neighbors_manifest["output_sha256"]:
        raise ValueError("semantic neighbor file hash mismatch")
    if sha256(candidate_path) != neighbors_manifest["candidate_sha256"]:
        raise ValueError("semantic candidate source hash mismatch")
    neighbors = read_jsonl(neighbors_path)
    if len(neighbors) != 394:
        raise ValueError(f"semantic neighbor count differs from manifest: {len(neighbors)}")

    selected = []
    for row in neighbors:
        if row["candidate"] not in candidates or row["candidate_question"] != candidates[row["candidate"]]["question"]:
            raise ValueError(f"semantic row does not match candidate source: {row['candidate']}")
        if row["kind"] == "history" and row["candidate"] in keep:
            selected.append(row)
        elif row["kind"] == "internal" and row["candidate"] in keep and row["match"] in keep:
            if row["match"] not in candidates or row["matched_question"] != candidates[row["match"]]["question"]:
                raise ValueError(f"internal match does not match candidate source: {row['match']}")
            selected.append(row)

    # Keep every candidate's history match and internal duplicate component in one review shard.
    parent = {question_id: question_id for question_id in keep}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in selected:
        if row["kind"] == "internal":
            union(row["candidate"], row["match"])
    groups: dict[str, list[dict]] = {}
    for row in selected:
        key = find(row["candidate"])
        item = {
            "pair_id": hashlib.sha256(f"{row['kind']}:{row['candidate']}:{row['match']}".encode("utf-8")).hexdigest()[:16],
            **row,
        }
        groups.setdefault(key, []).append(item)
    shards = [[], [], []]
    for index, key in enumerate(sorted(groups)):
        shards[index % len(shards)].extend(groups[key])

    OUT.mkdir(parents=True, exist_ok=True)
    batch_paths = []
    for index, rows in enumerate(shards, 1):
        path = OUT / f"semantic_pair_review_batch_{index:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        batch_paths.append(path)
    manifest = {
        "status": "manual_semantic_review_queue_only",
        "training_ready": False,
        "candidate_rows": len(candidates),
        "question_decisions": {
            label: sum(row["question_decision"] == label for row in review_rows)
            for label in ("keep", "uncertain", "exclude")
        },
        "semantic_neighbor_rows": len(neighbors),
        "review_pair_rows": len(selected),
        "review_pair_kinds": {
            label: sum(row["kind"] == label for row in selected)
            for label in ("history", "internal")
        },
        "review_shards": {path.name: {"rows": len(rows), "sha256": sha256(path)} for path, rows in zip(batch_paths, shards, strict=True)},
        "inputs_sha256": {
            "candidate_batch7.jsonl": sha256(candidate_path),
            "semantic_neighbors.jsonl": sha256(neighbors_path),
            "semantic_neighbors.manifest.json": sha256(neighbors_manifest_path),
            **{f"question_review_manual_0{i}.jsonl": sha256(BATCH / f"question_review_manual_0{i}.jsonl") for i in range(1, 4)},
        },
        "automatic_exclusion": False,
        "manual_review_rule": "question text and matched question only; similarity is a queue signal, not a duplicate decision",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"question_decisions": manifest["question_decisions"], "review_pair_rows": len(selected), "review_pair_kinds": manifest["review_pair_kinds"], "shards": {path.name: len(rows) for path, rows in zip(batch_paths, shards, strict=True)}, "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
