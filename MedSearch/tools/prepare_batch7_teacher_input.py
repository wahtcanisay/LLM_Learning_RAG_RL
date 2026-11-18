"""Apply reviewed batch7 question/semantic decisions before 7B rollout."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch7_20261001"
PAIR_REVIEW = BATCH / "semantic_pair_review"
OUT = BATCH / "teacher_input_review"
QUESTION_LEDGER_PATHS = [BATCH / f"question_review_manual_0{i}.jsonl" for i in range(1, 4)]
PAIR_FILES = [
    (PAIR_REVIEW / "semantic_pair_review_batch_01.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_batch_01.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_batch_01_manifest.json"),
    (PAIR_REVIEW / "semantic_pair_review_batch_02.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_batch_02.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_batch_02.manifest.json"),
    (PAIR_REVIEW / "semantic_pair_review_batch_03.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_03.jsonl", PAIR_REVIEW / "semantic_pair_review_manual_03_manifest.json"),
]
DUPLICATE_LABELS = {"same_task", "exact_near_duplicate"}
PAIR_LABELS = DUPLICATE_LABELS | {"related_distinct", "unclear"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_id_key(row: dict) -> tuple:
    value = row.get("source_id")
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite teacher-input package: {OUT}")
    candidate_path = BATCH / "candidate_batch7.jsonl"
    candidates = read_jsonl(candidate_path)
    candidate_by_id = {row["question_id"]: row for row in candidates}
    if len(candidates) != 1000 or len(candidate_by_id) != len(candidates):
        raise ValueError("candidate batch must contain 1,000 unique question IDs")

    question_reviews: dict[str, dict] = {}
    for path in QUESTION_LEDGER_PATHS:
        for row in read_jsonl(path):
            qid = row["question_id"]
            candidate = candidate_by_id.get(qid)
            quote = row.get("evidence_quote") or ""
            if candidate is None or not quote or quote not in candidate["question"]:
                raise ValueError(f"question review does not match source text: {qid}")
            if row.get("question_sha256") != text_sha256(candidate["question"]):
                raise ValueError(f"question hash mismatch: {qid}")
            if qid in question_reviews or row.get("question_decision") not in {"keep", "uncertain", "exclude"}:
                raise ValueError(f"duplicate ID or unknown question decision: {qid}")
            question_reviews[qid] = row
    if set(question_reviews) != set(candidate_by_id):
        raise ValueError("gold-blind question ledgers do not cover the 1,000 source candidates")

    semantic_manifest_path = BATCH / "semantic_neighbors.manifest.json"
    semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
    neighbor_path = BATCH / "semantic_neighbors.jsonl"
    if sha256(neighbor_path) != semantic_manifest["output_sha256"] or sha256(candidate_path) != semantic_manifest["candidate_sha256"]:
        raise ValueError("semantic neighbor/source hash mismatch")
    neighbor_rows = {}
    for path in (PAIR_REVIEW / f"semantic_pair_review_batch_0{i}.jsonl" for i in range(1, 4)):
        for row in read_jsonl(path):
            if row["pair_id"] in neighbor_rows:
                raise ValueError(f"duplicate pair ID in review queue: {row['pair_id']}")
            neighbor_rows[row["pair_id"]] = row

    reviewed_pairs = []
    pair_source_hashes = []
    for assignment_path, ledger_path, manifest_path in PAIR_FILES:
        assignments, ledger = read_jsonl(assignment_path), read_jsonl(ledger_path)
        review_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(assignments) != len(ledger) or sha256(assignment_path) != review_manifest.get("input_sha256", review_manifest.get("source_sha256")):
            raise ValueError(f"pair review input count/hash mismatch: {assignment_path.name}")
        if sha256(ledger_path) != review_manifest.get("output_ledger_sha256", review_manifest.get("ledger_sha256")):
            raise ValueError(f"pair review ledger hash mismatch: {ledger_path.name}")
        if [row["pair_id"] for row in assignments] != [row["pair_id"] for row in ledger]:
            raise ValueError(f"pair ID order/coverage mismatch: {ledger_path.name}")
        for source, review in zip(assignments, ledger, strict=True):
            if source["candidate"] not in candidate_by_id or source["candidate_question"] != candidate_by_id[source["candidate"]]["question"]:
                raise ValueError(f"semantic candidate text mismatch: {source['pair_id']}")
            label = review.get("relationship_label")
            if label not in PAIR_LABELS:
                raise ValueError(f"unknown pair relationship label: {source['pair_id']}")
            if "candidate_question" in review and review["candidate_question"] != source["candidate_question"]:
                raise ValueError(f"candidate question changed in ledger: {source['pair_id']}")
            if "matched_question" in review and review["matched_question"] != source["matched_question"]:
                raise ValueError(f"matched question changed in ledger: {source['pair_id']}")
            if review.get("candidate_question_sha256") not in (None, text_sha256(source["candidate_question"])):
                raise ValueError(f"candidate question hash mismatch: {source['pair_id']}")
            if review.get("matched_question_sha256") not in (None, text_sha256(source["matched_question"])):
                raise ValueError(f"matched question hash mismatch: {source['pair_id']}")
            if review.get("pair_kind") not in (None, source["kind"]):
                raise ValueError(f"pair kind mismatch: {source['pair_id']}")
            if source["candidate"] not in {qid for qid, row in question_reviews.items() if row["question_decision"] == "keep"}:
                raise ValueError(f"pair review includes a non-keep candidate: {source['candidate']}")
            if source["kind"] == "internal":
                if source["match"] not in candidate_by_id or source["match"] not in {qid for qid, row in question_reviews.items() if row["question_decision"] == "keep"}:
                    raise ValueError(f"internal pair endpoint not in keep set: {source['pair_id']}")
                if source["matched_question"] != candidate_by_id[source["match"]]["question"]:
                    raise ValueError(f"internal matched question mismatch: {source['pair_id']}")
            reviewed_pairs.append({**source, "relationship_label": label, "review_reason_zh": review.get("reason_zh", review.get("rationale_zh", ""))})
        pair_source_hashes.append({"assignment": assignment_path.name, "assignment_sha256": sha256(assignment_path), "ledger": ledger_path.name, "ledger_sha256": sha256(ledger_path)})

    if len(reviewed_pairs) != 231 or len({row["pair_id"] for row in reviewed_pairs}) != 231:
        raise ValueError("all 231 unique manually reviewed semantic pairs are required")
    pair_counts = Counter((row["kind"], row["relationship_label"]) for row in reviewed_pairs)
    dispositions = {}
    keep_ids = {qid for qid, row in question_reviews.items() if row["question_decision"] == "keep"}
    for qid, row in question_reviews.items():
        question_decision = row["question_decision"]
        initial = {"keep": "eligible_after_semantic_review", "uncertain": "hold_question_uncertain", "exclude": "exclude_question"}[question_decision]
        dispositions[qid] = {
            "question_id": qid,
            "source_id": candidate_by_id[qid].get("source_id"),
            "question_decision": question_decision,
            "semantic_disposition": initial,
            "pair_ids": [],
            "matched_ids": [],
            "matched_pools": [],
            "semantic_review_labels": [],
        }

    history_duplicates: set[str] = set()
    unclear: set[str] = set()
    internal_duplicate_edges = []
    internal_unclear_edges = []
    for row in reviewed_pairs:
        qid, match, kind, label = row["candidate"], row["match"], row["kind"], row["relationship_label"]
        record = dispositions[qid]
        record["pair_ids"].append(row["pair_id"])
        record["semantic_review_labels"].append(label)
        record["matched_ids"].append(match)
        if row.get("matched_pool"):
            record["matched_pools"].append(row["matched_pool"])
        if kind == "history" and label in DUPLICATE_LABELS:
            history_duplicates.add(qid)
        elif label == "unclear":
            unclear.add(qid)
            if kind == "internal":
                unclear.add(match)
                internal_unclear_edges.append((qid, match))
        elif kind == "internal" and label in DUPLICATE_LABELS:
            internal_duplicate_edges.append((qid, match))

    parent = {qid: qid for qid in keep_ids}

    def find(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left, right in internal_duplicate_edges:
        union(left, right)
    components: dict[str, set[str]] = defaultdict(set)
    for qid in keep_ids:
        components[find(qid)].add(qid)
    for component in components.values():
        if component & history_duplicates:
            history_duplicates |= component
        elif component & unclear:
            unclear |= component
        elif len(component) > 1:
            representative = min(component, key=lambda item: source_id_key(candidate_by_id[item]))
            for qid in component - {representative}:
                dispositions[qid]["semantic_disposition"] = "exclude_internal_duplicate"
                dispositions[qid]["matched_ids"].append(representative)
            dispositions[representative]["semantic_review_labels"].append("internal_duplicate_representative")

    for qid in history_duplicates:
        dispositions[qid]["semantic_disposition"] = "exclude_history_duplicate"
    for qid in unclear - history_duplicates:
        dispositions[qid]["semantic_disposition"] = "hold_semantic_unclear"
    for record in dispositions.values():
        record["pair_ids"] = sorted(set(record["pair_ids"]))
        record["matched_ids"] = sorted(set(record["matched_ids"]))
        record["matched_pools"] = sorted(set(record["matched_pools"]))
        record["semantic_review_labels"] = sorted(set(record["semantic_review_labels"]))

    rollout_rows = []
    for candidate in candidates:
        qid = candidate["question_id"]
        if dispositions[qid]["semantic_disposition"] != "eligible_after_semantic_review":
            continue
        row = dict(candidate)
        row["source_dataset_split"] = row.get("source_dataset_split") or row.get("split", "unassigned")
        row["split"] = "cold_start"
        rollout_rows.append(row)

    disposition_counts = Counter(row["semantic_disposition"] for row in dispositions.values())
    OUT.mkdir(parents=True, exist_ok=True)
    disposition_path = OUT / "batch7_semantic_dispositions.jsonl"
    rollout_path = OUT / "teacher_rollout_input_cold_start.jsonl"
    for path, rows in ((disposition_path, [dispositions[qid] for qid in sorted(dispositions)]), (rollout_path, rollout_rows)):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "teacher_rollout_input_ready_only",
        "training_ready": False,
        "teacher_rollout_started": False,
        "candidate_rows": len(candidates),
        "question_decision_counts": dict(Counter(row["question_decision"] for row in question_reviews.values())),
        "semantic_pair_rows": len(reviewed_pairs),
        "semantic_pair_kind_label_counts": {f"{kind}:{label}": count for (kind, label), count in sorted(pair_counts.items())},
        "candidate_disposition_counts": dict(disposition_counts),
        "teacher_rollout_rows": len(rollout_rows),
        "source_history_rows": semantic_manifest.get("reference_rows"),
        "history_screen_limitation": "MPNet Top-1 cosine >=0.92 is a manual screening queue, not proof of exhaustive semantic isolation; historical Stage A/B source files are unavailable per batch manifest",
        "inputs_sha256": {
            "candidate_batch7.jsonl": sha256(candidate_path),
            "semantic_neighbors.jsonl": sha256(neighbor_path),
            "semantic_neighbors.manifest.json": sha256(semantic_manifest_path),
            **{path.name: sha256(path) for path in QUESTION_LEDGER_PATHS},
            **{entry["assignment"]: entry["assignment_sha256"] for entry in pair_source_hashes},
            **{entry["ledger"]: entry["ledger_sha256"] for entry in pair_source_hashes},
        },
        "semantic_pair_sources": pair_source_hashes,
        "outputs_sha256": {disposition_path.name: sha256(disposition_path), rollout_path.name: sha256(rollout_path)},
        "rule": "only question keep candidates; manually reviewed same-task/exact-near duplicates against history excluded from this batch; internal duplicate components keep lower source_id; unclear pairs held; related-distinct remains eligible for teacher trajectory generation",
        "split_note": "cold_start labels the teacher-generation cohort only; source_dataset_split preserves original source split and does not assign final train/dev/test",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"question_decision_counts": manifest["question_decision_counts"], "semantic_pair_kind_label_counts": manifest["semantic_pair_kind_label_counts"], "candidate_disposition_counts": dict(disposition_counts), "teacher_rollout_rows": len(rollout_rows), "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
