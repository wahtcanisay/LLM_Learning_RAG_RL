"""Create a provisional Batch8 pool after human semantic duplicate review."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "MedSearch/data/medical_verifiable_v2_batch8_20260930"
SCREEN = BATCH / "semantic_screen_cpu"
OUTPUT = BATCH / "semantic_dedup_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def normalize(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def main() -> None:
    candidate_path = BATCH / "candidate_batch8.jsonl"
    queue_path = SCREEN / "semantic_review_queue.jsonl"
    pack_path = SCREEN / "human_pair_review_pack.jsonl"
    pack_manifest_path = SCREEN / "human_pair_review_pack_manifest.json"
    ledger_path = SCREEN / "human_pair_review_manual_02.jsonl"
    ledger_manifest_path = SCREEN / "human_pair_review_manual_02_manifest.json"
    source_manifest_path = BATCH / "manifest.json"
    candidates = read_jsonl(candidate_path)
    queue, pack, ledger = read_jsonl(queue_path), read_jsonl(pack_path), read_jsonl(ledger_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    ledger_manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))

    candidate_hash = file_sha(candidate_path)
    if len(candidates) != 1000 or source_manifest.get("candidate_sha256") != candidate_hash:
        raise ValueError("Batch8 candidate rows/hash do not match their source manifest")
    ids = [row["question_id"] for row in candidates]
    normalized = [normalize(row["question"]) for row in candidates]
    if len(set(ids)) != 1000 or len(set(normalized)) != 1000:
        raise ValueError("Batch8 source candidates are not exact-question unique")
    if not (len(queue) == len(pack) == len(ledger) == 66):
        raise ValueError("semantic queue, blind pack, and human ledger must each cover 66 pairs")
    if (pack_manifest.get("output_sha256") != file_sha(pack_path)
            or pack_manifest.get("input_semantic_queue_sha256") != file_sha(queue_path)
            or ledger_manifest.get("ledger", {}).get("sha256") != file_sha(ledger_path)
            or ledger_manifest.get("report", {}).get("sha256") != file_sha(SCREEN / "human_pair_review_manual_02_report.json")
            or ledger_manifest.get("coverage", {}).get("pair_ids") != 66
            or ledger_manifest.get("blindness", {}).get("gold_answer_reference_score_auto_label_fields_used") is not False):
        raise ValueError("semantic review package/ledger hash or blindness check failed")

    by_id = {row["question_id"]: row for row in candidates}
    history_exclusions: dict[str, dict] = {}
    internal_same_task: list[dict] = []
    counts = Counter()
    for index, (machine_pair, blind_pair, decision) in enumerate(zip(queue, pack, ledger)):
        pair_id = decision.get("pair_id")
        if decision.get("original_order_index") != index or blind_pair.get("pair_id") != pair_id:
            raise ValueError(f"semantic pair order/ID mismatch at row {index + 1}")
        if decision.get("pair_type") not in {"candidate_to_historical", "candidate_internal"}:
            raise ValueError(f"invalid semantic pair type at {pair_id}")
        if decision.get("relationship_label") not in {"same_task", "related_distinct"}:
            raise ValueError(f"unresolved semantic decision at {pair_id}")
        counts[(decision["pair_type"], decision["relationship_label"])] += 1
        pair_members = [decision.get("left", {}), decision.get("right", {})]
        for member in pair_members:
            if sha(member.get("question", "").encode("utf-8")) != member.get("question_sha256"):
                raise ValueError(f"question hash mismatch in {pair_id}")
        candidates_in_pair = [member for member in pair_members if member.get("source_category") == "batch8_candidate"]

        if decision["pair_type"] == "candidate_to_historical":
            if len(candidates_in_pair) != 1:
                raise ValueError(f"expected exactly one Batch8 candidate in {pair_id}")
            candidate_id = candidates_in_pair[0]["id"]
            if candidate_id not in by_id or by_id[candidate_id]["question"] != candidates_in_pair[0]["question"]:
                raise ValueError(f"historical pair candidate does not match Batch8 source: {candidate_id}")
            if decision["relationship_label"] == "same_task":
                history_exclusions[candidate_id] = {
                    "question_id": candidate_id,
                    "pair_id": pair_id,
                    "reason_zh": decision["reason_zh"],
                    "matched_historical_id": next((member["id"] for member in pair_members if member.get("source_category") != "batch8_candidate"), None),
                    "matched_question": next(member["question"] for member in pair_members if member.get("source_category") != "batch8_candidate"),
                }
        elif decision["pair_type"] == "candidate_internal":
            if len(candidates_in_pair) != 2 or any(member["id"] not in by_id for member in candidates_in_pair):
                raise ValueError(f"expected two Batch8 candidates in {pair_id}")
            if any(by_id[member["id"]]["question"] != member["question"] for member in candidates_in_pair):
                raise ValueError(f"internal pair candidate text mismatch in {pair_id}")
            if decision["relationship_label"] == "same_task":
                internal_same_task.append({
                    "pair_id": pair_id,
                    "candidate_ids": [member["id"] for member in candidates_in_pair],
                    "reason_zh": decision["reason_zh"],
                    "representative_selection": "defer until question and trajectory quality reviews; keep at most one",
                })

    # Historical duplicates leave the pool now; internal duplicates stay provisionally until quality ranking.
    retained = [row for row in candidates if row["question_id"] not in history_exclusions]
    if len(history_exclusions) != 57 or len(internal_same_task) != 3 or len(retained) != 943:
        raise ValueError("unexpected semantic dispositions or retained candidate count")
    duplicate_member_ids = {qid for group in internal_same_task for qid in group["candidate_ids"]}
    if len(duplicate_member_ids) != 6 or len(duplicate_member_ids) != sum(len(group["candidate_ids"]) for group in internal_same_task):
        raise ValueError("internal same-task pairs overlap or are not disjoint")
    max_unique = len(retained) - len(internal_same_task)
    if max_unique != 940:
        raise ValueError("internal same-task collapse count mismatch")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates_out = OUTPUT / "candidate_batch8_history_semantic_dedup.jsonl"
    exclusions_out = OUTPUT / "historical_same_task_exclusions.jsonl"
    groups_out = OUTPUT / "internal_same_task_groups.jsonl"
    report_out = OUTPUT / "report.md"
    candidates_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")
    exclusions_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history_exclusions.values()), encoding="utf-8")
    groups_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in internal_same_task), encoding="utf-8")
    if len(read_jsonl(candidates_out)) != 943 or len(read_jsonl(exclusions_out)) != 57 or len(read_jsonl(groups_out)) != 3:
        raise ValueError("derived Batch8 output coverage check failed")

    history_related = counts[("candidate_to_historical", "related_distinct")]
    internal_related = counts[("candidate_internal", "related_distinct")]
    report = (
        "# Batch8 provisional semantic filtering\n\n"
        "This is a provisional question inventory, not an SFT trajectory export.\n\n"
        f"- Source candidates: 1,000; historical same-task exclusions: {len(history_exclusions)}; retained questions: {len(retained)}.\n"
        f"- Three internal same-task pairs are recorded for later quality-based representative selection; at most {max_unique} unique tasks remain before question/trajectory review.\n"
        f"- Historical related-distinct pairs retained: {history_related}; internal related-distinct pairs retained: {internal_related}.\n"
        "- `training_ready=false`; Stage A/B exposure is unresolved, question review and Search-R1 trajectory review are incomplete, and no final split exists.\n"
    )
    report_out.write_text(report, encoding="utf-8")
    manifest = {
        "status": "provisional_semantic_filter_only",
        "training_ready": False,
        "final_split_created": False,
        "source_rows": 1000,
        "historical_same_task_excluded": len(history_exclusions),
        "retained_rows_before_question_review": len(retained),
        "internal_same_task_pairs_pending_representative_choice": len(internal_same_task),
        "maximum_distinct_question_count_before_quality_review": max_unique,
        "semantic_decision_counts": {
            "historical_same_task": counts[("candidate_to_historical", "same_task")],
            "historical_related_distinct": counts[("candidate_to_historical", "related_distinct")],
            "internal_same_task": counts[("candidate_internal", "same_task")],
            "internal_related_distinct": counts[("candidate_internal", "related_distinct")],
        },
        "inputs": {
            "candidate_sha256": candidate_hash,
            "queue_sha256": file_sha(queue_path),
            "blind_pack_sha256": file_sha(pack_path),
            "human_ledger_sha256": file_sha(ledger_path),
            "human_ledger_manifest_sha256": file_sha(ledger_manifest_path),
        },
        "outputs": {
            candidates_out.name: {"rows": len(retained), "sha256": file_sha(candidates_out)},
            exclusions_out.name: {"rows": len(history_exclusions), "sha256": file_sha(exclusions_out)},
            groups_out.name: {"rows": len(internal_same_task), "sha256": file_sha(groups_out)},
            report_out.name: {"sha256": file_sha(report_out)},
        },
        "internal_same_task_groups": internal_same_task,
        "limitations": [
            "Exact-history screening covers only available history; Stage A train/dev, Search100, and Stage B protocol dev are missing.",
            "Question review and Search-R1 trajectory review are incomplete.",
            "Other large raw SFT corpora were not included in this semantic embedding pass.",
        ],
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check = json.loads(manifest_path.read_text(encoding="utf-8"))
    if check["outputs"][candidates_out.name]["sha256"] != file_sha(candidates_out):
        raise ValueError("manifest output hash verification failed")
    print(json.dumps({"status": check["status"], "retained": len(retained), "history_excluded": len(history_exclusions), "internal_same_task_pairs": len(internal_same_task), "max_unique": max_unique, "training_ready": False, "manifest": manifest_path.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
