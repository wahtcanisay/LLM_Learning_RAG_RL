"""Freeze a fresh Huatuo reserve candidate batch against all visible boundaries."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import build_distillation_v2_batch3 as common


ROOT = common.ROOT
V2 = common.V2
BATCH2 = common.BATCH2
POOLS = common.POOLS
BATCH3 = ROOT / "MedSearch/data/medical_verifiable_v2_batch3_20260930"
SOURCE = common.SOURCE
SOURCE_REVISION = common.SOURCE_REVISION
SEED = 20261001
COUNT = 1000
OUT = ROOT / "MedSearch/data/medical_verifiable_v2_batch4_20261001"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {OUT}")
    source_manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    batch2_manifest = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    batch3_manifest = json.loads((BATCH3 / "manifest.json").read_text(encoding="utf-8"))
    pool_manifest = json.loads((POOLS / "manifest.json").read_text(encoding="utf-8"))
    raw_path = ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    reserve_path = V2 / "candidate_other.jsonl"
    batch3_candidate_path = BATCH3 / "candidate_batch3.jsonl"
    if source_manifest["source"] != SOURCE or source_manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("unexpected Huatuo source identity/revision")
    if common.sha256(batch3_candidate_path) != batch3_manifest["candidate_sha256"]:
        raise ValueError("batch3 candidate hash differs from its frozen manifest")

    histories = common.source_paths()
    expected_batch3_sources = batch3_manifest["history_input_sha256"]
    actual_source_hashes = {name: common.sha256(path) for name, path in histories.items()}
    if expected_batch3_sources != actual_source_hashes:
        missing = sorted(set(expected_batch3_sources) - set(actual_source_hashes))
        extra = sorted(set(actual_source_hashes) - set(expected_batch3_sources))
        changed = sorted(name for name in set(expected_batch3_sources) & set(actual_source_hashes)
                         if expected_batch3_sources[name] != actual_source_hashes[name])
        raise ValueError(f"batch3 history boundary does not reproduce: missing={missing}, extra={extra}, changed={changed}")
    histories["candidate_batch3"] = batch3_candidate_path

    for path, expected in (
        (raw_path, source_manifest["source_sha256"]),
        (reserve_path, source_manifest["outputs_sha256"]["candidate_other.jsonl"]),
        (V2 / "candidate_keep.jsonl", source_manifest["outputs_sha256"]["candidate_keep.jsonl"]),
        (BATCH2 / "candidate_batch2.jsonl", batch2_manifest["candidate_sha256"]),
        (POOLS / "cold_start.jsonl", pool_manifest["output_sha256"]["cold_start.jsonl"]),
    ):
        if common.sha256(path) != expected:
            raise ValueError(f"frozen input SHA256 changed: {path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    reserve = [row for row in read_jsonl(reserve_path) if row.get("status") == "reserve"]
    if len(reserve) != source_manifest["other_status_counts"]["reserve"]:
        raise ValueError("reserve status count differs from frozen source manifest")

    history_rows = [record for name, path in histories.items() for record in common.read_questions(name, path)]
    history_by_id: dict[str, set[str]] = defaultdict(set)
    history_by_question: dict[str, set[str]] = defaultdict(set)
    for row in history_rows:
        key = common.normalize(row["question"])
        if row["id"]:
            history_by_id[row["id"]].add(row["pool"])
        if key:
            history_by_question[key].add(row["pool"])

    eligible, excluded = [], Counter()
    seen_ids, seen_questions = set(), set()
    for row in reserve:
        index = row.get("source_id")
        if not isinstance(index, int) or not 0 <= index < len(raw) or row.get("question_id") != f"medical-o1-verifiable:{index}":
            raise ValueError(f"invalid source identity: {row.get('question_id')}")
        source = raw[index]
        answer = source.get("Ground-True Answer")
        question = source.get("Open-ended Verifiable Question")
        if row.get("question") != question or row.get("references") != [answer]:
            raise ValueError(f"question/gold differs from frozen raw source: {index}")
        key = common.normalize(question)
        item_id = row["question_id"]
        if item_id in history_by_id:
            excluded["historical_id"] += 1
            continue
        if key in history_by_question:
            excluded["historical_normalized_question"] += 1
            continue
        if item_id in seen_ids or key in seen_questions:
            excluded["internal_exact_duplicate"] += 1
            continue
        seen_ids.add(item_id)
        seen_questions.add(key)
        eligible.append(row)

    if len(eligible) < COUNT:
        raise ValueError(f"only {len(eligible)} exact-isolated reserve rows; need {COUNT}")
    ranked = sorted(
        eligible,
        key=lambda row: (hashlib.sha256(f"{SEED}:{row['source_id']}".encode("ascii")).digest(), row["source_id"]),
    )[:COUNT]
    candidates = [
        {"question_id": row["question_id"], "source": row["source"], "source_id": row["source_id"],
         "question": row["question"], "references": row["references"], "split": "unassigned"}
        for row in sorted(ranked, key=lambda row: row["source_id"])
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT / "candidate_batch4.jsonl"
    with candidate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    unique_history: dict[str, dict] = {}
    for row in history_rows:
        key = common.normalize(row["question"])
        if key:
            unique_history.setdefault(key, row)
    history_path = OUT / "semantic_history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unique_history.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidate_ids = {row["question_id"] for row in candidates}
    candidate_keys = {common.normalize(row["question"]) for row in candidates}
    history_ids = {row["id"] for row in unique_history.values() if row.get("id")}
    if candidate_ids & history_ids or candidate_keys & set(unique_history):
        raise AssertionError("exact isolation failed after write")
    if len(candidate_ids) != COUNT or len(candidate_keys) != COUNT:
        raise AssertionError("candidate count or internal uniqueness failed")

    input_hashes = {name: common.sha256(path) for name, path in histories.items()}
    report = {
        "status": "candidate_only_pending_semantic_pair_review_question_gold_review_and_teacher_generation",
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "raw_source_sha256": common.sha256(raw_path),
        "reserve_input_sha256": common.sha256(reserve_path),
        "reserve_rows": len(reserve),
        "exact_excluded_counts": dict(excluded),
        "exact_isolated_reserve_rows": len(eligible),
        "candidate_rows": len(candidates),
        "selection_seed": SEED,
        "selection_rule": "SHA256 ASCII(seed:source_id), smallest 1000 among reserve questions exact-isolated against batch1/2/3, existing SFT/RL/dev/test and source-reference datasets; output ascending source_id",
        "candidate_sha256": common.sha256(candidate_path),
        "semantic_history_path": str(history_path.relative_to(ROOT)),
        "semantic_history_rows_before_dedup": len(history_rows),
        "semantic_history_rows_after_exact_dedup": len(unique_history),
        "semantic_history_sha256": common.sha256(history_path),
        "history_source_counts": dict(Counter(row["pool"] for row in history_rows)),
        "history_input_sha256": input_hashes,
        "builder_script_sha256": common.sha256(Path(__file__)),
        "exact_isolation": "passed_against_visible_prior_SFT_RL_dev_test_and_candidate_batches; not a claim of complete hidden training exposure",
        "semantic_screen": "pending; MPNet cosine >=0.92 is only a manual review queue",
        "question_gold_review": "pending; source gold is not expert verified",
        "teacher_generation": "pending",
        "training_ready": False,
        "known_boundary_limitation": "historical Stage A/B source files are missing or unavailable in the current workspace audit; do not claim complete historical exposure isolation",
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in (
        "status", "reserve_rows", "exact_excluded_counts", "exact_isolated_reserve_rows",
        "candidate_rows", "semantic_history_rows_after_exact_dedup", "candidate_sha256", "training_ready"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
