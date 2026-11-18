"""Validate and summarize the frozen full-census SFT review."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from MedSearch.tools.check_distillation_inputs import check_split_isolation


ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "MedSearch/data/distillation/sft_environment_review_20260929_full"
PILOT = ROOT / "MedSearch/data/distillation/sft_environment_review_20260929_pilot"
ACCEPTED = ROOT / "MedSearch/data/distillation/teacher_cold1000_20260929_answered/accepted.jsonl"
ATTEMPTS = ROOT / "MedSearch/data/distillation/teacher_cold1000_20260929/teacher_attempts.jsonl"
COLD = ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean/cold_start.jsonl"
DEV = ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet"
TEST = ROOT / "MIRAGE/benchmark.json"
SEMANTIC_MODEL = ROOT / "MedicalGraphRAG/models/all-mpnet-base-v2"
REQUIRED = {
    "question_id", "reviewer", "question_decision", "question_rules", "question_reason_zh", "required_facts",
    "support_label", "support_map", "visible_support", "full_return_support",
    "exact_qa_leak", "exact_qa_leak_basis_zh", "environment_status", "environment_rationale_zh",
    "diagnostic_needed", "trajectory_status", "invalid_action", "duplicate_search",
    "duplicate_search_reason_zh", "teacher_answer_final", "teacher_answer_relation", "teacher_answer_issue",
    "dedup_status", "sft_recommendation", "recommendation_reason_zh",
    "invalid_action_original_review", "invalid_action_detail",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def apply_support_adjudications(combined: list[dict]) -> dict:
    agent_path = FULL / "sft_conflict_adjudication_agent04.jsonl"
    root_path = FULL / "support_conflict_adjudication_root_20260930.jsonl"
    override_path = FULL / "support_conflict_adjudication_root_overrides_20260930.jsonl"
    agent_rows = read_jsonl(agent_path)
    root_rows = read_jsonl(root_path)
    override_rows = read_jsonl(override_path)
    conflict_ids = {row["question_id"] for row in combined if row.get("support_label") == "conflicting"}

    adjudications = {}
    for item in agent_rows:
        qid = item["question_id"]
        if qid in adjudications or item.get("current_support_label") != "conflicting":
            raise ValueError(f"invalid or duplicate agent support adjudication: {qid}")
        adjudications[qid] = {
            "label": item["proposed_label"],
            "rationale_zh": item["rationale_zh"],
            "adjudicator": "independent batch reviewer; primary review incorporated",
            "full_context_reviewed": False,
        }

    for item in root_rows:
        qid = item["question_id"]
        if qid in adjudications or item.get("source_support_label") != "conflicting":
            raise ValueError(f"invalid or duplicate primary support adjudication: {qid}")
        adjudications[qid] = {
            "label": item["adjudicated_support_label"],
            "rationale_zh": item["adjudication_basis_zh"],
            "adjudicator": "primary reviewer",
            "full_context_reviewed": False,
        }

    deeper_ids = {item["question_id"] for item in agent_rows if item.get("needs_deeper_review")}
    override_ids = {item["question_id"] for item in override_rows}
    if deeper_ids != override_ids:
        raise ValueError(f"primary deep-review coverage mismatch ({len(override_ids)} vs {len(deeper_ids)})")
    for item in override_rows:
        qid = item["question_id"]
        if qid not in adjudications or item.get("source_support_label") != "conflicting":
            raise ValueError(f"invalid primary override: {qid}")
        adjudications[qid] = {
            "label": item["adjudicated_support_label"],
            "rationale_zh": item["adjudication_basis_zh"],
            "adjudicator": "primary reviewer; full observation and question rechecked",
            "full_context_reviewed": bool(item.get("deep_context_reviewed")),
            "support_map_replacements": item.get("support_map_replacements", []),
        }

    if len(adjudications) != len(agent_rows) + len(root_rows) or set(adjudications) != conflict_ids:
        raise ValueError(f"support adjudication coverage mismatch ({len(adjudications)} vs {len(conflict_ids)})")
    allowed = {"sufficient", "partial", "uncertain", "conflicting", "none", "unavailable"}
    changed = 0
    for row in combined:
        item = adjudications.get(row["question_id"])
        if item is None:
            continue
        if row.get("support_label") != "conflicting" or item["label"] not in allowed:
            raise ValueError(f"support adjudication source/label mismatch: {row['question_id']}")
        row["support_label"] = item["label"]
        row["visible_support"] = item["label"]
        for replacement in item.get("support_map_replacements", []):
            index = replacement["index"]
            if index < 0 or index >= len(row.get("support_map", [])):
                raise ValueError(f"support map replacement index out of range: {row['question_id']}")
            for key in ("exact_quote", "supported_fact", "rationale_zh"):
                if key in replacement:
                    row["support_map"][index][key] = replacement[key]
        row["primary_support_adjudication"] = item
        changed += 1

    return {
        "count": changed,
        "agent_first_pass_count": len(agent_rows),
        "primary_first_pass_count": len(root_rows),
        "primary_deep_review_count": len(override_rows),
        "sha256": {
            "agent_first_pass": sha256(agent_path),
            "primary_first_pass": sha256(root_path),
            "primary_deep_review": sha256(override_path),
        },
    }


def validate_semantic_screen(priority_ids: list[str], isolation_counts: dict) -> dict:
    manifest_path = FULL / "semantic_duplicate_queue.manifest.json"
    queue_path = FULL / "semantic_duplicate_queue.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        queue_rows = read_jsonl(queue_path)
        expected_review_hashes = {
            f"batch_{batch:02d}": sha256(FULL / f"review_batch_{batch:02d}.jsonl")
            for batch in range(1, 5)
        }
        expected_history_count = isolation_counts["frozen_dev"] + isolation_counts["frozen_final_test"]
        checks = {
            "candidate_count": manifest.get("candidate_count") == len(priority_ids),
            "candidate_ids": manifest.get("candidate_id_sha256") == hashlib.sha256("\n".join(sorted(priority_ids)).encode()).hexdigest(),
            "accepted": manifest.get("accepted_sha256") == sha256(ACCEPTED),
            "reviews": manifest.get("review_sha256") == expected_review_hashes,
            "pilot_review": manifest.get("pilot_review_sha256") == sha256(PILOT / "review.jsonl"),
            "frozen_splits": manifest.get("history_sha256") == {"dev": sha256(DEV), "final_test": sha256(TEST)},
            "history_count": manifest.get("history_count") == expected_history_count,
            "model": manifest.get("model_sha256") == sha256(SEMANTIC_MODEL / "pytorch_model.bin"),
            "tokenizer": manifest.get("tokenizer_sha256") == sha256(SEMANTIC_MODEL / "tokenizer.json"),
            "threshold": manifest.get("threshold") == 0.92,
            "queue_hash": manifest.get("queue_sha256") == sha256(queue_path),
            "queue_count": manifest.get("matches") == len(queue_rows),
        }
        valid = all(checks.values())
        return {
            "valid": valid,
            "complete": valid and manifest.get("matches") == 0 and not queue_rows,
            "candidate_count": manifest.get("candidate_count"),
            "history_count": manifest.get("history_count"),
            "matches": manifest.get("matches"),
            "threshold": manifest.get("threshold"),
            "method": manifest.get("method"),
            "queue_count": len(queue_rows),
            "checks": checks,
            "manifest_sha256": sha256(manifest_path),
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {"valid": False, "complete": False, "error": str(exc)}


def validate_manual_gate_review(combined: list[dict], recommendations: dict, source_by_id: dict) -> dict:
    path = FULL / "manual_gate_review_root_20260930.jsonl"
    rows = read_jsonl(path)
    exclusions = [row for row in rows if row.get("review_type") == "exclusion_confirmation"]
    priority_checks = [row for row in rows if row.get("review_type") == "priority_spot_check"]
    review_by_id = {row["question_id"]: row for row in combined}
    if len(exclusions) != 12 or {row["question_id"] for row in exclusions} != set(recommendations["exclude"]):
        raise ValueError("primary manual exclusion review does not cover the complete exclude list")
    if len(priority_checks) != 5 or not {row["question_id"] for row in priority_checks} <= set(recommendations["priority"]):
        raise ValueError("primary priority spot-check must cover five current priority records")
    for item in exclusions:
        if item.get("decision") != "confirm_exclude" or not item.get("rationale_zh"):
            raise ValueError(f"incomplete exclusion review: {item.get('question_id')}")
    for item in priority_checks:
        qid = item["question_id"]
        review = review_by_id[qid]
        ref = item.get("support_map_ref", {})
        matching_maps = [
            entry for entry in review.get("support_map", [])
            if entry.get("doc_label") == ref.get("doc_label")
            and entry.get("query") == ref.get("query")
            and entry.get("document_hash") == ref.get("document_hash")
        ]
        if item.get("decision") != "confirm_priority" or not item.get("rationale_zh") or not matching_maps:
            raise ValueError(f"incomplete priority spot-check: {qid}")
        if not quote_maps_to_visible_source(source_by_id[qid], matching_maps[0]):
            raise ValueError(f"priority spot-check source is not visible in original observation: {qid}")
        if review.get("sft_recommendation") != "priority" or review.get("support_label") != "sufficient":
            raise ValueError(f"priority spot-check no longer meets the high-precision gate: {qid}")
    return {
        "exclude_confirmed": len(exclusions),
        "priority_spot_checks": len(priority_checks),
        "sha256": sha256(path),
    }


def quote_maps_to_visible_source(source: dict, entry: dict) -> bool:
    label_match = re.fullmatch(r"Doc\s+(\d+)", str(entry.get("doc_label", "")))
    doc_index = int(entry.get("doc_index") or (label_match.group(1) if label_match else 0))
    query = str(entry.get("query", "")).strip()
    quote = str(entry.get("exact_quote", ""))
    expected_hash = entry.get("document_hash")
    if not doc_index or not query or not quote or not expected_hash:
        return False
    hash_ok = any(
        search.get("query", "").strip() == query
        and len(search.get("document_hashes", [])) >= doc_index
        and search["document_hashes"][doc_index - 1] == expected_hash
        for search in source.get("searches", [])
    )
    if not hash_ok:
        return False
    response = source.get("response", "")
    # The teacher response can contain literal `<search>` examples while
    # explaining that an earlier action was invalid. Use an overlapping scan
    # so a later, real search block is still considered after such examples.
    for pair in re.finditer(r"(?=(<search>(.*?)</search>\s*<information>(.*?)</information>))", response, re.S):
        if pair.group(2).strip() != query:
            continue
        info = pair.group(3)
        documents = list(re.finditer(r"Doc\s+(\d+)\s*\(Title:", info))
        for index, document in enumerate(documents):
            if int(document.group(1)) != doc_index:
                continue
            end = documents[index + 1].start() if index + 1 < len(documents) else len(info)
            if quote in info[document.start():end]:
                return True
    return False


def main() -> None:
    accepted = read_jsonl(ACCEPTED)
    accepted_by_id = {row["question_id"]: row for row in accepted}
    accepted_ids = set(accepted_by_id)
    attempts_ids = {row["question_id"] for row in read_jsonl(ATTEMPTS)}
    cold_ids = {row["question_id"] for row in read_jsonl(COLD)}
    if len(accepted_by_id) != len(accepted) or len(accepted) != 886:
        raise ValueError("accepted SFT input count/ID uniqueness changed")
    if len(accepted_ids - attempts_ids) or len(accepted_ids - cold_ids):
        raise ValueError("accepted SFT rows no longer cross-match attempts and frozen cold_start")
    pilot = read_jsonl(PILOT / "review.jsonl")
    pilot_review = [row for row in pilot if row["question_id"] in accepted_ids]
    pilot_ids = {row["question_id"] for row in pilot_review}

    sources: dict[int, dict[str, dict]] = {}
    inputs: dict[int, set[str]] = {}
    reviews: dict[int, list[dict]] = {}
    for batch in range(1, 5):
        rows = read_jsonl(FULL / "inputs" / f"input_batch_{batch:02d}.jsonl")
        sources[batch] = {row["question_id"]: row for row in rows}
        inputs[batch] = set(sources[batch])
        reviews[batch] = read_jsonl(FULL / f"review_batch_{batch:02d}.jsonl")
        expected = inputs[batch] - pilot_ids
        actual = {row["question_id"] for row in reviews[batch]}
        if len(actual) != len(reviews[batch]) or actual != expected:
            raise ValueError(f"batch {batch}: review coverage mismatch ({len(actual)} vs {len(expected)})")
    all_input_ids = [qid for batch in range(1, 5) for qid in inputs[batch]]
    if len(all_input_ids) != len(set(all_input_ids)) or set(all_input_ids) != accepted_ids:
        raise ValueError("frozen full-census input partition does not match accepted IDs")

    combined = pilot_review + [row for batch in range(1, 5) for row in reviews[batch]]
    combined_ids = [row["question_id"] for row in combined]
    if len(combined_ids) != len(set(combined_ids)) or set(combined_ids) != accepted_ids:
        raise ValueError(f"full review coverage mismatch ({len(set(combined_ids))}/{len(accepted_ids)})")
    support_adjudication = apply_support_adjudications(combined)

    source_by_id = {qid: row for batch_rows in sources.values() for qid, row in batch_rows.items()}
    citation_issues, schema_issues, priority_issues = [], [], []
    for review in combined:
        missing = REQUIRED - review.keys()
        if missing:
            schema_issues.append((review["question_id"], sorted(missing)))
        source = source_by_id.get(review["question_id"], accepted_by_id.get(review["question_id"]))
        maps = review.get("support_map") or []
        for entry in maps:
            if not quote_maps_to_visible_source(source, entry):
                citation_issues.append((review["question_id"], entry))
        unavailable = review.get("support_label") == "unavailable" and review.get("environment_status") == "unavailable"
        source_has_no_search = not source.get("searches") and not source.get("search_count", 0)
        no_search_record = review.get("support_label") == "none" and source_has_no_search
        if not maps and not (unavailable or no_search_record):
            citation_issues.append((review["question_id"], "missing support map"))
        if review.get("sft_recommendation") == "priority" and not (
            review.get("question_decision") == "keep"
            and review.get("support_label") == "sufficient"
            and review.get("trajectory_status") == "legal_answer"
            and review.get("teacher_answer_relation") == "equivalent"
            and not review.get("exact_qa_leak")
            and not review.get("invalid_action")
            and not review.get("duplicate_search")
        ):
            priority_issues.append(review["question_id"])

    if citation_issues or schema_issues or priority_issues:
        raise ValueError(json.dumps({
            "citation_issues": citation_issues[:20], "citation_issue_count": len(citation_issues),
            "schema_issues": schema_issues[:20], "schema_issue_count": len(schema_issues),
            "priority_issues": priority_issues,
        }, ensure_ascii=False))

    isolation = check_split_isolation(
        {"accepted_sft": ACCEPTED}, {"frozen_dev": DEV, "frozen_final_test": TEST}
    )
    if not isolation["exact_check_passed"]:
        raise ValueError("exact dev/test isolation failed")

    recommendations = {name: sorted(row["question_id"] for row in combined if row["sft_recommendation"] == name)
                       for name in ("priority", "reserve", "exclude")}
    if sum(map(len, recommendations.values())) != len(accepted_ids):
        raise ValueError("recommendation lists are not mutually exclusive and complete")
    for name, ids in recommendations.items():
        (FULL / f"{name}_ids.json").write_text(
            json.dumps({"count": len(ids), "question_ids": ids}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manual_gate_review = validate_manual_gate_review(combined, recommendations, source_by_id)
    semantic = validate_semantic_screen(recommendations["priority"], isolation["counts"])
    semantic_status = (
        f"screen complete; {semantic['matches']} matches at cosine >= {semantic['threshold']}; manual queue empty"
        if semantic.get("complete")
        else "screen pending or stale; see semantic_duplicate_queue.manifest.json"
    )
    (FULL / "review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(combined, key=lambda x: x["question_id"])),
        encoding="utf-8",
    )

    fields = ("question_decision", "support_label", "trajectory_status", "teacher_answer_relation", "sft_recommendation")
    counters = {field: dict(Counter(row[field] for row in combined)) for field in fields}
    counters["invalid_action"] = {"true": sum(bool(row["invalid_action"]) for row in combined), "false": sum(not row["invalid_action"] for row in combined)}
    counters["duplicate_search"] = {"true": sum(bool(row["duplicate_search"]) for row in combined), "false": sum(not row["duplicate_search"] for row in combined)}
    hashes = {"accepted": sha256(ACCEPTED), "attempts": sha256(ATTEMPTS), "cold_start": sha256(COLD)}
    report = {
        "status": (
            "manual_review_exact_isolation_and_semantic_screen_complete"
            if semantic.get("complete")
            else "manual_review_and_exact_isolation_complete_semantic_review_pending"
        ),
        "review_count": len(combined), "pilot_accepted_overlap": len(pilot_ids),
        "pilot_attempts_only_excluded_from_full_census": len(pilot) - len(pilot_review),
        "review_counts": counters, "citation_issues": 0, "schema_issues": 0,
        "support_conflict_adjudication": support_adjudication,
        "manual_gate_review": manual_gate_review,
        "semantic_screen": semantic,
        "exact_dev_test_isolation": {
            "counts": isolation["counts"], "conflict_count": len(isolation["conflicts"]),
            "exact_check_passed": isolation["exact_check_passed"],
            "semantic_near_duplicates": semantic_status,
        },
        "input_sha256": hashes,
        "scope_limits": [
            "原始 full-return 未保存；full_return_support 标 not_saved，不能判断原 observation 的截断损失。",
            "全量只审阅已有轨迹，不对 reserve 批量追加诊断检索；不把 reserve 说成语料无证据。",
            "语义近重复阈值扫描仅用于候选队列，阈值以下不证明绝无近重复。",
            "此审阅是项目内人工审查，不是医生核验，不证明训练收益或临床有效性。",
        ],
        "training_export_created": False,
        "training_ready": False,
    }
    (FULL / "manifest.json").write_text(
        json.dumps({**json.loads((FULL / "manifest.json").read_text(encoding="utf-8")), "status": report["status"], "review_summary": report}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table_lines = []
    for field, values in counters.items():
        for label, count in values.items():
            table_lines.append(f"| `{field}={label}` | {count}/{len(combined)} ({100 * count / len(combined):.1f}%) |")
    table = "\n".join(["| 项目 | 数量 / 全量 |", "|---|---:|"] + table_lines)
    if semantic.get("complete"):
        semantic_description = (
            f"在 {semantic['candidate_count']} 条 priority 候选和 {semantic['history_count']} 条冻结 dev/test 记录上，"
            f"使用 {semantic['method']}、cosine >= {semantic['threshold']} 筛查得到 {semantic['matches']} 个匹配，人工队列为空。"
            "该阈值筛查不等于穷尽证明。"
        )
    else:
        semantic_description = "语义近重复筛查尚未通过清单哈希核验，或仍有待人工裁定项；请查看 semantic_duplicate_queue.manifest.json。"
    (FULL / "report.md").write_text(
        "# 全量 SFT 医学质量与检索环境复审\n\n"
        "日期：2026-09-30。范围：固定 `accepted.jsonl` 中 886 条已有 search_distill 轨迹；源 JSONL 未改写。\n\n"
        f"## 总体计数\n\n- 已审唯一记录：{len(combined)}/886。\n- 试点中属于这 886 条的记录：{len(pilot_ids)}；仅来自 attempts、未进入 accepted 的试点边界项：{len(pilot)-len(pilot_review)}（不计入全量）。\n\n{table}\n\n"
        "## 候选使用规则\n\n"
        f"- `priority_ids.json`：{len(recommendations['priority'])} 条高精度候选，要求题面可用、原始可见 evidence 直接充分、teacher answer 与 gold 等价、合法最终答案、无非法动作、无完全重复 query、无已识别问答泄漏。\n"
        f"- `reserve_ids.json`：{len(recommendations['reserve'])} 条保留；可能是题目未决、证据 partial/none/conflicting/unavailable、答案关系未核实，或轨迹动作质量不达优先门槛。reserve 不等于坏题。\n"
        f"- `excluded_ids.json`：{len(recommendations['exclude'])} 条由题面缺失或问答对象矛盾等直接问题排除；原始行仍保留。\n\n"
        "## 冲突标签主审裁定\n\n"
        f"对原标记为 conflicting 的 {support_adjudication['count']} 条逐条复核：{counters['support_label'].get('conflicting', 0)} 条保留 conflicting，其余按证据相对 gold 的实际关系改标。独立初审 {support_adjudication['agent_first_pass_count']} 条，主审初审 {support_adjudication['primary_first_pass_count']} 条，其中 {support_adjudication['primary_deep_review_count']} 条回看完整题干与 observation。另主审确认 {manual_gate_review['exclude_confirmed']} 条排除项，并抽检 {manual_gate_review['priority_spot_checks']} 条 priority。逐条依据保存在复审目录的审查记录中。\n\n"
        "## 隔离与限制\n\n"
        f"对原始 886 条与冻结 dev（396）和 MIRAGE final test（5,456）做 ID/规范化题干精确核对：冲突 {len(isolation['conflicts'])}，通过。语义近重复筛查：{semantic_description}\n\n"
        "原始 BM25 full-return 未保存；全量只审阅旧轨迹的模型可见 observation，没有对所有 reserve 新跑补查，故 none/partial 只表示当前返回片段未支持答案，不表示整个语料没有证据。新增诊断检索若未来生成示范，必须单独生成并验收完整轨迹。\n\n"
        "本报告是项目内人工审阅，不是医生核验；没有生成新的训练文件、没有覆盖 accepted.jsonl、没有启动 SFT，也不证明模型能力提升。该质量审查不等于训练导出许可或模型效果验证。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
