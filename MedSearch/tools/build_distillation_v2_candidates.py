"""Rebuild a traceable V2 candidate pool without declaring medical approval."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from MedSearch.tools.check_distillation_inputs import _normalize, _records


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "FreedomIntelligence/medical-o1-verifiable-problem"
SOURCE_REVISION = "0b347655bd332537e6814d53c6fe4bc03b555bdc"


def _hard_quality_reasons(question: str, answer: str) -> list[str]:
    """Frozen conservative V1 rule screen; flags remain reviewable in V2."""
    q, a = question.lower().strip(), answer.strip()
    reasons = []
    if not q or not a:
        reasons.append("empty_question_or_answer")
    if len(a) > 120 or len(a.split()) > 18:
        reasons.append("answer_too_long_for_strict_em")
    if len(a) <= 1:
        reasons.append("answer_too_short")
    if re.search(r"\b(yes|no|normal|none|unknown|all of the above)\b", a.lower()):
        reasons.append("non_specific_answer")
    if re.search(r"\b(or|either|and/or)\b|/", a.lower()):
        reasons.append("multiple_answer_forms")
    if re.search(r"\b(which of the following|among the following|select|choose|all are true|some accurate statements)\b", q):
        reasons.append("option_dependent_question")
    if re.search(r"\b(as shown|shown below|image|figure|table|photograph|x[- ]?ray|radiograph|scan below|ecg shows|lab value of)\b", q):
        reasons.append("missing_visual_or_key_measurement_context")
    if re.search(r"\b(historically|in the past|old guideline|outdated|previous guideline)\b", q):
        reasons.append("potentially_outdated_management")
    if re.search(r"\b(explain|discuss|describe|why does|how does|what are the reasons|give examples)\b", q):
        reasons.append("open_explanation_question")
    if re.search(r"[.!?]", a) or a.count(",") >= 3 or ":" in a:
        reasons.append("answer_not_compact")
    if re.search(r"\b(all|various|several|multiple)\b", a.lower()) and len(a.split()) < 5:
        reasons.append("non_unique_answer")
    return sorted(set(reasons))


def select_candidates(
    raw: list[dict], existing_ids: set[str], existing_questions: set[str], limit: int
) -> tuple[list[dict], list[dict]]:
    """Keep clean first-pass candidates; retain every excluded row with a reason."""
    kept, other, seen = [], [], set()
    for index, source_row in enumerate(raw):
        question = source_row.get("Open-ended Verifiable Question")
        answer = source_row.get("Ground-True Answer")
        item_id = f"medical-o1-verifiable:{index}"
        reasons = []
        if not isinstance(question, str) or not isinstance(answer, str):
            reasons.append("invalid_source_fields")
            question = question if isinstance(question, str) else ""
            answer = answer if isinstance(answer, str) else ""
        key = _normalize(question)
        if item_id in existing_ids:
            reasons.append("existing_id")
        if key and key in existing_questions:
            reasons.append("existing_question")
        if key and key in seen:
            reasons.append("internal_exact_duplicate")
        if key:
            seen.add(key)
        reasons.extend(_hard_quality_reasons(question, answer))
        record = {
            "question_id": item_id,
            "source": SOURCE,
            "source_id": index,
            "question": question,
            "references": [answer],
            "split": "unassigned",
        }
        if not reasons and len(kept) < limit:
            kept.append(record)
        else:
            status = "excluded" if any(reason.startswith(("existing_", "internal_exact", "invalid_source")) for reason in reasons) else "quarantine"
            if not reasons:
                status = "reserve"
                reasons = ["candidate_limit_reached"]
            other.append({**record, "status": status, "reasons": sorted(set(reasons))})
    return kept, other


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "MedSearch/data/medical_verifiable_v2_20260919",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    source_path = ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    boundaries = {
        "rl_train": ROOT / "MedSearch/data/rl/train.parquet",
        "rl_dev": ROOT / "MedSearch/data/rl/dev.parquet",
        "final_test": ROOT / "MIRAGE/benchmark.json",
    }
    missing = [str(path) for path in (source_path, *boundaries.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required input assets missing: {missing}")
    existing_ids, existing_questions = set(), set()
    boundary_counts = {}
    for name, path in boundaries.items():
        count = 0
        for record in _records(name, path):
            count += 1
            if record["id"] is not None:
                existing_ids.add(record["id"])
            key = _normalize(record["question"])
            if key:
                existing_questions.add(key)
        boundary_counts[name] = count

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("raw medical question file must be a list")
    kept, other = select_candidates(raw, existing_ids, existing_questions, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.output_dir / "candidate_keep.jsonl"
    other_path = args.output_dir / "candidate_other.jsonl"
    _write_jsonl(kept_path, kept)
    _write_jsonl(other_path, other)
    status_counts = Counter(row["status"] for row in other)
    manifest = {
        "status": "candidate_only_pending_near_duplicate_and_medical_review",
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "source_sha256": _sha256(source_path),
        "source_rows": len(raw),
        "candidate_rows": len(kept),
        "other_rows": len(other),
        "other_status_counts": dict(status_counts),
        "boundary_counts": boundary_counts,
        "boundary_sha256": {name: _sha256(path) for name, path in boundaries.items()},
        "outputs_sha256": {path.name: _sha256(path) for path in (kept_path, other_path)},
        "selection": "source order after exact ID/text isolation and conservative V1 rule screen",
        "lexical_near_duplicates": "not_checked_in_v2",
        "semantic_near_duplicates": "not_checked",
        "medical_review": "not_done",
        "split_assignment": "not_done",
        "training_ready": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "candidate_rows": len(kept),
        "other_status_counts": dict(status_counts),
        "training_ready": False,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
