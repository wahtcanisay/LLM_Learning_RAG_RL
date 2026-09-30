"""Audit exact question and ID isolation before MedSearch distillation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def _normalize(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _rows(path: Path):
    if path.suffix == ".parquet":
        schema = pq.read_schema(path)
        columns = [name for name in ("id", "question_id", "source_id", "question") if name in schema.names]
        if "question" not in columns:
            raise ValueError(f"missing question column: {path}")
        for number, row in enumerate(pq.read_table(path, columns=columns).to_pylist(), 1):
            yield number, row
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield number, json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for number, row in enumerate(data, 1):
                yield number, row
        elif isinstance(data, dict) and {"medqa", "medmcqa"} <= data.keys():
            for split in ("medqa", "medmcqa"):
                for key, row in data[split].items():
                    yield f"{split}:{key}", row
        else:
            raise ValueError(f"unsupported JSON structure: {path}")
    else:
        raise ValueError(f"unsupported input format: {path}")


def _records(pool: str, path: Path):
    for row_number, row in _rows(path):
        if not isinstance(row, dict):
            raise ValueError(f"expected object at {path}:{row_number}")
        questions = [row["question"]] if "question" in row else [
            message["content"] for message in row.get("messages", []) if message.get("role") == "user"
        ]
        if not questions:
            raise ValueError(f"no question at {path}:{row_number}")
        item_id = row.get("question_id") or row.get("id") or row.get("source_id")
        for question in questions:
            if not isinstance(question, str):
                raise ValueError(f"non-text question at {path}:{row_number}")
            yield {
                "pool": pool,
                "path": str(path),
                "row": row_number,
                "id": str(item_id) if item_id is not None else None,
                "question": question,
            }


def check_split_isolation(
    candidates: dict[str, Path],
    history: dict[str, Path],
    required: dict[str, Path] | None = None,
) -> dict:
    """Compare candidate pools with each other and with historical boundaries.

    This is an exact ID/normalized-text audit, not a semantic similarity check.
    Missing assets make the result incomplete rather than silently passing.
    """
    if set(candidates) & set(history):
        raise ValueError("candidate and history pool names must be distinct")
    required = required or {}
    paths = {**candidates, **history}
    missing = [
        {"pool": pool, "path": str(path)}
        for pool, path in {**paths, **required}.items()
        if not path.exists()
    ]
    audited_files = {path.resolve() for path in candidates.values() if path.is_file()}
    unreviewed = [
        {"pool": pool, "path": str(path)}
        for pool, path in required.items()
        if path.exists() and (not path.is_file() or path.resolve() not in audited_files)
    ]
    counts = {}
    inputs = {}
    conflicts = []
    invalid_questions = []
    history_ids = defaultdict(list)
    history_questions = defaultdict(list)
    candidate_ids = defaultdict(list)
    candidate_questions = defaultdict(list)

    for pool, path in paths.items():
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"expected input file: {path}")
        with path.open("rb") as handle:
            inputs[pool] = {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
        records = list(_records(pool, path))
        counts[pool] = len(records)
        if pool in history:
            for record in records:
                if record["id"] is not None:
                    history_ids[record["id"]].append(record)
                key = _normalize(record["question"])
                if key:
                    history_questions[key].append(record)
                else:
                    invalid_questions.append(record)

    for pool, path in candidates.items():
        if not path.exists():
            continue
        for record in _records(pool, path):
            key = _normalize(record["question"])
            if not key:
                invalid_questions.append(record)
            matches = (
                ("id", record["id"], history_ids, candidate_ids),
                ("question", key, history_questions, candidate_questions),
            )
            for kind, value, historical, previous in matches:
                if not value:
                    continue
                for other in historical[value] + previous[value]:
                    conflicts.append({"kind": kind, "left": record, "right": other})
            if record["id"] is not None:
                candidate_ids[record["id"]].append(record)
            if key:
                candidate_questions[key].append(record)

    return {
        "counts": counts,
        "inputs": inputs,
        "missing": missing,
        "unreviewed": unreviewed,
        "invalid_questions": invalid_questions,
        "conflicts": conflicts,
        "semantic_near_duplicates": "not_checked",
        "exact_check_passed": bool(candidates) and not (missing or unreviewed or invalid_questions or conflicts),
    }


def check_v1_provenance(raw_path: Path, audited_path: Path, train_path: Path, dev_path: Path) -> dict:
    """Check that rule-screened V1 rows kept their original question and gold."""
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected raw question list: {raw_path}")
    formal = {}
    for path in (train_path, dev_path):
        if path.suffix == ".parquet":
            rows = pq.read_table(path, columns=["id", "question", "golden_answers"]).to_pylist()
        else:
            rows = (row for _, row in _rows(path))
        for row in rows:
            formal[row["id"]] = row
    issues = []
    seen = set()
    count = 0
    for _, row in _rows(audited_path):
        count += 1
        item_id = row["id"]
        index = row["source_index"]
        reasons = []
        if item_id in seen:
            reasons.append("duplicate_audited_id")
        seen.add(item_id)
        if not isinstance(index, int) or not 0 <= index < len(raw) or item_id != f"medical-o1-verifiable:{index}":
            reasons.append("invalid_source_identity")
        else:
            source = raw[index]
            if row["question"] != source.get("Open-ended Verifiable Question", "").strip():
                reasons.append("raw_question_mismatch")
            if row["answer"] != source.get("Ground-True Answer", "").strip():
                reasons.append("raw_answer_mismatch")
        if row["data"]["golden_answers"] != [row["answer"]]:
            reasons.append("embedded_answer_mismatch")
        current = formal.get(item_id)
        if current is None:
            reasons.append("missing_from_formal_rl")
        elif current["question"] != row["question"] or current["golden_answers"] != [row["answer"]]:
            reasons.append("formal_rl_content_mismatch")
        for reason in reasons:
            issues.append({"id": item_id, "reason": reason})
    return {
        "raw_rows": len(raw),
        "audited_rows": count,
        "issues": issues,
        "clinical_correctness": "not_checked",
    }


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"expected unique NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--history", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--require", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "MedSearch/docs/audits/distillation_split_isolation.json",
    )
    args = parser.parse_args()
    candidates = _named_paths(args.candidate) if args.candidate else {
        "existing_rl_train": ROOT / "MedSearch/data/rl/train.parquet",
        "existing_rl_dev": ROOT / "MedSearch/data/rl/dev.parquet",
    }
    history = _named_paths(args.history) if args.history else {
        "final_test": ROOT / "MIRAGE/benchmark.json",
    }
    required = _named_paths(args.require) if args.require else {
        "luna_v2": ROOT / "MedSearch/data/medical_verifiable_v2_20260919"
    }
    report = check_split_isolation(candidates, history, required)
    v1 = ROOT / "MedSearch/data/rl/medical_verifiable_expansion_v1/audited_keep.jsonl"
    raw = ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    if raw.exists() and v1.exists():
        report["v1_source_provenance"] = check_v1_provenance(
            raw, v1,
            ROOT / "MedSearch/data/rl/train.parquet",
            ROOT / "MedSearch/data/rl/dev.parquet",
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": report["counts"],
        "missing": report["missing"],
        "unreviewed": report["unreviewed"],
        "conflicts": len(report["conflicts"]),
        "invalid_questions": len(report["invalid_questions"]),
        "exact_check_passed": report["exact_check_passed"],
        "v1_source_issues": len(report.get("v1_source_provenance", {}).get("issues", [])),
        "report": str(args.report),
    }, ensure_ascii=False))
    if not report["exact_check_passed"] or report.get("v1_source_provenance", {}).get("issues"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
