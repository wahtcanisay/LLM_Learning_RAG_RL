"""Freeze a new high-isolation HuatuoGPT-o1 candidate batch; not a training export."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "MedSearch/data/medical_verifiable_v2_20260919"
BATCH2 = ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926"
POOLS = ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean"
SOURCE = "FreedomIntelligence/medical-o1-verifiable-problem"
SOURCE_REVISION = "0b347655bd332537e6814d53c6fe4bc03b555bdc"
SEED = 20260930
COUNT = 1000
OUT = ROOT / "MedSearch/data/medical_verifiable_v2_batch3_20260930"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def question_texts(row: dict) -> list[str]:
    for key in ("question", "Question", "instruction", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    for key in ("messages", "conversations"):
        messages = row.get(key)
        if isinstance(messages, list):
            found, fallback_system = [], []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", message.get("from", message.get("speaker", "")))).casefold()
                value = message.get("content", message.get("value"))
                if not isinstance(value, str) or not value.strip():
                    continue
                if role in {"user", "human", "instruction"}:
                    found.append(value)
                elif role == "system":
                    fallback_system.append(value)
            if found:
                return found
            if fallback_system:
                return fallback_system
    nested = row.get("data")
    if isinstance(nested, dict):
        return question_texts(nested)
    return []


def read_questions(name: str, path: Path) -> list[dict]:
    rows = []
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        for number, row in enumerate(table.to_pylist(), 1):
            for question in question_texts(row):
                rows.append({"id": str(row.get("question_id") or row.get("id") or row.get("source_id") or f"{name}:{number}"), "question": question, "pool": name})
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for question in question_texts(row):
                    rows.append({"id": str(row.get("question_id") or row.get("id") or row.get("source_id") or f"{name}:{number}"), "question": question, "pool": name})
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            iterable = enumerate(data, 1)
        elif isinstance(data, dict) and {"medqa", "medmcqa"} <= data.keys():
            iterable = ((f"{split}:{key}", row) for split in ("medqa", "medmcqa") for key, row in data[split].items())
        else:
            raise ValueError(f"unsupported JSON question file: {path}")
        for number, row in iterable:
            if not isinstance(row, dict):
                continue
            for question in question_texts(row):
                rows.append({"id": str(row.get("question_id") or row.get("id") or row.get("source_id") or f"{name}:{number}"), "question": question, "pool": name})
    else:
        raise ValueError(f"unsupported source file: {path}")
    if not rows:
        if path.suffix == ".parquet":
            detail = pq.read_schema(path).names
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                first = next((json.loads(line) for line in handle if line.strip()), {})
            detail = sorted(first.keys()) if isinstance(first, dict) else type(first).__name__
        else:
            detail = ""
        raise ValueError(f"no user questions read from {path}; fields={detail}")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_paths() -> dict[str, Path]:
    paths = {
        "candidate_keep_batch1": V2 / "candidate_keep.jsonl",
        "candidate_batch2": BATCH2 / "candidate_batch2.jsonl",
        "cold_start_previous": POOLS / "cold_start.jsonl",
        "base_rl_train": ROOT / "MedSearch/data/rl/train.parquet",
        "base_rl_dev": ROOT / "MedSearch/data/rl/dev.parquet",
        "new_rl_train": POOLS / "rl_train.parquet",
        "new_dev": POOLS / "dev.parquet",
        "final_test": ROOT / "MIRAGE/benchmark.json",
        # Keep these origin datasets in the semantic history. They are flagged
        # as source references rather than treated as proven model exposure.
        "raw_medqa_train": ROOT / "MedSearch/data/raw/medqa/train.parquet",
        "raw_medmcqa_train": ROOT / "MedSearch/data/raw/medmcqa/train.parquet",
    }
    for path in sorted((ROOT / "MedSearch/data/raw/sft").glob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            paths[f"raw_sft_{path.stem}"] = path
    expansion = ROOT / "MedSearch/data/rl/medical_verifiable_expansion_v1/audited_keep.jsonl"
    if expansion.is_file():
        paths["rl_expansion_v1"] = expansion
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required isolation sources missing: " + ", ".join(missing))
    return paths


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {OUT}")
    source_manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    batch2_manifest = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    pool_manifest = json.loads((POOLS / "manifest.json").read_text(encoding="utf-8"))
    raw_path = ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    reserve_path = V2 / "candidate_other.jsonl"
    if source_manifest["source"] != SOURCE or source_manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("unexpected source identity/revision")
    for path, expected in (
        (raw_path, source_manifest["source_sha256"]),
        (reserve_path, source_manifest["outputs_sha256"]["candidate_other.jsonl"]),
        (V2 / "candidate_keep.jsonl", source_manifest["outputs_sha256"]["candidate_keep.jsonl"]),
        (BATCH2 / "candidate_batch2.jsonl", batch2_manifest["candidate_sha256"]),
        (POOLS / "cold_start.jsonl", pool_manifest["output_sha256"]["cold_start.jsonl"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen input SHA256 changed: {path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    reserve = [row for row in read_jsonl(reserve_path) if row.get("status") == "reserve"]
    if len(reserve) != source_manifest["other_status_counts"]["reserve"]:
        raise ValueError("reserve status count differs from frozen manifest")
    histories = source_paths()
    history_rows = [record for name, path in histories.items() for record in read_questions(name, path)]
    history_by_id: dict[str, set[str]] = defaultdict(set)
    history_by_question: dict[str, set[str]] = defaultdict(set)
    for row in history_rows:
        key = normalize(row["question"])
        if row["id"]:
            history_by_id[row["id"]].add(row["pool"])
        if key:
            history_by_question[key].add(row["pool"])

    eligible, exact_excluded = [], Counter()
    seen_candidate_ids, seen_candidate_questions = set(), set()
    for row in reserve:
        index = row.get("source_id")
        if not isinstance(index, int) or not 0 <= index < len(raw) or row.get("question_id") != f"medical-o1-verifiable:{index}":
            raise ValueError(f"invalid source identity: {row.get('question_id')}")
        source = raw[index]
        answer = source.get("Ground-True Answer")
        question = source.get("Open-ended Verifiable Question")
        if row.get("question") != question or row.get("references") != [answer]:
            raise ValueError(f"question/gold differs from raw source: {index}")
        key = normalize(question)
        item_id = row["question_id"]
        if item_id in history_by_id:
            exact_excluded["historical_id"] += 1
            continue
        if key in history_by_question:
            exact_excluded["historical_normalized_question"] += 1
            continue
        if item_id in seen_candidate_ids or key in seen_candidate_questions:
            exact_excluded["internal_exact_duplicate"] += 1
            continue
        seen_candidate_ids.add(item_id)
        seen_candidate_questions.add(key)
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
    candidate_path = OUT / "candidate_batch3.jsonl"
    with candidate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Store one standardized, de-duplicated history file for the existing MPNet screen.
    history_path = OUT / "semantic_history.jsonl"
    unique_history: dict[str, dict] = {}
    for row in history_rows:
        key = normalize(row["question"])
        if key:
            unique_history.setdefault(key, row)
    with history_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unique_history.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Recheck exact isolation against the frozen, normalized history artifact.
    candidate_ids = {row["question_id"] for row in candidates}
    candidate_keys = {normalize(row["question"]) for row in candidates}
    history_ids = {row["id"] for row in unique_history.values() if row.get("id")}
    history_keys = set(unique_history)
    if candidate_ids & history_ids or candidate_keys & history_keys:
        raise AssertionError("exact isolation failed after write")
    if len(candidate_ids) != COUNT or len(candidate_keys) != COUNT:
        raise AssertionError("candidate count or internal uniqueness failed")

    input_hashes = {name: sha256(path) for name, path in histories.items()}
    report = {
        "status": "candidate_only_pending_semantic_pair_review_medical_review_and_teacher_generation",
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "raw_source_sha256": sha256(raw_path),
        "reserve_input_sha256": sha256(reserve_path),
        "reserve_rows": len(reserve),
        "exact_excluded_counts": dict(exact_excluded),
        "exact_isolated_reserve_rows": len(eligible),
        "candidate_rows": len(candidates),
        "selection_seed": SEED,
        "selection_rule": "SHA256 ASCII(seed:source_id), lexicographically smallest 1000 among exact-isolated reserve; output ascending source_id",
        "candidate_sha256": sha256(candidate_path),
        "semantic_history_path": str(history_path.relative_to(ROOT)),
        "semantic_history_rows_before_dedup": len(history_rows),
        "semantic_history_rows_after_exact_dedup": len(unique_history),
        "semantic_history_sha256": sha256(history_path),
        "history_source_counts": dict(Counter(row["pool"] for row in history_rows)),
        "history_input_sha256": input_hashes,
        "exact_isolation": "passed_against_prior_SFT_RL_dev_test_and_candidate_batches",
        "semantic_screen": "pending; MPNet cosine >=0.92 is a manual review queue, not automatic exclusion",
        "question_gold_review": "pending; source gold is not expert verified",
        "teacher_generation": "pending; no SFT trace is created by this candidate builder",
        "training_ready": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
