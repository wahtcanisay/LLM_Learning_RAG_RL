"""Build a small, frozen veRL dataset from Medical Verifiable Problems."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = "FreedomIntelligence/medical-o1-verifiable-problem"
SOURCE_REVISION = "0b347655bd332537e6814d53c6fe4bc03b555bdc"
SOURCE_FILE = "medical_o1_verifiable_problem.json"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/{SOURCE_REVISION}/{SOURCE_FILE}"
SOURCE_PAPER = "https://arxiv.org/abs/2412.18925"
INSTRUCTION = (
    "Answer the given medical question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you lack some knowledge, call "
    "the search engine with <search>query</search>; retrieved results will appear between "
    "<information> and </information>. You can search as many times as you want. If you find no "
    "further external knowledge needed, provide only the concise final answer inside <answer> "
    "and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_record(question: str, answer: str, index: int, split: str) -> dict:
    question = question.strip()
    answer = answer.strip()
    prompt = f"{INSTRUCTION}\nQuestion: {question}"
    return {
        "id": f"medical-o1-verifiable:{index}",
        "question": question,
        "golden_answers": [answer],
        "data_source": "medsearch_medical_verifiable",
        "ability": "medical_open_qa",
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": {"target": [answer]}},
        "extra_info": {
            "index": index,
            "split": split,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_split": "train",
        },
    }


def _length_stats(values: list[str]) -> dict[str, float | int]:
    return _number_stats([len(value) for value in values])


def _number_stats(values: list[int]) -> dict[str, float | int]:
    numbers = sorted(values)

    def percentile(q: float) -> float:
        position = (len(numbers) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(numbers[lower])
        return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)

    return {
        "min": numbers[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": numbers[-1],
    }


def build_audit(train: list[dict], dev: list[dict]) -> dict:
    records = train + dev
    ids_train = {row["id"] for row in train}
    ids_dev = {row["id"] for row in dev}
    question_keys = [normalize(row["question"]) for row in records]
    return {
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
            "paper": SOURCE_PAPER,
        },
        "integrity": {
            "train_count": len(train),
            "dev_count": len(dev),
            "disjoint_ids": ids_train.isdisjoint(ids_dev),
            "empty_questions": sum(not row["question"].strip() for row in records),
            "empty_answers": sum(not row["golden_answers"][0].strip() for row in records),
            "normalized_question_duplicates": len(question_keys) - len(set(question_keys)),
        },
        "lengths": {
            split: {
                "question_chars": _length_stats([row["question"] for row in rows]),
                "answer_chars": _length_stats([row["golden_answers"][0] for row in rows]),
            }
            for split, rows in (("train", train), ("dev", dev))
        },
    }


def select_manual_review(records: list[dict], target_count: int, seed: int) -> list[dict]:
    def flags(row: dict) -> list[str]:
        answer_length = len(row["golden_answers"][0])
        return [
            flag
            for condition, flag in ((answer_length <= 2, "short_answer"), (answer_length > 200, "long_answer"))
            if condition
        ]

    extreme = [row for row in records if flags(row)]
    ordinary = [row for row in records if not flags(row)]
    random.Random(seed).shuffle(ordinary)
    selected = extreme + ordinary[: max(0, target_count - len(extreme))]
    return [
        {
            "id": row["id"],
            "source_index": row["extra_info"]["index"],
            "question": row["question"],
            "answer": row["golden_answers"][0],
            "selection_flags": flags(row) or ["seeded_random"],
            "review_status": "pending",
            "question_valid": None,
            "answer_correct": None,
            "answer_unambiguous": None,
            "requires_missing_image_or_table": None,
            "notes": "",
        }
        for row in selected
    ]


def select_refill_review(records: list[dict], base_indices: set[int]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "source_index": row["extra_info"]["index"],
            "split": row["extra_info"]["split"],
            "question": row["question"],
            "answer": row["golden_answers"][0],
            "selection_flags": ["stable_refill"],
            "review_status": "pending",
            "question_valid": None,
            "answer_correct": None,
            "answer_unambiguous": None,
            "requires_missing_image_or_table": None,
            "notes": "",
        }
        for row in records
        if row["extra_info"]["index"] not in base_indices
    ]


def token_length_audit(
    train: list[dict], dev: list[dict], tokenizer, max_prompt_tokens: int
) -> dict:
    result = {
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "max_prompt_tokens": max_prompt_tokens,
    }
    for split, rows in (("train", train), ("dev", dev)):
        prompt_lengths = [
            len(tokenizer.encode(row["prompt"][0]["content"], add_special_tokens=False)) for row in rows
        ]
        answer_lengths = [
            len(tokenizer.encode(row["golden_answers"][0], add_special_tokens=False)) for row in rows
        ]
        result[split] = {
            "prompt_tokens": _number_stats(prompt_lengths),
            "answer_tokens": _number_stats(answer_lengths),
            "over_limit": sum(length > max_prompt_tokens for length in prompt_lengths),
        }
    return result


def find_lexical_near_duplicates(
    records: list[dict], pools: dict[str, list[str]], threshold: float
) -> list[dict]:
    entries = []
    postings: dict[str, set[int]] = defaultdict(set)
    for pool, questions in pools.items():
        for question in questions:
            tokens = set(normalize(question).split())
            if not tokens:
                continue
            entry_index = len(entries)
            entries.append((pool, question, tokens))
            for token in tokens:
                postings[token].add(entry_index)

    matches = []
    for row in records:
        candidate_tokens = set(normalize(row["question"]).split())
        possible = set().union(*(postings[token] for token in candidate_tokens if token in postings))
        for entry_index in possible:
            pool, question, reference_tokens = entries[entry_index]
            score = len(candidate_tokens & reference_tokens) / len(candidate_tokens | reference_tokens)
            if score >= threshold:
                matches.append(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "pool": pool,
                        "matched_question": question,
                        "jaccard": round(score, 6),
                    }
                )
    return sorted(matches, key=lambda item: (item["id"], item["pool"], -item["jaccard"]))


def select_records(
    raw: list[dict],
    forbidden: set[str],
    train_count: int,
    dev_count: int,
    seed: int,
    excluded_indices: set[int] | None = None,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    candidates = []
    seen = set()
    excluded_indices = excluded_indices or set()
    rejected = {"seen": 0, "invalid": 0, "duplicate": 0, "excluded": 0}
    for index, item in enumerate(raw):
        if index in excluded_indices:
            rejected["excluded"] += 1
            continue
        question = item.get("Open-ended Verifiable Question")
        answer = item.get("Ground-True Answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            rejected["invalid"] += 1
            continue
        key = normalize(question)
        if key in forbidden:
            rejected["seen"] += 1
            continue
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        candidates.append((question, answer, index))

    required = train_count + dev_count
    if len(candidates) < required:
        raise ValueError(f"need {required} clean rows, found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    chosen = candidates[:required]
    train = [build_record(*row, "train") for row in chosen[:train_count]]
    dev = [build_record(*row, "dev") for row in chosen[train_count:]]
    return train, dev, rejected


def select_records_with_stable_refill(
    raw: list[dict],
    forbidden: set[str],
    train_count: int,
    dev_count: int,
    seed: int,
    base_train_indices: list[int],
    base_dev_indices: list[int],
    excluded_indices: set[int],
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Keep accepted base rows in their split and fill holes from a fixed reserve."""
    if len(base_train_indices) != train_count or len(base_dev_indices) != dev_count:
        raise ValueError("base split sizes must match requested train/dev counts")
    if len(set(base_train_indices + base_dev_indices)) != train_count + dev_count:
        raise ValueError("base train/dev indices must be unique")

    candidates = []
    seen = set()
    rejected = {"seen": 0, "invalid": 0, "duplicate": 0, "excluded": 0}
    for index, item in enumerate(raw):
        question = item.get("Open-ended Verifiable Question")
        answer = item.get("Ground-True Answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            rejected["invalid"] += 1
            continue
        key = normalize(question)
        if key in forbidden:
            rejected["seen"] += 1
            continue
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        candidates.append((question, answer, index))

    random.Random(seed).shuffle(candidates)
    by_index = {row[2]: row for row in candidates}
    base_indices = set(base_train_indices + base_dev_indices)
    missing = base_indices - by_index.keys()
    if missing:
        raise ValueError(f"base rows are not eligible under current filters: {sorted(missing)}")

    rejected["excluded"] = len(excluded_indices & by_index.keys())
    retained_train = [by_index[index] for index in base_train_indices if index not in excluded_indices]
    retained_dev = [by_index[index] for index in base_dev_indices if index not in excluded_indices]
    reserve = [
        row for row in candidates if row[2] not in base_indices and row[2] not in excluded_indices
    ]
    train_holes = train_count - len(retained_train)
    dev_holes = dev_count - len(retained_dev)
    if len(reserve) < train_holes + dev_holes:
        raise ValueError(f"need {train_holes + dev_holes} reserve rows, found {len(reserve)}")

    train_rows = retained_train + reserve[:train_holes]
    dev_rows = retained_dev + reserve[train_holes : train_holes + dev_holes]
    train = [build_record(*row, "train") for row in train_rows]
    dev = [build_record(*row, "dev") for row in dev_rows]
    return train, dev, rejected


def forbidden_question_pools() -> dict[str, list[str]]:
    pools = {}
    benchmark_path = ROOT / "MIRAGE/benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        pools["final_test"] = [
            row["question"] for dataset in ("medqa", "medmcqa") for row in benchmark[dataset].values()
        ]
    return pools


def forbidden_questions() -> tuple[set[str], dict[str, int]]:
    pools = forbidden_question_pools()
    return {normalize(text) for texts in pools.values() for text in texts}, {name: len(texts) for name, texts in pools.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_excluded_indices(paths: list[Path]) -> set[int]:
    indices = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "source_index" in row:
                indices.add(int(row["source_index"]))
            else:
                indices.add(int(row["id"].rsplit(":", 1)[1]))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "MedSearch/data/raw/medical_o1_verifiable_problem.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "MedSearch/data/rl/medical_verifiable"
    )
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--dev-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--review-count", type=int, default=64)
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "models/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--exclude-findings", type=Path, action="append", default=[])
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        help="Keep accepted rows from these train/dev parquet files and only refill removed slots.",
    )
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    forbidden, pool_counts = forbidden_questions()
    excluded_indices = load_excluded_indices(args.exclude_findings)
    if args.base_data_dir:
        base_train_path = args.base_data_dir / "train.parquet"
        base_dev_path = args.base_data_dir / "dev.parquet"
        base_train = pq.read_table(base_train_path).to_pylist()
        base_dev = pq.read_table(base_dev_path).to_pylist()
        base_indices = {
            row["extra_info"]["index"] for row in base_train + base_dev
        }
        train, dev, rejected = select_records_with_stable_refill(
            raw,
            forbidden,
            args.train_count,
            args.dev_count,
            args.seed,
            [row["extra_info"]["index"] for row in base_train],
            [row["extra_info"]["index"] for row in base_dev],
            excluded_indices,
        )
    else:
        base_indices = set()
        train, dev, rejected = select_records(
            raw,
            forbidden,
            args.train_count,
            args.dev_count,
            args.seed,
            excluded_indices=excluded_indices,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    applied_exclusions_path = args.output_dir / "applied_exclusions.jsonl"
    applied_exclusions_path.write_text(
        "".join(
            json.dumps({"source_index": index}, ensure_ascii=False) + "\n"
            for index in sorted(excluded_indices)
        ),
        encoding="utf-8",
    )
    train_path = args.output_dir / "train.parquet"
    dev_path = args.output_dir / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(train), train_path)
    pq.write_table(pa.Table.from_pylist(dev), dev_path)
    audit = build_audit(train, dev)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    audit["token_lengths"] = token_length_audit(train, dev, tokenizer, args.max_prompt_tokens)
    lexical_matches = find_lexical_near_duplicates(
        train + dev, forbidden_question_pools(), threshold=0.80
    )
    audit["lexical_near_duplicates"] = {
        "metric": "token_set_jaccard",
        "threshold": 0.80,
        "count": len(lexical_matches),
    }
    review = select_manual_review(train, args.review_count, args.seed)
    audit["manual_review"] = {
        "split": "train",
        "target_count": args.review_count,
        "actual_count": len(review),
        "selection": "all answers with <=2 or >200 characters, then seeded random fill",
        "seed": args.seed,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manual_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review), encoding="utf-8"
    )
    refill_review = select_refill_review(train + dev, base_indices) if args.base_data_dir else []
    (args.output_dir / "refill_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in refill_review), encoding="utf-8"
    )
    (args.output_dir / "lexical_near_duplicates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lexical_matches), encoding="utf-8"
    )
    manifest = {
        "source": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "source_file": SOURCE_FILE,
        "source_fields": ["Open-ended Verifiable Question", "Ground-True Answer"],
        "paper": SOURCE_PAPER,
        "license": "apache-2.0",
        "seed": args.seed,
        "source_count": len(raw),
        "train_count": len(train),
        "dev_count": len(dev),
        "forbidden_pool_counts": pool_counts,
        "rejected": rejected,
        "exclusion_findings": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.exclude_findings
        ],
        "applied_exclusions": {
            "path": str(applied_exclusions_path),
            "count": len(excluded_indices),
            "sha256": _sha256(applied_exclusions_path),
        },
        "refill_review_count": len(refill_review),
        "base_data": (
            {
                "path": str(args.base_data_dir),
                "train_sha256": _sha256(args.base_data_dir / "train.parquet"),
                "dev_sha256": _sha256(args.base_data_dir / "dev.parquet"),
                "policy": "retain accepted base split members; refill holes from fixed seeded reserve",
            }
            if args.base_data_dir
            else None
        ),
        "input_sha256": _sha256(args.input),
        "train_sha256": _sha256(train_path),
        "dev_sha256": _sha256(dev_path),
        "reward": "medical verifier outcome reward; normalized exact match only for smoke",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
