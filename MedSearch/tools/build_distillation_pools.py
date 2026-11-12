"""Freeze weak-gold cold-start and RL pools without changing historical data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from MedSearch.tools.build_medical_verifiable_rl import INSTRUCTION, SOURCE_DATASET, SOURCE_REVISION
from MedSearch.tools.check_distillation_inputs import _normalize, _records, check_split_isolation


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260919


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _check_unique(rows: list[dict], label: str) -> None:
    ids = set()
    questions = set()
    for row in rows:
        item_id = row["id"]
        question = _normalize(row["question"])
        if not item_id or not question or item_id in ids or question in questions:
            raise ValueError(f"duplicate or empty {label} ID/question: {item_id}")
        ids.add(item_id)
        questions.add(question)


def _check_against(rows: list[dict], boundary: list[dict], label: str) -> None:
    ids = {row["id"] for row in boundary}
    questions = {_normalize(row["question"]) for row in boundary}
    for row in rows:
        if row["id"] in ids or _normalize(row["question"]) in questions:
            raise ValueError(f"{label} overlap: {row['id']}")


def build_pools(
    ledger_path: Path,
    old_train_path: Path,
    dev_path: Path,
    output_dir: Path,
    *,
    cold_count: int = 1000,
    history_paths: dict[str, Path] | None = None,
    ledger_manifest_path: Path | None = None,
    dev_pair_review_path: Path | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if ledger_manifest_path is not None:
        expected = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))["combined_ledger_sha256"]
        if _sha256(ledger_path) != expected:
            raise ValueError("combined ledger SHA256 differs from review manifest")

    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    decisions = Counter(row["decision"] for row in ledger)
    if set(decisions) - {"pending", "reject", "quarantine"}:
        raise ValueError(f"unexpected review decisions: {decisions}")
    eligible = [row for row in ledger if row["decision"] == "pending"]
    if len(eligible) < cold_count or cold_count < 0:
        raise ValueError("insufficient pending candidates for cold-start split")
    if any(row.get("screening_eligible", True) is not True for row in eligible):
        raise ValueError("pending row is not screening eligible")
    if any(not row.get("references") or not row.get("question") for row in eligible):
        raise ValueError("pending row has empty question or references")

    ranked = sorted(eligible, key=lambda row: (hashlib.sha256(f"{SEED}:{row['question_id']}".encode("ascii")).hexdigest(), row["question_id"]))
    cold_ids = {row["question_id"] for row in ranked[:cold_count]}
    old_table = pq.read_table(old_train_path)
    old_rows = old_table.to_pylist()
    dev_rows = pq.read_table(dev_path).to_pylist()
    excluded_old_ids = set()
    if dev_pair_review_path is not None:
        reviewed = [json.loads(line) for line in dev_pair_review_path.read_text(encoding="utf-8").splitlines() if line]
        old_ids = {row["id"] for row in old_rows}
        dev_ids = {row["id"] for row in dev_rows}
        reviewed_ids = set()
        for pair in reviewed:
            item_id = pair["train_id"]
            if item_id in reviewed_ids or item_id not in old_ids or pair["dev_id"] not in dev_ids:
                raise ValueError(f"invalid dev pair review identity: {item_id}")
            if pair["decision"] not in {"same_question", "different_question"} or not pair.get("reason"):
                raise ValueError(f"invalid dev pair review decision: {item_id}")
            reviewed_ids.add(item_id)
            if pair["decision"] == "same_question":
                excluded_old_ids.add(item_id)
        old_rows = [row for row in old_rows if row["id"] not in excluded_old_ids]
    new_rl = []
    cold = []
    assignments = []
    for row in eligible:
        item_id = row["question_id"]
        question = row["question"]
        answers = row["references"]
        split = "cold_start" if item_id in cold_ids else "rl_train"
        assignments.append({"question_id": item_id, "source_id": row["source_id"], "batch": row.get("batch"), "split": split})
        if split == "cold_start":
            cold.append({
                "question_id": item_id, "source": row.get("source", SOURCE_DATASET),
                "source_id": row["source_id"], "question": question,
                "references": answers, "split": split,
                "gold_status": "source_gold_not_expert_verified",
            })
        else:
            new_rl.append({
                "id": item_id, "question": question, "golden_answers": answers,
                "data_source": "medsearch_medical_verifiable", "ability": "medical_open_qa",
                "prompt": [{"role": "user", "content": f"{INSTRUCTION}\nQuestion: {question}"}],
                "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
                "extra_info": {
                    "index": row["source_id"], "split": "train",
                    "source_dataset": SOURCE_DATASET, "source_revision": SOURCE_REVISION,
                    "source_split": "train",
                },
            })

    cold_as_rows = [{"id": row["question_id"], "question": row["question"]} for row in cold]
    merged = old_rows + new_rl
    _check_unique(merged + cold_as_rows, "training")
    _check_unique(dev_rows, "dev")
    _check_against(merged + cold_as_rows, dev_rows, "dev")
    for name, path in (history_paths or {}).items():
        boundary = list(_records(name, path))
        _check_against(merged + cold_as_rows, boundary, name)

    output_dir.mkdir(parents=True)
    cold_path = output_dir / "cold_start.jsonl"
    cold_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cold), encoding="utf-8", newline="\n")
    assignments_path = output_dir / "new_assignments.jsonl"
    assignments_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in assignments), encoding="utf-8", newline="\n")
    rl_path = output_dir / "rl_train.parquet"
    pq.write_table(pa.Table.from_pylist(merged, schema=old_table.schema), rl_path)
    copied_dev_path = output_dir / "dev.parquet"
    shutil.copyfile(dev_path, copied_dev_path)
    if dev_pair_review_path is not None:
        shutil.copyfile(dev_pair_review_path, output_dir / "dev_pair_review.jsonl")

    audit = check_split_isolation(
        {"cold_start": cold_path, "rl_train": rl_path},
        {"dev": copied_dev_path, **(history_paths or {})},
    )
    if not audit["exact_check_passed"]:
        raise ValueError(f"written split isolation failed: {audit['conflicts'][:3]}")
    audit_path = output_dir / "exact_isolation.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {"cold_start": len(cold), "new_rl": len(new_rl), "old_rl": len(old_rows),
              "old_rl_excluded_for_dev": len(excluded_old_ids), "rl_total": len(merged), "dev": len(dev_rows)}
    manifest = {
        "status": "weak_gold_split_ready_not_medically_verified",
        "selection_seed": SEED,
        "selection_rule": "SHA256 ASCII(seed:question_id), smallest cold_count to cold_start; other pending to rl_train",
        "source_decisions": dict(decisions),
        "counts": counts,
        "gold_status": "original_source_gold_not_expert_verified",
        "medical_expert_review": "not_done",
        "near_duplicate_limit": "prior MPNet/lexical Top-1 review only; not exhaustive",
        "dev_pair_review": "manual question-text comparison, not medical gold review" if dev_pair_review_path else "not_provided",
        "exact_check_passed": True,
        "input_sha256": {"combined_ledger": _sha256(ledger_path), "old_rl_train": _sha256(old_train_path),
                         "dev": _sha256(dev_path),
                         **({"dev_pair_review": _sha256(dev_pair_review_path)} if dev_pair_review_path else {})},
        "output_sha256": {path.name: _sha256(path) for path in (cold_path, assignments_path, rl_path, copied_dev_path, audit_path,
                                                                 *((output_dir / "dev_pair_review.jsonl",) if dev_pair_review_path else ()))},
        "training_run_ready": False,
        "training_run_blockers": ["7B teacher identity/tokenizer check", "checkpoint save-reload-action smoke"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926/combined_review_ledger.jsonl")
    parser.add_argument("--ledger-manifest", type=Path, default=ROOT / "MedSearch/data/medical_verifiable_v2_batch2_20260926/review_ledger.manifest.json")
    parser.add_argument("--old-train", type=Path, default=ROOT / "MedSearch/data/rl/train.parquet")
    parser.add_argument("--dev", type=Path, default=ROOT / "MedSearch/data/rl/dev.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean")
    parser.add_argument("--cold-count", type=int, default=1000)
    parser.add_argument("--dev-pair-review", type=Path, default=ROOT / "MedSearch/docs/audits/distillation_dev_pair_review_20260926.jsonl")
    args = parser.parse_args()
    history = {
        "final_test": ROOT / "MIRAGE/benchmark.json",
    }
    manifest = build_pools(
        args.ledger, args.old_train, args.dev, args.output_dir,
        cold_count=args.cold_count, history_paths=history,
        ledger_manifest_path=args.ledger_manifest,
        dev_pair_review_path=args.dev_pair_review,
    )
    print(json.dumps({"counts": manifest["counts"], "exact_check_passed": manifest["exact_check_passed"], "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
