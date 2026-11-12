import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from MedSearch.tools.build_distillation_pools import build_pools


def test_build_pools_excludes_rejected_and_keeps_dev_separate(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {"question_id": f"medical-o1-verifiable:{i}", "source_id": i,
         "question": f"Question {i}?", "references": [f"Gold {i}"],
         "decision": "pending"}
        for i in range(10, 14)
    ] + [
        {"question_id": "medical-o1-verifiable:14", "source_id": 14,
         "question": "Rejected?", "references": ["bad"], "decision": "reject"},
        {"question_id": "medical-o1-verifiable:15", "source_id": 15,
         "question": "Quarantined?", "references": ["bad"], "decision": "quarantine"},
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    old_train = tmp_path / "old_train.parquet"
    dev = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "old:1", "question": "Old RL?"}]), old_train)
    pq.write_table(pa.Table.from_pylist([{"id": "dev:1", "question": "Dev?"}]), dev)

    result = build_pools(ledger, old_train, dev, tmp_path / "out", cold_count=2)

    cold = [json.loads(line) for line in (tmp_path / "out/cold_start.jsonl").read_text(encoding="utf-8").splitlines()]
    merged = pq.read_table(tmp_path / "out/rl_train.parquet").to_pylist()
    assert result["counts"] == {"cold_start": 2, "new_rl": 2, "old_rl": 1, "old_rl_excluded_for_dev": 0, "rl_total": 3, "dev": 1}
    assert {row["question_id"] for row in cold}.isdisjoint({row["id"] for row in merged})
    assert {row["id"] for row in merged} == {"old:1", *(f"medical-o1-verifiable:{i}" for i in range(10, 14) if f"medical-o1-verifiable:{i}" not in {row["question_id"] for row in cold})}
    assert all(row["id"] not in {"medical-o1-verifiable:14", "medical-o1-verifiable:15"} for row in merged)
    assert pq.read_table(dev).to_pylist() == [{"id": "dev:1", "question": "Dev?"}]


def test_build_pools_fails_on_dev_question_overlap(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"question_id": "new:1", "source_id": 1, "question": "Same question?", "references": ["gold"], "decision": "pending"}) + "\n", encoding="utf-8")
    old_train = tmp_path / "old_train.parquet"
    dev = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "old:1", "question": "Old RL?"}]), old_train)
    pq.write_table(pa.Table.from_pylist([{"id": "dev:1", "question": "same QUESTION ?"}]), dev)

    with pytest.raises(ValueError, match="dev"):
        build_pools(ledger, old_train, dev, tmp_path / "out", cold_count=0)


def test_build_pools_removes_reviewed_old_rl_dev_paraphrase(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"question_id": "new:1", "source_id": 1, "question": "New question?", "references": ["gold"], "decision": "pending"}) + "\n", encoding="utf-8")
    old_train = tmp_path / "old_train.parquet"
    dev = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": "old:duplicate", "question": "What drug treats condition A?"},
        {"id": "old:keep", "question": "Different question?"},
    ]), old_train)
    pq.write_table(pa.Table.from_pylist([{"id": "dev:1", "question": "Which drug treats condition A?"}]), dev)
    review = tmp_path / "review.jsonl"
    review.write_text(json.dumps({"train_id": "old:duplicate", "dev_id": "dev:1", "decision": "same_question", "reason": "same treatment asked"}) + "\n", encoding="utf-8")

    result = build_pools(ledger, old_train, dev, tmp_path / "out", cold_count=0, dev_pair_review_path=review)

    assert result["counts"]["old_rl_excluded_for_dev"] == 1
    assert {row["id"] for row in pq.read_table(tmp_path / "out/rl_train.parquet").to_pylist()} == {"old:keep", "new:1"}
    assert {row["id"] for row in pq.read_table(old_train).to_pylist()} == {"old:duplicate", "old:keep"}
