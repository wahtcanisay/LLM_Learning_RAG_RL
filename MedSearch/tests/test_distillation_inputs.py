import json

from MedSearch.tools.check_distillation_inputs import check_split_isolation, check_v1_provenance


def _jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_reports_both_id_and_normalized_question_conflicts(tmp_path):
    cold = _jsonl(
        tmp_path / "cold.jsonl",
        [{"id": "a", "question": "What is BP?"}],
    )
    rl = _jsonl(
        tmp_path / "rl.jsonl",
        [{"id": "a", "question": "Different question"}],
    )
    history = _jsonl(
        tmp_path / "history.jsonl",
        [{"id": "z", "question": "what   is bp"}],
    )

    report = check_split_isolation(
        {"cold": cold, "rl": rl}, {"history": history}
    )

    assert report["counts"] == {"cold": 1, "rl": 1, "history": 1}
    assert {item["kind"] for item in report["conflicts"]} == {"id", "question"}
    assert {item["left"]["pool"] for item in report["conflicts"]} <= {"cold", "rl"}
    assert all(item["left"]["row"] == 1 for item in report["conflicts"])
    assert report["exact_check_passed"] is False


def test_missing_input_is_explicitly_uncovered(tmp_path):
    existing = _jsonl(tmp_path / "dev.jsonl", [{"question": "Question"}])
    missing = tmp_path / "luna.jsonl"

    report = check_split_isolation({"dev": existing, "luna": missing}, {})

    assert report["counts"] == {"dev": 1}
    assert report["missing"] == [{"pool": "luna", "path": str(missing)}]
    assert report["exact_check_passed"] is False
    assert report["semantic_near_duplicates"] == "not_checked"


def test_existing_required_directory_is_not_treated_as_audited(tmp_path):
    existing = _jsonl(tmp_path / "dev.jsonl", [{"question": "Question"}])
    luna_dir = tmp_path / "luna"
    luna_dir.mkdir()
    _jsonl(luna_dir / "cold.jsonl", [{"question": "New question"}])

    report = check_split_isolation(
        {"dev": existing}, {}, {"luna_v2": luna_dir}
    )

    assert report["unreviewed"] == [{"pool": "luna_v2", "path": str(luna_dir)}]
    assert report["exact_check_passed"] is False


def test_reads_historical_chat_and_mirage_without_inventing_ids(tmp_path):
    candidate = _jsonl(tmp_path / "candidate.jsonl", [{"question_id": "q1", "question": "Same case"}])
    history = _jsonl(
        tmp_path / "history.jsonl",
        [{"messages": [{"role": "user", "content": "same case"}]}],
    )
    mirage = tmp_path / "benchmark.json"
    mirage.write_text(
        json.dumps({"medqa": {"one": {"question": "Unrelated"}}, "medmcqa": {}}),
        encoding="utf-8",
    )

    report = check_split_isolation(
        {"candidate": candidate}, {"history": history, "final_test": mirage}
    )

    assert report["counts"] == {"candidate": 1, "history": 1, "final_test": 1}
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["kind"] == "question"
    assert report["conflicts"][0]["right"]["id"] is None


def test_v1_provenance_detects_changed_gold(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps([{"Open-ended Verifiable Question": "Case?", "Ground-True Answer": "A"}]),
        encoding="utf-8",
    )
    audited = _jsonl(
        tmp_path / "audited_keep.jsonl",
        [{
            "id": "medical-o1-verifiable:0",
            "source_index": 0,
            "question": "Case?",
            "answer": "B",
            "data": {"golden_answers": ["B"]},
        }],
    )
    train = _jsonl(tmp_path / "train.jsonl", [{
        "id": "medical-o1-verifiable:0", "question": "Case?", "golden_answers": ["B"]
    }])
    dev = _jsonl(tmp_path / "dev.jsonl", [{
        "id": "other", "question": "Other?", "golden_answers": ["C"]
    }])

    report = check_v1_provenance(raw, audited, train, dev)

    assert report["audited_rows"] == 1
    assert report["issues"] == [{"id": "medical-o1-verifiable:0", "reason": "raw_answer_mismatch"}]
    assert report["clinical_correctness"] == "not_checked"
