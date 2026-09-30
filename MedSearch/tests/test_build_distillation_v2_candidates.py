from MedSearch.tools.build_distillation_v2_candidates import _write_jsonl, select_candidates


def test_jsonl_bytes_are_platform_independent(tmp_path):
    path = tmp_path / "items.jsonl"
    _write_jsonl(path, [{"x": 1}])
    assert path.read_bytes() == b'{"x": 1}\n'


def test_candidate_screen_preserves_gold_and_exposes_exclusions():
    raw = [
        {"Open-ended Verifiable Question": "Existing?", "Ground-True Answer": "A"},
        {"Open-ended Verifiable Question": "What is the enzyme?", "Ground-True Answer": "Lactase"},
        {"Open-ended Verifiable Question": "What is the enzyme?", "Ground-True Answer": "Lactase"},
        {"Open-ended Verifiable Question": "As shown in the image, what is it?", "Ground-True Answer": "Hepatitis"},
    ]

    kept, other = select_candidates(raw, {"medical-o1-verifiable:0"}, {"existing"}, limit=1)

    assert kept == [{
        "question_id": "medical-o1-verifiable:1",
        "source": "FreedomIntelligence/medical-o1-verifiable-problem",
        "source_id": 1,
        "question": "What is the enzyme?",
        "references": ["Lactase"],
        "split": "unassigned",
    }]
    assert [row["source_id"] for row in other] == [0, 2, 3]
    assert "existing_id" in other[0]["reasons"]
    assert "internal_exact_duplicate" in other[1]["reasons"]
    assert other[2]["status"] == "quarantine"
