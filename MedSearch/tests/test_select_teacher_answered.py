from MedSearch.tools.select_teacher_answered import select_answered


def record(*, answered, final_answer, source_status, reasons):
    return {
        "answered": answered,
        "final_answer": final_answer,
        "segments": [{"kind": "answer" if answered else "invalid_action"}],
        "response_ids": [1],
        "response_mask": [1],
        "status": source_status,
        "stop_reason": reasons,
    }


def test_keep_legal_answer_even_when_verifier_failed_or_earlier_action_invalid():
    wrong = record(answered=True, final_answer="wrong", source_status="rejected", reasons=["verifier_rejected", "invalid_action"])
    selected = select_answered(wrong)
    assert selected["status"] == "accepted"
    assert selected["prior_filter_status"] == "rejected"
    assert selected["prior_filter_reasons"] == ["verifier_rejected", "invalid_action"]
    assert selected["stop_reason"] == []
    assert selected["response_ids"] == wrong["response_ids"]


def test_exclude_trajectory_without_legal_final_answer():
    assert select_answered(record(answered=False, final_answer=None, source_status="rejected", reasons=["no_valid_answer"])) is None
