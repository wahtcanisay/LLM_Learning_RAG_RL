from MedSearch.tools.audit_token_f1_risk import count_high_f1, counterfactuals, token_f1


def test_direction_reversal_keeps_misleadingly_high_token_f1():
    variants = counterfactuals("right homonymous hemianopia")

    assert ("direction", "left homonymous hemianopia") in variants
    assert token_f1("right homonymous hemianopia", "left homonymous hemianopia") > 0.5


def test_plain_entity_has_no_counterfactual():
    assert counterfactuals("Thumb sign") == []


def test_numeric_answer_is_flagged_without_inventing_an_opposite():
    assert counterfactuals("2.5 mg/kg") == [("numeric", "")]


def test_high_f1_count_ignores_missing_scores():
    assert count_high_f1(["", None, 0.49, 0.5, 0.8]) == 2
