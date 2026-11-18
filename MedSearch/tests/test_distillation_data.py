from MedSearch.distillation.data import encode_distillation_trace


def test_encode_distillation_trace_preserves_ids_and_masks_environment():
    trace = {
        "prompt_ids": [10, 11],
        "response_ids": [20, 30, 21],
        "response_mask": [1, 0, 1],
        "golden_answers": [99],
    }

    encoded = encode_distillation_trace(trace)

    assert encoded["input_ids"] == [10, 11, 20, 30, 21]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]
    assert encoded["labels"] == [-100, -100, 20, -100, 21]
    assert "golden_answers" not in encoded
