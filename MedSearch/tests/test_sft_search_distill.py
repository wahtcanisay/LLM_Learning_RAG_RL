import pytest

from MedSearch.sft.data import SFTDataset


def test_search_distill_uses_original_token_ids_and_masks_environment():
    sample = {
        "prompt_ids": [10, 11],
        "response_ids": [20, 30, 21],
        "response_mask": [1, 0, 1],
        "references": ["gold must not be encoded"],
    }
    encoded = SFTDataset([sample], tokenizer=None, stage="search_distill", max_length=8)[0]
    assert encoded == {
        "input_ids": [10, 11, 20, 30, 21],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, 20, -100, 21],
    }


def test_search_distill_rejects_overlong_trace_instead_of_silent_truncation():
    sample = {"prompt_ids": [10, 11], "response_ids": [20, 21], "response_mask": [1, 1]}
    with pytest.raises(ValueError, match="max_length"):
        SFTDataset([sample], tokenizer=None, stage="search_distill", max_length=3)[0]
