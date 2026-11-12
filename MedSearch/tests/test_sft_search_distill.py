import pytest

from MedSearch.sft.data import SFTDataset, build_data_collator, load_samples


def test_collator_preserves_variable_topk_fields_and_ce_padding():
    import copy
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", padding_side="right",
    )
    samples = [
        {"prompt_ids": [10, 11], "response_ids": [20, 30, 21], "response_mask": [1, 0, 1],
         "target_positions": [2, 4], "teacher_topk_ids": [list(range(32))] * 2,
         "teacher_topk_logprobs": [[-5.] * 32] * 2},
        {"prompt_ids": [12], "response_ids": [22], "response_mask": [1],
         "target_positions": [1], "teacher_topk_ids": [list(range(32, 64))],
         "teacher_topk_logprobs": [[-6.] * 32]},
    ]
    dataset = SFTDataset(samples, tokenizer, "search_distill", 8)
    features = [dataset[0], dataset[1]]
    original = copy.deepcopy(features)
    collate = build_data_collator(tokenizer)
    batch = collate(features)
    assert features == original
    assert batch["input_ids"].tolist() == [[10, 11, 20, 30, 21], [12, 22, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 20, -100, 21], [-100, 22, -100, -100, -100]]
    for i, sample in enumerate(samples):
        for key, dtype in (("target_positions", torch.long), ("teacher_topk_ids", torch.long),
                           ("teacher_topk_logprobs", torch.float32)):
            assert isinstance(batch[key], list)
            assert batch[key][i].dtype == dtype
            assert batch[key][i].tolist() == sample[key]
        assert batch["teacher_topk_ids"][i].shape == (len(sample["target_positions"]), 32)
        assert (batch["labels"][i, batch["target_positions"][i]] != -100).all()

    ce_features = [{key: value for key, value in feature.items()
                    if key in ("input_ids", "attention_mask", "labels")} for feature in features]
    ce_batch = collate(ce_features)
    assert set(ce_batch) == {"input_ids", "attention_mask", "labels"}
    for key in ce_batch:
        assert torch.equal(ce_batch[key], batch[key])


def test_load_samples_reads_parquet_and_preserves_nested_topk_fields(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"trajectory_id": "second", "prompt_ids": [10], "response_ids": [20],
             "response_mask": [1], "target_positions": [1],
             "teacher_topk_ids": [list(range(32))], "teacher_topk_logprobs": [[-5.] * 32]},
            {"trajectory_id": "first", "prompt_ids": [11], "response_ids": [21],
             "response_mask": [1], "target_positions": [1],
             "teacher_topk_ids": [list(range(32, 64))], "teacher_topk_logprobs": [[-6.] * 32]}]
    path = tmp_path / "cache.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert load_samples(path) == rows


def test_load_samples_keeps_jsonl_support(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text('{"prompt_ids": [10]}\n\n{"prompt_ids": [20]}\n', encoding="utf-8")
    assert load_samples(path) == [{"prompt_ids": [10]}, {"prompt_ids": [20]}]


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
