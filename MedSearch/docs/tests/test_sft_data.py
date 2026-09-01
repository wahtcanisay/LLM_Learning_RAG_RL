import json
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MEDSEARCH_ROOT.parent
sys.path.insert(0, str(MEDSEARCH_ROOT))

from sft.data import SFTDataset, build_data_collator, load_samples
from sft.preprocess import encode_stage_a, encode_stage_b


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )


def test_load_samples_reads_jsonl_and_json(tmp_path):
    stage_a_samples = [
        {"id": "a1", "messages": [{"role": "user", "content": "q1"}]},
        {"id": "a2", "messages": [{"role": "user", "content": "q2"}]},
    ]
    stage_b_samples = [
        {
            "id": "b1",
            "question": "question?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "trajectory": [],
        }
    ]
    jsonl_path = tmp_path / "stage_a.jsonl"
    json_path = tmp_path / "stage_b.json"
    jsonl_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in stage_a_samples),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(stage_b_samples), encoding="utf-8")

    assert load_samples(jsonl_path) == stage_a_samples
    assert load_samples(json_path) == stage_b_samples


def test_dataset_dispatches_to_the_stage_encoder(tokenizer):
    stage_a_sample = {
        "messages": [
            {"role": "user", "content": "stage a question"},
            {"role": "assistant", "content": "stage a answer"},
        ]
    }
    stage_b_sample = {
        "question": "stage b question?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "trajectory": [
            {"type": "answer", "thinking": "reason", "answer": "A"},
        ],
    }

    stage_a_dataset = SFTDataset(
        [stage_a_sample], tokenizer, stage="stage_a", max_length=1024
    )
    stage_b_dataset = SFTDataset(
        [stage_b_sample], tokenizer, stage="stage_b", max_length=4096
    )

    assert len(stage_a_dataset) == len(stage_b_dataset) == 1
    assert stage_a_dataset[0] == encode_stage_a(stage_a_sample, tokenizer, 1024)
    assert stage_b_dataset[0] == encode_stage_b(stage_b_sample, tokenizer, 4096)


def test_dataset_rejects_unknown_stage(tokenizer):
    with pytest.raises(ValueError, match="unknown SFT stage"):
        SFTDataset([], tokenizer, stage="stage_c", max_length=1024)


def test_collator_dynamically_pads_inputs_masks_and_labels(tokenizer):
    collator = build_data_collator(tokenizer)
    features = [
        {
            "input_ids": [11, 12, 13],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 12, 13],
        },
        {
            "input_ids": [21],
            "attention_mask": [1],
            "labels": [21],
        },
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == batch["labels"].shape == (2, 3)
    if tokenizer.padding_side == "right":
        assert batch["input_ids"][1].tolist() == [21, tokenizer.pad_token_id, tokenizer.pad_token_id]
        assert batch["attention_mask"][1].tolist() == [1, 0, 0]
        assert batch["labels"][1].tolist() == [21, -100, -100]
    else:
        assert batch["input_ids"][1].tolist() == [tokenizer.pad_token_id, tokenizer.pad_token_id, 21]
        assert batch["attention_mask"][1].tolist() == [0, 0, 1]
        assert batch["labels"][1].tolist() == [-100, -100, 21]
