import json
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MEDSEARCH_ROOT.parent
sys.path.insert(0, str(MEDSEARCH_ROOT))

import sft.preprocess as preprocess
from sft.preprocess import encode_stage_a


def test_stage_b_prompt_keeps_structured_options_and_search_r1_instructions():
    assert hasattr(preprocess, "format_stage_b_prompt")

    prompt = preprocess.format_stage_b_prompt(
        {
            "question": "question marker?",
            "options": {
                "A": "option marker A",
                "B": "option marker B",
                "C": "option marker C",
                "D": "option marker D",
            },
        }
    )

    assert prompt == (
        "Answer the given multiple-choice question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as you want. "
        "If you find no further external knowledge needed, provide only the option letter inside "
        "<answer> and </answer>, for example <answer>B</answer>.\n"
        "Question: question marker?\n"
        "Options:\n"
        "A. option marker A\n"
        "B. option marker B\n"
        "C. option marker C\n"
        "D. option marker D\n"
    )


def test_stage_a_keeps_context_but_only_trains_on_assistant_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    sample = {
        "messages": [
            {"role": "system", "content": "system hidden marker"},
            {"role": "user", "content": "user hidden marker one"},
            {"role": "assistant", "content": "assistant visible marker one"},
            {"role": "user", "content": "user hidden marker two"},
            {"role": "assistant", "content": "assistant visible marker two"},
        ]
    }

    encoded = encode_stage_a(sample, tokenizer, max_length=1024)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = encoded["labels"]

    assert len(input_ids) == len(attention_mask) == len(labels)
    assert 0 < len(input_ids) <= 1024
    assert all(value == 1 for value in attention_mask)

    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(
        [token_id for token_id in labels if token_id != -100],
        skip_special_tokens=False,
    )

    assert "system hidden marker" in full_text
    assert "user hidden marker one" in full_text
    assert "assistant visible marker one" in target_text
    assert "assistant visible marker two" in target_text
    assert "system hidden marker" not in target_text
    assert "user hidden marker one" not in target_text
    assert "user hidden marker two" not in target_text


def test_stage_b_two_searches_mask_environment_and_train_model_actions():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    sample = {
        "question": "stage b question marker?",
        "options": {
            "A": "stage b option marker A",
            "B": "stage b option marker B",
            "C": "stage b option marker C",
            "D": "stage b option marker D",
        },
        "trajectory": [
            {
                "type": "search",
                "thinking": "first thinking marker",
                "query": "first search marker",
            },
            {"type": "information", "content": "first information hidden marker"},
            {
                "type": "search",
                "thinking": "second thinking marker",
                "query": "second search marker",
            },
            {"type": "information", "content": "second information hidden marker"},
            {
                "type": "answer",
                "thinking": "answer thinking marker",
                "answer": "B",
            },
        ],
    }

    encoded = preprocess.encode_stage_b(sample, tokenizer, max_length=1024)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = encoded["labels"]

    assert len(input_ids) == len(attention_mask) == len(labels)
    assert 0 < len(input_ids) <= 1024
    assert all(value == 1 for value in attention_mask)

    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(
        [token_id for token_id in labels if token_id != -100],
        skip_special_tokens=False,
    )

    assert full_text.index("first search marker") < full_text.index("first information hidden marker")
    assert full_text.index("first information hidden marker") < full_text.index("second search marker")
    assert full_text.index("second search marker") < full_text.index("second information hidden marker")
    assert full_text.index("second information hidden marker") < full_text.rindex("<answer>B</answer>")
    assert "<think>first thinking marker</think>" in target_text
    assert "<search>first search marker</search>" in target_text
    assert "<think>second thinking marker</think>" in target_text
    assert "<search>second search marker</search>" in target_text
    assert "<think>answer thinking marker</think>" in target_text
    assert "<answer>B</answer>" in target_text
    assert "stage b question marker" not in target_text
    assert "stage b option marker" not in target_text
    assert "information hidden marker" not in target_text
    assert labels.count(tokenizer.eos_token_id) == 1

    between_actions = full_text.split("</search>")
    assert all("<|im_end|>" not in part.split("<information>", 1)[0] for part in between_actions[1:])


def test_stage_b_rejects_unknown_trajectory_type():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    sample = {
        "question": "question?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "trajectory": [{"type": "unknown"}],
    }

    with pytest.raises(ValueError, match="unknown trajectory step type"):
        preprocess.encode_stage_b(sample, tokenizer, max_length=1024)


def test_frozen_stage_b_dataset_fits_4096_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    with (MEDSEARCH_ROOT / "data" / "stage_b" / "search100_training.json").open(
        encoding="utf-8"
    ) as file:
        samples = json.load(file)

    encoded_samples = [
        preprocess.encode_stage_b(sample, tokenizer, max_length=4096)
        for sample in samples
    ]

    assert len(encoded_samples) == 100
    assert max(len(sample["input_ids"]) for sample in encoded_samples) <= 4096
