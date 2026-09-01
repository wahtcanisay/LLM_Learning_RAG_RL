from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MEDSEARCH_ROOT))

from sft.evaluate_stage_a import (
    aggregate_nll,
    analyze_tags,
    calculate_loss_row,
    split_prompt_and_reference,
    stratified_sample,
    summarize_tag_records,
    validate_generation_settings,
    write_review_markdown,
)


def make_sample(group: str, index: int) -> dict:
    return {
        "id": f"{group}-{index}",
        "mixture_group": group,
        "source": "fixture",
        "messages": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ],
    }


def test_stratified_sample_is_deterministic_and_balanced():
    samples = [
        make_sample(group, index)
        for group in ("single", "reasoning")
        for index in range(4)
    ]

    selected = stratified_sample(samples, per_group=2, seed=7)

    assert [sample["id"] for sample in selected] == [
        sample["id"] for sample in stratified_sample(samples, per_group=2, seed=7)
    ]
    assert Counter(sample["mixture_group"] for sample in selected) == {
        "single": 2,
        "reasoning": 2,
    }


def test_split_prompt_removes_only_final_assistant_turn():
    sample = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "final answer"},
        ]
    }

    prompt, reference = split_prompt_and_reference(sample)

    assert prompt == sample["messages"][:-1]
    assert reference == "final answer"


def test_analyze_tags_accepts_one_complete_think_answer_pair():
    result = analyze_tags("<think>clinical reasoning</think>\n<answer>result</answer>")

    assert result["has_think"] is True
    assert result["complete_think"] is True
    assert result["malformed_think"] is False
    assert result["strict_think_answer"] is True


def test_analyze_tags_distinguishes_missing_and_malformed_think():
    missing = analyze_tags("plain medical answer")
    malformed = analyze_tags("<think>unfinished reasoning")

    assert missing["has_think"] is False
    assert missing["malformed_think"] is False
    assert malformed["has_think"] is True
    assert malformed["complete_think"] is False
    assert malformed["malformed_think"] is True


def test_aggregate_nll_is_token_weighted_and_grouped():
    result = aggregate_nll(
        [
            ("single", 4.0, 2),
            ("single", 3.0, 1),
            ("reasoning", 2.0, 2),
        ]
    )

    assert result["overall"]["nll"] == pytest.approx(9 / 5)
    assert result["overall"]["supervised_tokens"] == 5
    assert result["single"]["nll"] == pytest.approx(7 / 3)
    assert result["single"]["supervised_tokens"] == 3


def test_tag_summary_separates_consistency_from_correct_behavior():
    records = [
        {
            "sample_id": "reasoning-1",
            "mixture_group": "reasoning",
            "tag_analysis": analyze_tags("<think>x</think><answer>y</answer>"),
        },
        {
            "sample_id": "reasoning-1",
            "mixture_group": "reasoning",
            "tag_analysis": analyze_tags("plain"),
            "hit_max_new_tokens": True,
        },
        {
            "sample_id": "single-1",
            "mixture_group": "single",
            "tag_analysis": analyze_tags("plain"),
        },
        {
            "sample_id": "single-1",
            "mixture_group": "single",
            "tag_analysis": analyze_tags("plain again"),
        },
    ]

    result = summarize_tag_records(records)

    assert result["overall"]["tag_signature_consistency_rate"] == pytest.approx(0.5)
    assert result["overall"]["stable_expected_behavior_rate"] == pytest.approx(0.5)
    assert result["reasoning"]["missing_expected_think_answer_rate"] == pytest.approx(0.5)
    assert result["reasoning"]["max_token_truncation_rate"] == pytest.approx(0.5)
    assert result["single"]["unexpected_think_rate"] == 0.0


def test_write_review_markdown_keeps_answers_and_blank_rubric(tmp_path):
    record = {
        "sample_id": "single-1",
        "mixture_group": "single",
        "source": "fixture",
        "repeat_index": 0,
        "prompt_messages": [{"role": "user", "content": "Can I take this medicine?"}],
        "reference": "Reference answer",
        "generated": "Generated answer",
        "tag_analysis": analyze_tags("Generated answer"),
    }
    path = tmp_path / "review.md"

    write_review_markdown(path, [record], metadata={"variant": "base"})
    text = path.read_text(encoding="utf-8")

    assert "Can I take this medicine?" in text
    assert "Reference answer" in text
    assert "Generated answer" in text
    assert "医学正确性（0/1/2）：" in text
    assert "幻觉（是/否）：" in text
    assert "安全风险（是/否）：" in text


def test_validate_generation_repeats_requires_sampling():
    with pytest.raises(ValueError, match="temperature"):
        validate_generation_settings(repeats=3, temperature=0.0, top_p=0.9)

    validate_generation_settings(repeats=3, temperature=0.7, top_p=0.9)


def test_calculate_loss_row_counts_only_supervised_assistant_tokens():
    class FixedLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(2.0))

    encoded = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 2, 3],
    }

    row = calculate_loss_row(FixedLossModel(), encoded, group="single")

    assert row == ("single", 4.0, 2)
