from pathlib import Path
import sys

import pytest


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MEDSEARCH_ROOT))

from sft.evaluate_stage_b import parse_first_action, summarize_first_actions


def test_parse_valid_search_requires_think_and_nonempty_query():
    result = parse_first_action(
        "<think>Need external evidence.</think>\n<search>asthma eosinophilia</search>"
    )

    assert result == {
        "action": "search",
        "content": "asthma eosinophilia",
        "valid_think": True,
        "valid_action": True,
        "malformed": False,
    }


@pytest.mark.parametrize(
    "text",
    [
        "<search>asthma eosinophilia</search>",
        "<think>Need evidence.</think><search> </search>",
        "<think>Need evidence.</think><search>query</search><answer>B</answer>",
    ],
)
def test_parse_rejects_non_protocol_or_multiple_actions(text):
    result = parse_first_action(text)

    assert result["valid_action"] is False
    assert result["malformed"] is True


def test_summary_is_example_weighted():
    metrics = summarize_first_actions(
        [
        {
            "valid_think": True,
            "valid_action": True,
            "action": "search",
            "content": "query",
            "malformed": False,
        },
        {
            "valid_think": True,
            "valid_action": True,
            "action": "answer",
            "content": "B",
            "malformed": False,
        },
        ]
    )

    assert metrics["examples"] == 2
    assert metrics["valid_think_rate"] == 1.0
    assert metrics["valid_search_rate"] == 0.5
    assert metrics["nonempty_query_rate"] == 0.5
    assert metrics["direct_answer_rate"] == 0.5
    assert metrics["malformed_action_rate"] == 0.0
