import pyarrow.parquet as pq

from MedSearch.tools.build_medical_verifiable_rl import INSTRUCTION


SEARCH_R1_ANSWER_EXAMPLE = "For example, <answer> Beijing </answer>."


def test_rl_prompt_keeps_search_r1_answer_example():
    assert SEARCH_R1_ANSWER_EXAMPLE in INSTRUCTION

    for path in ("MedSearch/data/rl/train.parquet", "MedSearch/data/rl/dev.parquet"):
        prompts = pq.read_table(path, columns=["prompt"])["prompt"].to_pylist()
        assert prompts
        assert all(SEARCH_R1_ANSWER_EXAMPLE in prompt[0]["content"] for prompt in prompts)
