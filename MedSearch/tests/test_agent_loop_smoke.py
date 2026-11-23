import asyncio
import importlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import MedSearch.rl.medsearch_agent_loop as module
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        if ids == [61]:
            return "\n<think>need evidence</think><search>aspirin</search>"
        if ids == [61, 62]:
            return "<think>need evidence</think><search>aspirin</search><information>invented</information><answer>B</answer>"
        return {
            11: "<answer>A</answer>",
            21: "<search>aspirin</search>",
            22: "<answer>B</answer>",
            33: "<information><answer>gold</answer></information>",
            51: "invalid action",
        }[ids[0]]

    def encode(self, text, add_special_tokens=False):
        if text == "<answer>A</answer>":
            return [11]
        if text == "<search>aspirin</search>":
            return [21]
        if text == "<answer>B</answer>":
            return [22]
        if text == "invalid action":
            return [51]
        if text == "<information>DOC</information>":
            return [31, 32]
        if text == "<information><answer>gold</answer></information>":
            return [33]
        if text == "<think>need evidence</think><search>aspirin</search>":
            return [21]
        if "previous action is invalid" not in text and "search budget is exhausted" not in text:
            raise AssertionError(text)
        return [41, 42]


def test_import_does_not_overwrite_yaml_agent_config():
    config_path = Path(__file__).parents[1] / "configs" / "medsearch_agent.yaml"
    agent_config = OmegaConf.load(config_path)[0]
    previous = _agent_loop_registry.get(agent_config.name)
    try:
        _agent_loop_registry[agent_config.name] = agent_config
        importlib.reload(module)
        registered = _agent_loop_registry[agent_config.name]
        assert registered["search_url"] == "http://127.0.0.1:8000/retrieve"
        assert registered["max_observation_length"] == 500
    finally:
        if previous is None:
            _agent_loop_registry.pop(agent_config.name, None)
        else:
            _agent_loop_registry[agent_config.name] = previous


class FakeServer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        assert kwargs["sampling_params"]["max_tokens"] == 500
        output = self.outputs.pop(0)
        if isinstance(output, dict):
            token_ids = output["token_ids"]
            log_probs = output.get("log_probs")
        else:
            token_ids = output if isinstance(output, list) else [output]
            log_probs = None
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=log_probs,
            num_preempted=0,
            extra_fields={"min_global_steps": 7, "max_global_steps": 7},
        )


async def make_loop(outputs, max_turns=1):
    loop = object.__new__(module.MedSearchAgentLoop)
    loop.search_url = "fake"
    loop.top_k = 3
    loop.request_timeout = 1.0
    loop.max_turns = max_turns
    loop.max_observation_length = 1
    loop.max_action_length = 500
    loop.response_length = 64
    loop.tokenizer = FakeTokenizer()
    loop.server_manager = FakeServer(outputs)
    loop.assistant_logprobs = []

    async def build(self, messages):
        return [1, 2]

    async def merge(
        self,
        runtime_ids,
        assistant_ids,
        mask,
        logprobs=None,
        assistant_logprobs=None,
    ):
        self.assistant_logprobs.append(assistant_logprobs)
        return (
            SimpleNamespace(token_ids=runtime_ids + assistant_ids),
            mask + [1] * len(assistant_ids),
            logprobs,
        )

    loop.ct_build_initial_tokens = MethodType(build, loop)
    loop.ct_merge_assistant_token = MethodType(merge, loop)
    return loop


def test_direct_answer_and_one_search(monkeypatch):
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append(query)
        return "<information>DOC</information>"

    monkeypatch.setattr(module, "retrieve", fake_retrieve)

    async def run():
        direct = await make_loop([11])
        direct_out = await direct.run({}, raw_prompt=[{"role": "user", "content": "q"}])
        assert direct.server_manager.calls == 1
        assert calls == []
        assert direct_out.prompt_ids == [1, 2]
        assert direct_out.response_ids == [11]
        assert direct_out.response_mask == [1]
        assert direct_out.extra_fields["final_answer"] == "A"
        assert direct_out.extra_fields["min_global_steps"] == 7
        assert direct_out.extra_fields["max_global_steps"] == 7
        assert direct_out.num_turns == 2

        searched = await make_loop([21, 22])
        searched_out = await searched.run({}, raw_prompt=[{"role": "user", "content": "q"}])
        assert searched.server_manager.calls == 2
        assert calls == ["aspirin"]
        assert searched_out.prompt_ids == [1, 2]
        assert searched_out.response_ids == [21, 31, 41, 42, 22]
        assert searched_out.response_mask == [1, 0, 0, 0, 1]
        assert searched_out.num_turns == 4
        assert searched_out.extra_fields["valid_search_stats"] == 1
        assert searched_out.extra_fields["answered"] is True
        assert searched_out.extra_fields["final_answer"] == "B"

    asyncio.run(run())


def test_invalid_action_becomes_masked_correction(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "retrieve", lambda query, **kwargs: calls.append(query))

    async def run():
        loop = await make_loop([51, 11], max_turns=2)
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "q"}])

        assert loop.server_manager.calls == 2
        assert calls == []
        assert output.response_ids == [51, 41, 42, 11]
        assert output.response_mask == [1, 0, 0, 1]
        assert output.num_turns == 4
        assert output.extra_fields["invalid_action_stats"] == 1
        assert output.extra_fields["answered"] is True
        assert output.extra_fields["final_answer"] == "A"

    asyncio.run(run())


def test_two_searches_still_allow_final_answer(monkeypatch):
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append(query)
        return "<information>DOC</information>"

    monkeypatch.setattr(module, "retrieve", fake_retrieve)

    async def run():
        loop = await make_loop([21, 21, 22], max_turns=2)
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "q"}])

        assert loop.server_manager.calls == 3
        assert calls == ["aspirin", "aspirin"]
        assert output.response_ids == [21, 31, 21, 31, 41, 42, 22]
        assert output.response_mask == [1, 0, 1, 0, 0, 0, 1]
        assert output.extra_fields["valid_search_stats"] == 2
        assert output.extra_fields["answered"] is True
        assert output.extra_fields["final_answer"] == "B"

    asyncio.run(run())


def test_completed_search_is_truncated_before_environment_handoff(monkeypatch):
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append(query)
        return "<information>DOC</information>"

    monkeypatch.setattr(module, "retrieve", fake_retrieve)

    async def run():
        loop = await make_loop([{"token_ids": [61, 62], "log_probs": [0.1, 0.2]}, 22])
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "q"}])

        assert calls == ["aspirin"]
        assert output.extra_fields["invalid_action_stats"] == 0
        assert output.response_ids == [61, 31, 41, 42, 22]
        assert output.response_mask == [1, 0, 0, 0, 1]
        assert loop.assistant_logprobs == [[0.1], None]
        assert output.extra_fields["final_answer"] == "B"

    asyncio.run(run())


def test_observation_answer_does_not_become_final_answer(monkeypatch):
    monkeypatch.setattr(module, "retrieve", lambda *args, **kwargs: "<information><answer>gold</answer></information>")

    async def run():
        loop = await make_loop([21, 51])
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "q"}])
        assert output.extra_fields["answered"] is False
        assert output.extra_fields["final_answer"] is None
        assert output.response_ids == [21, 33, 41, 42, 51]
        assert output.response_mask == [1, 0, 0, 0, 1]

    asyncio.run(run())
