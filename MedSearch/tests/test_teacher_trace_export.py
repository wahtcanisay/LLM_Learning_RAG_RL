import json
from types import SimpleNamespace

from MedSearch.distillation.data import encode_distillation_trace
from MedSearch.tools.evaluate_agent_rollout import (
    State,
    append_environment,
    generate_actions,
    load_states,
    run_agent,
    run_teacher_export,
    teacher_record,
)


class CharTokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert "secret gold" not in messages[0]["content"]
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeLLM:
    def generate(self, prompts, params, use_tqdm):
        assert params[0].max_tokens == 500
        ids = [ord(c) for c in "<answer>ok</answer>"]
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids)])]


def test_cold_pool_prompt_omits_gold(tmp_path):
    path = tmp_path / "cold_start.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "question": "What?", "references": ["secret gold"], "split": "cold_start"}) + "\n", encoding="utf-8")
    states = load_states(path, CharTokenizer(), 1)
    assert len(states) == 1
    assert states[0].item_id == "q1"
    assert states[0].targets == ["secret gold"]
    assert states[0].runtime_ids == [1, 2]


def test_teacher_trace_preserves_generated_and_environment_ids(monkeypatch):
    import MedSearch.tools.evaluate_agent_rollout as rollout

    monkeypatch.setattr(rollout, "SamplingParams", lambda **kwargs: SimpleNamespace(**kwargs))
    state = State("q1", "What?", ["ok"], [1, 2], 2)
    append_environment(state, [50, 51], capture=True, kind="observation")
    actions = generate_actions(FakeLLM(), CharTokenizer(), [state], 4096, 4608, 0, 1, 42, max_action_length=500, capture=True)
    assert actions[0][1].kind == "answer"
    assert state.response_ids[:2] == [50, 51]
    assert state.response_mask == [0, 0] + [1] * len("<answer>ok</answer>")
    state.answered = True
    state.final_answer = "ok"
    state.search_records = [{"query": "x", "observation_sha256": "abc", "document_hashes": ["def"]}]
    record = teacher_record(state, CharTokenizer(), 0.9, 4096, "revision")
    assert record["status"] == "accepted"
    assert record["prompt_ids"] == [1, 2]
    assert record["response_ids"] == state.response_ids
    assert record["response_mask"] == state.response_mask
    assert record["segments"][0]["kind"] == "observation"
    assert record["segments"][1]["kind"] == "answer"
    assert record["final_answer"] == "ok"
    encoded = encode_distillation_trace(record)
    assert encoded["labels"][:len(record["prompt_ids"])] == [-100] * len(record["prompt_ids"])
    assert encoded["labels"][len(record["prompt_ids"]):] == [-100, -100] + record["response_ids"][2:]


def test_one_search_then_answer_has_real_observation_and_mask(monkeypatch):
    import MedSearch.tools.evaluate_agent_rollout as rollout

    monkeypatch.setattr(rollout, "SamplingParams", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(rollout, "request_retrieval", lambda *args: {"result": [[{"document": {"contents": "Title\nEvidence"}, "score": 1.0}]]})

    class TwoActions:
        calls = 0

        def generate(self, prompts, params, use_tqdm):
            self.calls += 1
            text = "<search>query</search>" if self.calls == 1 else "<answer>ok</answer>"
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[ord(c) for c in text])])]

    args = SimpleNamespace(
        distillation_dir="out", max_turns=2, response_length=4096, max_model_len=4608,
        temperature=0.0, top_p=1.0, seed=42, search_url="http://local/retrieve",
        top_k=5, observation_length=700,
    )
    state = State("q1", "What?", ["ok"], [1, 2], 2)
    run_agent(TwoActions(), CharTokenizer(), [state], args)
    assert state.answered and state.final_answer == "ok"
    assert state.searches == 1 and len(state.search_records[0]["document_hashes"]) == 1
    assert [segment["kind"] for segment in state.segments] == ["search", "observation", "answer"]
    assert state.response_mask == [int(segment["kind"] != "observation") for segment in state.segments for _ in range(segment["end"] - segment["start"])]
    assert teacher_record(state, CharTokenizer(), 0.9, 4096, "revision")["status"] == "accepted"


def test_export_writes_complete_manifest_and_split_files(monkeypatch, tmp_path):
    import MedSearch.tools.evaluate_agent_rollout as rollout

    monkeypatch.setattr(rollout, "SamplingParams", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(rollout, "request_retrieval", lambda *args: {"result": [[{"document": {"contents": "Title\nEvidence"}, "score": 1.0}]]})
    monkeypatch.setattr(rollout, "verifier_score", lambda response, targets, url: 0.9)

    class TwoActions:
        calls = 0

        def generate(self, prompts, params, use_tqdm):
            self.calls += 1
            text = "<search>query</search>" if self.calls == 1 else "<answer>ok</answer>"
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[ord(c) for c in text])])]

    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "cold_start.jsonl"
    data.write_text(json.dumps({"question_id": "q1", "question": "What?", "references": ["ok"], "split": "cold_start"}) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "export"
    args = SimpleNamespace(
        data=str(data), model=str(model), model_revision="fixed-revision", distillation_dir=str(output),
        retrieval_corpus=str(corpus),
        batch_size=1, max_turns=2, response_length=4096, max_model_len=4608,
        observation_length=700, top_k=5, search_url="http://local/retrieve",
        verifier_url="http://local/verify", temperature=0.0, top_p=1.0, seed=42,
    )
    run_teacher_export(TwoActions(), CharTokenizer(), load_states(data, CharTokenizer(), 1), args)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    record = json.loads((output / "accepted.jsonl").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"accepted": 1, "rejected": 0}
    assert record["config_hash"] == manifest["config_hash"]
    assert record["tokenizer_hash"] == manifest["config"]["tokenizer_sha256"]
    assert (output / "rejected.jsonl").read_text(encoding="utf-8") == ""
