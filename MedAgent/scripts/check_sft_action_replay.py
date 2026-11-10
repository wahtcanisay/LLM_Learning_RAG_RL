"""Replay a saved 3B SFT search and answer through the Harness on CPU."""

import hashlib
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer


MEDAGENT = Path(__file__).resolve().parents[1]
ROOT = MEDAGENT.parent
sys.path.insert(0, str(MEDAGENT))

from medagent.contracts import Evidence, SearchTask, Usage
from medagent.search_agent import ActionResponse, SearchAgent, parse_action


def main():
    distillation = ROOT / "MedSearch" / "outputs" / "distillation"
    index = json.loads((distillation / "FINALIZED_SFT_CHECKPOINTS.json").read_text(encoding="utf-8"))
    variant = index["variants"]["top32"]
    rollout_dir = ROOT / variant["evaluation_path"]
    manifest = json.loads((rollout_dir / "rollout_manifest.json").read_text(encoding="utf-8"))
    rollout = (rollout_dir / "rollouts.jsonl").read_bytes()
    assert manifest["adapter_sha256"] == variant["adapter_sha256"]
    assert hashlib.sha256(rollout).hexdigest() == manifest["rollout_sha256"]
    assert manifest["stop"] == ["</search>", "</answer>"]
    assert manifest["include_stop_str_in_output"]

    tokenizer = Tokenizer.from_file(str(ROOT / index["base_model_path"] / "tokenizer.json"))
    for line in rollout.splitlines():
        row = json.loads(line)
        segments = row["segments"]
        if [part["kind"] for part in segments] != ["search", "observation", "answer"]:
            continue
        response_ids = row["response_ids"]
        search, observation, answer = (
            tokenizer.decode(response_ids[part["start"]:part["end"]], skip_special_tokens=True)
            for part in segments
        )
        if search.find("<search>") > 0:
            break
    else:
        raise AssertionError("No complete search-observation-answer episode with a generated prefix")

    first, middle, last = segments
    assert first["start"] == 0 and first["end"] == middle["start"]
    assert middle["end"] == last["start"] and last["end"] == len(response_ids)
    assert row["response_mask"] == (
        [1] * first["end"] + [0] * (middle["end"] - middle["start"])
        + [1] * (last["end"] - last["start"])
    )
    assert tokenizer.decode(row["prompt_ids"], skip_special_tokens=False).endswith("<|im_start|>assistant\n")
    # A saved boundary token can include characters after the closing tag.
    assert "</search>" in search and "</answer>" in answer
    assert observation.startswith("<information>") and observation.endswith("</information>")
    assert parse_action(search).kind == "search"
    assert parse_action(answer).content == row["final_answer"]

    body = observation.removeprefix("<information>").removesuffix("</information>")
    replies = iter((search, answer))
    contexts = []
    queries = []

    def request_action(context):
        contexts.append(context)
        return ActionResponse(next(replies), Usage(None, None, None, False))

    def retrieve(query, _config, _timeout, *, top_k):
        queries.append(query)
        return [Evidence(body)]

    query = row["question"]
    task = SearchTask("Outer workflow question", query, 0, ("Outer workflow feedback",))
    result = SearchAgent(request_action, retrieve, {}, 1).run(task)
    assert len(contexts) == 2 and contexts[0] == query
    assert queries == [parse_action(search).content]
    assert observation in contexts[1]
    assert result.candidate_answer == row["final_answer"]
    generated_prefix = search[:search.index("<search>")]
    assert generated_prefix in contexts[1]
    assert result.context.startswith(contexts[1]) and result.context.endswith("</answer>")
    print(json.dumps({
        "mode": "archived_action_replay",
        "id": row["id"],
        "adapter_sha256": manifest["adapter_sha256"],
        "actions": len(contexts),
        "searches": len(queries),
        "answer_matches_archive": True,
        "original_second_request_tokens": len(row["prompt_ids"]) + middle["end"],
        "generated_prefix_retained_in_second_harness_context": True,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
