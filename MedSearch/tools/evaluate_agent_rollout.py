"""Evaluate MedSearch trajectories with one vLLM model copy."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from MedSearch.rl.medsearch_agent_loop import (
    FINAL_ANSWER_PROMPT,
    INVALID_ACTION_PROMPT,
    truncate_generated_action,
)
from MedSearch.rl.protocol import parse_action
from MedSearch.rl.retriever_client import (
    build_observation,
    encode_observation,
    request_retrieval,
    retrieve,
)
from MedSearch.rl.reward import compute_outcome_reward
from MedSearch.tools.build_medical_verifiable_rl import INSTRUCTION


@dataclass
class State:
    item_id: str
    question: str
    targets: list[str]
    runtime_ids: list[int]
    prompt_length: int
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    search_records: list[dict] = field(default_factory=list)
    searches: int = 0
    invalid_actions: int = 0
    answered: bool = False
    exhausted: bool = False
    final_answer: str | None = None
    stop_reason: str | None = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--response-length", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--observation-length", type=int, default=700)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--verifier-url", default="http://127.0.0.1:8002/verify")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distillation-dir", help="Write complete teacher trajectories for cold start")
    parser.add_argument("--model-revision", help="Fixed teacher checkpoint revision")
    parser.add_argument("--retrieval-corpus", help="Corpus file served by the retrieval endpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def verifier_score(response, targets, verifier_url):
    """Score the last closed answer with the frozen medical-o1 verifier."""
    import re

    answers = [
        answer.strip()
        for answer in re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer.strip() not in {"answer", "and"}
    ]
    if not answers:
        return 0.0
    request = Request(
        verifier_url,
        data=json.dumps({"answer": answers[-1], "references": targets}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response_handle:
        return float(json.load(response_handle)["prob"])


def as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def load_states(data_path, tokenizer, limit):
    data_path = Path(data_path)
    if data_path.suffix == ".jsonl":
        with data_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = pd.read_parquet(data_path).to_dict("records")
    if limit is not None:
        rows = rows[:limit]
    states = []
    for row in rows:
        if "prompt" in row:
            messages = [dict(message) for message in as_list(row["prompt"])]
            targets = as_list(row["golden_answers"])
            item_id = str(row["id"])
        else:
            if row.get("split") != "cold_start":
                raise ValueError(f"expected cold_start split: {row.get('question_id')}")
            messages = [{"role": "user", "content": f"{INSTRUCTION}\nQuestion: {row['question']}"}]
            targets = row["references"]
            item_id = str(row["question_id"])
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]
        states.append(
            State(
                item_id=item_id,
                question=str(row["question"]),
                targets=[str(target) for target in targets],
                runtime_ids=list(prompt_ids),
                prompt_length=len(prompt_ids),
            )
        )
    return states


def append_environment(state, token_ids, *, capture=False, kind="environment"):
    start = len(state.response_ids)
    state.runtime_ids.extend(token_ids)
    state.response_ids.extend(token_ids)
    if capture:
        state.response_mask.extend([0] * len(token_ids))
        state.segments.append({"kind": kind, "start": start, "end": len(state.response_ids)})


def generate_actions(llm, tokenizer, states, response_length, max_model_len, temperature, top_p, seed, *, max_action_length=None, capture=False):
    active = [
        state
        for state in states
        if not state.answered
        and not state.exhausted
        and len(state.response_ids) < response_length
        and len(state.runtime_ids) < max_model_len
    ]
    if not active:
        return []

    max_tokens = [
        min(
            response_length - len(state.response_ids),
            max_model_len - len(state.runtime_ids),
            max_action_length if max_action_length is not None else response_length,
        )
        for state in active
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=state.runtime_ids) for state in active],
        [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=budget,
                seed=seed,
            )
            for budget in max_tokens
        ],
        use_tqdm=False,
    )
    actions = []
    for state, output in zip(active, outputs, strict=True):
        generated_ids = list(output.outputs[0].token_ids)
        action_ids, action_text = truncate_generated_action(generated_ids, tokenizer)
        action = parse_action(action_text)
        if capture and parse_action(tokenizer.decode(action_ids, skip_special_tokens=True)) != action:
            action = None
        start = len(state.response_ids)
        state.runtime_ids.extend(action_ids)
        state.response_ids.extend(action_ids)
        if capture:
            state.response_mask.extend([1] * len(action_ids))
            state.segments.append({"kind": action.kind if action else "invalid_action", "start": start, "end": len(state.response_ids)})
            if action and action.kind == "answer":
                state.final_answer = action.content
        actions.append((state, action))
        if len(state.response_ids) >= response_length or len(state.runtime_ids) >= max_model_len:
            state.exhausted = True
            state.stop_reason = "length_exhausted"
    return actions


def run_agent(llm, tokenizer, states, args):
    capture = bool(args.distillation_dir)

    def add_context(state, ids, kind):
        if capture and (
            len(state.runtime_ids) + len(ids) >= args.max_model_len
            or len(state.response_ids) + len(ids) >= args.response_length
        ):
            state.exhausted = True
            state.stop_reason = "length_exhausted"
            return
        append_environment(state, ids, capture=capture, kind=kind)

    for _ in range(args.max_turns):
        for state, action in generate_actions(
            llm,
            tokenizer,
            states,
            args.response_length,
            args.max_model_len,
            args.temperature,
            args.top_p,
            args.seed,
            max_action_length=500 if capture else None,
            capture=capture,
        ):
            if action is not None and action.kind == "answer":
                state.answered = True
            elif action is not None and action.kind == "search":
                try:
                    if capture:
                        body = request_retrieval(action.content, args.search_url, args.top_k, 30.0)
                        observation = build_observation(body)
                        passages = body["result"][0]
                        state.search_records.append({
                            "query": action.content,
                            "search_url": args.search_url,
                            "top_k": args.top_k,
                            "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                            "document_hashes": [hashlib.sha256(json.dumps(p["document"], sort_keys=True).encode("utf-8")).hexdigest() for p in passages],
                        })
                    else:
                        observation = retrieve(action.content, search_url=args.search_url, topk=args.top_k, timeout=30.0)
                except Exception as exc:
                    if not capture:
                        raise
                    state.exhausted = True
                    state.stop_reason = f"retrieval_error:{type(exc).__name__}"
                    continue
                observation_ids = encode_observation(
                    observation, tokenizer, args.observation_length
                )
                add_context(state, observation_ids, "observation")
                state.searches += 1
            else:
                correction_ids = tokenizer.encode(
                    INVALID_ACTION_PROMPT,
                    add_special_tokens=False,
                )
                add_context(state, correction_ids, "correction")
                state.invalid_actions += 1

    needs_final = [state for state in states if not state.answered and not state.exhausted]
    final_ids = tokenizer.encode(FINAL_ANSWER_PROMPT, add_special_tokens=False)
    for state in needs_final:
        add_context(state, final_ids, "final_answer_prompt")
    for state, action in generate_actions(
        llm,
        tokenizer,
        [state for state in needs_final if not state.exhausted],
        args.response_length,
        args.max_model_len,
        args.temperature,
        args.top_p,
        args.seed,
        max_action_length=500 if capture else None,
        capture=capture,
    ):
        state.answered = action is not None and action.kind == "answer"


def teacher_record(state, tokenizer, verifier_prob, response_length, model_revision, tokenizer_hash=None, config_hash=None):
    """Keep one attempted teacher rollout, including rejected trajectories."""
    if len(state.response_ids) != len(state.response_mask):
        raise ValueError(f"response mask mismatch: {state.item_id}")
    reasons = []
    if state.stop_reason:
        reasons.append(state.stop_reason)
    if len(state.response_ids) > response_length:
        reasons.append("response_too_long")
    if any(token_id < 0 or token_id >= len(tokenizer) for token_id, mask in zip(state.response_ids, state.response_mask) if mask):
        reasons.append("unmapped_teacher_token")
    if not state.answered or not state.final_answer:
        reasons.append("no_valid_answer")
    if state.invalid_actions:
        reasons.append("invalid_action")
    if not state.search_records or not any(search["document_hashes"] for search in state.search_records):
        reasons.append("no_retrieval_evidence")
    if verifier_prob is None:
        reasons.append("verifier_error")
    elif verifier_prob <= 0.5:
        reasons.append("verifier_rejected")
    return {
        "trajectory_id": f"{state.item_id}:teacher:{model_revision[:12]}:0",
        "question_id": state.item_id,
        "policy_version": model_revision,
        "question": state.question,
        "references": state.targets,
        "prompt_ids": state.runtime_ids[:state.prompt_length],
        "response_ids": state.response_ids,
        "response_mask": state.response_mask,
        "segments": state.segments,
        "searches": state.search_records,
        "search_count": state.searches,
        "invalid_actions": state.invalid_actions,
        "final_answer": state.final_answer,
        "answered": state.answered,
        "status": "rejected" if reasons else "accepted",
        "stop_reason": reasons,
        "verifier_prob": verifier_prob,
        "acc": int(verifier_prob > 0.5) if verifier_prob is not None else None,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "config_hash": config_hash,
        "response": tokenizer.decode(state.response_ids, skip_special_tokens=True),
    }


def write_teacher_traces(states, tokenizer, args):
    if not args.model_revision:
        raise ValueError("--model-revision is required with --distillation-dir")
    if args.batch_size < 1 or args.max_turns != 2 or args.response_length != 4096:
        raise ValueError("teacher export requires batch-size >= 1, max-turns=2, response-length=4096")
    output = Path(args.distillation_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.data)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    tokenizer_path = Path(args.model) / "tokenizer.json"
    with tokenizer_path.open("rb") as handle:
        tokenizer_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    corpus_path = Path(args.retrieval_corpus)
    with corpus_path.open("rb") as handle:
        corpus_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    script_path = Path(__file__)
    with script_path.open("rb") as handle:
        script_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    config = {
        "data": str(source.resolve()), "data_sha256": source_hash, "model": str(Path(args.model).resolve()),
        "model_revision": args.model_revision, "tokenizer_sha256": tokenizer_hash,
        "retrieval_corpus": str(corpus_path.resolve()), "retrieval_corpus_sha256": corpus_hash,
        "export_script_sha256": script_hash,
        "response_length": args.response_length, "max_model_len": args.max_model_len,
        "max_turns": args.max_turns, "max_action_length": 500,
        "observation_length": args.observation_length, "top_k": args.top_k,
        "search_url": args.search_url, "verifier_url": args.verifier_url,
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
        "count": len(states),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = output / "manifest.json"
    attempts_path = output / "teacher_attempts.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_hash"] != config_hash or old["status"] != "incomplete":
            raise ValueError("existing export is complete or uses another configuration")
    elif any(output.iterdir()):
        raise ValueError("output directory is not empty and has no manifest")

    def save_manifest(status, processed):
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": status, "config_hash": config_hash, "config": config, "processed": processed}, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    prior = []
    if attempts_path.exists():
        with attempts_path.open(encoding="utf-8") as handle:
            prior = [json.loads(line)["question_id"] for line in handle if line.strip()]
        if prior != [state.item_id for state in states[:len(prior)]]:
            raise ValueError("saved attempt IDs are not the input prefix")
    save_manifest("incomplete", len(prior))
    with attempts_path.open("a", encoding="utf-8") as handle:
        for start in range(len(prior), len(states), args.batch_size):
            batch = states[start:start + args.batch_size]
            yield batch, handle, start, save_manifest, tokenizer_hash, config_hash


def run_teacher_export(llm, tokenizer, states, args):
    for batch, handle, start, save_manifest, tokenizer_hash, config_hash in write_teacher_traces(states, tokenizer, args):
        run_agent(llm, tokenizer, batch, args)
        for state in batch:
            try:
                answer = f"<answer>{state.final_answer}</answer>"
                prob = verifier_score(answer, state.targets, args.verifier_url) if state.answered else 0.0
            except Exception:
                prob = None
            handle.write(json.dumps(teacher_record(state, tokenizer, prob, args.response_length, args.model_revision, tokenizer_hash, config_hash), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        save_manifest("incomplete", start + len(batch))
    output = Path(args.distillation_dir)
    counts = {"accepted": 0, "rejected": 0}
    with (output / "teacher_attempts.jsonl").open(encoding="utf-8") as attempts, (output / "accepted.jsonl.tmp").open("w", encoding="utf-8") as accepted, (output / "rejected.jsonl.tmp").open("w", encoding="utf-8") as rejected:
        for line in attempts:
            status = json.loads(line)["status"]
            (accepted if status == "accepted" else rejected).write(line)
            counts[status] += 1
    (output / "accepted.jsonl.tmp").replace(output / "accepted.jsonl")
    (output / "rejected.jsonl.tmp").replace(output / "rejected.jsonl")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"] = counts
    manifest["output_sha256"] = {}
    for name in ("teacher_attempts.jsonl", "accepted.jsonl", "rejected.jsonl"):
        with (output / name).open("rb") as source:
            manifest["output_sha256"][name] = hashlib.file_digest(source, "sha256").hexdigest()
    manifest["status"] = "complete"
    (output / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "manifest.json.tmp").replace(output / "manifest.json")
    print(json.dumps({"total": len(states), **counts}))


def write_results(states, tokenizer, output_path, response_length, verifier_url):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    answered = 0
    searched = 0
    invalid = 0
    verifier_correct = 0
    with output.open("w", encoding="utf-8") as handle:
        for state in states:
            response_ids = state.response_ids[:response_length]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            reward = compute_outcome_reward(
                "medsearch_medical_verifiable",
                response,
                {"target": state.targets},
            )
            verifier_prob = verifier_score(response, state.targets, verifier_url)
            correct += int(reward["acc"])
            answered += int(state.answered)
            searched += int(state.searches > 0)
            invalid += int(state.invalid_actions > 0)
            verifier_correct += int(verifier_prob > 0.5)
            handle.write(
                json.dumps(
                    {
                        "id": state.item_id,
                        "question": state.question,
                        "ground_truth": state.targets,
                        "response": response,
                        "score": reward["score"],
                        "acc": reward["acc"],
                        "verifier_prob": verifier_prob,
                        "verifier_acc": int(verifier_prob > 0.5),
                        "answered": state.answered,
                        "searches": state.searches,
                        "invalid_actions": state.invalid_actions,
                        "response_tokens": len(response_ids),
                        "exhausted": state.exhausted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    total = len(states)
    print(
        json.dumps(
            {
                "total": total,
                "rule_acc": correct / total,
                "verifier_acc": verifier_correct / total,
                "closed_answer_rate": answered / total,
                "searched_rate": searched / total,
                "invalid_action_rate": invalid / total,
            },
            ensure_ascii=False,
        )
    )


def main():
    args = parse_args()
    if args.distillation_dir and (not args.model_revision or not args.retrieval_corpus or args.limit is None):
        raise ValueError("teacher export requires --model-revision, --retrieval-corpus and --limit")
    if not args.distillation_dir and not args.output:
        raise ValueError("evaluation requires --output")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    tokenizer = llm.get_tokenizer()
    states = load_states(args.data, tokenizer, args.limit)
    if args.distillation_dir:
        if len(states) != args.limit:
            raise ValueError(f"requested {args.limit} teacher questions, found {len(states)}")
        run_teacher_export(llm, tokenizer, states, args)
        return
    run_agent(llm, tokenizer, states, args)
    write_results(
        states,
        tokenizer,
        args.output,
        args.response_length,
        args.verifier_url,
    )


if __name__ == "__main__":
    main()
