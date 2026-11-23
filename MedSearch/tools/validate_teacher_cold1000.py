"""Validate one frozen cold-start teacher export without changing its contents."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from MedSearch.tools.build_medical_verifiable_rl import INSTRUCTION


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["processed"] == args.expected
    assert manifest["config"]["count"] == args.expected
    assert manifest["config"]["data_sha256"] == sha256(source)
    assert manifest["config"]["model_revision"] == "74d46a43224daa612b5f27174e1c58b35587ee92"
    assert manifest["config"]["retrieval_corpus_sha256"] == sha256(Path(manifest["config"]["retrieval_corpus"]))
    source_rows = read_jsonl(source)[:args.expected]
    attempts = read_jsonl(output / "teacher_attempts.jsonl")
    accepted = read_jsonl(output / "accepted.jsonl")
    rejected = read_jsonl(output / "rejected.jsonl")
    assert len(source_rows) == len(attempts) == args.expected
    assert len(accepted) + len(rejected) == args.expected
    assert len({row["question_id"] for row in attempts}) == args.expected
    assert [row["question_id"] for row in attempts] == [row["question_id"] for row in source_rows]
    assert accepted == [row for row in attempts if row["status"] == "accepted"]
    assert rejected == [row for row in attempts if row["status"] == "rejected"]
    assert manifest["counts"] == {"accepted": len(accepted), "rejected": len(rejected)}
    for name, expected in manifest["output_sha256"].items():
        assert sha256(output / name) == expected

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    masks = {"answer": 1, "search": 1, "invalid_action": 1, "observation": 0, "correction": 0, "final_answer_prompt": 0}
    for source_row, attempt in zip(source_rows, attempts, strict=True):
        expected_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\nQuestion: {source_row['question']}"}],
            tokenize=True, add_generation_prompt=True,
        )
        if not isinstance(expected_prompt, list):
            expected_prompt = expected_prompt["input_ids"]
        assert attempt["prompt_ids"] == expected_prompt
        assert attempt["question"] == source_row["question"]
        assert attempt["references"] == source_row["references"]
        ids, mask = attempt["response_ids"], attempt["response_mask"]
        assert len(ids) == len(mask) <= manifest["config"]["response_length"]
        assert len(attempt["prompt_ids"]) + len(ids) <= manifest["config"]["max_model_len"]
        assert attempt["response"] == tokenizer.decode(ids, skip_special_tokens=True)
        position = 0
        for segment in attempt["segments"]:
            assert segment["start"] == position and 0 <= segment["start"] < segment["end"] <= len(ids)
            assert segment["kind"] in masks
            assert all(bit == masks[segment["kind"]] for bit in mask[segment["start"]:segment["end"]])
            position = segment["end"]
        assert position == len(ids)
        assert attempt["model_revision"] == manifest["config"]["model_revision"]
        assert attempt["tokenizer_hash"] == manifest["config"]["tokenizer_sha256"]
        assert attempt["config_hash"] == manifest["config_hash"]
        assert attempt["search_count"] == len(attempt["searches"])
        for search in attempt["searches"]:
            assert search["search_url"] == manifest["config"]["search_url"]
            assert search["top_k"] == manifest["config"]["top_k"]
            assert len(search["observation_sha256"]) == 64
            assert search["document_hashes"] and all(len(value) == 64 for value in search["document_hashes"])
        if attempt["status"] == "accepted":
            assert attempt["answered"] and attempt["final_answer"]
            assert attempt["searches"] and attempt["verifier_prob"] > 0.5
            assert not attempt["stop_reason"]
        else:
            assert attempt["stop_reason"]
    print(json.dumps({"validated": len(attempts), "status_counts": dict(Counter(row["status"] for row in attempts)), "source_sha256": sha256(source), "attempts_sha256": sha256(output / "teacher_attempts.jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
