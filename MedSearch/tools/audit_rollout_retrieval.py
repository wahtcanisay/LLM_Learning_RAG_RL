"""Audit BM25 passages for the first search in saved agent rollouts."""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from MedSearch.rl.retriever_client import build_observation, encode_observation, request_retrieval


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_answer(text, answers):
    haystack = normalized(text)
    return any(normalized(answer) in haystack for answer in answers if normalized(answer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--search-url", default="http://127.0.0.1:8001/retrieve")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--observation-length", type=int, default=700)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = [json.loads(line) for line in args.results.open(encoding="utf-8")]
    searchable = []
    for record in records:
        queries = re.findall(r"<search>(.*?)</search>", record["response"], re.S)
        if queries:
            searchable.append((record, queries[0].strip()))

    audits = []
    for start in range(0, len(searchable), 16):
        batch = searchable[start : start + 16]
        # One HTTP request per query uses the exact Search-R1 /retrieve contract.
        for record, query in batch:
            payload = request_retrieval(query, args.search_url, args.top_k, 30)
            passages = payload["result"][0]
            assert len(passages) <= args.top_k
            full = build_observation(payload)
            visible = tokenizer.decode(
                encode_observation(full, tokenizer, args.observation_length),
                skip_special_tokens=True,
            )
            docs = [item["document"]["contents"] for item in passages]
            audits.append(
                {
                    "id": record["id"],
                    "query": query,
                    "question": record["question"],
                    "ground_truth": record["ground_truth"],
                    "answer_correct": record["acc"],
                    "verifier_correct": record["verifier_acc"],
                    "full_gold_lexical_hit": contains_answer(" ".join(docs), record["ground_truth"]),
                    "visible_gold_lexical_hit": contains_answer(visible, record["ground_truth"]),
                    "full_observation_tokens": len(tokenizer.encode(full, add_special_tokens=False)),
                    "visible_doc_count": visible.count("Doc "),
                    "docs": docs,
                }
            )
        print(f"audited={len(audits)}/{len(searchable)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in audits:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "searched": len(audits),
                "full_gold_lexical_hit": sum(item["full_gold_lexical_hit"] for item in audits),
                "visible_gold_lexical_hit": sum(item["visible_gold_lexical_hit"] for item in audits),
                "over_budget": sum(item["full_observation_tokens"] > args.observation_length for item in audits),
            }
        )
    )


if __name__ == "__main__":
    main()
