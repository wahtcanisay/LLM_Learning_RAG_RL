"""Sample frozen dev questions and compare BM25 top-k from textbook and wiki corpora."""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from pyserini.search.lucene import LuceneSearcher


TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "what", "which", "with", "would", "most", "likely", "does",
}


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def textbook_search(url: str, query: str, topk: int) -> list[dict]:
    payload = json.dumps({"queries": [query], "topk": topk, "return_scores": True}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)["result"][0]


def wiki_hits(searcher: LuceneSearcher, query: str, topk: int) -> list[dict]:
    return [{"docid": str(hit.docid), "score": float(hit.score)} for hit in searcher.search(query, topk)]


def read_wiki_lines(path: Path, wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if str(line_no) not in wanted:
                continue
            row = json.loads(line)
            found[str(row.get("id", line_no))] = row
            if len(found) == len(wanted):
                break
    return found


def compact_hit(hit: dict, answer_tokens: set[str]) -> dict:
    doc = hit.get("document", {})
    text = doc.get("content") or doc.get("contents") or ""
    present = answer_tokens & tokens(text)
    return {
        "id": str(doc.get("id", "")),
        "title": doc.get("title", ""),
        "score": float(hit.get("score", 0.0)),
        "chars": len(text),
        "answer_tokens_present": sorted(present),
        "snippet": re.sub(r"\s+", " ", text)[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, default=Path("MedSearch/data/rl/dev.parquet"))
    parser.add_argument("--wiki-index", type=Path, default=Path("Search-R1/data/wiki-18-bm25-index/bm25"))
    parser.add_argument("--wiki-corpus", type=Path, default=Path("Search-R1/data/wiki-18-corpus-xet/extracted/wiki_dump.jsonl"))
    parser.add_argument("--textbook-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pq.read_table(args.dev).to_pylist()
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.sample_size, len(rows)))
    searcher = LuceneSearcher(str(args.wiki_index))
    records = []
    wiki_ids: set[str] = set()
    for row in sample:
        question = row["question"]
        answers = row.get("golden_answers") or row.get("reward_model", {}).get("ground_truth", {}).get("target", [])
        answer_tokens = set().union(*(tokens(str(answer)) for answer in answers))
        tb_raw = textbook_search(args.textbook_url, question, args.topk)
        wh = wiki_hits(searcher, question, args.topk)
        wiki_ids.update(item["docid"] for item in wh)
        records.append({
            "id": row.get("id"),
            "question": question,
            "answers": answers,
            "textbook": {"hits": tb_raw, "compact": [compact_hit(hit, answer_tokens) for hit in tb_raw]},
            "wiki": {"hits": wh},
        })

    wiki_docs = read_wiki_lines(args.wiki_corpus, wiki_ids)
    textbook_ok = wiki_ok = 0
    textbook_top1 = wiki_top1 = 0
    for record in records:
        answer_tokens = set().union(*(tokens(str(answer)) for answer in record["answers"]))
        tb_compact = record["textbook"]["compact"]
        wiki_compact = []
        for hit in record["wiki"]["hits"]:
            row = wiki_docs.get(hit["docid"], {})
            contents = row.get("contents", "")
            title, _, body = contents.partition("\n")
            present = answer_tokens & tokens(contents)
            wiki_compact.append({
                "id": hit["docid"], "title": title.strip('"'), "score": hit["score"],
                "chars": len(body), "answer_tokens_present": sorted(present),
                "snippet": re.sub(r"\s+", " ", body)[:500],
            })
        record["wiki"]["compact"] = wiki_compact
        tb_flags = [bool(set(hit["answer_tokens_present"]) >= answer_tokens) if answer_tokens else False for hit in tb_compact]
        wiki_flags = [bool(set(hit["answer_tokens_present"]) >= answer_tokens) if answer_tokens else False for hit in wiki_compact]
        record["diagnostics"] = {
            "textbook_answer_all_tokens_in_top5": any(tb_flags),
            "wiki_answer_all_tokens_in_top5": any(wiki_flags),
            "textbook_answer_any_token_top1": bool(set(tb_compact[0]["answer_tokens_present"]) & answer_tokens) if tb_compact else False,
            "wiki_answer_any_token_top1": bool(set(wiki_compact[0]["answer_tokens_present"]) & answer_tokens) if wiki_compact else False,
        }
        textbook_ok += record["diagnostics"]["textbook_answer_all_tokens_in_top5"]
        wiki_ok += record["diagnostics"]["wiki_answer_all_tokens_in_top5"]
        textbook_top1 += record["diagnostics"]["textbook_answer_any_token_top1"]
        wiki_top1 += record["diagnostics"]["wiki_answer_any_token_top1"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {"dev": str(args.dev), "sample_size": len(records), "seed": args.seed, "topk": args.topk,
                   "textbook_url": args.textbook_url, "wiki_index": str(args.wiki_index), "wiki_corpus": str(args.wiki_corpus)},
        "summary": {"textbook_all_answer_tokens_top5": textbook_ok, "wiki_all_answer_tokens_top5": wiki_ok,
                    "textbook_any_answer_token_top1": textbook_top1, "wiki_any_answer_token_top1": wiki_top1},
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample": len(records), "wiki_docs": len(wiki_docs), "summary": {
        "textbook_all_answer_tokens_top5": textbook_ok, "wiki_all_answer_tokens_top5": wiki_ok,
        "textbook_any_answer_token_top1": textbook_top1, "wiki_any_answer_token_top1": wiki_top1}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
