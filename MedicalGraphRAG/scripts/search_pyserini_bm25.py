import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any


def package_version(
    package: str,
    resolver: Callable[[str], str] = distribution_version,
) -> str:
    return resolver(package)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_hit_counts(
    rows: list[dict[str, object]],
    *,
    requested_top_k: int,
) -> dict[str, object]:
    counts = [len(row["hits"]) for row in rows]
    histogram = Counter(counts)
    return {
        "requested_top_k": requested_top_k,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(count < requested_top_k for count in counts),
        "hit_count_histogram": {
            str(count): frequency for count, frequency in sorted(histogram.items())
        },
    }


def search_one(
    searcher: Any,
    question: Mapping[str, object],
    chunk_to_doc: Mapping[str, str],
    top_k: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    raw_hits = searcher.search(str(question["question"]), k=top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = []
    for rank, hit in enumerate(raw_hits, start=1):
        if hit.docid not in chunk_to_doc:
            raise ValueError(f"unknown chunk_id from Pyserini: {hit.docid}")
        hits.append(
            {
                "chunk_id": hit.docid,
                "doc_id": chunk_to_doc[hit.docid],
                "chunk_rank": rank,
                "score": float(hit.score),
            }
        )
    return {
        "query_id": str(question["query_id"]),
        "split": str(question["split"]),
        "latency_ms": latency_ms,
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)
    questions = _read_jsonl(args.questions)
    metadata = _read_jsonl(args.metadata)
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in metadata}
    rows = [search_one(searcher, row, chunk_to_doc, args.top_k) for row in questions]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "command": sys.argv,
                "query_count": len(rows),
                **hit_summary,
                "k1": args.k1,
                "b": args.b,
                "pyserini_version": package_version("pyserini"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
