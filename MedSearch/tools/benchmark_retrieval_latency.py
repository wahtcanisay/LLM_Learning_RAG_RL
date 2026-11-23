"""Measure paired HTTP retrieval latency on saved first-search queries."""

import argparse
import json
import statistics
import time
from pathlib import Path

from MedSearch.rl.retriever_client import request_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    selected = set(pd.read_parquet(args.ids)["id"].astype(str))
    queries = [
        row["query"]
        for row in map(json.loads, args.audit.open(encoding="utf-8"))
        if row["id"] in selected
    ]
    urls = {
        "bm25": "http://127.0.0.1:8001/retrieve",
        "hybrid": "http://127.0.0.1:8003/retrieve",
    }
    times = {name: [] for name in urls}
    for url in urls.values():
        request_retrieval(queries[0], url, 5, 30)
    for index, query in enumerate(queries):
        names = list(urls) if index % 2 == 0 else list(urls)[::-1]
        for name in names:
            started = time.perf_counter()
            request_retrieval(query, urls[name], 5, 30)
            times[name].append((time.perf_counter() - started) * 1000)
    summary = {
        name: {
            "queries": len(values),
            "mean_ms": round(statistics.mean(values), 1),
            "median_ms": round(statistics.median(values), 1),
            "p95_ms": round(sorted(values)[int(0.95 * (len(values) - 1))], 1),
        }
        for name, values in times.items()
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
