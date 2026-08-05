"""Build a frozen retrieval dataset from any BEIR dataset (HuggingFace parquet).

Run inside the container (needs pandas + pyarrow). Reads downloaded raw files in
`--raw-dir` (corpus.parquet, queries.parquet, {qrels-split}.tsv) and writes a
frozen dataset under `--output-dir` in the same contract as `nfcorpus_v1`:
documents.jsonl, chunks.jsonl (one per document), questions.jsonl (the qrels
split), qrels.tsv (multi-gold relevance=1), manifest.json.

Usage: python build_beir.py --raw-dir data/raw/scifact --output-dir data/processed/scifact_v1 --source beir/scifact --qrels-split test
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl, write_qrels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="beir")
    parser.add_argument("--qrels-split", default="test")
    args = parser.parse_args()

    corpus = pd.read_parquet(args.raw_dir / "corpus.parquet")
    queries = pd.read_parquet(args.raw_dir / "queries.parquet")
    qrels = pd.read_csv(
        args.raw_dir / f"{args.qrels_split}.tsv",
        sep="\t",
        skiprows=1,
        names=["qid", "did", "score"],
    )
    qrels["score"] = pd.to_numeric(qrels["score"], errors="coerce")
    qrels = qrels[qrels["score"] >= 1]

    documents = [
        {
            "doc_id": str(row["_id"]),
            "title": str(row["title"]),
            "content": str(row["text"]),
            "source": args.source,
            "year": None,
        }
        for row in corpus.to_dict("records")
    ]
    chunks = [
        {
            "chunk_id": str(row["_id"]),
            "doc_id": str(row["_id"]),
            "order": 0,
            "title": str(row["title"]),
            "content": str(row["text"]),
            "source": args.source,
        }
        for row in corpus.to_dict("records")
    ]
    query_by_id = {
        str(row["_id"]): str(row["text"]) for row in queries.to_dict("records")
    }
    test_query_ids = [str(qid) for qid in sorted(qrels["qid"].unique())]
    questions = [
        {
            "query_id": qid,
            "question": query_by_id[qid],
            "answer": "",
            "long_answer": "",
            "split": "test",
        }
        for qid in test_query_ids
    ]
    qrel_rows = [(str(r["qid"]), str(r["did"]), 1) for r in qrels.to_dict("records")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    write_qrels(args.output_dir / "qrels.tsv", qrel_rows)

    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    artifact_hashes = {name: sha256_file(args.output_dir / name) for name in names}
    counts = {
        "questions": len(questions),
        "documents": len(documents),
        "chunks": len(chunks),
        "qrels": len(qrel_rows),
    }
    write_json(
        args.output_dir / "manifest.json",
        {"counts": counts, "artifact_hashes": artifact_hashes},
    )
    per_query = qrels.groupby("qid").size()
    print(json.dumps(counts))
    print(
        f"relevant docs per query: mean {per_query.mean():.2f}, median {per_query.median():.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
