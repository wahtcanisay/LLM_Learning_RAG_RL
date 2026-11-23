"""Build a compact MedCPT FAISS index over the existing BM25 JSONL corpus."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).cuda().eval()
    index = faiss.IndexFlatIP(model.config.hidden_size)
    batch = []

    def add_batch():
        if not batch:
            return
        encoded = tokenizer(
            batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            vectors = model(**encoded).last_hidden_state[:, 0, :]
        index.add(np.ascontiguousarray(vectors.float().cpu().numpy()))
        batch.clear()

    with args.corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if args.limit is not None and line_number >= args.limit:
                break
            row = json.loads(line)
            if row["id"] != str(line_number):
                raise ValueError(f"Corpus row {line_number} has id {row['id']}")
            batch.append([row["title"], row["content"]])
            if len(batch) >= args.batch_size:
                add_batch()
                if index.ntotal % 4800 == 0:
                    print(f"indexed={index.ntotal}", flush=True)
    add_batch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(args.output)
    print(json.dumps({"indexed": index.ntotal, "dimension": index.d, "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
