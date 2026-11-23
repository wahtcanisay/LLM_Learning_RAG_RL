"""Search-R1 compatible MedCPT Dense or BM25+MedCPT RRF retrieval."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


def rrf_rank(bm25_ids, dense_ids, topk, constant=60):
    scores = {}
    for ranked_ids in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (constant + rank)
    selected = sorted(scores, key=lambda doc_id: -scores[doc_id])[:topk]
    return selected, [scores[doc_id] for doc_id in selected]


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int = 5
    return_scores: bool = False


def create_app(args):
    torch.set_num_threads(4)
    corpus = [json.loads(line) for line in Path(args.corpus).open(encoding="utf-8")]
    index = faiss.read_index(args.index)
    if index.ntotal != len(corpus):
        raise ValueError(f"Index has {index.ntotal} rows; corpus has {len(corpus)}")
    tokenizer = AutoTokenizer.from_pretrained(args.query_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.query_model, local_files_only=True).eval()
    bm25 = None
    if args.mode == "hybrid":
        from pyserini.search.lucene import LuceneSearcher

        bm25 = LuceneSearcher(args.bm25_index)

    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        outputs = []
        for query in request.queries:
            tokens = tokenizer(
                [query], truncation=True, padding=True, max_length=64, return_tensors="pt"
            )
            with torch.inference_mode():
                vector = model(**tokens).last_hidden_state[:, 0, :]
            vector = np.ascontiguousarray(vector.float().numpy())
            count = max(request.topk, args.candidate_k)
            dense_scores, dense_rows = index.search(vector, count)
            dense_ids = [str(row) for row in dense_rows[0] if row >= 0]
            if bm25 is None:
                selected = dense_ids[: request.topk]
                scores = dense_scores[0, : len(selected)].tolist()
            else:
                bm25_ids = [hit.docid for hit in bm25.search(query, count)]
                selected, scores = rrf_rank(bm25_ids, dense_ids, request.topk)
            docs = [corpus[int(doc_id)] for doc_id in selected]
            outputs.append(
                [{"document": doc, "score": score} for doc, score in zip(docs, scores)]
                if request.return_scores else docs
            )
        return {"result": outputs}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--corpus", default="MedSearch/data/retrieval/medrag_textbooks.jsonl")
    parser.add_argument("--index", default="MedSearch/data/retrieval/medcpt_textbooks.index")
    parser.add_argument("--query-model", default="MedSearch/data/retrieval/medcpt_models/query")
    parser.add_argument("--bm25-index", default="MedSearch/data/retrieval/medrag_textbooks/bm25")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
