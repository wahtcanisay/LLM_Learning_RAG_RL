"""Task 6: 使用真实 toy BM25/Dense 结果验证 MedRAG 的 RRF merge。"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

# 直接执行 docs/task6_toy_rrf.py 时，Python 默认把 docs/ 放入 sys.path；
# 手动加入项目根目录，才能导入同级的 src.utils。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import RetrievalSystem


COLLECTION = Path(__file__).parent / "task4_toy_collection"
BM25_INDEX = Path("/tmp/medrag_task4_index")
DENSE_INDEX = Path("/tmp/medrag_task5_dense/faiss.index")
DENSE_METADATA = Path("/tmp/medrag_task5_dense/metadatas.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "facial nerve"
RRF_K = 100


def load_item(source, index):
    path = COLLECTION / f"{source}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[index])


def normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def main():
    # BM25：Lucene 返回 docid 和 BM25 原始分数。
    bm25 = LuceneSearcher(str(BM25_INDEX))
    bm25_hits = bm25.search(QUERY, k=3)
    bm25_texts = []
    bm25_scores = []
    for hit in bm25_hits:
        parts = hit.docid.split("_")
        source = "_".join(parts[:-1])
        index = int(parts[-1])
        bm25_texts.append(load_item(source, index))
        bm25_scores.append(float(hit.score))

    # Dense：加载 Task 5 的 FAISS index 和同一个 query encoder。
    dense_index = faiss.read_index(str(DENSE_INDEX))
    metadata = [
        json.loads(line)
        for line in DENSE_METADATA.read_text(encoding="utf-8").splitlines()
    ]
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [QUERY], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    query_embedding = normalize_rows(query_embedding).astype("float32")
    dense_scores, dense_positions = dense_index.search(query_embedding, k=3)

    dense_texts = []
    dense_score_list = []
    for score, position in zip(dense_scores[0], dense_positions[0]):
        meta = metadata[int(position)]
        dense_texts.append(load_item(meta["source"], meta["index"]))
        dense_score_list.append(float(score))

    print("BM25_RESULTS", [(item["id"], score) for item, score in zip(bm25_texts, bm25_scores)])
    print("DENSE_RESULTS", [(item["id"], score) for item, score in zip(dense_texts, dense_score_list)])

    # merge() 需要“检索器 × 语料库”的嵌套输入；这里分别提供一套 BM25 和 Dense 结果。
    # 不调用 RetrievalSystem.__init__，避免再次触发真实语料/索引初始化。
    system = object.__new__(RetrievalSystem)
    system.retriever_name = "RRF-2"
    system.corpus_name = "Textbooks"
    texts = [[bm25_texts], [dense_texts]]
    scores = [[bm25_scores], [dense_score_list]]
    merged_texts, merged_scores = system.merge(texts, scores, k=3, rrf_k=RRF_K)

    print("RRF_RESULTS")
    for rank, (item, score) in enumerate(zip(merged_texts, merged_scores), start=1):
        print({"rank": rank, "id": item["id"], "rrf_score": score, "title": item["title"]})


if __name__ == "__main__":
    main()
