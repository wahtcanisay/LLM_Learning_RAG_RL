"""Task 5: 用三条 toy JSONL 文档验证 Dense Embedding + FAISS 映射闭环。

这个脚本不调用 MedRAG.Retriever，避免初始化时下载完整语料；
它只复现 embed()、construct_index() 和 Dense query 的核心数据流。
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DOC_PATH = Path(__file__).parent / "task4_toy_collection" / "toy.jsonl"
INDEX_DIR = Path("/tmp/medrag_task5_dense")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_rows(vectors):
    """将向量归一化，使内积等价于 cosine 相似度。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def main():
    items = [json.loads(line) for line in DOC_PATH.read_text(encoding="utf-8").splitlines()]
    texts = [item["contents"] for item in items]

    # 文档和 query 必须使用同一个 Embedding 模型与相同的归一化方式。
    model = SentenceTransformer(MODEL_NAME)
    doc_embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    doc_embeddings = normalize_rows(doc_embeddings).astype("float32")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", doc_embeddings)

    # IndexFlatIP 对 toy 数据做精确内积检索；不使用 HNSW，便于理解映射。
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    # metadata 的行顺序必须与 index.add() 的向量顺序完全一致。
    metadata = []
    for position, item in enumerate(items):
        metadata.append({"index": position, "source": "toy"})
    (INDEX_DIR / "metadatas.jsonl").write_text(
        "\n".join(json.dumps(item) for item in metadata) + "\n",
        encoding="utf-8",
    )
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # query 也必须经过同一个模型和归一化处理。
    query_embedding = model.encode(
        ["facial nerve"], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    query_embedding = normalize_rows(query_embedding).astype("float32")
    scores, positions = index.search(query_embedding, k=3)

    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        meta = metadata[int(position)]
        item = items[meta["index"]]
        print(
            {
                "rank": rank,
                "faiss_position": int(position),
                "score": float(score),
                "source": meta["source"],
                "index": meta["index"],
                "id": item["id"],
                "title": item["title"],
            }
        )


if __name__ == "__main__":
    main()
