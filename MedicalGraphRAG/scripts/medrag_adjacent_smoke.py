"""MedRAG 长上下文启动测试(smoke):验证 chunk+adjacent 模式在真实长文本教材上可建图可检索。

用法:
    /opt/venv/bin/python scripts/medrag_adjacent_smoke.py \
        <medrag_textbooks_chunk_dir> <output_dir>

从 MedRAG textbooks/statpearls chunk 取前 N 个文件、每个文件前 K 个 chunk,经
medrag_adapter 转为 doc_id/order,构造冻结数据集,跑 build_graph_index(chunk, adjacent),
校验 expected==actual 相邻边,并做一次 PPR 检索。无 qrels 时不报 Recall/MRR/nDCG。
"""
import hashlib
import json
import sys
from pathlib import Path

from medical_graphrag.data.medrag_adapter import adapt_medrag_file
from medical_graphrag.data.io import sha256_file, write_json, write_jsonl, write_qrels
from medical_graphrag.retrieval.graph import (
    GraphBuildConfig,
    GraphConfig,
    LinearGraphRetriever,
    build_graph_index,
)

MAX_FILES = 3
MAX_CHUNKS_PER_DOC = 20


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frozen_dataset(
    chunk_dir: Path, dataset_dir: Path, *, max_files: int, max_chunks: int
) -> Path:
    files = sorted(chunk_dir.glob("*.jsonl"))[:max_files]
    if not files:
        raise SystemExit(f"no MedRAG chunk files under {chunk_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    for path in files:
        passages = adapt_medrag_file(path)
        doc_id = path.stem
        chosen = passages[:max_chunks]
        # document content = 首段 content(仅用于满足冻结契约,adjacent 建图用 chunks)
        content = chosen[0].content if chosen else ""
        documents.append({
            "doc_id": doc_id, "title": doc_id, "content": content,
            "source": "medrag", "year": None,
        })
        for p in chosen:
            chunks.append({
                "chunk_id": p.passage_id, "doc_id": p.doc_id, "order": p.order,
                "title": p.title, "content": p.content, "source": p.source,
            })

    questions = [{
        "query_id": "q1", "question": "What is the function of white blood cells in the immune response?",
        "answer": "yes", "long_answer": "x", "split": "test",
    }]
    qrels = [("q1", documents[0]["doc_id"], 1)]
    write_jsonl(dataset_dir / "documents.jsonl", documents)
    write_jsonl(dataset_dir / "chunks.jsonl", chunks)
    write_jsonl(dataset_dir / "questions.jsonl", questions)
    write_qrels(dataset_dir / "qrels.tsv", qrels)
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    manifest = {
        "dataset": "medrag_adjacent_smoke",
        "counts": {
            "questions": len(questions),
            "documents": len(documents),
            "chunks": len(chunks),
            "qrels": len(qrels),
        },
        "artifact_hashes": {name: _sha(dataset_dir / name) for name in names},
    }
    write_json(dataset_dir / "manifest.json", manifest)
    return dataset_dir


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    chunk_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    dataset_dir = build_frozen_dataset(
        chunk_dir, output_dir / "dataset",
        max_files=MAX_FILES, max_chunks=MAX_CHUNKS_PER_DOC,
    )
    index_dir = output_dir / "index"

    build_config = GraphBuildConfig(
        retrieval_unit="chunk",
        passage_edge_mode="adjacent",
        embedding_model="models/all-mpnet-base-v2",
        ner_model="en_ner_bc5cdr_md",
    )
    report = build_graph_index(dataset_dir, index_dir, build_config=build_config)
    print("== graph build ==")
    print("profile:", report["graph_profile"])
    print("retrieval_unit:", report["retrieval_unit"])
    print("passage_edge_mode:", report["passage_edge_mode"])
    print("edge_by_type:", report["edge_count_by_type"])
    print("adjacent_diag:", report["passage_passage_diagnostics"])
    assert report["graph_profile"] == "linearrag_adjacent_v1"
    assert report["passage_edge_mode"] == "adjacent"
    assert report["edge_count_by_type"]["adjacent"] > 0
    assert report["edge_count_by_type"]["similarity"] == 0
    # expected == actual 已在 build 内校验;这里再打印确认
    diag = report["passage_passage_diagnostics"]
    assert diag["expected_edge_count"] == diag["actual_edge_count"]

    retriever = LinearGraphRetriever(
        index_dir,
        config=GraphConfig(
            embedding_model="models/all-mpnet-base-v2",
            ner_model="en_ner_bc5cdr_md",
        ),
    )
    assert retriever.retrieval_unit == "chunk"
    assert retriever.passage_edge_mode == "adjacent"
    top_ids, scores = retriever.search(
        "What is the function of white blood cells in the immune response?", top_k=5
    )
    print("== search ==")
    print("top passages:", top_ids[:5])
    print("scores:", [round(s, 4) for s in scores[:5]])
    assert len(top_ids) == 5
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
