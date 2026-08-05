# MedicalGraphRAG

**可复现的医学检索（RAG）研究与评测项目** —— 用统一、可审计的框架实现并对比多种医学文档检索器。

> 你问医学问题 → 系统从医学语料检索相关文档 → 交给生成模型作答。本项目聚焦**检索这一步**，用真实实验回答：BM25、向量检索、混合融合、图检索，哪种找法在什么场景下最好。

完整项目总览见根目录 [`README.md`](../README.md)（含路线图、数据集价值、评测方案、Hybrid2 构想）。

---

## 已实现的检索器

| 检索器 | 文件 | 原理 | 状态 |
|---|---|---|---|
| **BM25** | `src/medical_graphrag/retrieval/bm25.py` | Lucene 关键词检索 | ✅ 已评测 |
| **Dense** | `src/medical_graphrag/retrieval/dense.py` | all-mpnet + FAISS 向量检索 | ✅ 已评测 |
| **Hybrid** | `src/medical_graphrag/retrieval/hybrid.py` | BM25 + Dense 的 RRF 排名融合 | ✅ 已评测 |
| **Graph** | `src/medical_graphrag/retrieval/graph.py` | 医学实体图 + PPR（移植 LinearRAG） | ✅ 已评测 |
| **Reranker（Hybrid2）** | `src/medical_graphrag/retrieval/reranker.py` | Qwen3-Reranker 三路候选重排 | 🔜 开发中 |

## 评测（多相关 qrels，可独立复算）

| 方法 | pubmedqa R@10 | nfcorpus R@10 | nfcorpus nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.984 | 0.150 | 0.313 |
| Dense | **0.994** | 0.159 | 0.325 |
| Hybrid | 0.992 | **0.172** | **0.354** |
| Graph | 0.958 | 0.157 | 0.314 |

> 结论：单答案基准上 Dense 最优；多相关基准上 Hybrid（RRF 融合）最优。基准选择会改变"哪种检索器更好"的结论。

## 快速开始

环境：容器 `llm-pytorch`，Python 环境 `/opt/venv`（含 igraph / scispacy / sentence-transformers / faiss）。

```bash
# 评测 Dense（示例）
/opt/venv/bin/python scripts/build_faiss_dense_index.py \
  --dataset-dir data/processed/nfcorpus_v1 \
  --output-dir outputs/nfcorpus_v1/dense_abstract_only \
  --embedding-model models/all-mpnet-base-v2

/opt/venv/bin/python scripts/search_faiss_dense.py \
  --index outputs/nfcorpus_v1/dense_abstract_only/index.faiss \
  --embeddings outputs/nfcorpus_v1/dense_abstract_only/chunk_embeddings.npy \
  --questions data/processed/nfcorpus_v1/questions.jsonl \
  --metadata outputs/nfcorpus_v1/dense_abstract_only/chunk_metadata.jsonl \
  --output outputs/nfcorpus_v1/dense_abstract_only/raw_rankings.jsonl \
  --report outputs/nfcorpus_v1/dense_abstract_only/search_run.json \
  --index-report outputs/nfcorpus_v1/dense_abstract_only/index_build.json \
  --top-k 100 --embedding-model models/all-mpnet-base-v2

/opt/venv/bin/python -m medical_graphrag.cli evaluate-dense \
  --dataset-dir data/processed/nfcorpus_v1 \
  --metadata outputs/nfcorpus_v1/dense_abstract_only/chunk_metadata.jsonl \
  --rankings outputs/nfcorpus_v1/dense_abstract_only/raw_rankings.jsonl \
  --index-report outputs/nfcorpus_v1/dense_abstract_only/index_build.json \
  --search-report outputs/nfcorpus_v1/dense_abstract_only/search_run.json \
  --output-dir experiments/nfcorpus_v1/dense_abstract_only \
  --git-commit $(git rev-parse HEAD) \
  --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

## 数据集

- `data/processed/pubmedqa_hard_v1/`：自建，1,000 题 / 5,000 篇，单答案；
- `data/processed/nfcorpus_v1/`：BEIR 公开基准，323 题 / 3,633 篇，多相关；
- 均为哈希冻结、可重建（`scripts/build_*.py`），git 忽略。

## 项目结构

```text
src/medical_graphrag/
  data/          # 数据加载、切块、冻结数据
  retrieval/     # bm25 / dense / hybrid / graph / reranker
  evaluation/    # 指标（含多相关）+ 各检索器评测
scripts/         # 容器内构建/检索/重排脚本
experiments/     # 实验结果 metrics.json（可审计）
```

## 铁律

只改一个变量、指标必须真实脚本输出、哈希审计、不编造结果。
