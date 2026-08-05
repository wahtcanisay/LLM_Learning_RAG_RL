# MedicalGraphRAG

**可复现的医学检索（RAG）研究与评测项目** —— 用统一、可审计的框架实现并对比多种医学文档检索器，为后续医学问答智能体（MedSearch-R1）打牢检索地基。

> 简单说：这是一个"医学资料怎么找最准"的实验平台。你问医学问题 → 系统从医学语料里检索相关文档 → 交给生成模型作答。本项目聚焦**检索这一步**，用真实实验回答：BM25、向量检索、混合融合、图检索，哪种找法在什么场景下最好。

---

## 核心特性

- **统一检索接口**：四种检索器（BM25 / Dense / Hybrid / Graph）实现同一 `search(query, top_k)` 契约，可互换、可公平对比；
- **全链路审计**：数据冻结加哈希、每步产物 SHA-256 绑定、指标可独立复算——**不编造结果**是项目铁律；
- **双基准评测**：自建 `pubmedqa_hard_v1`（单答案）＋ 国际标准 BEIR/NFCorpus（多相关答案）；
- **多相关指标**：评测泛化支持 BEIR 风格多相关 qrels（Recall@k / MRR@10 / nDCG@10）；
- **图检索实现**：忠实移植 LinearRAG 的医学实体图检索（BC5CDR NER → 实体-文章图 → PPR），含诚实负结果。

---

## 架构

```text
                    ┌─────────────────────────────────────┐
                    │          冻结数据（可复现）           │
                    │  pubmedqa_hard_v1  /  nfcorpus_v1   │
                    └──────────────────┬──────────────────┘
                                       ↓
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   │   BM25       │    Dense     │   Hybrid     │    Graph     │
   │  (Lucene)    │  (FAISS)     │  (RRF 融合)   │(BC5CDR+PPR)  │
   └──────────────┴──────────────┴──────────────┴──────────────┘
                                       ↓
                       统一 raw rankings（chunk → document）
                                       ↓
                    evaluate_rankings（Recall@k / MRR / nDCG）
                                       ↓
                    metrics.json + 审计链（哈希绑定，可复算）
```

四个检索器各自解决一个问题：

| 检索器 | 原理 | 强项 |
|---|---|---|
| **BM25** | 关键词词频/文档频率统计 | 术语明确的问题 |
| **Dense** | 文本向量化后找语义最近 | 换说法也能匹配 |
| **Hybrid** | BM25 + Dense 排名做 RRF 融合 | 两路信号互补时 |
| **Graph** | 医学实体建图，PPR 沿边传播 | 多跳/跨实体（理论优势） |

---

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | MedRAG 检索基础 | ✅ 完成 |
| 2 | LinearRAG 图检索迁移 | ✅ 完成 |
| 3 | MedicalGPT 领域微调（SFT） | 🔜 进行中（并行线预习中） |
| 4 | Search-R1 搜索强化学习 | ⏳ 未开始 |
| 5 | MedSearch-R1 医学搜索智能体 | ⏳ 未开始 |

**检索层已全部落地并审计**：4 检索器 × 2 基准 × 真实指标。

---

## 数据集

### pubmedqa_hard_v1（自建，单答案）

- 1,000 道医学选择题 + 5,000 篇医学摘要（1,000 gold + 4,000 干扰）；
- 每题恰 1 篇正确答案，文档级 qrels，整包哈希冻结；
- **用途**：可控、自建，验证检索器在"语义匹配为主"场景的表现。

### nfcorpus_v1（BEIR 公开基准，多答案）—— **有用，且关键**

- 国际标准检索基准 NFCorpus 的医学子集：323 道题 / 3,633 篇文章，**每题平均 38 篇相关文档**，官方多相关 qrels；
- **它有没有用？很有用**，理由：
  1. **有权威标注** —— 解决了 GraphRAG-Bench 无文档级 gold 的痛点，能算出可信的 Recall@k；
  2. **补上多答案视角** —— 单答案基准看不出"多相关召回"能力；
  3. **改变了结论** —— 正是在它上面，我们发现 Hybrid（RRF 融合）真正胜出，而在自建单答案基准上是 Dense 胜出。**这证明了数据集选择会直接改变"哪种检索器更好"的结论。**

> 数据可重建：`MedicalGraphRAG/scripts/build_nfcorpus.py` + 原始下载，全部产物带哈希。

---

## 实验结果（诚实版）

### pubmedqa_hard_v1（test 500）

| 方法 | Recall@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.946 | 0.955 |
| **Dense** | **0.966** | **0.994** | **0.978** | **0.982** |
| Hybrid | 0.960 | 0.992 | 0.974 | 0.979 |
| Graph | 0.800 | 0.958 | 0.857 | 0.881 |

### nfcorpus_v1（test 323）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| **Hybrid** | **0.172** | **0.552** | **0.354** |
| Graph | 0.157 | 0.483 | 0.314 |

**结论**：
- 单答案、语义为主 → **Dense** 最优；
- 多相关、信号均衡 → **Hybrid（RRF 融合）**最优；
- **图检索在两个基准均未胜出**（实现正确、有审计背书，但当前数据非多跳推理场景，价值未体现——诚实负结果）；
- 所有指标独立复算误差 < 1e-15。

---

## 如何评测 RAG 效果

分三层，逐层加码：

1. **检索层（已完成）**：Recall@k / MRR@10 / nDCG@10，在两个基准上对比检索器。新增检索器 = 实现 `search(query, top_k)` → 接入统一评测即可。
2. **端到端问答层（阶段 3 后）**：检索器召回 top-k → 喂给生成模型 → **QA Accuracy**（答案正确率）。这才能回答"检索命中是否真的带来了更好的答案"。计划用 `Base / SFT / SFT+RAG` 对比。
3. **工程层**：延迟、索引大小、显存峰值、token 成本——尤其对最终智能体（阶段 5）至关重要。

原则：**每次只改一个变量**（换检索器、换融合、换生成模型分开测），保证能定位提升来源。

---

## 快速开始

环境：WSL2 Docker 容器 `llm-pytorch`，Python 环境 `/opt/venv`（含 igraph / scispacy / sentence-transformers / faiss）。

```bash
# 评测一个检索器（以 Dense 为例）
cd MedicalGraphRAG

# 1) 建向量索引
/opt/venv/bin/python scripts/build_faiss_dense_index.py \
  --dataset-dir data/processed/nfcorpus_v1 \
  --output-dir outputs/nfcorpus_v1/dense_abstract_only \
  --embedding-model models/all-mpnet-base-v2

# 2) 检索
/opt/venv/bin/python scripts/search_faiss_dense.py \
  --index outputs/nfcorpus_v1/dense_abstract_only/index.faiss \
  --embeddings outputs/nfcorpus_v1/dense_abstract_only/chunk_embeddings.npy \
  --questions data/processed/nfcorpus_v1/questions.jsonl \
  --metadata outputs/nfcorpus_v1/dense_abstract_only/chunk_metadata.jsonl \
  --output outputs/nfcorpus_v1/dense_abstract_only/raw_rankings.jsonl \
  --report outputs/nfcorpus_v1/dense_abstract_only/search_run.json \
  --index-report outputs/nfcorpus_v1/dense_abstract_only/index_build.json \
  --top-k 100 --embedding-model models/all-mpnet-base-v2

# 3) 评测
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

> 详细命令与各检索器差异见 `docs/superpowers/specs/` 的设计文档与根目录 `HANDOFF_*.md` 交接文档。

---

## 项目结构

```text
MedicalGraphRAG/           # 主项目：检索、评测、实验
  src/medical_graphrag/
    data/                  # 数据加载、切块、冻结数据
    retrieval/             # bm25 / dense / hybrid / graph 四种检索器
    evaluation/            # 指标（含多相关）+ 各检索器评测
  scripts/                 # 容器内构建/检索脚本
  experiments/             # 实验结果 metrics.json（可审计）
  data/processed/          # 冻结数据集（可重建，git 忽略）

LinearRAG/  MedRAG/        # 只读参考实现与数据源
MedicalGPT/                # 阶段 3 并行线（SFT 预习）
docs/superpowers/specs/    # 各阶段设计文档
STUDY_PROGRESS.md          # 学习/项目进度
HANDOFF_*.md               # 各阶段交接文档
```

---

## 路线图（Roadmap）

- [x] 检索层：BM25 / Dense / Hybrid / Graph 四路基线 × 双基准
- [x] 评测泛化：多相关 qrels 指标
- [ ] **Hybrid2：三路检索 + Qwen 重排**（构想，未实现）
  - 思路：第一路融合 BM25 + Dense + Graph 得到候选，再上 **Qwen3-Reranker**（cross-encoder 专用重排模型）对 top-50 候选重排，取 top-10 进生成；
  - 与当前 RRF（只看排名）不同，reranker 是"把 query 和文档拼一起喂模型"的联合打分，理论上精度更高；
  - 依赖：Qwen3-Reranker 模型下载 + 统一检索接口扩展；排在 Dense/Hybrid/LinearRAG 全部基线后。
- [ ] 阶段 3：MedicalGPT 领域微调（进行中）
- [ ] 阶段 4：Search-R1 搜索强化学习
- [ ] 阶段 5：MedSearch-R1 医学搜索智能体

---

## 许可

研究用途，非临床产品。不提供诊断建议。
