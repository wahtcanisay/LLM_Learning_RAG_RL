# 项目交接文档（单一文档，聚合全部阶段）

更新时间：2026-08-05
项目根目录：`D:\code_list\some tricks\LLMLeanring`
分支：`main`（已推送）／ `feat/reranker-hybrid2`（进行中）

> 本文件是唯一交接文档。之前的 `HANDOFF_DENSE_BASELINE.md`、`HANDOFF_HYBRID.md`、`HANDOFF_LINEARRAG.md`、`HANDOFF_NFCORPUS.md`、`HANDOFF_DEEPSEEK_V4_FLASH.md` 已合并删除。

## 1. 项目概览与当前进度

**MedicalGraphRAG** —— 可复现的医学检索（RAG）研究与评测项目。用统一、可审计的框架实现并对比多种医学检索器，为最终医学搜索智能体（MedSearch-R1）打地基。项目总览见根目录 [`README.md`](README.md)。

路线（5 阶段）：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | MedRAG 检索基础 | ✅ 完成 |
| 2 | LinearRAG 图检索迁移 | ✅ 完成 |
| 3 | MedicalGPT 领域微调 SFT | 🔜 进行中（另一条并行线，GPT 在预习） |
| 4 | Search-R1 搜索强化学习 | ⏳ |
| 5 | MedSearch-R1 医学搜索智能体 | ⏳ |

**检索层全部完成并审计**：4 检索器（BM25 / Dense / Hybrid / Graph）× 2 基准（pubmedqa_hard_v1 / nfcorpus_v1）。**当前正在做 Hybrid2：Qwen3-Reranker 三路候选重排**（`feat/reranker-hybrid2` 分支）。

## 2. 运行环境（重要）

- 容器：`llm-pytorch`（WSL2 Docker）。
- Python 环境：**`/opt/venv`**（`--system-site-packages`，继承 torch / sentence-transformers / faiss；新增 igraph、scispacy、en_ner_bc5cdr_md、Qwen3-Reranker）。**检索/构建/重排脚本一律用 `/opt/venv/bin/python`**。
- 本地模型（git 忽略）：`MedicalGraphRAG/models/all-mpnet-base-v2`（Dense/图用）、`MedicalGraphRAG/models/Qwen3-Reranker-0.6B`（Hybrid2 用，下载中）。
- 评测 CLI：`python -m medical_graphrag.cli evaluate-{bm25,dense,hybrid,graph,reranker}`。
- 测试：`/opt/venv/bin/python -m pytest`（当前 75+ 测试）。

## 3. 已完成阶段（精简）

### 3.1 冻结数据

**pubmedqa_hard_v1**（自建，单答案）：1,000 题 / 5,000 文档 / 7,562 chunks / 1,000 qrels（500 dev + 500 test）。manifest SHA `cf9b7591...`。构建自 PubMedQA PQA-L + MedRAG PubMed 干扰采样（seed `20260803`）。

**nfcorpus_v1**（BEIR 公开，多相关）：323 test / 3,633 文档 / 12,334 qrels（每题平均 38 相关）。构建自 `scripts/build_nfcorpus.py` + 原始 parquet（git 忽略，可重建）。

**scifact_v1 / trec_covid_v1**（BEIR，2026-08-05 接入）：泛化 `scripts/build_beir.py`；scifact 300 题/5,183 文档；trec_covid 50 题/171,332 文档。**bioasq 因 HF 仓库 401 门禁未接入**。

两者均为哈希冻结，可独立复算。

### 3.2 检索器与结果

| 检索器 | 文件 | 实现 |
|---|---|---|
| BM25 | `retrieval/bm25.py` | Pyserini/Lucene，k1=0.9 b=0.4，Top-100 chunks → max 折叠 |
| Dense | `retrieval/dense.py` | all-mpnet + FAISS IndexFlatIP，归一化 |
| Hybrid | `retrieval/hybrid.py` | BM25+Dense 的 RRF 排名融合（k=60） |
| Graph | `retrieval/graph.py` | BC5CDR 医学 NER → Entity-Passage 图 → PPR（移植 LinearRAG） |
| Reranker (Hybrid2) | `retrieval/reranker.py` | Qwen3-Reranker-0.6B 三路候选重排（CrossEncoder） |

**结果（真实、可复算）**：

*pubmedqa_hard_v1（test 500）*

| 方法 | R@1 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.946 | 0.955 |
| **Dense** | **0.966** | **0.994** | **0.978** | **0.982** |
| Hybrid | 0.960 | 0.992 | 0.974 | 0.979 |
| Graph | 0.800 | 0.958 | 0.857 | 0.881 |

*nfcorpus_v1（test 323，多相关）*

| 方法 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| Hybrid | 0.172 | 0.552 | 0.354 |
| Graph | 0.157 | 0.483 | 0.314 |
| **Hybrid2 (Qwen rerank)** | **0.189** | **0.584** | **0.384** |

*scifact_v1（test 300，~1.1 相关/题）*

| 方法 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.791 | 0.630 | 0.664 |
| Dense | 0.769 | 0.594 | 0.633 |
| **Hybrid** | **0.838** | **0.669** | **0.705** |

（scifact 的 Graph/Hybrid2 待补；trec_covid 待跑。）

### 3.3 关键结论

1. **单答案基准上 Dense 最优；多相关基准上 Hybrid（RRF 融合）最优** —— 基准选择会改变"哪种检索器更好"的结论；
2. **Hybrid 在 pubmedqa 上是负结果**（< Dense，因 BM25 太弱被稀释），在 nfcorpus 上胜出（两路势均力敌、互补）；
3. **图检索两个基准均未胜出**（实现正确、有审计背书，但当前数据非多跳推理场景，价值未体现 —— 诚实负结果）；
4. 所有指标独立复算误差 < 1e-15，每个阶段都过独立代码审查。

### 3.4 测评适配

评测已泛化支持**多相关 qrels**（BEIR 风格）：`Recall@k=|gold∩topk|/|gold|`、`MRR`=首个相关文档、`nDCG` 多相关；**单 gold 完全向后兼容**。`evaluate_rankings` / `read_qrels` / `first_gold_rank` 在 `evaluation/retrieval.py`。

## 4. 审计链与纪律

- 每个阶段：数据冻结哈希 → 索引/排名哈希绑定 → 评测校验（query 集合、split、命中分布、连续 rank、有限 score、各阶段 SHA）；
- 独立复算 + 独立代码审查是每个阶段的标配；
- **铁律**：只改一个变量、指标必须真实脚本输出、不编造结果、封闭基准不外推、大规模下载/模型先汇报、新实验用隔离分支审计后合并 `main`。

## 5. 下一步

1. **BEIR 数据接入（进行中）**：scifact 补 Graph + Hybrid2（凑齐五路）；trec_covid 跑检索（171k 文档，Graph 太重跳过）；bioasq 找替代源（HF 401）。
2. **接入 HotpotQA + FRAMES**（多跳硬 gold，写清洗脚本 `scripts/build_*.py`），验证图检索多跳价值。
3. **MIRAGE 端到端**（仓库已拉取到 `MIRAGE/`，git 忽略可重建；等阶段 3 有 LLM 再跑 QA Accuracy）。
4. 阶段 3 MedicalGPT SFT（GPT 并行线预习中）。之后阶段 4 Search-R1、阶段 5 MedSearch-R1。

**关键结论**：检索层 5 路（BM25/Dense/Hybrid/Graph/Hybrid2）× 多基准（pubmedqa/nfcorpus/scifact）完备。基准选择显著影响结论：pubmedqa Dense 最优、nfcorpus Hybrid2 最优、scifact Hybrid 最优；图检索在已有基准均未胜出。

## 6. 给下一位 Agent 的开场指令

```text
先读根目录 README.md、STUDY_PROGRESS.md、HANDOFF.md。
检索层已完备（BM25/Dense/Hybrid/Graph × pubmedqa + nfcorpus，全部审计合并），
评测支持多相关 qrels。当前分支 feat/reranker-hybrid2 在做 Hybrid2：
Qwen3-Reranker 三路候选重排（代码已写，模型下载 + 加载适配中）。
环境用 /opt/venv/bin/python；不要重跑或重构已审计基线。
任何新指标必须来自真实脚本输出 + 哈希审计。
```
