# MedicalGraphRAG

**可复现的医学检索（RAG）研究与评测项目** —— 用统一、可审计的框架实现并对比多种医学文档检索器，为后续医学问答智能体（MedSearch-R1）打牢检索地基。

> 简单说：这是一个"医学资料怎么找最准"的实验平台。你问医学问题 → 系统从医学语料里检索相关文档 → 交给生成模型作答。本项目聚焦**检索这一步**，用真实实验回答：BM25、向量检索、混合融合、图检索、重排，哪种找法在什么场景下最好。

---

## 核心特性

- **统一检索接口**：五种检索器（BM25 / Dense / Hybrid / Graph / Hybrid2-Reranker）实现同一 `search(query, top_k)` 契约，可互换、可公平对比；
- **全链路审计**：数据冻结加哈希、每步产物 SHA-256 绑定、指标可独立复算——**不编造结果**是项目铁律；
- **多基准评测**：4 个数据集 × 5 种检索器 × 真实指标（详见[数据集](#数据集)与[实验结果](#实验结果)）；
- **多相关指标**：评测泛化支持 BEIR 风格多相关 qrels（Recall@k / MRR@10 / nDCG@10）；
- **图检索实现**：忠实移植 LinearRAG 的医学实体图检索（BC5CDR NER → 实体-文章图 → PPR），含诚实负结果。

---

## 架构

```text
                    ┌─────────────────────────────────────┐
                    │          冻结数据（可复现）           │
                    │  pubmedqa / nfcorpus / scifact /     │
                    │  hotpotqa                            │
                    └──────────────────┬──────────────────┘
                                       ↓
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │  BM25    │  Dense   │ Hybrid   │  Graph   │ Hybrid2  │
   │ (Lucene) │ (FAISS)  │ (RRF)    │(BC5CDR   │(Qwen3-   │
   │          │          │          │ +PPR)    │ Reranker)│
   └──────────┴──────────┴──────────┴──────────┴──────────┘
                                       ↓
                       统一 raw rankings（chunk → document）
                                       ↓
                    evaluate_rankings（Recall@k / MRR / nDCG）
                                       ↓
                    metrics.json + 审计链（哈希绑定，可复算）
```

五种检索器各自解决一个问题：

| 检索器 | 原理 | 强项 |
|---|---|---|
| **BM25** | 关键词词频/文档频率统计 | 术语明确的问题 |
| **Dense** | 文本向量化后找语义最近 | 换说法也能匹配 |
| **Hybrid** | BM25 + Dense 排名做 RRF 融合 | 两路信号互补时 |
| **Graph** | 医学实体建图，PPR 沿边传播 | 多跳/跨实体（理论优势） |
| **Hybrid2** | Qwen3-Reranker 三路候选重排 | 精度最高，多跳场景价值最显著 |

---

## 数据集

共 4 个冻结数据集（`data/processed/`，哈希冻结、可重建，git 忽略）。下面各抽 1-2 个真实样例辅助理解。

### pubmedqa_hard_v1（自建，单答案）

- **1,000 道医学选择题** + 5,000 篇医学摘要（1,000 gold + 4,000 干扰），每题恰 1 篇正确答案；
- **用途**：可控、自建，验证检索器在"语义匹配为主"场景的表现。

**样例**：
```
Q: Storage of vaccines in the community: weak link in the cold chain?
A: maybe（gold 文档：PMID:1571683，标题同问题）
```

### nfcorpus_v1（BEIR 公开基准，多答案）

- **323 道题 / 3,633 篇文章**，每题平均 38 篇相关文档，官方多相关 qrels；
- **价值**：国际标准检索基准 + 权威标注，补上"多相关召回"视角——正是在它上面我们发现 Hybrid 真正胜出，证明**数据集选择会改变"哪种检索器更好"的结论**。

**样例**：
```
Q: deafness
（gold 文档之一：MED-10 | Statin Use and Breast Cancer Survival…）
```

### scifact_v1（BEIR 公开基准，科学声明核查）

- **300 道科学声明 + 5,183 篇论文摘要**，约 1.1 相关/题；
- **价值**：科学声明核查场景，词项匹配与语义匹配各有胜负，RRF 融合仍胜。

**样例**：
```
Q: 0-dimensional biomaterials show inductive properties.
（gold 文档：4983 | Microstructural development of human newborn cerebral white matter…）
```

### hotpotqa_v1（多跳硬 gold）

- **7,405 道多跳问题 + 66,581 篇维基段落**（每问 10 段候选，其中 2 段 gold + 8 段干扰），每题恰 2.0 篇 gold 文档；
- **价值**：真正的多跳推理场景——问题需要**跨两段文档**才能回答。这正是图检索（Graph）理论上最该发力的场景，也验证了 Hybrid/Hybrid2 在多跳场景的融合价值。

**样例**：
```
Q: Were Scott Derrickson and Ed Wood of the same nationality?
A: yes
gold 文档（2 篇）：
  - Ed Wood        （维基段落：关于导演 Ed Wood）
  - Scott Derrickson（维基段落：关于导演 Scott Derrickson）
```

---

## 实验结果（诚实版）

### pubmedqa_hard_v1（test 500）

| 方法 | Recall@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.946 | 0.955 |
| **Dense** | **0.966** | **0.994** | **0.978** | **0.982** |
| Hybrid | 0.960 | 0.992 | 0.974 | 0.979 |
| Graph | 0.790 | 0.982 | 0.864 | 0.894 |

### nfcorpus_v1（test 323，多相关）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| Hybrid | 0.172 | 0.552 | 0.354 |
| Graph | 0.158 | 0.477 | 0.312 |
| **Hybrid2 (Qwen3)** | **0.189** | **0.584** | **0.384** |

### scifact_v1（test 300）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.791 | 0.630 | 0.664 |
| Dense | 0.769 | 0.594 | 0.633 |
| Hybrid | 0.838 | 0.669 | 0.705 |
| Graph | 0.705 | 0.456 | 0.509 |
| **Hybrid2 (Qwen3)** | **0.895** | **0.740** | **0.772** |

### hotpotqa_v1（test 7405，多跳硬 gold）—— 最适合图检索的场景

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.738 | 0.798 | 0.663 |
| Dense | 0.711 | 0.817 | 0.668 |
| Graph | 0.695 | 0.797 | 0.650 |
| Hybrid | 0.800 | 0.852 | 0.731 |
| **Hybrid2 (Qwen3)** | **0.898** | **0.955** | **0.865** |

### 关键结论

1. **单答案基准上 Dense 最优**；**多相关/多跳基准上 Hybrid 或 Hybrid2 最优**——基准选择显著影响结论；
2. **Hybrid2（Qwen3-Reranker 重排）在三个数据集上都是最优**，多跳场景（hotpotqa）价值最显著（R@10 0.898，比 Hybrid 0.800 提升 0.098）；
3. **Hybrid RRF 在多跳场景大幅胜出单路**（hotpotqa R@10 0.800 vs Dense 0.711），词项+语义两路互补；
4. **图检索（Graph）在全部 4 个基准均未胜出**——实现正确、有审计背书，但 BC5CDR 医学 NER 在通用维基（hotpotqa）上实体稀疏，实体传播信号弱。**诚实负结果**，图检索价值在现有数据上未体现；
5. 所有指标独立复算误差 < 1e-15。

---

## LinearRAG 适配与论文复现

本项目图检索是对 [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)（ICLR 2026，arXiv 2510.10114）的**忠实移植**：`get_seed_entities → 实体传播（BFS）→ passage 先验 → PPR`，与官方默认非向量化分支逐行一致。

### 论文声称的检索效果

LinearRAG 论文的检索评测集中在 **Medical 数据集**（源自 GraphRAG-Bench / NCCN 临床指南，2,062 问 × 4 种题型），用 **Evidence Recall / Context Relevance**（RAGAS 风格，LLM 判断）报告，top-k=5，embedding 用 all-mpnet-base-v2（与本项目相同）：

| 题型 | Evidence Recall | Context Relevance |
|---|---:|---:|
| Fact Retrieval | 88.86 | 86.09 |
| Complex Reasoning | 87.03 | 81.58 |
| Contextual Summarize | 89.13 | 87.89 |
| Creative Generation | 89.08 | 72.74 |

论文主表另报 **QA Accuracy**（GPT-4o-mini 生成 + 评测）：HotpotQA Contain-Acc 64.30 / GPT-Acc 66.50，全面超过 HippoRAG2 等 GraphRAG 基线。

### 可复现性判断

1. **检索指标层面**：论文的 Evidence Recall 依赖 GraphRAG-Bench 官方数据（`Datasets/Corpus/medical.json` + `medical_questions.json`，HF 可下载）+ **LLM 评判（GPT-4o-mini）**。我们当前无 LLM API（阶段 3 之后才有），**暂无法复现**；
2. **QA 准确率层面**：同样需要 LLM 生成，暂无法复现；
3. **标准 IR 指标层面**：我们已在 hotpotqa_v1（多跳硬 gold）上用标准指标（R@10/MRR/nDCG）评测了 Graph 检索——这是论文没有提供的视角（论文只报 Medical 的 context_recall + 各数据集 QA accuracy），**不是复现而是补充**。

> 结论：LinearRAG 论文的检索效果（High Recall + High Relevance）我们**逻辑上复现（实现一致），数值上暂未复现（缺 LLM API + GraphRAG-Bench 数据未接入）**。等阶段 3 有 LLM 后可端到端复现。

---

## 如何评测 RAG 效果

分三层，逐层加码：

1. **检索层（已完成）**：Recall@k / MRR@10 / nDCG@10，在 4 个基准上对比检索器。新增检索器 = 实现 `search(query, top_k)` → 接入统一评测即可；
2. **端到端问答层（阶段 3 后）**：检索器召回 top-k → 喂给生成模型 → **QA Accuracy**。计划用 `Base / SFT / SFT+RAG` 对比；
3. **工程层**：延迟、索引大小、显存峰值、token 成本——尤其对最终智能体（阶段 5）至关重要。

原则：**每次只改一个变量**（换检索器、换融合、换生成模型分开测），保证能定位提升来源。

---

## 快速开始

环境：WSL2 Docker 容器 `llm-pytorch`，Python 环境 `/opt/venv`（含 igraph / scispacy / sentence-transformers / faiss）。

一条命令跑完 构建 → 检索 → 评测 全流程（`cli run` 是唯一 evaluate 入口）：

```bash
cd MedicalGraphRAG

# 评测 Dense（示例）：建 FAISS 索引 → 检索 → 产出 metrics.json
/opt/venv/bin/python -m medical_graphrag.cli run dense \
  --dataset nfcorpus_v1 \
  --git-commit $(git rev-parse HEAD) \
  --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

# 其余检索器同理
/opt/venv/bin/python -m medical_graphrag.cli run bm25      --dataset hotpotqa_v1 --git-commit $(git rev-parse HEAD)
/opt/venv/bin/python -m medical_graphrag.cli run graph     --dataset scifact_v1 --git-commit $(git rev-parse HEAD)
/opt/venv/bin/python -m medical_graphrag.cli run hybrid    --dataset nfcorpus_v1 --git-commit $(git rev-parse HEAD)
/opt/venv/bin/python -m medical_graphrag.cli run reranker  --dataset hotpotqa_v1 --git-commit $(git rev-parse HEAD)
```

`retriever` ∈ {bm25, dense, hybrid, graph, reranker}；`--dataset` 是 `data/processed/` 下的冻结数据集名。产物与审计链与原先分步脚本完全一致：`outputs/<ds>/`（索引/排名/报告）+ `experiments/<ds>/metrics.json`（可审计）。

---

## 项目结构

```text
MedicalGraphRAG/
  src/medical_graphrag/
    data/          # 数据加载、切块、冻结数据
    retrieval/     # bm25 / dense / hybrid / graph / reranker
    evaluation/    # 指标（含多相关）+ 各检索器评测
  scripts/         # 容器内构建/检索/重排脚本
  experiments/     # 实验结果 metrics.json（可审计）
  data/processed/  # 冻结数据集（可重建，git 忽略）
```

---

## 路线图（Roadmap）

- [x] 检索层：BM25 / Dense / Hybrid / Graph 四路基线 × 双基准
- [x] 评测泛化：多相关 qrels 指标
- [x] **Hybrid2：三路检索 + Qwen 重排**（R@10 在 nfcorpus/scifact/hotpotqa 三基准最优）
- [x] **多跳基准接入**：hotpotqa_v1（7,405 问多跳硬 gold），验证图检索多跳价值
- [ ] 阶段 3：MedicalGPT 领域微调（进行中）
- [ ] 阶段 4：Search-R1 搜索强化学习
- [ ] 阶段 5：MedSearch-R1 医学搜索智能体

---

## 许可

研究用途，非临床产品。不提供诊断建议。
