# MedRAG vs LinearRAG vs 我们的 RAG:数据集、评测与结果全景

> 目标:把三个方法用的数据集类型、评测方式讲透,回答"我们的 hotpotqa 结果能不能直接用于 LinearRAG 多跳对比",并列出我们项目所有数据集与真实结果。

---

## 1. LinearRAG(ICLR 2026)

**是什么**:Relation-free 图检索。建图**不用 LLM**(轻量 NER 抽实体 + 句子语义桥),线性时间/空间复杂度,能一次检索做到多跳推理。QA 生成用 OpenAI API。

### 1.1 数据集(从 HF `Zly0523/linear-rag` 下载)

| 数据集 | 类型 | 内容 | 本地状态 |
|---|---|---|---|
| **Medical** | 长文档 | GraphRAG-Bench/NCCN 临床指南,2,062 题 × 4 题型 | ✅ 已下载 |
| **HotpotQA** | 多跳 | 维基多跳问答,答案需跨两段文档 | ❌ 未下载 |
| **Novel** | 长文档 | 长篇小说(run.py 默认) | ❌ 未下载 |

**关键**:LinearRAG 的数据是**长文档、多 chunk**。`medical` 是临床指南切 chunk,同一篇指南几十上百个 chunk——它的"文档内相邻边(权 1)"正是为此设计,让 PPR 沿上下文传播。

### 1.2 评测方式

- **检索层**(Medical):**Evidence Recall / Context Relevance**(RAGAS 风格,**LLM 打分**),top-k=5,embedding all-mpnet:
  | 题型 | Evidence Recall | Context Relevance |
  |---|---|---:|
  | Fact Retrieval | 88.86 | 86.09 |
  | Complex Reasoning | 87.03 | 81.58 |
  | Contextual Summarize | 89.13 | 87.89 |
  | Creative Generation | 89.08 | 72.74 |
- **QA 层**(HotpotQA):**QA Accuracy**(GPT-4o-mini 生成),Contain-Acc 64.30 / GPT-Acc 66.50,超过 HippoRAG2 等。

**一句话**:LinearRAG 的数据是**长文档**(临床指南/小说/多跳维基),评测**都依赖 LLM**(相关性要 LLM 打分,QA 要 LLM 生成)。

---

## 2. MedRAG(MIRAGE 基准)

**是什么**:端到端医学 RAG 工具包。检索 top-k → 喂 LLM(Meditron 等)→ 生成选择题答案。

### 2.1 数据集

**语料池(无 qrels,只是检索池)**:

| 语料 | 领域 | 体量 | 粒度 |
|---|---|---|---|
| PubMed | 生物医学**摘要** | 66G | 短单段 |
| StatPearls | 临床决策支持 | 463M | 文章切块 |
| Textbooks | 医学教材 | 202M | 长书切块 |
| Wikipedia | 通用 | 43G | 段落 |
| MedCorp | 四者合并 | 全部 | 混合 |

**评测 QA**:MIRAGE 基准 = **5 个医学 QA 集、7,663 题**(MedQA、PubMedQA、MMLU 等),用于测端到端问答。

### 2.2 评测方式

- **端到端 QA 准确率**:检索 top-k → LLM 生成 → 选择题答案准确率。论文用了 1.8 万亿 prompt token、41 种语料×检索器×LLM 组合。
- 检索器:BM25 / Contriever / SPECTER / **MedCPT** / RRF-2 / RRF-4。
- **不直接评测检索质量**——检索好坏只体现在最终答对率上。

**一句话**:MedRAG 的主数据是**短摘要(pubmed)+ 长书**,评测是**端到端 QA 准确率**,检索只是中间一环。

---

## 3. 多跳数据集:我们的 hotpotqa 能不能直接用?

**结论:不能直接对表,但能用做"检索层补充"。**

| | LinearRAG 的 HotpotQA | 我们的 hotpotqa_v1 |
|---|---|---|
| 评测类型 | **QA Accuracy**(Contain/GPT-acc,GPT-4o-mini 生成) | **标准 IR 指标**(Recall@k/MRR/nDCG) |
| 是否需 LLM | 是 | 否 |
| 我们是否对齐 | ❌ 类型不同,不能直接比数字 | — |

- 我们的 hotpotqa_v1(7,405 问 / 66,581 维基段落,每问 2 个 gold)是**检索层**评测:Graph R@10 0.695 vs Dense 0.711(图在多跳检索上反而略弱,因为医学 NER 在维基上实体稀疏)。
- LinearRAG 论文只报 hotpotqa 的 **QA 准确率**,没报检索指标——我们的结果正好是论文没提供的**检索层视角**(README 已定位为补充,非复现)。
- 要对齐(比 QA 准确率),需要跑**端到端 QA**:ds-v4 回答 → GPT 打分,这正是下一步要写的脚本。

---

## 4. 我们 RAG 项目的全部数据集 + 结果

### 4.1 冻结数据集

| 数据集 | 来源 | 规模 | 类型 | qrels |
|---|---|---|---|---|
| `pubmedqa_hard_v1` | 自建(MedRAG pubmed 采样) | 1000 问 / 5000 摘要 | **短摘要**,单 gold | ✅ |
| `nfcorpus_v1` | BEIR | 323 问 / 3633 文档 | 医学文档,多相关 | ✅ |
| `scifact_v1` | BEIR | 300 问 / 5183 摘要 | **论文摘要**,单 gold | ✅ |
| `hotpotqa_v1` | BEIR | 7405 问 / 66581 段落 | **多跳维基**,2 gold/问 | ✅ |

### 4.2 检索器(统一 `search(query, top_k)` 契约)

- **chunk 级**(历史):BM25 / Dense / Hybrid(RRF)/ Graph(LinearRAG 忠实移植)/ Hybrid2(Qwen3-Reranker)。
- **document 级**(本轮新增):BM25-doc / Dense-doc / Hybrid-doc / **Graph-EP-doc**(=LinearRAG 原生)/ **Graph-Sim-doc**(=LinearRAG 软边)/ reranker-doc(Hybrid2)。

### 4.3 结果

**chunk 级(test 指标,来自 README)**:

| 基准 | 指标 | BM25 | Dense | Hybrid | Graph | Hybrid2 |
|---|---|---:|---:|---:|---:|---:|
| pubmedqa | MRR / nDCG | 0.946/0.955 | **0.978/0.982** | 0.974/0.979 | 0.864/0.894 | — |
| nfcorpus | MRR / nDCG | 0.516/0.313 | 0.502/0.325 | 0.552/0.354 | 0.477/0.312 | **0.584/0.384** |
| scifact | MRR / nDCG | 0.630/0.664 | 0.594/0.633 | 0.669/0.705 | 0.456/0.509 | **0.740/0.772** |
| hotpotqa | MRR / nDCG | 0.798/0.663 | 0.817/0.668 | 0.852/0.731 | 0.797/0.650 | **0.955/0.865** |

**document 级(test,本轮实测)**:

| 基准 | BM25-doc | Dense-doc | Hybrid-doc | Graph-EP-doc | Graph-Sim-doc |
|---|---:|---:|---:|---:|---:|
| pubmedqa MRR | 0.962 | 0.959 | **0.974** | 0.768 | 0.756 |
| pubmedqa R@10 | 0.990 | 0.992 | **0.992** | 0.972 | 0.938 |
| nfcorpus MRR | 0.516 | 0.506 | **0.557** | 0.480 | 0.477 |
| scifact MRR | 0.630 | 0.605 | **0.659** | 0.460 | 0.449 |

### 4.4 核心发现

1. **图检索在摘要型数据上稳定弱于 Dense/Hybrid**:三个摘要数据集(pubmedqa/nfcorpus/scifact)上 Graph-EP/Graph-Sim 的 MRR 全部垫底。
2. **软边效果跨数据集一致**:相似度软边**中性或有害**——pubmedqa(0.756<0.768)有害、nfcorpus(0.477≈0.480)中性、scifact(0.449<0.460)轻微有害。原因:摘要数据里干扰往往主题相近,kNN 软边把 PPR 引向干扰。
3. **Hybrid-doc 始终最优**:词项+语义 RRF 互补在三个数据集上都赢。
4. **诚实负结果**:图检索(LinearRAG 同源)在摘要短文档上未体现价值——这是方法对数据类型的固有结果,不是实现缺陷。

---

## 5. 下一步

- 完成 document 级 scifact + pubmedqa reranker 三变体(后台运行中)。
- **端到端 QA 评测脚本**(LinearRAG 格式):检索 top-k → ds-v4 回答 → GPT-5.4-mini 打分,对齐 LinearRAG/MedRAG 的 QA 准确率口径。
- 长文本(MedRAG textbooks/statpearls)检索性验证已通过(adjacent 启动测试)。
