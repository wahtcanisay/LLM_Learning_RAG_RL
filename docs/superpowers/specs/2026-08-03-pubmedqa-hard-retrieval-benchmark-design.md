# PubMedQA 医学检索硬指标子集设计

## 1. 项目定位

本设计为 Medical GraphRAG 的第一套可程序化验证的医学检索基准。它不下载或索引完整 MedRAG PubMed 语料，而是使用 PubMedQA 的专家标注文章作为 gold documents，并从本地 MedRAG PubMed 语料固定抽取干扰文章，构造规模可控、qrels 明确、可供 BM25、Dense、Hybrid 与 LinearRAG 公平比较的封闭检索集合。

该基准主要回答两个问题：

1. 不同检索器能否把与问题对应的 gold PubMed 文章排进 Top-k；
2. 检索质量变化是否能转化为 PubMedQA yes/no/maybe 问答质量变化。

第一版只验证单跳文档检索，不把它包装成多跳医学 GraphRAG 结论。LinearRAG 的医学图结构价值将在后续带多证据 qrels 的数据上单独验证。

### 1.1 项目目录边界

所有新代码、配置、测试和生成数据统一放在根仓库的 `MedicalGraphRAG/` 目录。该目录与现有 `MedRAG/`、`LinearRAG/` 并列，但不单独执行 `git init`，继续由根仓库统一版本控制。

- `MedRAG/`：只读数据源与 BM25/Dense/RRF 参考实现；
- `LinearRAG/`：只读图检索参考实现与官方 medical 冒烟入口；
- `MedicalGraphRAG/`：新的可维护项目，承载清洗、统一检索接口、评测和后续医学迁移。

重构不复制两套官方仓库的全部源码，也不在其中继续加入项目功能。确需复用的算法应通过清晰接口重新实现，并以测试验证行为。

## 2. 已确认的数据事实

### 2.1 PubMedQA

PubMedQA PQA-L 共包含 1,000 条专家标注样本，以 PMID 为键。每条样本至少提供：

- `QUESTION`：检索查询和问答题干；
- `CONTEXTS`：对应论文摘要中去除结论后的结构化段落；
- `LABELS`：段落类型；
- `LONG_ANSWER`：论文结论；
- `final_decision`：`yes`、`no` 或 `maybe`。

官方另提供 500 个测试 PMID 及其答案标签。其余 500 条 PQA-L 样本作为开发集，官方测试 500 题只用于最终评测，不用于阈值、检索超参数或 Prompt 选择。

### 2.2 本地 MedRAG PubMed

本地 MedRAG PubMed 约 65.20 GiB，每条通常是一篇独立论文摘要。相邻 JSONL 行不代表同一篇连续文档。本地已下载版本使用 `pubmed23n0001_0` 一类分片位置 ID，而不是可直接与 PubMedQA 对接的 `PMID:*`。

因此，第一版不尝试恢复全量本地 PubMed 的 PMID：

- gold documents 直接由 PubMedQA `CONTEXTS` 构造，使用官方 PMID；
- MedRAG PubMed 只提供干扰 documents，沿用其本地稳定 ID；
- 不同论文之间不添加相邻 Passage 边。

### 2.3 LinearRAG medical

现有 LinearRAG medical 数据有 2,062 个问题和 225 个大 chunk，但存在跨文档拼接，且问题只带改写后的 `evidence`，没有 `gold_chunk_id`。它继续用于官方流程冒烟，不作为第一套 Recall@k qrels 来源。

## 3. 范围与非目标

### 3.1 第一版范围

- 1,000 篇 PubMedQA gold documents；
- 4,000 篇从本地 MedRAG PubMed 固定抽样的干扰 documents；
- 500 道开发问题和 500 道官方测试问题；
- 文档级 qrels；
- BM25、Dense、Hybrid 和 LinearRAG 使用完全相同的问题、语料和指标；
- `abstract_only` 主设置与 `title_abstract` 对照设置；
- Recall@1/5/10、MRR@10、binary nDCG@10、延迟；
- 生成阶段记录 Accuracy 和 Macro-F1。

### 3.2 第一版非目标

- 不索引完整 65 GiB PubMed；
- 不从 BioASQ 构造多证据 qrels；
- 不加入 BM25 hard negatives；
- 不把随机未标注文档声称为人工确认的负例；
- 不改动 LinearRAG 的实体传播、PPR、Embedding 和生成模型等多个主要变量；
- 不根据官方测试结果选择随机种子、chunk 参数或检索超参数；
- 不使用 LLM Judge 代替程序化检索指标或三分类答案指标。

## 4. 数据集构造

### 4.1 规模

固定语料库包含 5,000 个 document：

```text
PubMedQA PQA-L gold documents       1,000
MedRAG PubMed sampled distractors   4,000
                                       ─────
Total documents                     5,000
```

所有开发和测试检索器共享同一个 5,000-document corpus。每道 PubMedQA 问题只有一个已标注 relevant document，即与 query PMID 相同的 PubMedQA document。

### 4.2 干扰文档抽样

干扰文档使用固定随机种子 `20260803`，采用两级确定性抽样，避免为了 4,000 篇干扰文档扫描完整 65 GiB 语料：

1. 从按文件名排序的非空 MedRAG PubMed JSONL 分片中，使用该种子无放回抽取 40 个分片；
2. 对每个入选分片按文件行顺序执行 reservoir sampling，抽取 120 条候选，共得到至多 4,800 条；
3. 完成下述去重后，对候选的 `SHA-256(seed + doc_id)` 升序排列，取前 4,000 条；
4. 若不足 4,000 条，则按同一种子生成的后续分片顺序逐个增加分片并重复 reservoir sampling，直到补足。

抽样输入不依赖文件系统枚举顺序，也不读取 query 内容或答案。manifest 保存入选分片名、各分片候选数和最终入选数。

抽样后执行以下去重：

1. `doc_id` 完全重复时只保留第一条；
2. 规范化 `title + content` 的 SHA-256 完全重复时只保留第一条；
3. 与任一 PubMedQA gold document 的规范化内容哈希相同时排除；
4. 与 PubMedQA gold title 规范化后完全相同时排除。

除上述确定性去重外，不使用 query、BM25、Embedding 或测试答案筛选干扰文档，避免 query-conditioned corpus bias。未进入 qrels 的文档在本基准中按未标注文档处理；报告中不得声称它们经过人工确认全部不相关。

### 4.3 20 题审计门

在生成完整数据前，先固定检查官方测试集中排序后的前 20 个 PMID。审计表必须包含：

- PMID 与 split；
- question 与 final decision；
- `CONTEXTS` 数量和空值情况；
- 生成的 document/chunk ID；
- qrels 是否唯一指向同 PMID document；
- title 泄漏风险标记；
- 是否与干扰文档发生哈希或标题重复。

只有 20/20 题通过 schema、qrels 和去重检查，才允许扩展到全部 1,000 题和 5,000 篇语料。

## 5. 数据契约

### 5.1 Questions

文件：`questions.jsonl`

```json
{
  "query_id": "21645374",
  "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
  "answer": "yes",
  "long_answer": "...",
  "split": "test"
}
```

约束：

- `query_id` 必须等于官方 PMID 字符串；
- `answer` 只能为 `yes`、`no`、`maybe`；
- 不把 `CONTEXTS` 或 `LONG_ANSWER` 拼进检索 query；
- 测试 split 以官方 `test_ground_truth.json` 的 PMID 集合为准。

### 5.2 Documents

文件：`documents.jsonl`

PubMedQA gold document：

```json
{
  "doc_id": "PMID:21645374",
  "title": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
  "content": "...",
  "source": "pubmedqa",
  "year": "2011"
}
```

MedRAG distractor：

```json
{
  "doc_id": "MEDRAG:pubmed23n0001_2",
  "title": "Effect of etafenone on total and regional myocardial blood flow.",
  "content": "...",
  "source": "medrag_pubmed",
  "year": null
}
```

`source` 只用于审计和结果分析，不进入检索文本，也不允许作为模型特征。

### 5.3 Chunks

文件：`chunks.jsonl`

```json
{
  "chunk_id": "PMID:21645374#0",
  "doc_id": "PMID:21645374",
  "order": 0,
  "title": "...",
  "content": "...",
  "source": "pubmedqa"
}
```

PubMedQA 的每个 `CONTEXTS` 元素先作为自然段落候选。超过 512 tokens 的段落再按 512 tokens、64 tokens overlap 切分；短段落不与下一篇文章拼接。MedRAG 干扰摘要采用相同的最大长度与 overlap 规则。

清洗阶段固定使用 `sentence-transformers/all-mpnet-base-v2` tokenizer；manifest 必须记录解析到的模型 revision。所有检索器复用同一 chunks 文件，禁止各检索器自行重新切块。后续更换 Dense 模型时也不重新切块，除非建立新的数据集版本并重新运行全部基线。

### 5.4 Qrels

文件：`qrels.tsv`

```text
query_id\tdoc_id\trelevance
21645374\tPMID:21645374\t1
```

每个 query 必须恰好有一个 relevance=1 的 document。qrels 在文档层定义，不把 gold document 的每一个 chunk 分别当成独立 relevant document。

### 5.5 Manifest

文件：`manifest.json`

至少记录：

- PubMedQA 与 MedRAG 来源和版本；
- 清洗脚本 Git commit；
- 随机种子；
- dev/test query 数；
- gold/distractor document 数；
- chunk 数、最大长度、overlap 和 tokenizer；
- 去重前后数量与原因；
- questions/documents/chunks/qrels 文件 SHA-256；
- 20 题审计结果路径。

## 6. 检索设置

### 6.1 主设置：abstract_only

检索文本只使用 chunk `content`。这是主结果，因为 PubMedQA 的问题经常来自论文标题；如果把 document title 一并检索，gold document 可能依靠近乎相同的标题被轻易命中。

### 6.2 对照设置：title_abstract

检索文本使用 `title + content`，用于对照 MedRAG 常见的 `contents` 检索契约。结果必须单列，不能与 `abstract_only` 混成同一主表。

### 6.3 LinearRAG 图边

- Entity–Passage 边按 LinearRAG 原方法构建；
- 相邻 Passage 边只允许连接相同 `doc_id` 且 `order` 连续的 chunk；
- 不同 PMID、不同 MedRAG 摘要之间禁止相邻边；
- PubMedQA 与 MedRAG 来源之间不通过数据顺序连边；
- 是否有 Seed Entity、是否回退 Dense、图节点/边数均写入实验日志。

### 6.4 Chunk 到 document 聚合

所有检索器先输出 chunk 排名，再用同一规则聚合为 document 排名：

```text
document_score = max(score of retrieved chunks belonging to the document)
```

同一 document 只保留一次，并按最高 chunk score 排序。第一版不比较 mean、sum 或学习式聚合，避免增加主要变量。

## 7. 评测指标

### 7.1 检索指标

在 document 排名上计算：

- Recall@1；
- Recall@5；
- Recall@10；
- MRR@10；
- binary nDCG@10；
- 平均检索延迟；
- P50/P95 检索延迟。

由于每个 query 只有一个 gold document，Recall@k 与 Hit@k 数值相同。报告中统一使用 Recall@k，并明确这一数据性质。

### 7.2 问答指标

生成模型只能输出 `yes`、`no`、`maybe`，计算：

- Accuracy；
- Macro-F1；
- 非法输出率。

同时保存 No-RAG 与 Oracle-context 结果：No-RAG 衡量模型自身知识，Oracle-context 给出当前生成模型在 gold 摘要已知时的上界参考。检索结果与生成结果分别评测，不能以 QA Accuracy 替代 Recall@k。

### 7.3 结果分层

除总表外，至少按以下维度分析：

- `yes`、`no`、`maybe`；
- gold document rank 为 1、2–5、6–10、未进 Top-10；
- `abstract_only` 与 `title_abstract`；
- 有 Seed Entity 与 Dense fallback；
- PubMedQA context 段落数。

## 8. 基线与实验顺序

实验固定按以下顺序推进：

1. Oracle qrels 与指标单元测试；
2. BM25；
3. Dense；
4. BM25 + Dense Hybrid；
5. LinearRAG；
6. 固定生成模型上的 No-RAG、Oracle、各检索器 RAG。

第一版不加入 Reranker。每次只改变检索方法，保持 corpus、chunks、query、top-k、生成模型、Prompt 和随机种子一致。

## 9. 校验、失败处理与防泄漏

清洗脚本遇到以下任一情况必须失败并返回非零退出码：

- 官方测试 PMID 不在 PQA-L；
- query 缺少 question 或 final decision；
- final decision 不属于三类合法标签；
- gold document 缺少 context；
- query 没有 qrels 或有多个 gold document；
- document/chunk ID 重复；
- dev/test PMID 重叠；
- 最终 document 数不等于 5,000；
- 最终 gold document 数不等于 1,000；
- SHA-256 manifest 与现有产物不一致但脚本试图静默复用。

测试集不得参与：

- 干扰文档选择；
- chunk 参数选择；
- top-k、融合权重或图传播超参数选择；
- Prompt 修改；
- 模型 checkpoint 选择。

所有这些选择只能依据开发集或固定规则完成。

## 10. 产物目录

项目内数据目录：

```text
MedicalGraphRAG/
├── data/
│   ├── raw/pubmedqa/
│   │   ├── ori_pqal.json
│   │   └── test_ground_truth.json
│   └── processed/pubmedqa_hard_v1/
│       ├── questions.jsonl
│       ├── documents.jsonl
│       ├── chunks.jsonl
│       ├── qrels.tsv
│       ├── manifest.json
│       ├── audit_20.json
│       └── README.md
├── src/medical_graphrag/
├── tests/
├── configs/
├── pyproject.toml
└── README.md
```

规范的第一版输出目录为 `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/`。`MedicalGraphRAG/data/raw/` 和大规模 `processed/` 产物默认通过项目 `.gitignore` 排除；生成脚本、配置、数据说明和小型审计结果应进入版本控制。所有被忽略的数据必须可由 manifest 记录的来源、参数和命令确定性重建。默认 MedRAG 干扰语料路径为 `../MedRAG/corpus/pubmed/chunk`，但必须可通过 CLI 参数覆盖。

## 11. 分阶段完成标准

### 11.1 数据审计完成

- 前 20 个官方测试 PMID 全部成功解析；
- 20/20 qrels 唯一且指向正确 PMID；
- 无空 context、重复 ID 或跨文档 chunk；
- 生成审计文件并人工阅读至少 3 条 yes、3 条 no、3 条 maybe 样本；若前 20 条无法覆盖三类，则从前 20 条外按 PMID 排序补足人工阅读样本，但不改变 20 题自动审计集合。

### 11.2 清洗完成

- 1,000 questions、1,000 gold documents、4,000 distractors；
- dev/test 各 500 题且无交集；
- 每题一个 qrels；
- chunk 不跨 document；
- manifest 数量、参数和 SHA-256 完整；
- 同一命令重复运行得到相同文件哈希。

### 11.3 检索基线完成

- BM25、Dense、Hybrid、LinearRAG 使用统一命令和统一 evaluator；
- 输出真实 Recall@1/5/10、MRR@10、nDCG@10 和延迟；
- 保留至少 5 个成功案例和 5 个失败案例；
- 明确 `abstract_only` 与 `title_abstract` 的差异；
- 不把 5,000-document 封闭基准的结果外推为全 PubMed 检索性能。

## 12. 后续扩展边界

第一版稳定后，后续版本可按单变量原则选择一个方向：

1. 增加固定规模的 BM25 hard-negative 挑战集；
2. 使用 BioASQ 构造多 gold document qrels；
3. 将 StatPearls 或 Textbooks 加入更连续的医学图检索语料；
4. 恢复本地 MedRAG PubMed 的 PMID 元数据并扩大候选库。

这些扩展均不进入本设计的第一版完成条件。
