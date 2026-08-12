# 合并建库 + 建库开关 + MedRAG 基线 设计（阶段 2 · 方案 A）

> [!CAUTION]
> **状态：已于 2026-08-08 废弃，不得进入 implementation plan 或代码实现。**
> 用户决定放弃合并大库，改为每个数据集独立建库：摘要型数据按完整 document 构建 Similarity 软边，长文本保留同文档 Adjacent `1.0` 边。替代规格见 `2026-08-08-per-dataset-document-edge-policy-design.md`。

## 1. 目标

把多个医疗数据集合并成**一个大库**，通过**建库开关（`edge_mode`）**为图检索注入多跳能力：

- **长文本数据集**（临床指南、教材、笔记）→ 同文档相邻边，边权固定 1.0（LinearRAG 核心机制）；
- **短文本数据集**（PubMed 摘要）→ 相似度软边（kNN + 阈值隔断），不设权 1 直接边。

以 **MedRAG 为基线**，建立 **LLM-free 的检索评测闭环**（标准 IR 指标，不依赖生成模型）。为阶段 5 的 MedSearch-R1 打检索地基。

承接 `2026-08-04-linearrag-graph-design.md` 的诚实负结果：移植版 Graph 在 4 个基准全未胜出，根因是医学 NER 在通用语料上实体稀疏 + 未开 passage-passage 边。本次设计让图检索在**医学数据大库**上获得真正可传播的结构。

## 2. 决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| 大库范围 | **受控子集**（现有冻结语料 pubmedqa/nfcorpus/scifact + MedRAG 子集 textbooks/statpearls） | 建库/评测小时级可迭代；全量 100G+ 留待后续 |
| 成功标准 | **先闭环再优化**（Phase 1 工程闭环 + 诚实基线；Phase 2 图检索优化） | 不赌图检索翻盘，先把可复现的基线立住 |
| 建库开关 | **方案 A：显式 config 开关**（per-dataset `edge_mode` 写进冻结 manifest） | 语义清晰、可审计、符合项目"冻结+哈希"文化 |
| 规模风险 | 记录在案，先做，出现问题再优化 | 规模双刃剑，用 scale 对比实验验证 |
| 评测口径 | Phase 1 保留 per-dataset 受控对比；merged 评测若指标异常启用**双段评测** | 大库召回指标下降是规模成本，不是回归 |

## 3. 架构总览

```text
merge_config.json                          // 新增：合并建库入口
  ├─ datasets: [{dir, edge_mode, doc_field}]
  │     dir        → 冻结数据集目录或 MedRAG chunk 目录
  │     edge_mode  → "adjacent" | "similarity"
  │     doc_field  → 文档标识字段（adjacent 模式用它分文档）
  ▼
build_graph_index(merge_config)            // 改造：单数据集 → 多数据集
  ├─ 读各数据集 chunks + manifest
  ├─ 打 provenance 列：dataset / doc_id / chunk_idx
  ├─ NER（BC5CDR）→ entities + 句子桥
  ├─ 三类边：
  │     Entity–Passage  （通用，归一化共现）
  │     Adjacent        （edge_mode=adjacent，同文档相邻，权=1）
  │     Similarity      （edge_mode=similarity，kNN 软边）
  ├─ igraph → graph.graphml + stores
  └─ 冻结报告：每数据集 edge_mode + provenance 哈希
        ▼
LinearGraphRetriever.search()              // 轻改：读图 + PPR 参数按报告加载
```

**组件边界**：建库（`build_graph_index`）是改造重点；检索（`LinearGraphRetriever`）是轻改——它本就只读图边+权重跑 PPR，新增边类型对其透明。评测、审计链不变。

## 4. 建库开关 `edge_mode`

合并配置示例：

```jsonc
{
  "datasets": [
    {"dir": "data/processed/pubmedqa_hard_v1", "edge_mode": "similarity"},
    {"dir": "data/processed/nfcorpus_v1",      "edge_mode": "similarity"},
    {"dir": "data/processed/scifact_v1",       "edge_mode": "similarity"},
    {"dir": "medrag/textbooks_chunk",          "edge_mode": "adjacent", "doc_field": "id"},
    {"dir": "medrag/statpearls_chunk",         "edge_mode": "adjacent", "doc_field": "id"}
  ]
}
```

- `edge_mode ∈ {"adjacent", "similarity"}`；`doc_field` 只在 `adjacent` 模式使用（分文档标识）。
- 每个数据集的 `edge_mode` 与 provenance 写进**冻结 manifest**（SHA-256 绑定），建边策略可审计、可复算。
- 判定 long/short 是**入库时的配置决策**，不由构建期猜测——这是方案 A 的核心取舍。

## 5. 边构建细节

### 5.1 Entity–Passage（通用）

沿用现实现（`graph.py:105-127`）：`weight = count(entity in passage) / sum(count of all entities in passage)`。

### 5.2 Adjacent（长文本 → 权 1.0）

- 按 `doc_field` 把 chunks 分组 → 组内按 `chunk_idx` 排序 → 相邻对加边，权重恒 1.0；
- **严格限同文档内**——直接修掉原版 LinearRAG 的全局 `enumerate` bug（`LinearRAG.py:71` 会把文档 A 末段连到文档 B 首段）。
- 只有当数据集的 `edge_mode=adjacent` 时才建。

### 5.3 Similarity（短文本 → 软边）

复用已有的 passage embeddings：

- 每段取 **top-k 近邻**（k≈5-10），边权 = cosine 相似度；
- **阈值隔断**：低于阈值（初值 0.5-0.6）的边不建——这就是"隔断"的实现：不相干摘要连边都没有；
- **hub 惩罚**：邻接度高的段落降权，防止 PPR 概率被综述类摘要吸走（fatal-link 问题）；
- **边权归一化**：相似度边也按 passage 归一化（出边权和=1），与实体边同量纲，避免 PPR 转移概率偏向某一类边。

三个参数（k / 阈值 / hub 惩罚）作为 `GraphConfig` 扩展，Phase 2 扫参。

## 6. retrieve 改动（轻改）

核心链路 **seed → 实体传播（BFS 桥）→ passage 先验 → PPR 一行不改**。改动三处：

1. **GraphConfig 从 build 报告加载合并库参数组**：合并库的 `damping` / `passage_node_weight` / `passage_ratio` 存入 build 报告，检索时读回（`graph.py:50-58` 已可配置，加"合并库参数组"读取路径）。相似度边让图变密，预期需扫更高 `damping`；
2. **hub 感知预留**：若实验发现 hub 噪声，在 `_calculate_passage_scores` 或 PPR reset 上按节点度降权（先留接口，不默认开）；
3. **规模性能预留**：大库 seed-entity 匹配是全实体 argmax 点积（`entity_embeddings` 常驻内存，一次矩阵乘；几十万实体可接受），标注上限；无实体 → Dense 回退不变。

## 7. MedRAG 基线移植

移植 3 个 MedRAG 检索器进统一 `search(query, top_k)` 契约：

| MedRAG 检索器 | 移植方式 |
|---|---|
| BM25 | 复用项目已有 Pyserini BM25（两者等价） |
| **MedCPT** | 新增：MedCPT-Query-Encoder + CLS pooling + FAISS IndexFlatIP（与 all-mpnet 不同的新嵌入模型，真正的新基线） |
| **RRF-2** | BM25 + MedCPT 的 RRF 融合（项目 Hybrid 的换嵌模型版本） |

## 8. LLM-free 评测

- **语料**：合并冻结库（pubmedqa_hard_v1 + nfcorpus_v1 + scifact_v1 + MedRAG textbooks/statpearls 子集）；
- **查询与 qrels**：各数据集用自己的 test 查询 + 各自 qrels，**按数据集分别算指标**（避免跨库污染），另报 pooled；
- **指标**：Recall@k / MRR@10 / nDCG@10，走现有 `evaluate_rankings`；
- **对比表**：项目五路（BM25 / Dense / Hybrid / Graph / Hybrid2）+ MedCPT + RRF-2；
- **跨库污染说明**：A 库的干扰 chunk 可能是 B 库的 gold。由于按数据集 qrels 单独评测，跨库命中记为该数据集假正例——这是规模风险的一部分，记录不掩盖。

## 9. 评测口径演进（风险记录）

**问题**：用大库检索对应数据集 QA，Recall@10 / MRR 指标预期不好看。

**判断：这是评测体制切换，不是检索回归**：

1. per-dataset qrels 是在小型干扰集上设计的（如 pubmedqa_hard_v1 = 1 gold + 4,000 采样干扰）；合并后 gold 与大库全部候选竞争 top-k 名额，同题相似干扰变多 → 指标下降 = 规模成本；
2. Recall@10 不是大库的正确观察窗口。规模上去后应分两段评测：

| 层 | 指标 | LLM 依赖 |
|---|---|---|
| 候选召回层 | Recall@候选池（R@100/1000） | 无 |
| 精确排序层 | precision@10（Reranker 负责） | 无 |

3. 端到端 QA 准确率是最终智能体（MedSearch-R1）口径，需 LLM——留给 roadmap 阶段 3/5；
4. 用**相对指标**（merged vs per-dataset delta）解释规模成本。

**决策**：Phase 1 保留 per-dataset 评测做受控对比；merged 评测若指标异常，启用双段评测（候选池 Recall + 重排 precision）。

## 10. 规模风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| 候选污染 | 大库同题相似干扰变多，precision/MRR 降 | scale 对比实验 + 双段评测 |
| hub 爆炸 | 实体枢纽随规模增长，PPR 局域性崩塌 | hub 惩罚 + 度分布监控 |
| 实体歧义 | 跨数据集同名实体 merge 成错误节点 | 必要时 per-dataset 实体命名空间 |
| 延迟/成本 | 大图 PPR、全实体 argmax、FAISS 规模 | 受控子集先行；性能标注上限 |

**决策**：先做受控子集，规模影响留到出现问题再优化（用户确认）。

## 11. 实验设计

### Phase 1 闭环（先立基线）

1. 写 `merge_config.json`（5 数据集：3 similarity + 2 adjacent）；
2. 扩展 `build_graph_index` 支持多数据集 + provenance + 按 edge_mode 建边；
3. 移植 MedCPT / RRF-2 检索器；
4. 跑全部 7 个检索器 → 产出诚实基线（per-dataset + pooled）。

**验收**：管线跑通、审计链完整、指标不退化、MedCPT 在医学语料上具备竞争力（基线 sanity）。

### Phase 2 优化（单独一轮）

- 建库开关开/关对比（有/无 similarity 边、有/无 adjacent 边）；
- 相似度边参数扫描：k / 阈值 / hub 惩罚；
- adjacent 文档内作用域正确性验证；
- **规模对比实验**：受控子集 vs 更大合并 vs 各自建库——用数据回答"过大库会不会变差"；
- PPR 参数（damping / passage_node_weight）在合并库上的重调。

**目标**：图检索至少在一个场景超过非图基线，或得到诚实负结果并定位原因（NER、边策略、参数）。

## 12. 验收标准

- **Phase 1**：`merge_config.json` 可复现构建（哈希冻结）；7 个检索器全部出指标；无 LLM 参与评测；
- **Phase 2**：图检索在合并库上要么翻盘、要么产出诚实负结果 + 可复算的失败原因分析。

## 13. 打开项

- 相似度边三参数（k / 阈值 / hub 惩罚）的初值实验；
- 合并库 PPR 参数重调；
- MedCPT embedding 计算成本（子集规模）；
- 更大库的评测口径切换点（何时启用双段评测）；
- MedRAG 全量语料（pubmed 66G 等）的架构预留（增量建库、分片索引）。
