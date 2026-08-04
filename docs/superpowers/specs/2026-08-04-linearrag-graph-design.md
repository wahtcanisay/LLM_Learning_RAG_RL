# LinearGraphRetriever 图检索设计（阶段 2 · 方案 A）

## 1. 目标

在冻结数据 `pubmedqa_hard_v1` 上实现并评测第 4 条检索基线：LinearGraphRetriever（关系无关图检索）。得到与 BM25 / Dense / Hybrid 的**同数据四路对比**。忠实移植 LinearRAG 默认非向量化检索主干的核心算法（Entity→Sentence→Entity 语义传播 + Dense Passage 先验 + Personalized PageRank），适配我们的冻结 chunk 数据与统一评测接口。

图检索的实体类型用**医学 NER**（BC5CDR：疾病 + 化学物/药物），避免通用 NER 只提数字导致的空图。

## 2. 运行环境

- 容器 `llm-pytorch`，venv `/opt/venv`（`--system-site-packages`，继承 torch/st/faiss/igraph/spacy）。
- 新增依赖：`igraph`、`scispacy`、`en_ner_bc5cdr_md`（医学 NER 模型，装入 venv）。
- 图检索脚本一律用 `/opt/venv/bin/python`。

## 3. 架构总览

```text
离线构建（一次）：
frozen chunks.jsonl
  → 医学 NER（BC5CDR）提取每个 chunk 的实体
  → 句子切分 + all-mpnet 句子 embedding
  → Entity–Passage 边（归一化共现）+ 可选相邻 Passage 边
  → igraph（Entity + Passage 节点，带 weight）
  → 持久化 graph + entity/passage/sentence store + 报告（哈希）

在线检索（每 query）：
query
  → all-mpnet query embedding + question NER
  → seed entities（argmax 对齐到语料实体）
  → Entity→Sentence→Entity BFS 传播（语义桥）
  → passage 先验（归一化 Dense + 实体奖励）
  → PPR（personalized_pagerank, damping）
  → top-k chunks → 折叠成文档排名 → 评测
```

## 4. 离线图构建

### 4.1 实体抽取

- 用 `en_ner_bc5cdr_md` 对每个 chunk 的 `content` 做 NER；实体 = span 文本（保序去重，保留大小写）。
- Passage→Entity：chunk 包含哪些实体；Sentence→Entity：切分句子后，句子包含哪些实体（语义桥需要）。
- 记录实体去重后的稳定 ID（hash 或枚举）。

### 4.2 句子语义桥

- 每个 chunk 用 `sent_tokenize` 切句。
- 每句用 `all-mpnet-base-v2` 编码（归一化）。
- 建 `entity → sentence_ids`、`sentence → entity_ids` 双向映射（与 LinearRAG `entity_hash_id_to_sentence_hash_ids` / `sentence_hash_id_to_entity_hash_ids` 对应）。

### 4.3 Entity–Passage 边

与 LinearRAG `add_entity_to_passage_edges` 相同：

```text
edge_weight = count(entity in passage) / sum(count of all entities in passage)
```

### 4.4 相邻 Passage 边（默认关闭，单变量实验）

- 反思结论：pubmedqa 短摘要无相邻上下文，且直接套用会跨文档错边。
- v1 默认**不建相邻边**（图只含 Entity–Passage 边）。相邻边作为后续单变量实验（按 `(doc_id, order)` 只在文档内连接）单独验证，不进 v1 主结果。

### 4.5 igraph 构建与持久化

- 节点：Entity 节点 + Passage 节点；边：Entity–Passage（weight = 归一化共现）。
- 顶点属性 `name`（节点 ID）、`content`（原文）；记录 `passage_node_indices`。
- 持久化到 `indexes/pubmedqa_hard_v1/graph_abstract_only/`：graphml + entity/passage/sentence store + 报告（含各产物 SHA-256，沿用审计纪律）。

## 5. 在线检索 search(query, top_k)

移植 LinearRAG 默认非向量化分支：

1. `get_seed_entities(query)`：question NER → 每个问题实体与全部语料实体 embedding 点积 → argmax 选分数最高的一个语料实体。无实体 → 返回空，走 Dense 回退。
2. `calculate_entity_scores(...)`：BFS。seed 实体获得初始分数；每个 active 实体取与其关联句子中与问题最相似的 `top_k_sentence` 句（语义桥），传播分数 = 实体分数 × 句子相似度，阈值剪枝，`used_sentence` 去重，`max_iterations` 上限。
3. `calculate_passage_scores(...)`：
   `passage_score = passage_ratio × 归一化 Dense 相似度 + log(1 + Σ entity_score × log(1+occurrence) / tier)`，再乘 `passage_node_weight`。
4. `run_ppr(node_weights)`：`graph.personalized_pagerank(damping, weights='weight', reset=node_weights)`，只保留 Passage 节点排序。
5. 返回 top-k chunks → 折叠成文档 → 与既有检索器同协议。

### 关键参数（初始值，参照 LinearRAG medical 配置）

| 参数 | 初值 | 说明 |
|---|---|---|
| `damping` | 0.85 | PPR 沿边传播概率 |
| `passage_ratio` | 1.5 | Dense 先验权重 |
| `passage_node_weight` | 0.5 | passage 顶点重启权重 |
| `iteration_threshold` | 0.5 | 实体传播剪枝阈值 |
| `top_k_sentence` | 1 | 每个实体选的桥接句数 |
| `max_iterations` | 3 | 传播轮数上限 |
| chunk top-k | 100 | 与 BM25/Dense 候选深度一致 |

无 seed 实体时回退 Dense（复用现有 `dense_passage_retrieval`）。

## 6. 评测与四路对比

- chunk 排名用现有 `collapse_chunk_hits` 折叠成文档排名，`evaluate_rankings` 算 Recall@1/5/10、MRR@10、nDCG@10。
- dev/test 严格分开；主结果 official test。
- 与 BM25 / Dense / Hybrid 同表对比。
- 记录失败案例；观察 PubMedQA 短模糊 query（11570976 / 18359123）图检索表现。

## 7. 组件与文件边界

- `src/medical_graphrag/retrieval/graph.py`：离线构建（NER、句子、边、igraph）+ 在线检索核心。
- `scripts/build_graph_index.py`、`scripts/search_graph.py`：容器内执行（用 `/opt/venv/bin/python`）。
- `src/medical_graphrag/evaluation/graph.py`：`evaluate_graph_run`（绑定图索引报告哈希 + rankings 哈希，与单路同契约）。
- CLI：`evaluate-graph`。
- 产物：`experiments/pubmedqa_hard_v1/graph_abstract_only/`。

## 8. 测试策略

1. 医学 NER 在已知句子上的实体抽取（疾病/药物）；
2. Entity–Passage 边权与手算一致；
3. igraph 节点/边结构与预期一致（Entity + Passage，无边则报错）；
4. seed entities argmax 逻辑；
5. BFS 传播数学（两条路径分数累加）与阈值剪枝；
6. passage prior（Dense + 实体奖励）手算；
7. PPR 结果在 toy 图上的确定性与排序；
8. 折叠与指标复用；哈希绑定；无 seed 回退。

## 9. 完成标准

- 图索引构建成功并持久化（带哈希报告）；
- 1000 题全部产生 chunk 排名与文档排名；
- 真实输出四路对比表（BM25/Dense/Hybrid/Graph）；
- pytest 全过；独立复算指标一致；
- 学习者能解释：图检索的实体传播、语义桥、PPR 与 Dense 先验如何结合；
- 更新 `STUDY_PROGRESS.md` 与 `HANDOFF_LINEARRAG.md`。
