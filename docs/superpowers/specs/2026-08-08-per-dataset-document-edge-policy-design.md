# 单数据集独立建库与文档级边策略设计（DS 代码修改交接规格）

## 1. 文档状态

- 状态：**待用户书面复核后实施**。
- 日期：2026-08-08。
- 适用项目：`MedicalGraphRAG/`。
- 替代文档：`2026-08-07-merged-library-build-switch-design.md`。
- 核心决定：放弃合并大库；每个数据集独立建库、独立检索、独立使用自己的 queries/qrels。

本文件是 DS 后续编写 implementation plan 和修改代码时的唯一设计依据。实施者不得继续实现 `merge_config.json`、跨数据集图、跨库实体传播或 pooled merged 指标。

## 2. 目标与非目标

### 2.1 目标

1. 保持 `pubmedqa_hard_v1`、`nfcorpus_v1`、`scifact_v1` 等数据集各自独立的冻结语料和评测闭环。
2. 对摘要型数据，以**完整 document/abstract** 为一个 Passage 节点，通过 embedding cosine similarity 构建 kNN 软边。
3. 对 textbooks、statpearls 等长文本数据，以 chunk 为 Passage 节点，仅在同一文档内构建相邻边，边权固定为 `1.0`，保留 LinearRAG 的邻接边语义。
4. 保留 Entity–Passage、Entity→Sentence→Entity 传播、Dense passage prior 和 PPR 主链路，使本轮实验只改变 Passage–Passage 边。
5. 为每个图索引冻结输入哈希、检索粒度、边策略、边参数、边统计和产物哈希。
6. 通过相同 retrieval unit 的 BM25、Dense、Hybrid 与 Graph 对比，避免用 document-level Graph 对比 chunk-level 基线。

### 2.2 非目标

- 不建立合并大库，不做跨数据集 kNN、跨库实体 merge、跨库 ID namespace 或 merged qrels。
- 不下载或索引全量 PubMed/Wikipedia。
- 不在本轮加入 hub penalty、学习型边权、关系抽取、UMLS 实体链接或自定义 PPR transition matrix。
- 不在本轮同时引入 MedCPT、修改 NER 模型、修改 reranker 或生成模型。
- 不把图检索胜出作为工程验收条件；允许诚实负结果。
- 不删除或覆盖已经完成的 chunk-level 历史实验。

## 3. 数据集与策略映射

| 数据集/语料 | 冻结单位 | `retrieval_unit` | `passage_edge_mode` | 说明 |
|---|---|---|---|---|
| `pubmedqa_hard_v1` | dataset | `document` | `similarity` | `documents.jsonl` 每行是一篇完整摘要；不使用拆碎的 chunks 计算相似边 |
| `nfcorpus_v1` | dataset | `document` | `similarity` | 当前基本是一文档一 chunk，但仍显式读取 `documents.jsonl` |
| `scifact_v1` | dataset | `document` | `similarity` | 完整论文摘要作为 Passage |
| MedRAG textbooks | 单本教材文件 | `chunk` | `adjacent` | 只连接同一本书中 `order` 连续的 chunk |
| MedRAG statpearls | 单篇 article 文件 | `chunk` | `adjacent` | 只连接同一 article 中 `order` 连续的 chunk |

`hotpotqa_v1`、`frames_v1` 及其他已有数据不在本次修改范围内，保持现状。不得依据“文本看起来长/短”在运行期自动猜测策略；策略必须由配置显式指定并写入报告。

## 4. 总体架构

```text
data/processed/<dataset>/
  ├─ manifest.json
  ├─ documents.jsonl
  ├─ chunks.jsonl
  ├─ questions.jsonl
  └─ qrels.tsv
          │
          ▼
load_retrieval_passages(dataset_dir, retrieval_unit)
  ├─ document → passage_id = doc_id, text = full document.content
  └─ chunk    → passage_id = chunk_id, doc_id + order retained
          │
          ▼
build_graph_index(..., GraphBuildConfig)
  ├─ NER + sentence bridge
  ├─ Entity–Passage edges
  ├─ passage_edge_mode=similarity → document kNN soft edges
  ├─ passage_edge_mode=adjacent   → same-document chunk edges, weight=1.0
  └─ passage_edge_mode=none       → no Passage–Passage edges
          │
          ▼
LinearGraphRetriever.search(query, top_k)
          │
          ▼
dataset-local evaluation with the same qrels
```

每次调用只接收一个 `dataset_dir`，构建一个索引。不得新增接受多个 dataset 的 builder API。

## 5. 统一 Passage 加载契约

新增以下可单元测试的加载边界：

```python
@dataclass(frozen=True)
class RetrievalPassage:
    passage_id: str
    doc_id: str
    order: int | None
    title: str
    content: str
    source: str


def load_retrieval_passages(
    dataset_dir: Path,
    retrieval_unit: Literal["document", "chunk"],
) -> list[RetrievalPassage]:
    ...
```

### 5.1 `retrieval_unit=document`

- 输入必须是冻结的 `documents.jsonl`。
- `passage_id = doc_id`。
- `content` 必须使用完整 `document.content`；v1 不拼接 title，保持 `abstract_only` 口径。
- `order = None`。
- `passage_id`、`doc_id` 必须非空且唯一。
- build report 必须绑定 `documents.jsonl` 的 SHA-256，而不是 `chunks.jsonl`。
- qrels 已是 document ID，Graph 结果不再执行 chunk-to-document collapse。

“完整摘要”禁止通过 tokenizer 静默截断实现。document embedding 必须覆盖 `content` 的全部 token：

1. 用 embedding 模型自己的 tokenizer 得到不截断 token IDs；
2. 令每个 window 加入 special tokens 后的长度不超过模型实际 `max_seq_length`，按该有效长度切成确定性 token windows，`overlap_tokens=32`；
3. 每个 window 单独编码并 L2 normalize；
4. 对全部 window embeddings 做算术平均，再对结果做一次 L2 normalize，得到唯一 document embedding；
5. 记录每篇文档 window count，并报告被截断 token 数，后者必须为 0。

该派生 document embedding 必须同时供 Dense-document、Similarity kNN 建边和 Graph passage prior 使用，不允许三处各自重新编码出不同向量。BM25-document 和 NER 仍读取未截断的完整 `content`。

### 5.2 `retrieval_unit=chunk`

- 输入使用冻结的 `chunks.jsonl`，字段为 `chunk_id/doc_id/order/title/content/source`。
- `passage_id = chunk_id`。
- `doc_id` 与 `order` 是 Adjacent 构建的唯一依据；不得按全局 enumerate 建边。
- 对 MedRAG 原始语料，必须先经过 source adapter 生成上述字段，不允许把原始 `id` 直接当 `doc_id`。

MedRAG adapter 的规则必须基于文件身份：textbook 的文件 stem 是文档 ID，StatPearls 的 article 文件 stem 是文档 ID；原始 `id` 末尾的整数只用于校验和提取 chunk order。无法解析、前缀不匹配、order 重复时构建失败，不能静默跳过。

## 6. 图 profile 与边权契约

### 6.1 `abstract_similarity_v1`

```text
retrieval_unit     = document
passage_edge_mode  = similarity
adjacent edges     = disabled
similarity unit    = full document.content
```

该 profile 是 MedicalGraphRAG 的实验改进，不得称为“忠实 LinearRAG”。它保留 Entity–Passage 边，但用 document-document similarity 软边替代固定 `1.0` 的 Passage 邻接边。

### 6.2 `linearrag_adjacent_v1`

```text
retrieval_unit     = chunk
passage_edge_mode  = adjacent
similarity edges   = disabled
adjacent weight    = 1.0
```

该 profile 保留 LinearRAG 的邻接边权语义，同时修复原版全局排序可能造成的跨文档错边。只有 `doc_id` 相同且 `next.order == current.order + 1` 时才建边。

### 6.3 互斥规则

`passage_edge_mode ∈ {none, similarity, adjacent}`，同一个索引只能选择一个值。禁止在一个图中同时加入 Similarity 与 Adjacent 边，从设计上消除两类 Passage 边权混合冲突。

Entity–Passage 边继续沿用：

```text
weight(entity, passage)
  = count(entity in passage)
    / sum(count(all extracted entities in passage))
```

不得声称 Entity–Passage 权重与 Similarity 权重“已经同量纲”。v1 只是把两类真实权重交给现有 weighted PPR；报告必须分别记录两类边的权重总量和分布。`similarity_edge_scale` 固定为 `1.0`，本轮不调参。

## 7. 摘要 Similarity 软边算法

### 7.1 固定 v1 参数

```text
embedding_model       = models/all-mpnet-base-v2
normalize_embeddings  = true
document_aggregation   = mean_window_then_l2
window_overlap_tokens  = 32
index_type            = FAISS IndexFlatIP
k                      = 5
minimum_cosine         = 0.50
neighbor_policy        = union_knn
graph_direction        = undirected
similarity_edge_scale  = 1.0
self_edges              = forbidden
parallel_edges          = forbidden
```

这些值是首轮工程配置，不代表最优参数。若后续扫参，必须在独立 dev split 上一次只改变一个变量；SciFact/NFCorpus 当前没有项目内 dev split 时，不得使用 test 指标选择参数。

### 7.2 构建步骤

1. 按 `documents.jsonl` 的稳定顺序读取完整摘要。
2. 按第 5.1 节的 token-window 全覆盖规则生成并冻结 document embeddings。
3. 用 `IndexFlatIP` 请求 `k+1` 个邻居，删除自身 ID。
4. 丢弃 cosine `< minimum_cosine` 的候选。
5. 将有向候选 `(i, j)` 转为无向 pair `(min_id, max_id)`；任一方向进入 top-k 即保留，即 union-kNN。
6. 相同 pair 只写一条边；cosine 本身对称，边权为重新从冻结 embedding 计算的 dot product，再乘固定 `similarity_edge_scale=1.0`。
7. 不补齐低于阈值的邻居。没有合格邻居的 document 允许成为 similarity-isolated 节点，但仍可通过 Entity–Passage 边和 PPR reset 获得分数。

### 7.3 必须记录的诊断

- similarity edge count；
- isolated document count/rate；
- degree min/mean/P50/P95/P99/max；
- edge weight min/mean/P50/P95/max；
- exact-content duplicate pair count（只报告，不在本轮折叠 qrels）；
- document window count min/mean/P95/max，以及 truncated token count（必须为 0）；
- Entity–Passage 与 Similarity 两类边各自的 count、weight sum 和 weight percentiles。

不实现 hub penalty。若 P99/max degree 或失败案例显示 hub 污染，再另开单变量设计，不得在本轮“顺手优化”。

## 8. 长文本 Adjacent 边算法

1. 按 `doc_id` 分组。
2. 每组按数值 `order` 升序排序。
3. 校验 `(doc_id, order)` 唯一，order 为非负整数。
4. 只对 `next.order == current.order + 1` 的相邻项建无向边。
5. 每条边权严格为 `1.0`。
6. order 出现空洞时记录 gap，并禁止跨 gap 连边；无法解析 order 时构建失败。
7. 不建立跨文档边，不建立 Similarity 边。

若输入包含每篇文档的完整连续 chunks，则：

```text
adjacent_edge_count = Σ_doc max(chunk_count(doc) - 1, 0)
```

build report 必须同时记录 expected 与 actual；二者不一致时失败。

## 9. 配置与冻结报告

必须将构图参数从在线 `GraphConfig` 中分离为 `GraphBuildConfig`，避免把离线边策略与在线 PPR 参数混为一组：

```python
@dataclass(frozen=True)
class GraphBuildConfig:
    retrieval_unit: Literal["document", "chunk"]
    passage_edge_mode: Literal["none", "similarity", "adjacent"]
    embedding_model: str
    ner_model: str
    similarity_k: int = 5
    similarity_min_cosine: float = 0.50
    similarity_edge_scale: float = 1.0
    window_overlap_tokens: int = 32
```

配置校验：

- `document + adjacent` 非法；
- `chunk + similarity` 在 v1 非法；
- similarity 参数只在 similarity mode 生效，但仍完整写入 config hash；
- `k >= 1`、`0 <= minimum_cosine <= 1`、`similarity_edge_scale > 0`；
- `0 <= window_overlap_tokens < effective_window_tokens`；
- dataset manifest、retrieval source artifact、模型标识和完整 build config 必须进入索引身份哈希。

`graph_build.json` 至少新增：

```text
graph_schema_version
graph_profile
retrieval_unit
passage_edge_mode
source_artifact
source_artifact_sha256
entity_passage_edge_count
passage_passage_edge_count
edge_count_by_type
edge_weight_stats_by_type
similarity diagnostics 或 adjacent diagnostics
build_config
build_config_sha256
graph_sha256
embedding artifact hashes
document window coverage statistics
Git commit（由实验运行报告绑定）
```

搜索阶段必须从 build report 恢复并验证 retrieval unit、embedding model、NER model 和在线 PPR 参数，不允许 CLI 静默覆盖成另一个实验。

## 10. 公平基线与实验命名

摘要型新实验必须让所有对比方法读取相同的 `documents.jsonl`、使用相同 `abstract_only` 文本，并直接输出 document IDs：

| 方法 | retrieval unit | 说明 |
|---|---|---|
| BM25-document | document | 完整摘要建 Lucene index |
| Dense-document | document | 完整摘要 all-mpnet + IndexFlatIP |
| Hybrid-document | document | 上述两路 RRF，沿用并记录固定 `rrf_k=60` |
| Graph-EP-document | document | Entity–Passage，无 Passage–Passage 边 |
| Graph-Similarity-document | document | Entity–Passage + similarity soft edges |
| Reranker-document | document | 只在已有 reranker 流程自然支持时运行，不得阻塞首轮闭环 |

旧的 `graph_abstract_only` chunk-level 结果保留，不覆盖、不重命名成新结果。新目录必须包含 retrieval unit/profile，并采用以下名称：

```text
indexes/<dataset>/graph_document_ep_v1/
indexes/<dataset>/graph_document_similarity_v1/
experiments/<dataset>/bm25_document_v1/
experiments/<dataset>/dense_document_v1/
experiments/<dataset>/hybrid_document_v1/
experiments/<dataset>/graph_document_ep_v1/
experiments/<dataset>/graph_document_similarity_v1/
```

不得把 document-level 新指标与历史 chunk-level 指标直接做“提升 X%”结论。主对比只能发生在同一 retrieval unit、同一数据 manifest、同一 query split 之间。

## 11. 评测口径

每个数据集只使用自己的 queries/qrels，不做 pooled merged 指标。

首轮继续报告：

- Recall@1/5/10；
- MRR@10；
- nDCG@10；
- mean/P50/P95 latency；
- 索引构建时间、索引空间、CPU RAM 与 GPU VRAM（适用时）；
- 每个 query 返回结果数量分布；
- 成功和失败案例。

NFCorpus 多相关 qrels 必须保留全部 relevant documents，不能退化成 first-gold 指标。所有指标继续做独立复算并要求在 `1e-12` 绝对误差内一致。

对没有 queries/qrels 的 textbooks/statpearls，本轮只验收数据适配、图结构和可搜索性，不报告 Recall/MRR/nDCG，也不得用合成 query 宣称正式检索提升。

## 12. 代码边界（交给 DS）

DS 的 implementation plan 必须按以下职责拆分，不得把全部逻辑继续堆进单个 `graph.py`：

1. Passage loader：document/chunk 统一加载与输入校验。
2. Passage edge builders：纯函数实现 similarity 与 adjacent，两者互斥。
3. Graph builder orchestration：NER、embedding、三类节点/边、持久化、报告。
4. Search contract：识别 document 与 chunk unit，只有 chunk unit 执行 collapse。
5. Baseline collection export：BM25/Dense 使用同一 retrieval passages。
6. Evaluation/audit：绑定正确 source artifact hash 与 profile。
7. CLI/run pipeline：显式配置 profile，产物目录不覆盖历史实验。

必须新增或修改的主要区域：

```text
MedicalGraphRAG/src/medical_graphrag/data/
  retrieval_passages.py       # document/chunk loader + window coverage
MedicalGraphRAG/src/medical_graphrag/retrieval/graph.py
MedicalGraphRAG/src/medical_graphrag/retrieval/graph_edges.py
MedicalGraphRAG/src/medical_graphrag/retrieval/search_graph.py
MedicalGraphRAG/src/medical_graphrag/run_pipeline.py
MedicalGraphRAG/src/medical_graphrag/evaluation/graph.py
MedicalGraphRAG/configs/
MedicalGraphRAG/tests/
```

不得修改 `LinearRAG/` 官方快照来实现新策略；它只作为语义对照。不得删除现有 GraphConfig 默认值或改变历史实验解释。

## 13. 测试门禁

### 13.1 Passage loader

- document 模式确实读取完整 `documents.jsonl`，PubMedQA 多 chunk 文档只生成一个 Passage；
- 超过模型最大长度的 document 会生成多个 token windows，首尾 token 均被覆盖，truncated token count 为 0；
- document embedding 等于 fixture 中所有 window embeddings 的 `mean_then_l2` 手算结果；
- document ID 重复、空 content、manifest hash 不匹配时报错；
- chunk 模式保留 doc/order，字段缺失时报错；
- MedRAG adapter 能把 `Anatomy_Gray_0/1/2` 分到一个 doc，并生成 order 0/1/2；
- StatPearls 同理，且不同 article 绝不混组。

### 13.2 Similarity edges

- 使用手写归一化 embedding fixture 验证 cosine、top-k、阈值；
- 自环被移除；
- 同一 pair 双向命中时不产生平行边；
- union-kNN 任一方向命中即可保留；
- 所有权重有限且落在 `[minimum_cosine, 1 + 1e-6]`；
- 节点无合格邻居时构建仍成功并正确报告 isolated；
- 相同输入产生稳定节点/边顺序和一致产物哈希。

### 13.3 Adjacent edges

- 两个文档各三段时恰好四条边；
- 文档 A 末段不连接文档 B 首段；
- 每条权重严格为 `1.0`；
- order gap 不跨越；
- duplicate order/不可解析 order 失败。

### 13.4 互斥、搜索与评测

- similarity profile 中 Adjacent edge count 必须为 0；
- adjacent profile 中 Similarity edge count 必须为 0；
- document 模式搜索直接返回 qrels doc IDs，不走 collapse；
- chunk 模式保持当前 collapse 行为；
- index/search/evaluation 三层 hash 不匹配均失败；
- 现有历史测试全部继续通过。

## 14. 分阶段实施与验收

### Phase 1：PubMedQA document-level 闭环

只做 `pubmedqa_hard_v1`：

1. document Passage loader；
2. Graph-EP-document；
3. Graph-Similarity-document；
4. BM25/Dense/Hybrid document-level 公平基线；
5. dev/test 指标、结构报告和失败案例。

验收只要求闭环、测试、审计和真实结果，不要求 Similarity 图胜过 Dense/Hybrid。

### Phase 2：SciFact/NFCorpus 复用验证

不修改算法参数，直接复用固定 v1 配置，验证同一实现能在两个数据集独立运行。不得根据 test 结果回调 k/threshold。

### Phase 3：长文本 Adjacent 结构验证

为一个小型 textbook fixture 和一个小型 StatPearls fixture 建索引，先验证 adapter、同文档边和 PPR 可搜索；没有 qrels 时不进入正式质量对比。

只有 Phase 1 通过代码审阅后才能进入 Phase 2；只有摘要 profile 稳定后才实现长文本 adapter，避免同时修改两条建边链。

## 15. DS 提交后的审阅包

DS 完成代码修改后必须提供以下材料，再交回本 Agent 审阅：

1. implementation plan 路径；
2. Git commit 与完整 `git diff --stat`；
3. 修改文件列表及每个文件职责；
4. 单元测试命令和未截断的 pass/fail 摘要；
5. Phase 1 实际运行命令；
6. build/search/evaluation 报告路径与 SHA-256；
7. document-level 五路真实指标表；
8. similarity 图边数、isolated rate、degree/weight 分布；
9. 至少 3 个改善案例和 3 个退化案例；
10. 已知限制、未完成项和任何偏离本规格的理由。

不得只报告“pytest passed”或“指标提升”。本 Agent 将重点复核：完整摘要是否真的作为单节点、是否仍有跨文档 Adjacent 错边、Similarity 是否被错误写成固定权重、document-level 基线是否公平、test split 是否被用于调参、审计哈希是否覆盖真正输入。

## 16. 完成定义

本设计完成不等于实验结论为正。代码修改只有同时满足以下条件才可接受：

- 单数据集索引边界未被破坏；
- document/chunk retrieval unit 可由报告明确区分；
- Similarity 与 Adjacent 两类 Passage 边从配置、代码和产物上互斥；
- PubMedQA 使用完整摘要，而不是碎片 chunk 计算相似度；
- 长文本 Adjacent 只在同文档连续 order 内，权重为 `1.0`；
- 同粒度基线齐全，真实指标、成本和失败案例均保留；
- 全部配置、输入、图、排名与指标可通过哈希复核；
- DS 提交的实现通过本 Agent 的代码和实验审阅。
