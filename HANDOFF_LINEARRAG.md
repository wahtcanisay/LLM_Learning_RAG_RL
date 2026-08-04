# LinearGraphRetriever 交接文档（阶段 2 · 方案 A：pubmedqa_hard_v1 四路对比）

更新时间：2026-08-04
项目根目录：`D:\code_list\some tricks\LLMLeanring`
当前分支：`main`（方案 A 已合并；下一步方案 B 需新建分支）

## 1. 当前状态

阶段 1 + 阶段 2 方案 A 全部完成。`pubmedqa_hard_v1` 上**四路检索基线对比**（official test 500 题）：

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.926 | 0.974 | 0.984 | 0.945825 | 0.955147 | 1.36 ms |
| Dense（最优） | **0.966** | **0.992** | **0.994** | **0.977786** | **0.981885** | 16.08 ms |
| Hybrid RRF | 0.960 | 0.990 | 0.992 | 0.973833 | 0.978454 | 离线 n/a |
| Graph (BC5CDR + PPR) | 0.800 | 0.942 | 0.958 | 0.861929 | 0.885909 | 168.7 ms |

**结论**：Dense > Hybrid > BM25 > **Graph**。图检索在短摘要 + 事实题基准上垫底（且最慢），这是预期负结果——印证 `pubmedqa_hard_v1` 不是图的主场，需要方案 B。

## 2. 本轮做了什么

### 2.1 实现（新代码，忠实移植 LinearRAG 默认非向量化主干）

- `src/medical_graphrag/retrieval/graph.py`：
  - `build_graph_index`：医学 NER（scispaCy BC5CDR，疾病+化学物）→ 句子切分（spacy sentencizer）→ 句子 embedding → Entity↔Sentence 语义桥 → Entity-Passage 边（**按实体在原文出现次数归一化**，匹配 LinearRAG）→ igraph（Entity + Passage 节点）→ 持久化 + 哈希报告。
  - `LinearGraphRetriever`：`search(query, top_k)` = question NER → seed entities（argmax 实体匹配）→ Entity→Sentence→Entity BFS 传播 → passage 先验（min-max 归一化 Dense + 实体奖励）→ `igraph.personalized_pagerank` → top-k chunks。
- `src/medical_graphrag/evaluation/graph.py`：复用 `collapse_chunk_hits` / `evaluate_rankings`，绑定 graph 索引报告 + rankings 哈希。
- `scripts/build_graph_index.py`、`scripts/search_graph.py`（**必须用 `/opt/venv/bin/python`**，含 igraph + scispacy 模型）。
- CLI `evaluate-graph`；`tests/test_graph.py` 4 个测试；全套 `72 passed`。

### 2.2 依赖（容器 venv）

- 容器 `llm-pytorch`，venv `/opt/venv`（`--system-site-packages`）。
- 新增：`igraph 1.0.0`、`scispacy`、`en_ner_bc5cdr_md`（医学 NER）。
- 图检索/构建脚本一律用 `/opt/venv/bin/python`。

### 2.3 真实运行与审计

- 图：8378 实体、19178 Entity-Passage 边、42342 句子桥；相邻边 v1 关闭。
- 独立复算误差 ~1e-15；独立代码审查发现 8 项全部修复（运行时配置绑定、sidecar 哈希、边权按出现次数、log 域钳制、空实体守卫、PPR NaN 守卫、句子级实体过滤、死代码清理）。
- 修复后重建索引重跑，指标：test R@1 0.800、R@10 0.958。

## 3. 关键文件与产物

```text
MedicalGraphRAG/src/medical_graphrag/retrieval/graph.py
MedicalGraphRAG/src/medical_graphrag/evaluation/graph.py
MedicalGraphRAG/src/medical_graphrag/cli.py
MedicalGraphRAG/scripts/build_graph_index.py
MedicalGraphRAG/scripts/search_graph.py
MedicalGraphRAG/tests/test_graph.py
MedicalGraphRAG/experiments/pubmedqa_hard_v1/graph_abstract_only/{metrics,run_manifest,cases}.json
MedicalGraphRAG/indexes/pubmedqa_hard_v1/graph_abstract_only/   # ignored，图索引 + 报告
docs/superpowers/specs/2026-08-04-linearrag-graph-design.md
```

## 4. 下一步（方案 B：linearrag_medical_v1）

图检索在短摘要上垫底，需在其"主场"验证价值：

1. 从 `LinearRAG/dataset/medical/`（GraphRAG-Bench Medical：225 长指南 chunk、2062 题含 509 多跳、evidence 标注）构建 `linearrag_medical_v1` 冻结数据：documents/chunks/qrels（evidence 文本 → chunk 映射）+ manifest + 哈希。
2. 复用 `LinearGraphRetriever` 与统一评测；相邻边按 `(doc_id, order)` 文档内连接并作单变量实验。
3. 与 BM25/Dense/Hybrid（需要时在同样本上重跑）对比，重点看多跳题（Complex Reasoning）的图检索增益。

## 5. 可直接复制给下一位 Agent 的开场指令

```text
阶段 1（BM25/Dense/Hybrid）与阶段 2 方案 A（LinearGraphRetriever on pubmedqa_hard_v1）
全部完成并合并 main。四路对比：Dense > Hybrid > BM25 > Graph（图垫底，预期负结果）。
下一步是方案 B：构建 linearrag_medical_v1（GraphRAG-Bench Medical）并验证图检索在多跳上的价值。
依赖在容器 venv /opt/venv/bin/python。不要重跑或重构已审计基线。
任何新指标必须来自真实脚本输出。
```
