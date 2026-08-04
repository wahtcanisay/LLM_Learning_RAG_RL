# Dense 医学检索基线设计

## 1. 目标

在冻结的 `pubmedqa_hard_v1` 上落地第二条可复现检索基线：Dense/FAISS。与 BM25 基线的唯一差异是检索器本身；冻结数据、dev/test 划分、qrels、chunk→document 折叠、评测接口与审计纪律全部保持一致。本轮最终报告文档级 Recall@1/5/10、MRR@10、nDCG@10、延迟与显存峰值。

主设置仍为 `abstract_only`（索引 chunk 的 `content`）；`title_abstract` 只作后续泄漏对照。

## 2. 已冻结的输入

输入目录为 `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/`：

- `questions.jsonl`：1,000 条问题，500 dev、500 test；
- `chunks.jsonl`：7,562 个 chunk；
- `documents.jsonl`：5,000 个文档；
- `qrels.tsv`：1,000 条文档级相关性标注；
- `manifest.json`：数据参数与输入文件 SHA-256。

运行前必须校验 manifest 数量和哈希，不一致直接失败。

## 3. 范围与非目标

本轮包含：

1. 用 `sentence-transformers/all-mpnet-base-v2` 编码 7,562 个 chunk；
2. 构建 FAISS `IndexFlatIP`（精确内积，向量归一化后等价余弦）；
3. 对 dev/test 全部问题编码并检索 Top-100 chunk；
4. 以最大 chunk 分数聚合为文档排名（复用现有折叠）；
5. dev、test 分别评测并记录运行环境与显存峰值；
6. 保存成功/失败案例，供下一步与 BM25 对照。

本轮不包含 Hybrid、Reranker、LinearRAG、生成模型或 QA Accuracy；不调 embedding 参数追逐 test 指标；不把 5,000-document 封闭结果外推为全 PubMed 性能。

**候选深度一致：** Dense 检索同样只取 Top-100 chunk 后折叠，而不是把全部 7,562 个 chunk 都计入聚合。这样与 BM25 的可比性成立——差异只来自检索器打分质量，而不是"Dense 看到了更多候选"。BM25 因稀疏命中而少于 100 chunk 的问题依旧存在；Dense 天然每个 query 都能返回 100 个 chunk。

## 4. 组件与文件边界

### 4.1 项目代码

- `src/medical_graphrag/retrieval/dense.py`
  - chunk 编码与 query 编码；
  - FAISS `IndexFlatIP` 构建与检索；
  - 复用 `collapse_chunk_hits` 做 chunk→document 折叠；
  - 提供稳定、可单元测试的数据结构。
- `src/medical_graphrag/evaluation/retrieval.py`
  - 复用现有 `evaluate_rankings`，与 BM25 完全相同。
- `src/medical_graphrag/cli.py`
  - 增加 Dense embedding、检索、评测相关子命令。
- `scripts/`
  - 容器内 Dense 脚本只承担编码与检索，不复制指标公式，也不决定 dev/test 规则。

### 4.2 生成产物

大体积、可重建产物保持忽略：

```text
MedicalGraphRAG/indexes/pubmedqa_hard_v1/dense_abstract_only/
MedicalGraphRAG/outputs/pubmedqa_hard_v1/dense_abstract_only/
```

可提交的小型实验记录放在：

```text
MedicalGraphRAG/experiments/pubmedqa_hard_v1/dense_abstract_only/
├── metrics.json
├── run_manifest.json
└── cases.json
```

## 5. 数据流

```text
frozen chunks.jsonl
        ↓ all-mpnet-base-v2 编码（归一化）
chunk_embeddings.npy + chunk_metadata
        ↓ FAISS IndexFlatIP
dense_index
        ↓ 每题 query 编码 + 检索 Top-100 chunk
raw_rankings.jsonl
        ↓ max-score collapse by doc_id（与 BM25 相同）
unique document rankings
        ↓ split-aware evaluator
dev metrics + official test metrics + cases
```

## 6. 编码与排名契约

- embedding 模型固定为 `sentence-transformers/all-mpnet-base-v2`，与 LinearRAG 的 `all-mpnet-base-v2` 及本数据切块 tokenizer 一致；维度 768。
- 所有 embedding 使用 `normalize_embeddings=True`，FAISS 用 `IndexFlatIP`，点积即余弦相似度。
- query 文本固定为 `question` 字段原文，与 BM25 输入一致，不拼 context/options。
- 每 query 请求 Top-100 chunk，`document_score = max(score of retrieved chunks with same doc_id)`，排序键为：分数降序 → 最佳 chunk rank 升序 → `doc_id` 字典序升序（复用现有折叠函数，保持与 BM25 完全一致的确定性）。
- 显存峰值本轮适用：记录真实 GPU/CPU 峰值；CPU-only 时如实记录为"CPU only"并给显存峰值 0，不估计。

## 7. 数据集隔离与指标

dev 与 test 严格分开，主结果为 official test 500 题。文档级指标复用 `evaluate_rankings`：Recall@1/5/10、MRR@10、binary nDCG@10。延迟记录检索调用本身（编码+FAISS 搜索），不包含数据导出与索引构建；分别报告 mean、P50、P95。

## 8. 实验记录

`run_manifest.json` 至少包含：

- 运行时间、Git commit、操作系统与 Docker image；
- Python、transformers、sentence-transformers、FAISS 版本；
- embedding 模型名与维度、归一化设置、IndexFlatIP、chunk top-k；
- 数据集 manifest SHA-256 和输入 artifact SHA-256；
- query/chunk/document 数量与 dev/test 数量；
- 编码、检索、评测命令；
- 索引时间、索引空间、embedding 文件 SHA-256、显存峰值。

`metrics.json` 写入样本数与 split。`cases.json` 保存 5 个 test 成功、5 个 test 失败案例（与 BM25 同字段），并额外记录"BM25 成功但 Dense 失败"与"Dense 成功但 BM25 失败"的对照，便于下一步分析两种信号差异。

## 9. 错误处理

以下情况必须返回非零退出码：

- 数据哈希或数量与 manifest 不一致；
- chunk embedding 数量与 chunk 数不一致；
- query 编码失败或返回 NaN；
- 任一问题缺少排名记录或查询集合与 split 不一致；
- dev/test 交集非空；
- 指标或延迟出现 NaN、负值；
- FAISS 索引与冻结 chunk 元数据不一致。

## 10. 测试与验证策略

先测试后代码，至少覆盖：

1. 全部 7,562 chunk 都有 embedding，维度为 768；
2. FAISS `IndexFlatIP` 检索结果等于逐点暴力内积（小样本对照）；
3. query 编码结果稳定、无 NaN；
4. 同一文档多个 chunk 采用最大分数折叠（与 BM25 折叠测试共享）；
5. dev/test qrels 严格分离；
6. 已知小排名 Recall/MRR/nDCG 精确值；
7. CLI 参数、输出目录与错误码；
8. 容器内集成冒烟：真实编码 → FAISS → 折叠 → 指标。

最终运行前后均执行完整 pytest；真实 test 指标必须来自命令输出及落盘文件。

## 11. 本轮完成标准

- 7,562 chunk 全部编码并建立可复现 FAISS `IndexFlatIP` 索引；
- dev/test 各 500 题产生 Top-100 chunk 与唯一文档排名；
- pytest 全部通过，容器集成冒烟通过；
- 真实输出 Recall@1/5/10、MRR@10、nDCG@10、mean/P50/P95 延迟与显存峰值；
- 记录模型、版本、命令、Git commit、输入/输出哈希；
- 保存 BM25↔Dense 交叉成功/失败案例；
- 更新 `STUDY_PROGRESS.md`，学习者能够解释 Dense 与 BM25 在检索信号上的差异，以及为什么候选深度保持一致。
