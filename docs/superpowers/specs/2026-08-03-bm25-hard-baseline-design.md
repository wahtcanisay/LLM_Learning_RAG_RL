# BM25 医学硬检索基线设计

## 1. 目标

在已经冻结的 `pubmedqa_hard_v1` 上落地第一条可复现的真实检索基线。BM25 使用 WSL Docker 容器中的 Pyserini/Lucene，对 7,562 个固定 chunk 建索引；每个问题先召回 Top-100 chunk，再按 `doc_id` 聚合成唯一文档排名，最终报告文档级 Recall@1/5/10、MRR@10、nDCG@10 与延迟。

本轮只做 `abstract_only` 主设置，即索引 chunk 的 `content`。`title_abstract` 仅作为后续标题泄漏对照，不进入本轮主结果。

## 2. 已冻结的输入

输入目录为 `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/`：

- `questions.jsonl`：1,000 条问题，500 dev、500 test；
- `chunks.jsonl`：7,562 个 chunk；
- `qrels.tsv`：1,000 条文档级相关性标注；
- `manifest.json`：数据参数与输入文件 SHA-256。

运行前必须校验 manifest 中的数量和哈希。任何不一致均直接失败，不能静默使用变化后的数据。

## 3. 范围与非目标

本轮包含：

1. 导出 Pyserini JSON collection；
2. 用 Lucene 构建 BM25 索引；
3. 对 dev/test 全部问题检索 Top-100 chunk；
4. 以最大 chunk 分数聚合为文档排名；
5. dev、test 分别评测并记录运行环境；
6. 保存成功/失败案例，供下一步理解 BM25 行为。

本轮不包含 Dense、Hybrid、Reranker、LinearRAG、生成模型或 QA Accuracy；不修改 BM25 参数来追逐 test 指标；不把该 5,000-document 封闭集合结果外推为全 PubMed 性能。

## 4. 组件与文件边界

### 4.1 项目代码

新增或扩展以下模块：

- `src/medical_graphrag/retrieval/bm25.py`
  - 导出 Pyserini collection；
  - 读取 chunk 命中并执行 chunk→document 聚合；
  - 提供稳定、可单元测试的数据结构。
- `src/medical_graphrag/evaluation/retrieval.py`
  - 复用现有 `evaluate_rankings`；
  - 增加按 split 选择 qrels/rankings 与结果审计所需的辅助逻辑。
- `src/medical_graphrag/cli.py`
  - 增加 `export-pyserini` 与 `evaluate-bm25` 子命令。
- `scripts/search_pyserini_bm25.py`
  - 只承担容器内 Pyserini 查询；
  - 不复制指标公式，也不决定 dev/test 规则。

### 4.2 生成产物

大体积、可重建产物保持忽略：

```text
MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only/
MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/
```

其中 collection 使用两个文件，避免依赖 Pyserini 是否保留额外 JSON 字段：

- `collection/chunks.jsonl`：每行只含 `id=chunk_id` 与 `contents=content`；
- `chunk_metadata.jsonl`：保存 `chunk_id`、`doc_id`、`order`、`title`、`source`。

可提交的小型实验记录放在：

```text
MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/
├── metrics.json
├── run_manifest.json
└── cases.json
```

原始逐题 Top-100 排名放在 ignored `outputs/`，不提交到 Git。

## 5. 数据流

```text
frozen chunks.jsonl
        ↓ export-pyserini
collection/chunks.jsonl + chunk_metadata.jsonl
        ↓ Pyserini Lucene indexer
BM25 index
        ↓ 每题 Top-100 chunk
raw_rankings.jsonl
        ↓ max-score collapse by doc_id
unique document rankings
        ↓ split-aware evaluator
dev metrics + official test metrics + cases
```

导出阶段必须保证：

- collection 恰有 7,562 行且 `id` 唯一；
- metadata 与 collection 的 chunk ID 集合完全相同；
- `contents` 只来自 chunk `content`，不拼接 title、source 或答案字段；
- 空 query、空 content、未知 chunk ID 或未知 doc ID 均报错退出。

## 6. BM25 与排名契约

第一版在查询脚本中显式调用 `LuceneSearcher.set_bm25(k1=0.9, b=0.4)`。这两个值已经从容器内 Pyserini 0.22.1 的实际函数签名与源码确认，也是该版本的默认值；显式设置可避免后续升级造成默认值漂移。任何参数调整只能依据 dev 集，且必须形成新的实验配置。

每个 query 请求最多 Top-100 chunk。Lucene BM25 只返回至少包含一个匹配词项的文档，因此稀有词 query 可能得到少于 100 个 hit；这种短排名必须原样参与指标，未返回文档视为未命中，禁止用零分文档补齐。运行报告必须保存实际 hit 数分布和短排名问题数。文档聚合规则固定为：

```text
document_score = max(score of retrieved chunks with the same doc_id)
```

排序键固定为：

1. `document_score` 降序；
2. 该文档最佳 chunk 的原始排名升序；
3. `doc_id` 字典序升序。

这保证重复文档只保留一次，且分数并列时结果可重现。短于 10 个唯一文档的合法 BM25 排名仍参与 Recall@10；报告必须显式暴露其长度，不能丢弃该题或伪造无词项匹配的候选。

原始排名每行至少保存：

```json
{
  "query_id": "21645374",
  "latency_ms": 12.3,
  "hits": [
    {
      "chunk_id": "PMID:21645374#0",
      "doc_id": "PMID:21645374",
      "chunk_rank": 1,
      "score": 9.87
    }
  ]
}
```

## 7. 数据集隔离与指标

dev 与 test 共用冻结语料和索引，但必须单独报告：

- dev 500 题：调试、检查失败案例、未来选择 BM25 或融合参数；
- test 500 题：最终主结果，只在实现和配置冻结后运行与报告；
- 不把两者合并成一个 1,000 题主指标。

文档级指标由现有 `evaluate_rankings` 统一计算：

- Recall@1；
- Recall@5；
- Recall@10；
- MRR@10；
- binary nDCG@10。

由于每题恰有一个 gold document，Recall@k 在数值上等于 Hit@k，但报告名称仍固定为 Recall@k。延迟记录搜索调用本身，不包含 Docker 启动、collection 导出和索引构建；分别报告 mean、P50、P95。索引时间与索引目录大小单独记录。

## 8. 实验记录

`run_manifest.json` 至少包含：

- 运行时间、Git commit、操作系统与 Docker image；
- Python、Java、Pyserini/Lucene 版本；
- 数据集 manifest SHA-256 和四个输入 artifact SHA-256；
- query/chunk/document 数量与 dev/test 数量；
- text mode、chunk top-k、聚合规则、BM25 `k1`/`b`；
- 索引命令、检索命令、评测命令；
- 索引时间、索引空间与原始排名文件 SHA-256。

`metrics.json` 必须同时写入样本数和 split，防止脱离上下文引用数字。`cases.json` 至少保存：

- 5 个 test 成功案例；
- 5 个 test 失败案例；
- gold document rank（若进入 Top-100）；
- Top-10 文档 ID、最佳 chunk ID、分数、标题；
- gold 标题和 gold chunk 摘要，供人工分析术语匹配失败原因。

若失败案例少于 5 个，则保存全部失败案例并如实注明数量，不能为了凑数改写定义。

## 9. 错误处理

以下情况必须返回非零退出码：

- 数据哈希或数量与 manifest 不一致；
- query、chunk 或 qrel ID 重复；
- qrel 指向不存在的文档；
- Pyserini 返回未知 chunk ID；
- 任一问题缺少排名记录，或短排名数量分布未写入 search report；
- dev/test 交集非空；
- 排名文件 query 集合与目标 split 不一致；
- 指标或延迟出现 NaN、负值；
- test 运行沿用了未记录的参数覆盖。

中途失败保留日志和临时排名用于定位，但不得生成看似完整的 `metrics.json`。

## 10. 测试与验证策略

实现阶段遵循先测试后代码，至少覆盖：

1. collection 只索引 `content`，不泄漏 title；
2. collection/metadata ID 一一对应；
3. 同一文档多个 chunk 采用最大分数；
4. 相同分数按最佳 chunk rank、doc ID 稳定排序；
5. 少于请求 Top-k 的合法 Lucene 排名被保留并写入分布统计；
6. dev/test qrels 严格分离；
7. 已知小排名的 Recall、MRR、nDCG 精确值；
8. CLI 参数、输出目录与错误码；
9. 在现有小型 Lucene fixture 或临时索引上完成一次容器内集成冒烟。

最终运行前后均执行完整 pytest；真实 test 指标必须来自命令输出及落盘文件，不能手工填写。

## 11. 本轮完成标准

只有同时满足以下条件，BM25 硬基线才算完成：

- 7,562 个 chunk 成功建立可复现 Lucene 索引；
- dev/test 各 500 题均产生最多 Top-100 chunk 和唯一文档排名，短排名分布可审计；
- pytest 全部通过，容器集成冒烟通过；
- 真实输出 Recall@1/5/10、MRR@10、nDCG@10、mean/P50/P95 延迟；
- 记录索引时间、空间、版本、命令、Git commit 与输入/输出哈希；
- 保存并人工阅读成功和失败案例；
- 更新 `STUDY_PROGRESS.md`，由学习者能够解释 BM25 为何不是向量检索，以及 chunk 排名为什么需要聚合到文档级后再对 qrels 评测。
