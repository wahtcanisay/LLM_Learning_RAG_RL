# Hybrid RRF 检索基线设计

## 1. 目标

在冻结的 `pubmedqa_hard_v1` 上落地第三条检索基线：Hybrid = RRF(BM25 + Dense)。复用 BM25 与 Dense 两条已审计基线的真实 raw rankings，在**文档排名层**用 Reciprocal Rank Fusion 融合，目标是验证：融合两种互补信号（词项匹配 + 语义相似度）能否整体超过任一路单路，并观察短模糊 query（如 `11570976` "Is it Crohn's disease?"、`18359123` "Is it better to be big?"）是否被缓解。

本轮**不重新运行任何检索器**——融合是一个纯后处理层，输入是两路已落盘的 raw rankings。

## 2. 已冻结的输入

- `pubmedqa_hard_v1` 冻结数据：questions / documents / chunks / qrels / manifest（哈希固定）。
- BM25 真实排名：`outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl` + `search_run.json`。
- Dense 真实排名：`outputs/pubmedqa_hard_v1/dense_abstract_only/raw_rankings.jsonl` + `search_run.json`。
- chunk metadata：两路运行的 `metadata_sha256` 相同（`7c664c2c...`），任一路的 `chunk_metadata.jsonl` 即可，评测时校验其哈希。

## 3. 范围与非目标

本轮包含：

1. 读取两路 raw rankings，各自按 `doc_id` 取最大 chunk 分数折叠成文档排名；
2. 对每题的文档排名做 RRF 融合（k 固定）；
3. dev/test 分别评测 Recall@1/5/10、MRR@10、nDCG@10；
4. 与 BM25、Dense 单路对比，并单独检查两个已知失败 query 的排名变化；
5. 保存融合配置、两路源排名哈希与结果。

本轮不包含：重新建索引或重跑检索；Reranker、生成模型、LinearRAG；调 k 追逐 test 指标（k 只在 dev 上验证敏感性）。

## 4. RRF 融合契约

对每题，设 `R_b` 为 BM25 折叠后的文档排名，`R_d` 为 Dense 折叠后的文档排名（均为 `doc_id` 序列，缺失文档视为排名无穷大）：

```text
rrf_score(doc) = 1/(k + rank_b(doc)) + 1/(k + rank_d(doc))
```

- `k` 固定为 `60`（RRF 论文常用值）；缺失于某路的文档该路贡献 0。
- 融合后按 `rrf_score` 降序排列；分数并列时按 `doc_id` 字典序升序，保证确定性。
- 两路都折叠自 Top-100 chunk（BM25 部分题少于 100 是真实短排名，Dense 恒为 100），候选深度一致。

## 5. 组件与文件边界

- `src/medical_graphrag/retrieval/hybrid.py`
  - `fuse_rrf(bm25_doc_ids, dense_doc_ids, k=60) -> list[doc_id]`：纯函数，可单元测试。
- `src/medical_graphrag/evaluation/hybrid.py`
  - `evaluate_hybrid_run(...)`：读取两路 raw rankings → 各自折叠 → RRF 融合 → 评测 → 写 metrics/manifest/cases；校验两路 query 集合一致、源排名哈希绑定。
- `src/medical_graphrag/cli.py`
  - 新增 `evaluate-hybrid` 子命令。
- 产物：`experiments/pubmedqa_hard_v1/hybrid_rrf/{metrics.json,run_manifest.json,cases.json}`。

## 6. 数据流

```text
BM25 raw_rankings.jsonl ─┐
                         ├→ 各自 max-chunk-score 折叠 → 文档排名 → RRF(k=60) → 融合文档排名
Dense raw_rankings.jsonl ┘                                              ↓
                                                              split-aware evaluator
                                                      dev metrics + test metrics + cases
```

## 7. 数据集隔离与指标

dev 与 test 严格分开；主结果 official test 500 题。指标复用 `evaluate_rankings`：Recall@1/5/10、MRR@10、binary nDCG@10。延迟不适用（融合是离线后处理，不是在线检索调用；若报告，仅记录融合计算耗时，且必须与检索延迟分开标注）。

## 8. 实验记录

`run_manifest.json` 至少包含：

- 运行时间、Git commit、两路源 rankings 的 SHA-256；
- 两路 search/index report 的摘要（embedding model、BM25 k1/b、text mode、chunk top-k）；
- k 值、融合规则、聚合规则、dev/test 数量；
- 数据集 manifest SHA-256。

`cases.json` 至少保存：5 个 test 成功、5 个 test 失败案例（与单路同字段），并**显式记录 `11570976` 与 `18359123` 两题的 gold rank（融合前 BM25/Dense 各多少、融合后多少）**。

## 9. 错误处理

- 两路 raw rankings 的 query 集合不一致 → 报错；
- 任一 source ranking 缺失、哈希不匹配 → 报错；
- 融合后出现空排名或 NaN 分数 → 报错；
- dev/test 与冻结 questions 的 split 不一致 → 报错。

## 10. 测试与验证策略

1. RRF 数学：两路已知小排名，手算 rrf_score 与排序；
2. 缺失文档：只在一路出现的文档正确获得单路贡献；
3. 并列分数按 doc_id 稳定排序；
4. k 为 60 时与手算一致；
5. 折叠复用：与 BM25/Dense 单路折叠结果一致；
6. 两路 query 集合不一致、源排名被替换 → 报错；
7. 指标与单路在相同数据上可复算。

## 11. 本轮完成标准

- 两路真实 raw rankings 融合成功，dev/test 各 500 题有融合文档排名；
- pytest 全部通过；
- 真实输出 Recall@1/5/10、MRR@10、nDCG@10，与 BM25、Dense 单路同表对比；
- `11570976` / `18359123` 的 gold rank 变化显式记录；
- 记录 k、两路源排名 SHA 与配置；
- 更新 `STUDY_PROGRESS.md`，学习者能解释 RRF 为何用排名而非分数、融合在哪些 query 上带来增益或损失。
