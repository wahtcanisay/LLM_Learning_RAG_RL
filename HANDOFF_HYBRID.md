# Hybrid RRF 交接文档（阶段 1 · Hybrid = RRF(BM25+Dense)）

更新时间：2026-08-04
项目根目录：`D:\code_list\some tricks\LLMLeanring`
当前分支：`feat/hybrid-rrf`（审计通过后合并回 `main`）

## 1. 当前状态

阶段 1 三条检索基线全部完成，均在冻结数据 `pubmedqa_hard_v1` 上、同一评测契约：

| 方法 | test Recall@1 | test Recall@10 | test MRR@10 | test nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.945825 | 0.955147 |
| **Dense（最优单路）** | **0.966** | **0.994** | **0.977786** | **0.981885** |
| Hybrid RRF (k=60) | 0.960 | 0.992 | 0.973833 | 0.978454 |

**负结果：Hybrid 全面低于 Dense 单路。** 不深挖、不做无依据调参，如实记录后进入阶段 2。

## 2. 本轮做了什么

### 2.1 实现（新代码）

- `src/medical_graphrag/retrieval/hybrid.py`：`fuse_rrf(first, second, k=60)` 纯函数——按排名位置融合两路文档排名，缺失文档贡献 0，并列按 doc_id 字典序确定性排序。
- `src/medical_graphrag/evaluation/hybrid.py`：`evaluate_hybrid_run(...)`——读取两路已审计 raw rankings，各自按 max-chunk-score 折叠，RRF 融合，`evaluate_rankings` 评测；绑定两路源排名 SHA-256、两路 dataset manifest、metadata；manifest 记录 k、两路检索器配置。
- `src/medical_graphrag/cli.py`：新增 `evaluate-hybrid`。
- 测试：`tests/test_hybrid.py` 7 个用例（RRF 手算、缺失文档、tie-break、k 校验、评测契约、哈希绑定、已知失败查询记录）；全套 `65 passed`。

### 2.2 真实运行

- 复用 BM25（`outputs/pubmedqa_hard_v1/bm25_abstract_only_audited/`）与 Dense（`outputs/pubmedqa_hard_v1/dense_abstract_only/`）的 raw rankings，**不重跑任何检索器**。
- 独立复算：dev/test 五项指标与 metrics.json 最大绝对误差 `0 / 1e-16`。
- 已知失败查询 `11570976`（Dense 63→Hybrid 114）与 `18359123`（57→113）在融合后更糟。

## 3. 负结果分析与结论

- **为什么融合反而差**：此基准上 BM25 严格弱于 Dense（PubMedQA 摘要偏语义匹配，词项信号是噪音）。RRF 让 BM25 高位错误文档获得高融合分，稀释了更强的 Dense 信号；短模糊 query 上 BM25 gold 完全不在 top-100，只贡献噪音。
- **结论**：设计假设"融合词项+语义互补信号能超过单路"在 `pubmedqa_hard_v1` 上被证伪。**Dense 是阶段 1 最优检索器**。
- 教训：RRF 融合只有在两路信号强度相近且互补时才可能增益；一路显著弱于另一路时，融合会回归或更差。这是可写入简历的真实负结果案例（诚实报告）。

## 4. 关键文件与产物

```text
MedicalGraphRAG/src/medical_graphrag/retrieval/hybrid.py
MedicalGraphRAG/src/medical_graphrag/evaluation/hybrid.py
MedicalGraphRAG/src/medical_graphrag/cli.py
MedicalGraphRAG/tests/test_hybrid.py
MedicalGraphRAG/experiments/pubmedqa_hard_v1/hybrid_rrf/metrics.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/hybrid_rrf/run_manifest.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/hybrid_rrf/cases.json
docs/superpowers/specs/2026-08-04-hybrid-rrf-design.md
STUDY_PROGRESS.md
HANDOFF_HYBRID.md
```

## 5. 下一步

**阶段 2：LinearRAG 图检索迁移。** 要点见 `STUDY_PROGRESS.md`「下一步」：实现统一 `LinearGraphRetriever.search(query, top_k)` 接入同一评测；数据考虑 `linearrag_medical_v1`（GraphRAG-Bench Medical，长文档 + 多跳题，图检索主场）；相邻边按 `(doc_id, order)` 文档内连接并作单变量实验。Reranker（Qwen3-Reranker）排在全部检索基线之后。

## 6. 可直接复制给下一位 Agent 的开场指令

```text
阶段 1 三条检索基线已完成（BM25 / Dense / Hybrid RRF），Dense 最优，
Hybrid 为负结果（已记录，不重做）。不要重跑或重构任何基线。
下一步是阶段 2 LinearRAG 图检索迁移：
实现 LinearGraphRetriever.search(query, top_k) 复用同一评测，
先用官方 medical 小数据或构建 linearrag_medical_v1 验证，
再决定是否在 pubmedqa_hard_v1 上迁移。任何新指标必须来自真实脚本输出。
```
