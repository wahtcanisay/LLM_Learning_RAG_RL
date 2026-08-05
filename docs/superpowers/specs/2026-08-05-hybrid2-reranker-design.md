# Hybrid2 设计：Qwen3-Reranker 三路检索重排

## 1. 目标

在既有检索基线（BM25 / Dense / Graph，均已在 pubmedqa_hard_v1 与 nfcorpus_v1 上评测）之上，新增 **Hybrid2**：用专用 cross-encoder 重排器 **Qwen3-Reranker** 对三路检索的候选文档**联合重排**，目标是超越纯 RRF 融合（Hybrid）与任一路单检索器。

关键区别：
- **Hybrid（RRF）**：只看排名位置，不做语义级联合打分；
- **Hybrid2（Reranker）**：把 `query` 与 `document` **拼在一起**喂给 cross-encoder，做真正的语义相关度打分。

## 2. 流程

```text
BM25 raw rankings ─┐
Dense raw rankings ─┼→ 各自折叠成文档排名 → 每路取 top-N → 候选文档并集
Graph raw rankings ─┘
                                    ↓
        每题 (query, candidate_doc) 成对 → Qwen3-Reranker 打分
                                    ↓
                  按 reranker 分数排序 → top-10 文档排名
                                    ↓
                  evaluate_rankings（多相关，复用既有评测）
```

- 候选深度：每路取 top-N（N=50），三路并集去重后约 ≤150 候选/题；
- 重排器：`Qwen/Qwen3-Reranker-0.6B`（0.6GB，先跑通；4B 作可选加强）；
- 输出：**文档级排名**（cross-encoder 在整篇文档上打分，不是 chunk）；
- 评测：复用 `evaluate_rankings` 多相关指标，与既有基线同表对比。

## 3. 为什么可能有效（假设）

- RRF 对"某路把错误文档排很高"敏感（noisy 高排名被抬升）；
- cross-encoder 直接看 query↔doc 词级交互，能区分"两路都排前面但都不相关"的候选；
- 在 nfcorpus（多相关、信号均衡）上最可能看到提升。

## 4. 组件与文件

- `src/medical_graphrag/retrieval/reranker.py`：加载 Qwen3-Reranker，`rerank(query, candidates) -> 排序后的 doc_ids + scores`；
- `scripts/rerank_candidates.py`：读取三路 raw rankings → 折叠 → 候选并集 → 重排 → 写 doc 级 rankings + 报告（哈希绑定）；
- `src/medical_graphrag/evaluation/reranker.py`：`evaluate_reranker_run`，读 doc 级 rankings + qrels → 指标；
- CLI：`evaluate-reranker`；
- 产物：`experiments/{dataset}/reranker_qwen/{metrics,run_manifest,cases}.json`。

## 5. 运行环境

- 容器 venv `/opt/venv/bin/python`；
- 新增依赖：`Qwen3-Reranker-0.6B` 模型（0.6GB，从 HF 镜像下载到 `models/`）；
- 重排脚本必须在容器内跑（模型 + transformers）。

## 6. 完成标准

- 三路候选并集 + Qwen reranker 重排，两个基准（pubmedqa / nfcorpus）都有真实指标；
- 与 Hybrid（RRF）、三路单路同表对比，记录提升/持平/下降；
- 独立复算指标、代码审查、pytest 全过；
- 学习者能解释 cross-encoder 与 bi-encoder / 排名融合的本质区别。
