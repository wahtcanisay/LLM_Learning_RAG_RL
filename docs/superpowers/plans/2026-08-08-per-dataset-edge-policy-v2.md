# Per-Dataset Document Edge Policy v2 实施计划（仅 Phase 0/1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 关闭审阅文档（`docs/superpowers/reviews/2026-08-08-per-dataset-edge-policy-plan-review.md`）全部 P0-1..8 与 P1-1..5，在 `MedicalGraphRAG/` 实现仅覆盖 PubMedQA Phase 1 的 document 级图检索与五路公平基线，产出真实指标与审计产物。

**Architecture:** 冻结设计 `docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md` 是唯一依据。核心是 **P0-8 唯一 document embedding artifact**：`documents.jsonl → build_document_embeddings()` 产出 `document_embeddings.npy + document_embedding_metadata.jsonl + document_embedding_report.json`，Dense index / Similarity kNN / Graph passage prior 三个 consumer 只加载同一 artifact，报告引用同一 `embedding_report_sha256`。document hit schema 统一为 `{doc_id, rank, score}`，`validate_hit_rows` 按 `retrieval_unit` 分支。

**Tech Stack:** Python 3.12 / igraph / sentence-transformers(all-mpnet)/ faiss(IndexFlatIP)/ spacy(BC5CDR)/ Pyserini(Lucene)/ numpy。容器 `llm-pytorch`，`/opt/venv/bin/python`。

**实施 commit:** `9fa9a42`（代码），v1 计划 `7b4c6cb` 仅作历史保留，不直接执行。NFCorpus/SciFact（Phase 2）与 MedRAG adjacent（Phase 3）**不在本轮实施**（P1-1）。

---

## 1. P0/P1 关闭矩阵

| 审阅项 | 关闭方式（实现位置） | 测试证据 |
|---|---|---|
| **P0-1** 覆盖证明 | `retrieval/document_embeddings.py`：`_window_ranges` 用索引区间、`_coverage` 用布尔位置 mask 计算 truncated；`embed_documents_full` 对空文本与未覆盖 fail closed | `tests/test_document_embeddings.py::test_window_ranges_*`、`test_embed_documents_full_*` |
| **P0-2** Similarity 边 | `retrieval/graph_edges.py`：唯一 ID、2-D/行数/finite/L2 归一化校验；union-knn、去重、并列确定性、权重域 `[min_cosine, 1+1e-6]` | `tests/test_graph_edges.py`（含 `test_similarity_edges_validates_inputs`） |
| **P0-3** GraphBuildConfig | `retrieval/graph.py::GraphBuildConfig.__post_init__` 运行时互斥校验；document→{none,similarity}、chunk→{none,adjacent}；`graph_profile` 区分 EP/Sim；`to_dict()` 供 config hash | `tests/test_graph_document.py::test_graph_build_config_*` |
| **P0-4** 完整 build 重构 | `retrieval/graph.py::build_graph_index`：loader→NER→句子桥→三类边→节点→持久化→完整报告，无占位；embedder/nlp 可注入 | `tests/test_graph_document.py::test_build_document_*` |
| **P0-5** 统一 ranking schema | `evaluation/retrieval.py::validate_hit_rows(retrieval_unit=...)`：document 校验 rank/唯一 doc_id/禁 chunk_id；chunk 校验 chunk_rank；`search_graph`/`search_document` 产出同构 doc schema | `tests/test_document_baselines.py::test_validate_hit_rows_document_schema` |
| **P0-6** 哈希链 | `search_graph.run_search` 写 `graph_build_report_sha256 = sha256_file(index_report)`；`evaluation/graph._validate_run_context` 校验其等于 `run_context["index_report_sha256"]` | `tests/test_graph_document.py::test_evaluate_graph_run_*`（含错误的 graph_sha256 被拒） |
| **P0-7** 五路完整基线 | `run_pipeline.py`：`run_bm25_document`/`run_dense_document`/`run_graph_document(profile)`/`run_hybrid_document` 各自完整 build→search→evaluate；`configs/pubmedqa_hard_v1_document.json` 已创建 | `tests/test_document_baselines.py`；CLI `run --help` 注册校验 |
| **P0-8** 唯一 embedding artifact | `retrieval/document_embeddings.py`：Dense(`build_dense_document_index`)、Graph(`build_graph_index` document unit) 均只加载并记录同一 `embedding_report_sha256`；`ensure_document_embeddings` 幂等 | `tests/test_document_baselines.py::test_p0_8_three_consumers_same_embedding_report_sha` |
| **P1-1** 阶段拆分 | v2 仅 Phase 0/1；Phase 0 删除"提交已提交计划"步骤，改为记录基线 commit 与测试日志 | 本文件 |
| **P1-2** Adjacent 语义 | `graph_edges.build_adjacent_edges`：同 doc 内 `next.order==order+1`、gap 不跨、expected/actual 一致性校验在 `build_graph_index` | `tests/test_graph_edges.py::test_adjacent_*` |
| **P1-3** MedRAG adapter | 不在本轮实施；规则记录为后续计划要求（prefix mismatch fail closed、空 content/非法 order/重复 ID/重复 order 拒绝） | — |
| **P1-4** 工程指标 | `run_manifest.json` 记录 latency 统计；索引 `index_bytes`/`elapsed_seconds`；正式运行日志单独计时 build/search/eval | Phase 1 证据 |
| **P1-5** 成对案例 | `evaluation/graph.py::write_graph_pair_cases`：逐题 EP/Sim gold rank、delta、新相似邻居+权重、改善/退化/不变 | `tests/test_document_baselines.py::test_write_graph_pair_cases`；`cli graph-pairs` |

## 2. 文件结构（实施后）

```text
MedicalGraphRAG/src/medical_graphrag/
  data/retrieval_passages.py          # 新建: RetrievalPassage + load_retrieval_passages
  retrieval/document_embeddings.py    # 新建: 唯一 embedding artifact(P0-8/P0-1)
  retrieval/graph_edges.py            # 新建: similarity/adjacent 纯函数边构建(P0-2/P1-2)
  retrieval/graph.py                  # 改: GraphBuildConfig + build_graph_index 重构 + LinearGraphRetriever 读 unit
  retrieval/search_graph.py           # 改: document 分支 + retrieval_unit 报告
  retrieval/dense.py                  # 改: build_dense_document_index(消费 artifact)
  retrieval/bm25.py                   # 改: export_document_collection
  retrieval/search_bm25.py            # 改: build_lucene_document_index
  retrieval/search_document.py        # 新建: document 级 Dense/BM25 检索
  evaluation/retrieval.py             # 改: validate_hit_rows 按 retrieval_unit 分支(P0-5)
  evaluation/graph.py                 # 改: document unit 评测 + P0-6 + write_graph_pair_cases(P1-5)
  evaluation/document.py              # 新建: 共享 document 评测核心
  evaluation/bm25.py / dense.py / hybrid.py  # 改: document 级评测入口
  run_pipeline.py                     # 改: 五路 runner + graph-pairs
  cli.py                              # 改: run --profile + graph-pairs 子命令
  configs/pubmedqa_hard_v1_document.json      # 新建
  tests/                              # 新建 6 个测试文件 + mocks.py(35 个新测试)
```

## 3. 测试门禁（全部在容器内通过）

- 单元测试不加载真实模型：`tests/mocks.py` 提供确定性 MockEmbedder/MockNlp，`build_graph_index`/`LinearGraphRetriever` 支持注入；
- 全量 `pytest tests/` = **114 passed**（历史 79 + 新增 35，无回归）；
- 真实模型仅在 integration/正式运行阶段使用。

## 4. Phase 1 正式运行命令

```bash
cd MedicalGraphRAG && GIT=$(git rev-parse HEAD)
IMG=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
/opt/venv/bin/python -m medical_graphrag.cli run bm25-document   --dataset pubmedqa_hard_v1 --git-commit $GIT --docker-image $IMG
/opt/venv/bin/python -m medical_graphrag.cli run dense-document   --dataset pubmedqa_hard_v1 --git-commit $GIT --docker-image $IMG
/opt/venv/bin/python -m medical_graphrag.cli run graph-document   --dataset pubmedqa_hard_v1 --profile ep --git-commit $GIT --docker-image $IMG
/opt/venv/bin/python -m medical_graphrag.cli run graph-document   --dataset pubmedqa_hard_v1 --profile similarity --git-commit $GIT --docker-image $IMG
/opt/venv/bin/python -m medical_graphrag.cli run hybrid-document  --dataset pubmedqa_hard_v1 --git-commit $GIT --docker-image $IMG
/opt/venv/bin/python -m medical_graphrag.cli graph-pairs          --dataset pubmedqa_hard_v1
```

产物目录（不覆盖历史 chunk 实验）：`experiments/pubmedqa_hard_v1/{bm25,dense,hybrid,graph_document_ep_v1,graph_document_similarity_v1}/` + `graph_ep_vs_sim_v1/paired_cases.jsonl`。

## 5. 验收证据（审阅 §6，2026-08-08 实测）

**1. v2 计划 commit**：`docs/superpowers/plans/2026-08-08-per-dataset-edge-policy-v2.md` @ `0758b4a`。

**2. 代码实现 commit**（起止）：`9fa9a42`（主实现）→ `0659f38`（embedding 批量优化）→ `a92078d`（model_name 修复）→ `396dfc3`（document_count 字段）→ `2ca1e56`（dense import）→ `094c4dd`（实验产物入库）。

**3. `git status --short`**：`M STUDY_PROGRESS.md`（用户维护，未动）；`?? docs/superpowers/reviews/`（审阅文档）；无其他未提交代码改动。

**4. pytest**：`/opt/venv/bin/python -m pytest tests/ -q` → **114 passed, 5 warnings, 31.33s**（历史 79 + 新增 35，无回归）。

**5. document embedding artifact**：
- 路径：`MedicalGraphRAG/outputs/pubmedqa_hard_v1/document_embeddings_v1/`
- report：`document_embedding_report.json`，文件 SHA-256 = `adbb1ad56330a62d11a166fb501a109c6b223c4397ac4fa12e708ee8ee89ed8d`
- `embeddings_sha256 = 9a4f8376dffdee6b2399db478a7536addd23af8ea2495ab231c1a236c6cf3408`；`metadata_sha256 = 306a0a46...`
- 5000 文档，dim 768，`window_coverage.truncated_token_count = 0`，window 均值 1.21

**6. 三 consumer 同 hash（P0-8 证明）**：Dense index、Graph-EP、Graph-Sim 三份报告的 `embedding_report_sha256` 均为 `adbb1ad...`（= artifact report 文件 hash），`embedding_embeddings_sha256` 均为 `9a4f8376...`（= npy 文件 hash）。

**7. 五路 runner 输出**（`experiments/pubmedqa_hard_v1/`，test 500）：

| 方法 | R@1 | R@5 | R@10 | MRR | nDCG |
|---|---:|---:|---:|---:|---:|
| BM25-document | 0.944 | 0.988 | 0.990 | 0.962 | 0.969 |
| Dense-document | 0.938 | 0.984 | 0.992 | 0.959 | 0.967 |
| Hybrid-document | 0.960 | 0.992 | 0.992 | 0.974 | 0.979 |
| Graph-EP-document | 0.660 | 0.910 | 0.972 | 0.768 | 0.818 |
| Graph-Similarity-document | 0.674 | 0.874 | 0.938 | 0.756 | 0.800 |

运行命令见 §4；产物目录含 `metrics.json / run_manifest.json / cases.json`。logs：`MedicalGraphRAG/logs/{dense_document,graph_ep,graph_sim,bm25_doc,hybrid_doc,graph_pairs}.log`。

**8. 正式运行配置**：`configs/pubmedqa_hard_v1_document.json`；manifest `data/processed/pubmedqa_hard_v1/manifest.json`（SHA `cf9b7591...`）；raw rankings 在 `outputs/pubmedqa_hard_v1/*_v1/raw_rankings.jsonl`。

**9. Graph-EP vs Graph-Sim 成对案例**：`experiments/pubmedqa_hard_v1/graph_ep_vs_sim_v1/paired_cases.jsonl`（713KB，500 题：**improve 35 / degrade 63 / no_change 402**）。

**10. 新增/修改文件**：见 §2；`git diff --stat 7b4c6cb..HEAD` = 24 files, +3019/-97（代码）+ 15 个实验产物文件。

**11. 已知限制 / 偏差**：
- 查询编码静默截断：pubmedqa 个别 query 超 384 token（transformers 警告 432>384），五路用同一查询编码，公平但长查询尾部信息丢失；
- **诚实负结果**：Similarity 软边比 Graph-EP 更差（R@10 0.938 vs 0.972，MRR 0.756 vs 0.768）；paired cases 显示 degrade 63 > improve 35。pubmedqa 的干扰摘要本就主题相近，kNN 相似边把 PPR 引向干扰摘要——本数据集上相似度边是负作用；
- Phase 2（NFCorpus/SciFact）与 Phase 3（MedRAG adjacent）不在本轮（P1-1/P1-3）。

## 6. 已知限制 / 未完成项

- Phase 2 / Phase 3 待 Phase 1 代码审阅通过后另写计划。
- 相似度边参数（k/min_cosine/scale）为固定 v1 值，未扫参（审阅 §7.1 禁止用 test 调参）；后续可在独立 dev split 上单变量扫描。
