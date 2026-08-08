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

## 5. 验收证据（审阅 §6）

- [x] 代码实现 commit：`9fa9a42`
- [x] pytest 全量：114 passed（日志见容器 `MedicalGraphRAG`，`pytest -q` 输出）
- [x] embedding artifact + report 路径与 SHA-256（见 §6）
- [x] 三 consumer 同 hash 证明：`test_p0_8_three_consumers_same_embedding_report_sha`
- [ ] 五路正式运行产物（见 §6，运行中）
- [ ] Graph-EP vs Graph-Sim 成对案例文件
- [ ] 新增/修改文件清单（§2）

## 6. 已知限制 / 未完成项

- Phase 2（NFCorpus/SciFact）与 Phase 3（MedRAG adjacent）不在本轮，待 Phase 1 代码审阅通过后另写计划（P1-1/P1-3）。
- 正式实验若因环境未跑，将如实标注"未运行"及阻塞原因（审阅 §6 末条）。
