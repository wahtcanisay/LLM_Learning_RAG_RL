# Dense Baseline 交接文档（阶段 1 · Dense/FAISS）

更新时间：2026-08-04
项目根目录：`D:\code_list\some tricks\LLMLeanring`
当前分支：`feat/dense-baseline`（尚未审计合并；审计通过后合并回 `main`）

## 1. 接手后的第一步

```powershell
cd "D:\code_list\some tricks\LLMLeanring"
git branch --show-current        # 预期 feat/dense-baseline 或合并后的 main
Get-Content -Raw -Encoding UTF8 STUDY_PROGRESS.md
Get-Content -Raw -Encoding UTF8 HANDOFF_DENSE_BASELINE.md
```

路线保持不变：

```text
阶段 1：MedRAG 基础检索 → BM25（完成）→ Dense（完成）→ Hybrid（下一步）
→ 阶段 2：LinearRAG 图结构检索与医学迁移
→ 阶段 3：MedicalGPT SFT → 阶段 4：Search-R1 RL → 阶段 5：MedSearch-R1
```

Reranker（专用 Qwen3-Reranker）排在 Dense/Hybrid/LinearRAG 三者全部完成后。

## 2. 当前状态

阶段 1 的正式检索基线已完成两条：BM25（`bm25_abstract_only`）与 **Dense（`dense_abstract_only`）**，均在冻结数据 `pubmedqa_hard_v1` 上。下一步是 Hybrid = RRF(BM25 + Dense)。

LinearRAG 仍只完成默认主干源码阅读，未做医学迁移实验。

## 3. 本轮完成的事情

### 3.1 Dense 实现（新代码）

- `src/medical_graphrag/retrieval/dense.py`：`export_chunk_metadata` + `build_dense_index`（懒加载 sentence-transformers/faiss，本地 CLI 可无重依赖导入）。
- `src/medical_graphrag/evaluation/dense.py`：`evaluate_dense_run`，与 BM25 评测共用 `evaluate_rankings` / `collapse_chunk_hits` / `read_jsonl` / `percentile` / `read_qrels`。
- `scripts/build_faiss_dense_index.py`、`scripts/search_faiss_dense.py`：容器内编码 + 建索引 + 检索，自带哈希绑定校验。
- `scripts/download_embedding_model.py`：断点续传 + 多源（hf-mirror → huggingface.co）重试下载器。
- CLI 新增 `evaluate-dense`（与 `evaluate-bm25` 同构，共用 `_evaluate_command`）。
- 测试：新增 10 个（scripts 校验、评测契约、哈希绑定、导出顺序），全套 `57 passed`。

### 3.2 模型获取（网络坑）

- `all-mpnet-base-v2` 权重（438MB）HF 直连多次中途断线、hf-mirror 无 ETag 被 huggingface_hub 拒绝。
- 最终用 `scripts/download_embedding_model.py`（Range 断点续传 + 无限重试）下到 gitignored 的 `MedicalGraphRAG/models/all-mpnet-base-v2/`。
- 报告里 embedding_model 记录为本地相对路径 `models/all-mpnet-base-v2`（从 `MedicalGraphRAG/` 运行）。

### 3.3 真实运行参数

```text
embedding_model = models/all-mpnet-base-v2（= LinearRAG 同款 all-mpnet-base-v2）
dim = 768
normalize = true（点积 = 余弦）
index = FAISS IndexFlatIP（精确）
chunk_top_k = 100（与 BM25 候选深度对齐）
折叠 = max(chunk_score) by doc_id（与 BM25 相同）
运行环境 = llm-pytorch 容器，Python 3.12.3，sentence-transformers 2.2.2，faiss-cpu 1.8.0
```

### 3.4 正式结果

| 指标（official test 500） | BM25 | Dense | Δ |
|---|---:|---:|---:|
| Recall@1 | 0.926 | **0.966** | +0.040 |
| Recall@5 | 0.974 | **0.992** | +0.018 |
| Recall@10 | 0.984 | **0.994** | +0.010 |
| MRR@10 | 0.945825 | **0.977786** | +0.032 |
| nDCG@10 | 0.955147 | **0.981885** | +0.027 |
| Mean latency | 1.359 ms | 13.309 ms | Dense 约 10× 慢 |

dev（500 题）：Recall@1 0.966、Recall@5 0.994、Recall@10 0.994、MRR@10 0.978233、nDCG@10 0.982254。

### 3.5 案例观察（cross-retriever）

- test 上 Dense 仅 3 题 gold 掉出 Top-10（BM25 8 题）：`11570976`（"Is it Crohn's disease?" rank 63）、`18359123`（"Is it better to be big?" rank 57）、`18708308`（rank 15）。
- **前两题与 BM25 失败的是同一批**：query 过短/模糊，两个检索器都难以区分同主题候选——可作为下一步 Hybrid 是否缓解的观察点。
- `7482275`（necrotizing fasciitis / hyperbaric oxygenation）BM25 与 Dense 都排 gold 第 1。

## 4. 关键文件与产物

### Git 跟踪文件（待审计合并）

```text
MedicalGraphRAG/src/medical_graphrag/retrieval/dense.py
MedicalGraphRAG/src/medical_graphrag/evaluation/dense.py
MedicalGraphRAG/src/medical_graphrag/evaluation/retrieval.py   # 新增共享辅助
MedicalGraphRAG/src/medical_graphrag/evaluation/bm25.py       # 改用共享辅助（无回归）
MedicalGraphRAG/src/medical_graphrag/cli.py                    # 新增 evaluate-dense
MedicalGraphRAG/scripts/build_faiss_dense_index.py
MedicalGraphRAG/scripts/search_faiss_dense.py
MedicalGraphRAG/scripts/download_embedding_model.py
MedicalGraphRAG/tests/test_dense_scripts.py
MedicalGraphRAG/tests/test_dense_evaluation.py
MedicalGraphRAG/tests/test_dense_export.py
MedicalGraphRAG/experiments/pubmedqa_hard_v1/dense_abstract_only/metrics.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/dense_abstract_only/run_manifest.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/dense_abstract_only/cases.json
docs/superpowers/specs/2026-08-03-dense-baseline-design.md
STUDY_PROGRESS.md
HANDOFF_DENSE_BASELINE.md
```

### ignored 真实证据

```text
MedicalGraphRAG/models/all-mpnet-base-v2/                        # 本地模型（438MB）
MedicalGraphRAG/outputs/pubmedqa_hard_v1/dense_abstract_only/    # embeddings/index/rankings
MedicalGraphRAG/indexes/pubmedqa_hard_v1/dense_abstract_only/    # （未单独建目录，索引在 outputs/）
```

关键哈希：

```text
dataset manifest:  cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa
chunk embeddings:  6954766a5e0589145c61ac32f7da163c40bd11a84d4e949119e17be96823f332
index:             733dad620e843530c07fe7930ad6a90a0cc9ab2a02b9df4902cb578709173faf
raw rankings:      21d08cbfac28760d758382efea42205e73f47b3d611f9e7486ee1ebac5a19039
```

评测审计：`evaluate-dense` 校验 search/index 报告与 rankings/metadata/questions 的 SHA-256 绑定、embedding 配置一致性（model/dim/normalized/index_type）、命中分布、连续 rank、有限 score。

## 5. 下一步唯一任务

**Hybrid = RRF(BM25 + Dense)**：

1. 复用同一冻结数据、dev/test、qrels、document 折叠与 `evaluate_rankings`。
2. RRF：`score(doc) = Σ 1/(k + rank_r(doc))`，k 固定（如 60），按 rank 融合两路文档排名，不标定分数。
3. 与 BM25、Dense 单路对比；记录 Recall@1/5/10、MRR@10、nDCG@10、延迟。
4. 观察 `11570976` / `18359123` 这类短 query 是否被融合缓解。
5. 不引入 Reranker、生成模型或 LinearRAG。

## 6. 边界（必须保留）

- 封闭 5,000 文档 / 1,000 题基准，不把 Recall@10=0.994 外推为全 PubMed 性能。
- PubMedQA 题目与 gold 摘要主题高度重合，高指标不等于"搜索很强"。
- 检索指标，不是 QA Accuracy；生成模型未运行。
- Dense 每 query 延迟约 13ms 是单次编码 + FAISS 检索的真实值，不含批处理编码的摊薄。
- 模型在本地相对路径，换机器/容器需重下或重指 `--embedding-model`。

## 7. 可直接复制给下一位 Agent 的开场指令

```text
当前在阶段 1，BM25 与 Dense 两条正式基线已完成（冻结数据 pubmedqa_hard_v1）。
不要重跑或重构 BM25/Dense，也不要直接进入 LinearRAG 医学迁移。
今天唯一任务是设计并实现 Hybrid = RRF(BM25 + Dense)：
复用同一冻结数据、折叠与评测接口，融合两路 document 排名，
与两条单路基线对比并记录真实指标与失败案例。
任何新指标必须来自真实脚本输出和落盘文件。
```
