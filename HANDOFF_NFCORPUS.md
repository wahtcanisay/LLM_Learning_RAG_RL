# NFCorpus 评测适配交接文档（多相关 qrels + BEIR 基准）

更新时间：2026-08-04
项目根目录：`D:\code_list\some tricks\LLMLeanring`
当前分支：`main`（已合并）

## 1. 背景：为什么从 GraphRAG-Bench 转到 NFCorpus

方案 B 原计划用 GraphRAG-Bench Medical 验证图检索"主场"价值，但发现其**无文档级 gold**：
- corpus 是 1MB 大字符串，questions 只有改写过的文本 evidence；
- 逐字匹配仅 4/500、归一化 11/500、embedding 映射 top-1 不可靠；
- **无法构建可靠 qrels 的检索基准**。

改用**公开带标注的 BEIR/NFCorpus**（医学 IR，多相关 qrels，能算 Recall@k）。

## 2. 本轮做了什么

### 2.1 评测泛化：单 gold → 多相关 qrels

- `evaluation/retrieval.py`：
  - `evaluate_rankings` 泛化：`Recall@k = |gold ∩ top-k| / |gold|`；`MRR@10` = 首个相关文档倒数；`nDCG@10` = 多相关 DCG/IDCG。**单 gold 完全向后兼容**（pubmedqa 指标不变）。
  - `read_qrels` 返回 `{query: [doc_ids]}`（每题多个相关）。
  - `first_gold_rank` 助手。
- 4 个 eval 模块（bm25/dense/hybrid/graph）适配：`qrels[query_id]` 列表化、空 split 跳过、cases 取 `qrels[query_id][0]`。
- `retrieval/bm25.py` 的 `validate_frozen_dataset`：允许每题多个相关文档。

### 2.2 NFCorpus 数据与四路检索

- `scripts/build_nfcorpus.py`：从 BEIR parquet 构建 `nfcorpus_v1` 冻结数据（**323 test queries / 3633 docs / 12334 多相关 qrels**，每题平均 38 相关）。
- 四路检索全部复用既有实现：BM25（Pyserini）、Dense（all-mpnet + FAISS）、Hybrid（RRF）、Graph（BC5CDR + PPR）。

## 3. 真实结果（test 323）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| **Hybrid RRF** | **0.172** | **0.552** | **0.354** |
| Graph | 0.157 | 0.483 | 0.314 |

**关键洞察（两基准对比）：**
- `pubmedqa_hard_v1`：Dense > Hybrid > BM25 > Graph（Dense 碾压，BM25 太弱被 RRF 稀释）。
- `nfcorpus_v1`：**Hybrid > Dense > BM25 ≈ Graph**（BM25/Dense 势均力敌，RRF 融合真正互补）。
- **基准选择显著影响"融合是否有用"的结论**；图检索在两个基准均未胜出。

## 4. 审计

- 独立复算（自己实现多相关指标公式）：四路指标误差 0/1e-16。
- 独立代码审查：**未发现真实缺陷**；泛化向后兼容、read_qrels 列表化用法全迁移、validate 放宽无漏洞、build 数据正确。修了一个低严重度健壮性项（read_qrels 空行/字段处理）。
- 75 测试全过。

## 5. 关键文件

```text
MedicalGraphRAG/src/medical_graphrag/evaluation/retrieval.py   # 多相关指标泛化
MedicalGraphRAG/src/medical_graphrag/evaluation/{bm25,dense,hybrid,graph}.py
MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py          # validate 放宽
MedicalGraphRAG/scripts/build_nfcorpus.py
MedicalGraphRAG/experiments/nfcorpus_v1/*/metrics.json          # 四路结果
MedicalGraphRAG/data/raw/nfcorpus/ + data/processed/nfcorpus_v1/  # ignored，可重建
```

## 6. 下一步

阶段 3 MedicalGPT SFT（并行线已开始预习）。Reranker（Qwen3-Reranker）排在检索基线完备后。图检索的价值未在这两个基准上体现——若继续，需一个真正多跳 + 文档级 gold 的医学基准（如 HotpotQA 医学版），或接受"图检索在我们测试的医学检索场景无增益"的结论。

## 7. 可直接复制给下一位 Agent 的开场指令

```text
检索基线已完备并在两个基准评测：
pubmedqa_hard_v1（单 gold）与 nfcorpus_v1（BEIR 多相关 qrels）。
结论：pubmedqa 上 Dense 最优，nfcorpus 上 Hybrid（RRF）最优；图检索均未胜出。
评测已泛化支持多相关 qrels（向后兼容）。
不要重跑或重构已审计基线。下一步阶段 3 MedicalGPT SFT。
任何新指标必须来自真实脚本输出。
```
