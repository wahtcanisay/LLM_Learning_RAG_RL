# 项目交接文档（单一文档，聚合全部阶段）

更新时间：2026-08-09
项目根目录：`D:\code_list\some tricks\LLMLeanring`
分支：`main`（本地领先 origin **18 个 commit 未推送**，见 §7）

> 本文件是唯一交接文档。之前的 `HANDOFF_DENSE_BASELINE.md`、`HANDOFF_HYBRID.md`、`HANDOFF_LINEARRAG.md`、`HANDOFF_NFCORPUS.md`、`HANDOFF_DEEPSEEK_V4_FLASH.md` 已合并删除。

## 1. 项目概览与当前进度

**MedicalGraphRAG** —— 可复现的医学检索（RAG）研究与评测项目。用统一、可审计的框架实现并对比多种医学检索器，为最终医学搜索智能体（MedSearch-R1）打地基。项目总览见 [`MedicalGraphRAG/README.md`](MedicalGraphRAG/README.md)（含全部数据集、实验结果、LinearRAG 适配、简历版总结）。

路线（5 阶段）：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | MedRAG 检索基础 | ✅ 完成 |
| 2 | LinearRAG 图检索迁移 | ✅ 完成 |
| 3 | MedicalGPT 领域微调 SFT | 🔜 SFT 调用链学习已完成，正式训练待规划 |
| 4 | Search-R1 搜索强化学习 | ⏳ 仓库/论文已拉取，模型选型已决策，训练待规划 |
| 5 | MedSearch-R1 医学搜索智能体 | ⏳ 设计文档已定，实施待阶段 4 后 |

**检索层全部完成并审计**：5 检索器（BM25 / Dense / Hybrid / Graph / Hybrid2）× 4 基准（pubmedqa_hard_v1 / nfcorpus_v1 / scifact_v1 / hotpotqa_v1）。统一入口 `cli run <retriever> --dataset <name>`。

## 2. 运行环境（重要）

- 容器：`llm-pytorch`（WSL2 Docker）。
- Python 环境：**`/opt/venv`**（`--system-site-packages`，继承 torch / sentence-transformers / faiss；新增 igraph、scispacy、en_ner_bc5cdr_md、Qwen3-Reranker）。**一律用 `/opt/venv/bin/python`**。
- 本地模型（git 忽略）：`MedicalGraphRAG/models/all-mpnet-base-v2`、`MedicalGraphRAG/models/Qwen3-Reranker-0.6B`。
- 统一评测入口：`python -m medical_graphrag.cli run <retriever> --dataset <name>`。
- 测试：`/opt/venv/bin/python -m pytest`（当前 79+ 测试）。

## 3. 已完成阶段（精简）

### 3.1 冻结数据

| 数据集 | 规模 | 说明 |
|---|---|---|
| `pubmedqa_hard_v1` | 1,000 问 / 5,000 文档 / 7,562 chunks / 1,000 qrels | 自建单答案，seed `20260803` |
| `nfcorpus_v1` | 323 test / 3,633 文档 / 12,334 qrels | BEIR 多相关，均 38 相关/问 |
| `scifact_v1` | 300 问 / 5,183 文档 | BEIR 科学声明核查 |
| `hotpotqa_v1` | 7,405 问 / 66,581 文档 / 14,810 qrels | 多跳硬 gold，每问恰 2.0 gold |
| `trec_covid_v1` | 数据已建 | **已放弃检索**（24.6% 空文本） |
| `frames_v1` | 824 问 + 金标映射 | 轻接，不建语料 |

**bioasq 因 HF 仓库 401 门禁未接入**。全部哈希冻结、可重建（`scripts/build_*.py`）。

### 3.2 检索器与结果

| 检索器 | 文件 | 实现 |
|---|---|---|
| BM25 | `retrieval/bm25.py` | Pyserini/Lucene，k1=0.9 b=0.4 |
| Dense | `retrieval/dense.py` | all-mpnet + FAISS IndexFlatIP |
| Hybrid | `retrieval/hybrid.py` | BM25+Dense RRF（k=60） |
| Graph | `retrieval/graph.py` | BC5CDR NER → 实体-段落图 → PPR（移植 LinearRAG，已对齐官方参数） |
| Hybrid2 | `retrieval/reranker.py` | Qwen3-Reranker-0.6B 三路候选重排 |

**结果（真实、可复算，chunk 级五路）**：

*pubmedqa_hard_v1（test 500）*

| 方法 | R@1 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.946 | 0.955 |
| **Dense** | **0.966** | **0.994** | **0.978** | **0.982** |
| Hybrid | 0.960 | 0.992 | 0.974 | 0.979 |
| Graph | 0.790 | 0.982 | 0.864 | 0.894 |

*nfcorpus_v1（test 323）*

| 方法 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| Hybrid | 0.172 | 0.552 | 0.354 |
| Graph | 0.158 | 0.477 | 0.312 |
| **Hybrid2** | **0.189** | **0.584** | **0.384** |

*scifact_v1（test 300）*

| 方法 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.791 | 0.630 | 0.664 |
| Dense | 0.769 | 0.594 | 0.633 |
| Hybrid | 0.838 | 0.669 | 0.705 |
| Graph | 0.705 | 0.456 | 0.509 |
| **Hybrid2** | **0.895** | **0.740** | **0.772** |

*hotpotqa_v1（test 7405，多跳）*

| 方法 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.738 | 0.798 | 0.663 |
| Dense | 0.711 | 0.817 | 0.668 |
| Graph | 0.695 | 0.797 | 0.650 |
| Hybrid | 0.800 | 0.852 | 0.731 |
| **Hybrid2** | **0.898** | **0.955** | **0.865** |

> **并行线新增（2026-08-08 起，Codex/DS 实施）**：document 级五路基线 + Per-Dataset Edge Policy（graph profile `ep`/`similarity`，reranker `--sources bd/bdg/bde`）。详见 §5.2。

### 3.3 关键结论

1. **基准选择显著影响结论**：pubmedqa（单答案）Dense 最优；nfcorpus/scifact/hotpotqa（多相关/多跳）Hybrid2 最优；
2. **Hybrid2 (Qwen3-Reranker) 全面最优**，多跳场景（hotpotqa）价值最显著（R@10 0.898 vs Hybrid 0.800）；
3. **Hybrid RRF 大幅超过单路**（hotpotqa R@10 0.800 vs Dense 0.711）——词项+语义两路互补；
4. **图检索在全部四个基准均未胜出**（诚实负结果；HotpotQA 上 BC5CDR 医学 NER 在通用维基实体稀疏）；
5. 所有指标独立复算误差 < 1e-15，每阶段过独立代码审查。

### 3.4 测评适配

评测泛化支持**多相关 qrels**（BEIR 风格）：`Recall@k=|gold∩topk|/|gold|`、`MRR`=首个相关、`nDCG` 多相关；单 gold 向后兼容。`evaluate_rankings` / `read_qrels` / `first_gold_rank` 在 `evaluation/retrieval.py`。

## 4. 审计链与纪律

- 每个阶段：数据冻结哈希 → 索引/排名哈希绑定 → 评测校验（query 集合、split、命中分布、连续 rank、有限 score、各阶段 SHA）；
- 独立复算 + 独立代码审查是标配；
- **铁律**：只改一个变量、指标必须真实脚本输出、不编造结果、封闭基准不外推、大规模下载/模型先汇报、新实验用隔离分支审计后合并 `main`。

## 5. 下一步（交给 GPT 规划）

### 5.1 阶段 3 MedicalGPT SFT（主推进方向）

- **调用链学习已完成**（2026-08-09，见 STUDY_PROGRESS）：4 概念题复述通过（LoRA/QLoRA、`train_on_inputs` loss mask、切模型显存杠杆、GRPO≠Search-R1）；4 处学习注释已补（commit `48f6dfb`）。
- **待规划**：正式 SFT 训练。数据 `MedicalGPT/data/sft/` 已有样例（medical_sft_1K_format.jsonl 等，git 忽略）。默认模型 `Qwen/Qwen3.5-0.8B` 过小；建议按 Search-R1 选型走 3B。

### 5.2 并行线 Per-Dataset Edge Policy（进行中，Codex 审阅，DS 实施）

- 2026-08-08 设计冻结（`docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md`）；合并大库方案废弃。
- DS implementation plan `7b4c6cb` 被 Codex 审阅判定有阻断项；v2 计划 + 代码待 DS 回交。
- 代码已有 document 级五路 runner、graph profile（`ep`/`similarity`）、reranker `--sources`。**下门禁：DS 回交代码 commit + pytest 日志后由 Codex 审阅。**

### 5.3 阶段 4 Search-R1（材料已备，训练待规划）

- **仓库已拉取** `Search-R1/`（PeterGriffinJin/Search-R1，veRL GRPO/PPO/REINFORCE++）；**论文** `papers/Search-R1_2503.09516.pdf`（commit `3d4832d`）。
- **模型选型已决策**（2026-08-05 调研，写进 MedSearch-R1 设计文档 §7）：官方用 Qwen2.5-7B 主（+26%）/3B（+21%）；本地 RTX 5090 32GB；**3B 起步 → 稳定后 4B/7B**；rollout≥4（3B≈11GB、7B≈20GB）；MedicalGPT 内置 GRPO（TRL，QLoRA）。

### 5.4 其他

- MIRAGE 端到端（仓库已拉取，等阶段 3 有 LLM 再跑 QA Accuracy）。
- FRAMES 轻接已完成（`frames_v1/questions_with_gold.json`）。

## 6. 给下一位 Agent（GPT）的开场指令

```text
先读 MedicalGraphRAG/README.md、STUDY_PROGRESS.md、HANDOFF.md。

【已完成】检索层 5 路 × 4 基准 chunk 级全部审计合并；统一 evaluate 入口
cli run <retriever> --dataset <name>。Graph 已对齐 LinearRAG 官方参数。
Search-R1 仓库+论文已拉取，模型选型已决策（3B 起步→4B/7B，rollout≥4）。
MedicalGPT SFT 调用链学习已完成（4 概念题复述通过）。

【并行线】Per-Dataset Edge Policy 由 Codex 审阅、DS 实施（进行中），
代码已有 document 级 runner，勿破坏。

【待你规划】阶段 3 正式 SFT 训练（数据 MedicalGPT/data/sft/ 已有样例）、
阶段 4 Search-R1 3B GRPO 训练。

环境用 /opt/venv/bin/python；不要重跑或重构已审计基线。
任何新指标必须来自真实脚本输出 + 哈希审计。
```

## 7. Git 状态与推送（重要）

- **本地领先 origin 18 个 commit 未推送**（含 Search-R1 拉取、模型选型、edge policy 实施、注释等）。
- **push 被 git TLS 握手卡住**（`unexpected eof while reading`），根因是代理链路对 github.com 的时效性网络问题，**不是 git 配置或认证**（gh 已登录 wahtcanisay）。之前 push 成功过（`51609b7..fe89a64`）。
- **解决**：换代理节点/重启代理后 `git push origin main`。git 全局配置已恢复原状（无 sslBackend 覆盖，系统 schannel 默认；代理 7890 保留）。
