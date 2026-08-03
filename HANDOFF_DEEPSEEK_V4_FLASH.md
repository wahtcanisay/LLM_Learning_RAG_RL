# DeepSeek V4 Flash 交接文档

更新时间：2026-08-03  
项目根目录：`D:\code_list\some tricks\LLMLeanring`  
当前分支：`main`  
BM25 功能与实验基线：`3bfdb8f`（`exp: bind final BM25 rankings evidence`）；本交接文档提交后 `main` HEAD 会更新到文档提交。

## 1. 接手后的第一步

必须先执行：

```powershell
cd "D:\code_list\some tricks\LLMLeanring"
git branch --show-current
git status --short
Get-Content -Raw -Encoding UTF8 STUDY_PROGRESS.md
Get-Content -Raw -Encoding UTF8 HANDOFF_DEEPSEEK_V4_FLASH.md
```

预期：分支为 `main`。本交接文档提交后，工作区应当干净。

项目路线必须保持：

```text
阶段 1：MedRAG 基础检索
→ 阶段 2：LinearRAG 图结构检索与医学迁移
→ 阶段 3：MedicalGPT LoRA/QLoRA SFT
→ 阶段 4：Search-R1 3B 搜索强化学习
→ 阶段 5：MedSearch-R1 医学证据搜索 Agent
```

不要因为 LinearRAG 源码已经读过就跳过阶段 1 的 Dense、Hybrid、Reranker 和 QA 基线。

## 2. 当前究竟处于什么状态

当前仍属于阶段 1，但正式 BM25 检索基线的代码、真实运行、硬指标、审计链、案例和文档已经完成并合并到 `main`。

阶段 1 尚未整体完成，原因是正式 Dense、Hybrid、Hybrid + Reranker、生成与 QA Accuracy 都还没有落地。LinearRAG 目前只完成默认主干源码的第一轮阅读，没有完成官方 medical 数据端到端运行，也没有医学迁移实验。

当前唯一应继续的任务不是立刻写 Dense，而是让学习者完成 BM25 案例解释门禁，确认其能够解释已经跑出的真实结果。

## 3. 本轮已经完成的事情

### 3.1 新项目与冻结数据

已建立独立项目：`MedicalGraphRAG/`，与 `MedRAG/`、`LinearRAG/` 平级。

已实现：

- PubMedQA PQA-L 加载；
- MedRAG PubMed 确定性 distractor 采样；
- 文档边界安全切块；
- `questions.jsonl`、`documents.jsonl`、`chunks.jsonl`、`qrels.tsv` 构建；
- 原子文件写入、manifest、SHA-256 和结构审计；
- 检索指标计算和案例保存。

正式冻结数据 `pubmedqa_hard_v1`：

| 项目 | 数量 |
|---|---:|
| Questions | 1,000 |
| Documents | 5,000 |
| Gold documents | 1,000 |
| MedRAG PubMed distractors | 4,000 |
| Chunks | 7,562 |
| Qrels | 1,000 |
| Dev | 500 |
| Official test | 500 |

关键配置：seed `20260803`、max tokens `512`、overlap `64`、主文本模式 `abstract_only`。

数据 manifest SHA-256：

```text
cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa
```

冻结数据位于 ignored 目录：

```text
MedicalGraphRAG/data/processed/pubmedqa_hard_v1/
```

不要重新清洗或改写这批数据后继续沿用 `pubmedqa_hard_v1` 名称；若数据发生变化，必须新建版本并重新生成 manifest。

### 3.2 正式 BM25 基线

运行环境：WSL2 Docker 容器 `llm-pytorch`，Pyserini `0.22.1`，Python `3.12.3`，Java `21.0.11`。BM25/Lucene 使用 CPU，不使用 GPU，因此 GPU 显存峰值为“不适用”。

正式参数：

```text
text_mode = abstract_only
stemmer = porter
remove_stopwords = true
k1 = 0.9
b = 0.4
chunk_top_k = 100
document aggregation = max(chunk_score) by doc_id
```

流程：

```text
frozen chunks
→ Pyserini collection
→ Lucene index
→ Top-100 chunk hits
→ 按 doc_id 折叠
→ document ranking
→ Recall / MRR / nDCG
```

Lucene 不返回零词项匹配文档。真实运行中 995/1,000 题返回 100 hits，另外 5 题分别返回 3、29、58、71、72 hits。这些短排名被原样评测，没有补零、伪造候选或丢弃问题。

### 3.3 正式结果

| Split | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| dev（500） | 0.920 | 0.972 | 0.978 | 0.943250 | 0.951905 | 1.394 ms |
| official test（500） | 0.926 | 0.974 | 0.984 | 0.945825 | 0.955147 | 1.359 ms |

official test 是主结果。独立复算时，dev/test 的五项检索指标与 `metrics.json` 最大绝对误差为 `0.0`。

必须保留以下解释边界：

- 这是 5,000 文档的封闭检索基准，不是全量 PubMed；
- PubMedQA 问题与 gold article 主题高度一致，因此高 Recall 不得外推；
- 这些是检索指标，不是 QA Accuracy；
- 尚未运行生成模型，不能声称问答性能提升；
- 主实验不索引 title，`title_abstract` 只能作为后续单变量/泄漏敏感性对照。

### 3.4 审计链与修复

已经闭合以下 provenance chain：

```text
frozen dataset/artifact hashes
→ questions SHA
→ collection SHA + metadata SHA
→ index SHA
→ index report SHA
→ raw rankings SHA
→ final evaluation manifest
```

评测会验证：query 集合、split、命中数量分布、短排名数量、连续 rank、有限 score、chunk→document 映射、Top-k、k1、b、text mode 和各阶段 SHA。替换 questions、index、metadata 或保持命中数量不变地替换 rankings 都会失败。

本轮解决过的关键问题：

- U+2029 被 `str.splitlines()` 错误当作 JSONL 边界；
- `pyserini.__version__` 不存在，改读 distribution metadata；
- 少于 Top-100 的真实短排名不能按固定长度报错；
- Docker 中脚本无法导入未安装的本项目包；
- 评测最初硬编码 Top-k/k1/b/text mode；
- 最初缺少 dataset→index→search→evaluation 的完整哈希绑定。

当前完整测试：

```powershell
cd "D:\code_list\some tricks\LLMLeanring\MedicalGraphRAG"
python -m pytest -q
```

最近真实结果：`47 passed`。

## 4. 关键文件与产物位置

### Git 跟踪文件

```text
STUDY_PROGRESS.md
MedicalGraphRAG/README.md
MedicalGraphRAG/configs/pubmedqa_hard_v1.json
MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py
MedicalGraphRAG/src/medical_graphrag/evaluation/bm25.py
MedicalGraphRAG/scripts/build_pyserini_index.py
MedicalGraphRAG/scripts/search_pyserini_bm25.py
MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/run_manifest.json
MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/cases.json
docs/superpowers/specs/2026-08-03-bm25-hard-baseline-design.md
docs/superpowers/plans/2026-08-03-bm25-hard-baseline.md
```

### ignored 但已经保留的真实证据

```text
MedicalGraphRAG/data/processed/pubmedqa_hard_v1/
MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only_audited/
MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only_audited_2c7bcbc/
```

关键哈希：

```text
questions:    cb957619d30d8885e685e334652abbc6376263278c7de337c35cf3537ce56982
collection:   8651101da23e625c4324e6e0d97018039c2cefd97f539c74bfd69d7fb202360c
metadata:     7c664c2c94fa7ab9aaf497716cbe40be71c53aa3e2ec957c90029d5dadb7649f
index:        24d98c4f6ce12c6aba2e8f7e7aa34c9b5594b92c0caca7a67c122b70f927a274
raw rankings: 3c2376b93f9c7982c28e2d706d942da0e3390c27e1fa9d9e092c09639aa28487
```

注意：`run_manifest.json` 保存的是当时隔离 worktree 中的历史执行命令，worktree 已在合并后删除；真实 ignored 产物已经复制到上面的主项目路径。历史命令用于审计，不应原样重跑。重新运行时应使用 `MedicalGraphRAG/README.md` 和当前主项目绝对路径，并接受新运行会产生新的时间与 rankings SHA。

## 5. 关键 Git 历史

```text
3bfdb8f exp: bind final BM25 rankings evidence
49f655e fix: bind search inputs and rankings to reports
8b407ca exp: record audited BM25 baseline rerun
2c7bcbc fix: make Pyserini scripts standalone
57707ca fix: audit BM25 retrieval provenance chain
9c08e49 exp: record PubMedQA BM25 hard baseline
ec7f3cf fix: preserve sparse BM25 short rankings
c9ac8ee fix: read Pyserini version from package metadata
0c5354d fix: preserve Unicode separators in JSONL records
```

BM25 功能分支已经 fast-forward 合并到 `main` 并删除，隔离 worktree 也已清理。不要寻找 `codex/bm25-hard-baseline` 分支。

## 6. 接下来唯一要做的事情

不要立即改代码。先让学习者用自己的话回答：

1. 为什么 query `7482275`（necrotizing fasciitis / hyperbaric oxygenation）能让 BM25 把 gold 排到第 1？
2. 为什么 query `11570976`（`Is it Crohn's disease?`）会让 gold 掉出 Top-100？
3. 为什么 chunk ranking 不能直接与 document-level qrels 比较？
4. 为什么按 `doc_id` 折叠时使用最高 chunk BM25 score？
5. Recall@k、MRR@10、nDCG@10 分别衡量什么？
6. BM25 依赖什么信息，为什么它不是向量检索？

完成标准：学习者明确提到词项匹配、词频/文档频率、长度归一化、高区分度术语，并说明 BM25 不使用 embedding 或向量语义相似度；还要正确区分 chunk-level ranking 与 document-level qrels。

检查通过后：

1. 在 `STUDY_PROGRESS.md` 记录学习者的真实解释和纠正点；
2. 将“今日唯一任务”改为 Dense baseline 设计；
3. 先讨论设计和单变量边界，再动代码；
4. Dense 必须复用同一冻结数据、dev/test、qrels、document 折叠与评测接口；
5. 第一版只改变检索器，不同时引入 Hybrid、Reranker、生成模型或 LinearRAG。

## 7. 还没有做的事情

### 阶段 1 未完成

- [ ] 学习者完成 BM25 案例和指标解释门禁；
- [ ] 正式 Dense Retriever 设计、实现、建索引和真实指标；
- [ ] BM25 与 Dense 在相同数据/切分上的成功和失败案例对比；
- [ ] 正式 Hybrid Retrieval（RRF 或预先定义的单变量融合）；
- [ ] Reranker 基线；
- [ ] No-RAG、BM25、Dense、Hybrid、Hybrid + Reranker 的统一命令；
- [ ] 本地 Qwen 3B/4B 生成流程；
- [ ] QA Accuracy 与检索指标的分离评测；
- [ ] 平均延迟、索引时间、索引空间、显存峰值的完整总表；
- [ ] 阶段 1 总结和可写入简历的真实描述。

当前 Dense/FAISS、BM25 和 RRF 的旧结果只有 3 条 toy 文档，只能说明接口曾跑通，不得作为正式基线。

### 阶段 2 未完成

- [ ] 确认 LinearRAG 官方 medical 数据真实来源和字段契约；
- [ ] 在官方小数据上端到端构建索引并检索；
- [ ] 记录真实耗时、显存、实体数、边数、索引空间和指标；
- [ ] 实现统一 `LinearGraphRetriever.search(query, top_k)`；
- [ ] 把 `pubmedqa_hard_v1` 或后续明确医学子集迁移到图索引；
- [ ] 普通 NER 与 scispaCy/医学 NER 单变量对比；
- [ ] BM25、Dense、Hybrid、LinearRAG 同协议比较。

### 阶段 3～5 均未开始正式实验

- [ ] MedicalGPT LoRA/QLoRA SFT；
- [ ] Search-R1 3B 搜索强化学习；
- [ ] MedSearch-R1 医学搜索 Agent。

不要提前跳到这些阶段。

## 8. 督导与实验纪律

- 每次会话先读 `STUDY_PROGRESS.md` 和本交接文件；
- 每天只安排一个 30 分钟至 2 小时的核心任务；
- 用户说“跑通了”不算完成，必须检查日志、产物、配置和指标；
- 所有指标必须由真实脚本输出，保留随机种子、Git commit、命令和失败案例；
- 每次只改变一个主要变量；
- 检索命中不等于答案正确；
- 不编造训练时间、显存、QA Accuracy 或提升比例；
- 不将封闭 5,000 文档结果外推到全量 PubMed；
- 大规模下载、模型下载或新索引构建前先汇报方案；
- 新实验优先使用隔离分支/worktree，通过测试和审查后再合并 `main`。

## 9. 可直接复制给下一位 Agent 的开场指令

```text
先读取项目根目录的 AGENTS.md、STUDY_PROGRESS.md 和
HANDOFF_DEEPSEEK_V4_FLASH.md。当前 main 已完成正式 BM25 检索基线，
不要重跑或重构 BM25，也不要直接进入 LinearRAG 医学迁移。
今天唯一任务是检查我对 BM25 成功/失败案例、chunk→document 折叠、
Recall/MRR/nDCG 以及“BM25 不是向量检索”的解释。
检查通过后更新 STUDY_PROGRESS.md，再与我讨论 Dense baseline 的单变量设计。
任何新指标必须来自真实脚本和日志。
```
