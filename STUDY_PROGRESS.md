# 当前阶段

- **阶段 1：MedRAG / MedicalGraphRAG 基础检索**——已完成可运行的 BM25、Dense、Hybrid、Graph、Hybrid2（Reranker）检索框架及四套基准评测；生成式 QA 评测尚未开展。
- **阶段 2：LinearRAG 图结构检索迁移**——已完成官方主干代码阅读、默认参数对齐和医学/多跳检索实验；官方 medical 数据无法构造可靠 IR qrels，因此它只保留定性验证，`hotpotqa_v1` 是标准多跳 IR 补充基准。
- **阶段 3：MedicalGPT LoRA/QLoRA SFT**——已完成代码阅读和 100 条样本 LoRA smoke test；正式数据清洗、独立测试集评测及医学质量结论尚未完成，不能标记为阶段完成。
- **阶段 4：Search-R1 搜索强化学习**——仅开始只读预习，尚未安装 veRL/vLLM、准备 NQ/Wikipedia/e5 检索环境或启动 RL；不视为已进入正式训练。
- **阶段 5：MedSearch-R1**——未开始。

路线保持为：MedRAG → LinearRAG → MedicalGPT → Search-R1 → MedSearch-R1。阶段 3、4 的预习不会替代前序阶段的正式验收。

# 当前项目

**Medical GraphRAG + Agent**：以 `MedicalGraphRAG/` 为统一检索框架，复用 MedRAG 的医学语料与基线，吸收 LinearRAG 的图检索思想；MedicalGPT 提供领域 SFT，Search-R1 提供多轮搜索 RL，最终组合为 MedSearch-R1。

主文档和真实产物优先级：

- 检索框架、运行命令和完整结果：`MedicalGraphRAG/README.md`、`MedicalGraphRAG/experiments/`。
- 图边策略与代码审阅：`docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md`、`docs/superpowers/reviews/2026-08-08-per-dataset-edge-policy-code-review.md`。
- MedicalGPT smoke 的配置、日志和 adapter：`MedicalGPT/logs/sft/`、`MedicalGPT/outputs/sft/`、`MedicalGPT/experiments/sft/`。

# 本周目标

1. 完成 Search-R1 的数据协议与训练入口阅读，不启动未经缩放的官方 8 卡配置。
2. 回到 MedicalGPT：制定正式数据清洗、训练/验证/独立测试隔离和医学质量评测方案。
3. 对 `MedicalGraphRAG` 的 document-level v2 结果保持门禁：Dense、Hybrid、Graph 未生成同版本产物前，不把 v2 BM25 与旧实验横向比较。

# 今日唯一任务

**2026-08-11：同步 Search-R1 官方代码与第一版检索资产。**

只下载官方 `wiki-18` 语料及配套 E5 索引；不配置 Lead/veRL/vLLM，不运行 NQ 预处理、检索服务或 GRPO。

# 完成标准

- 本地 `Search-R1/` 与官方源码 HEAD 逐文件对齐（允许 CRLF/LF 差异）；
- 官方语料仓库与 E5 索引仓库已启动下载，目标路径明确且受 Git 忽略；
- 不产生环境安装、NQ parquet 或训练产物；
- 记录官方来源和下载状态，完成后再单独校验文件大小与哈希。

# 已完成

## MedicalGraphRAG：统一检索框架与已核验结果

- 已建立独立的 `MedicalGraphRAG/`，提供统一的 `cli run <retriever> --dataset <name>` 入口；支持 BM25、Dense、RRF Hybrid、Graph 与 Qwen3-Reranker Hybrid2。
- 数据、qrels、检索报告、评测、审计和哈希校验均落盘；检索层当前支持 PubMedQA、NFCorpus、SciFact、HotpotQA 的多相关 qrels。
- `pubmedqa_hard_v1` 是 **5,000 document 的封闭基准**，不代表全量 PubMed 检索能力。其旧版 abstract/chunk official test 中 Dense 最优：Recall@10 `0.994`、MRR@10 `0.977786`、nDCG@10 `0.981885`；RRF Hybrid 略低于 Dense，说明融合并非必然增益。
- Cross-encoder Hybrid2 在 NFCorpus（nDCG@10 `0.384`）、SciFact（Recall@10 `0.895`、MRR@10 `0.740`、nDCG@10 `0.772`）和 HotpotQA（`0.898`、`0.955`、`0.865`）胜出；HotpotQA 的 RRF Hybrid 也显著高于单路检索。
- BC5CDR Entity–Passage + PPR 图检索在四个已测基准均未胜出。当前证据指向：PubMedQA 短摘要和事实题中图传播是噪声；HotpotQA 等通用维基语料上医学 NER 实体稀疏。该负结果必须保留，不能以调参掩盖。
- 已对齐 LinearRAG 默认参数（`damping=0.5`、`passage_node_weight=0.05`、`passage_ratio=2`，并排除 `ORDINAL/CARDINAL`）；对齐后 Graph 的 Recall@10 为 PubMedQA `0.982`、SciFact `0.705`、HotpotQA `0.695`、NFCorpus `0.158`。
- 已完成接口重构（commit `5efb6e8`）和 README/简历版总结（commit `fe89a64`）；历史统计见各实验目录，不在本文件重复维护。

## Document-level v2 边策略

- 旧版完整摘要 embedding 存在“decode 后重新分词”窗口漂移：PubMedQA `6,037` 个窗口中 `205` 个不一致；旧 v1 结果不得作为最终 document-level 结论。
- v2 直接冻结 token-ID 窗口，Dense、Graph、Similarity 共用同一 embedding artifact；Similarity 由冻结向量重算，Adjacent 只连同文档连续 `order`，权重固定 `1.0`。
- v2 BM25-document 已完成：PubMedQA test Recall@10 `0.990`、MRR@10 `0.9619`、nDCG@10 `0.9689`。Dense、Hybrid、Graph 尚未在此版本生成产物，禁止横向宣称提升。
- 测试状态：Windows `46 passed`；WSL Docker `125 passed, 2 failed`，失败均为 spaCy `3.8` 与 BC5CDR `3.7` 模型不兼容，不是检索指标结果。

## MedRAG 与 LinearRAG 阅读边界

- MedRAG 已阅读生成入口、模板、BM25/Dense/RRF、数据处理和检索接口，并完成 toy 闭环；toy 结果只验证接口，不作为医学实验指标。
- LinearRAG 已阅读初始化、embedding 缓存、NER、离线 `index()`、图构建、在线 `retrieve()`、实体传播、passage 先验、PPR、`qa()` 与评测调用链；官方源码记录版本为 `bcc94e66c221f798801255efba09311d6fbcd8d6`。
- LinearRAG 内置 medical 的 evidence 是改写文本，和 chunk 的全量匹配率仅 `0.35%`，无法构造可信 qrels；因此不把其定性演示包装成定量复现。
- 已删除 `MedRAG/corpus/pubmed/`（约 `65.20 GiB`）与 `MedRAG/corpus/wikipedia/`（约 `42.54 GiB`），释放约 `107.74 GiB`。如需全量语料实验，必须重新下载并核验版本；`statpearls` 和 `textbooks` 仍保留。

## MedicalGPT SFT：已验证范围

- 已完成 `scripts/run_sft.sh → training/supervised_finetuning.py` 调用链阅读，能够解释 chat template、assistant-only loss mask、LoRA/QLoRA、gradient checkpointing、collator 和 Trainer 保存恢复；对源码的学习注释提交为 `48f6dfb`。
- 已下载 `Qwen/Qwen2.5-3B` Base 和 `shibing624/medical` 英文数据。固定种子 `20260809` 的 100 条审计样本结构完整；90/10 ShareGPT 拆分只用于 smoke test。
- 真实 smoke 配置：单卡 RTX 5090、BF16 LoRA、rank `8`、alpha `16`、batch `2`、gradient accumulation `4`、`model_max_length=1024`、1 epoch、12 steps、seed `20260809`。结果：train loss `2.332629`、eval loss `2.008307`、perplexity `7.450692`、runtime `13.9074s`；整卡显存最低/峰值 `2,132/17,113 MiB`，训练增量约 `14,981 MiB`。
- adapter 位于 `MedicalGPT/outputs/sft/qwen2.5-3b-medical-smoke-seed20260809-memprobe2/`；行为对比显示 3 条 validation 中 2 条输出变化。它只证明训练管线和 adapter 加载有效，**不证明医学能力提升**。

## Search-R1：已阅读范围

- 本地 `Search-R1/` 是总仓库引入的源码快照（commit `3d4832d`），不含官方独立 Git 历史；2026-08-11 已与官方 HEAD `598e61bd1d36895726d28a8d06b3a15bed19f5d3` 逐文件比对，内容一致（仅 CRLF/LF 换行差异）。
- 官方 `PeterJinGo/wiki-18-corpus` 与 `PeterJinGo/wiki-18-e5-index` 正在下载至本地临时目录；完成校验后放入 `Search-R1/data/wiki-18/`。该目录被 Git 忽略，语料和索引不会推送到 GitHub。
- README 主流程：NQ 处理为 parquet → 本地检索服务 → `<think>/<search>/<information>/<answer>` 交错 rollout → 规则化 QA EM reward → GRPO/PPO。
- 目前只阅读 `scripts/data_process/nq_search.py`。官方 `train_grpo.sh` 面向 8 GPU、batch `512`、5 rollouts、15 epochs、max_turns `2`，不能直接在单张 RTX 5090 上运行。

# 遇到的问题

| 问题 | 状态与处理 |
|---|---|
| PubMedQA document-level embedding 窗口漂移 | 已修复为 token-ID window v2；旧 v1 指标冻结为历史，不作结论。 |
| WSL 的 BC5CDR 测试失败 | 未解决；spaCy `3.8` 与模型 `3.7` 不兼容。Windows 针对性测试通过。 |
| Datasets 默认 cache 被历史 file lock 卡住 | 已定位为 `filelock.acquire()`；smoke 改用独立 `cache/sft_smoke_datasets`，不删除全局 cache。 |
| LinearRAG medical 无可靠 qrels | 已确认；只做定性验证，标准 IR 使用 HotpotQA 补充。 |
| Search-R1 官方配置超过单卡预算 | 未解决但预期正常；后续需设计 3B、小 batch、少 rollout 的单卡缩放配置。 |
| Hugging Face 直接 HTTPS 连接超时 | 改由官方 Git/LFS 仓库下载；当前后台传输中，完成前不运行任何处理脚本。 |

# 下一步

1. 等待并校验 `wiki-18` 语料和 E5 索引下载完成；不做解压、预处理或环境配置。
2. 回到阶段 3，先定义正式 MedicalGPT 数据清洗规则、训练/验证/独立测试隔离，以及可复现的医学质量评测；不得把 100 条 smoke 外推为正式实验。
3. 为 Search-R1 制定单卡 3B 最小配置：先验证 parquet、工具协议和检索环境，再启动小规模 rollout；不能复制官方 8 卡脚本。
4. 等 v2 Dense、Hybrid、Graph 产物和审计齐备后，再报告 document-level 检索对比。

# 待补知识

- QA Accuracy 与 Recall@k、MRR、nDCG 的边界，以及生成正确和检索命中的差异；
- scispaCy/医学 NER 与 BC5CDR 在医学、通用语料上的实体覆盖差异；
- Search-R1 rollout、retrieved-token mask、EM reward、GRPO 和检索服务之间的数据流；
- SFT 正式评测的格式遵循、医学问答质量和幻觉分析设计。

# 实验结果总表

| 数据集 / 版本 | 当前最优或关键结论 | 结论边界 |
|---|---|---|
| PubMedQA abstract/chunk official test | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 document 封闭基准；不是全 PubMed。 |
| PubMedQA document-level v2 | BM25：R@10 `0.990`、MRR `0.9619`、nDCG `0.9689` | Dense/Hybrid/Graph 尚未完成，不能比较。 |
| NFCorpus | Hybrid2 nDCG `0.384` | 多相关 qrels；R@10 的理论上限受相关文档数影响。 |
| SciFact | Hybrid2：R@10 `0.895`、MRR `0.740`、nDCG `0.772` | Graph 未胜出。 |
| HotpotQA | Hybrid2：R@10 `0.898`、MRR `0.955`、nDCG `0.865` | 每题 2 个 gold；Graph 未胜出。 |
| MedicalGPT 100 条 smoke | eval loss `2.008307`、PPL `7.450692` | 只验证管线，不能说明医学能力。 |

# 失败案例

| 场景 | 原因 | 当前结论 |
|---|---|---|
| PubMedQA 上 RRF Hybrid 低于 Dense | BM25 的高位错误结果稀释 Dense 信号 | 融合不是默认更好；保留负结果。 |
| 图检索在四套基准未胜出 | 任务/语料与实体传播信号不匹配，通用语料医学 NER 稀疏 | 不以无依据调参追逐指标；后续从实体识别和任务类型分析。 |
| TREC-COVID 未接入检索 | 24.6% 文档正文为空，部分 qrels 指向空文本 | 不用标题回退伪造连续文本检索实验。 |
| MedicalGPT `skip_memory_metrics=False` 训练停滞 | Trainer 路径异常，未形成完整产物 | 已终止且不计入；改用外部只读显存采样完成 smoke。 |
