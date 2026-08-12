# 当前阶段

**当前主线：MedSearch-R1 Mini 后训练项目。**

- **阶段 1：MedRAG / MedicalGraphRAG 基础检索**——已完成 BM25、Dense、Hybrid、Graph、Hybrid2（Reranker）统一框架及四套真实基准评测；它作为当前 RL 项目的检索基础和第一项简历项目保留，不再继续扩展未验收的图检索方案。
- **阶段 2：LinearRAG 图结构检索迁移**——已完成代码阅读、默认参数对齐、医学/多跳实验和负结果分析；图检索未胜出，这一结论保留，但不阻塞后训练主线。
- **阶段 3～5：Medical SFT → Search-R1 → MedSearch-R1**——合并为当前唯一开发主线。MedicalGPT 的 100 条 LoRA smoke 已证明训练链路可运行，但正式 SFT、MedQA 数据隔离、医疗 Search 环境、GRPO 和独立评测均尚未完成。

2026-08-11 路线调整后，不再把“依次学习三个仓库”当作三个割裂项目，而是交付一个完整闭环：

```text
Qwen2.5-3B Base
  → Medical LoRA SFT
  → MedQA + Medical Textbooks BM25
  → Search / Answer 多轮 rollout（最多 2 次检索）
  → outcome-based GRPO
  → Medical-SFT / Fixed RAG / MedSearch-R1 对照评测
```

3B 是第一版闭环模型；7B 或后续 9B 只在 3B 的数据、训练、评测和资源记录全部跑通后作为规模扩展，不提前并行开线。

# 当前项目

**算法主项目：MedSearch-R1 Mini——面向医疗问答的多轮搜索强化学习。**

项目只训练一个医疗 Search SubAgent：给定 MedQA 问题，自主选择 `SEARCH` 或 `ANSWER`，生成检索 query，最多检索两轮，并以可程序化验证的最终选项正确性作为第一版 reward。第一版固定使用 Qwen2.5-3B、LoRA、Medical Textbooks 和 CPU BM25；暂不加入多工具、Tool-Calling SFT、复杂奖励、LLM Judge 或新 RL 算法。

项目分工与投递顺序：

1. 先完成 MedSearch-R1 Mini 的 SFT → RL → Evaluation 闭环；
2. 用已完成的 Medical GraphRAG 真实基线和 MedSearch-R1 后训练结果整理第一版简历并开始投递；
3. 投递期间继续完善 `D:/code_list/some tricks/Agent/项目书.md` 中的 PiAgent 二次开发与 Medical Harness；
4. RL 模型以后作为 `medical_search_agent(question)` 接入 Pi Main Agent。Pi 是产品展示和编排层，MedSearch-R1 是当前算法主体；
5. GitHub 高质量 Agent 项目与论文学习作为持续辅助任务，只选择能直接补强当前阶段的问题，不另起会打断主线的新项目。

主文档和真实产物优先级：

- 当前后训练范围原文：`C:/Users/dell/Downloads/MEDSEARCH_R1_MINI_PLAN.md`（仓库外，待归档到 `docs/`）。
- 检索框架、运行命令和完整结果：`MedicalGraphRAG/README.md`、`MedicalGraphRAG/experiments/`。
- 当前稳定代码基线：`MedicalGraphRAG/` 固定为 README 结果对应的 commit `fe89a64`。
- MedicalGPT smoke 的配置、日志和 adapter：`MedicalGPT/logs/sft/`、`MedicalGPT/outputs/sft/`、`MedicalGPT/experiments/sft/`。
- 后续 Agent 产品层设计：`D:/code_list/some tricks/Agent/项目书.md`。

# 本周目标

1. 冻结 MedSearch-R1 Mini 第一版范围，并把方案逐项映射到 Search-R1/veRL 的真实代码入口。
2. 定义 MedQA 的训练、验证、独立测试隔离及近重复检查，明确 Medical SFT 数据与 RL/测试题的隔离规则。
3. 保持 `MedicalGraphRAG` 在 README 已验证基线；第一版 Search 环境只复用 Medical Textbooks + CPU BM25，不同时改 Retriever。

# 今日唯一任务

**2026-08-12：完成 Search-R1 第一批主干源码的学习型中文注释与阅读地图。**

只覆盖数据预处理、Parquet 数据协议、多轮 Search rollout、检索服务、EM reward、训练装配和 GRPO 主循环；给出 P0/P1/P2 阅读优先级与外部包基础介绍。本任务不安装 veRL/vLLM、不启动检索服务或训练。

# 完成标准

- `Search-R1/docs/source_code_learning_zh.md` 能从脚本入口追到 actor update；
- 8 个核心源码文件具有模块、函数、关键字段、状态机和外部依赖注释；
- Python 编译、Bash 语法、comment-only AST/命令对比和最小协议行为检查通过；
- 注释分支已提交并推送，且不把 `.claude/` 纳入提交；
- 不声称 Search-R1 环境、检索服务或 GRPO 已运行。

# 已完成

## 2026-08-12：Search-R1 第一批主干源码注释

- 建立 `Search-R1/docs/source_code_learning_zh.md`，按 P0/P1/P2 区分算法主干、后续必读和当前可跳过的 veRL/vLLM/Megatron 框架底座。
- 注释范围：`nq_search.py`、`rl_dataset.py`、`generation.py`、`tensor_helper.py`、`retrieval_server.py`、`qa_em.py`、`main_ppo.py`、`ray_trainer.py` 的 Search-R1 主链，以及官方 `train_grpo.sh` 参数入口。
- 重点确认：`max_turns=2` 后仍可能有一次禁止真实搜索的 final rollout；information token 可被 attention 读取但通过 state masking 不参与 actor loss；NQ prompt 的 answer 示例与模型答案共同满足 reward parser 的“双 answer 标签”前提。
- 新增 `Search-R1/scripts/verify_source_notes.py`，以 commit `3d4832d` 的官方快照为固定基线，验证去除 docstring 后 Python AST 与去除注释后 Bash 命令保持不变，并检查 3 个 reward 与 3 个 action parser 案例。
- 本轮只验证注释与纯协议逻辑；当前 Python 环境缺少 `tensordict`，未安装 Search-R1 重型依赖，也未运行检索服务或 GRPO。
- 运行命令：`python -m compileall -q ...`、`"C:\\Program Files\\Git\\bin\\bash.exe" -n Search-R1/train_grpo.sh`、`python Search-R1/scripts/verify_source_notes.py`、`git diff --check`。
- 结果与指标：无训练/评测指标；comment-only 对比为 8 个 Python + 1 个 Bash 文件，协议行为检查 6/6 通过（以最终验证输出为准）。
- 显存峰值：未运行模型，不适用。
- 代码或日志位置：`Search-R1/docs/source_code_learning_zh.md`、`Search-R1/scripts/verify_source_notes.py`。

## 2026-08-11：MedSearch-R1 Mini 路线收束

- 今天完成的是**方案决策与范围冻结**，不是 RL 实验：确定以小模型领域后训练为当前主线，先做 Qwen2.5-3B 完整闭环，7B/9B 作为后续可选规模实验。
- 第一版动作空间只保留 `SEARCH` 与 `ANSWER`，沿用 `<search>`、`<information>`、`<answer>` 协议，最大检索次数为 2。
- 第一版数据与环境暂定为 MedQA + Medical Textbooks + CPU BM25；Medical SFT 不承担 Tool-Calling 教学，搜索行为由后续 GRPO 学习。
- 第一版 reward 只使用可解析的最终选项正确性；非法或不可解析答案记 0。暂不加入 LLM Judge、过程奖励、搜索成本奖励或 query 质量奖励。
- 第一版核心对照固定为 Medical-SFT、Medical-SFT + Fixed RAG、MedSearch-R1；核心研究问题是 adaptive search policy 是否在独立测试集上优于固定检索。
- 明确 PiAgent/Medical Harness 不与 RL 同时开工：RL 闭环完成后，将它封装为 `medical_search_agent(question)` 接入 Pi Main Agent；`D:/code_list/some tricks/Agent/项目书.md` 保留为后续 Agent 二次开发设计依据。
- 求职节奏调整为：先用已有 RAG 结果与完成后的后训练项目形成简历并投递，再在投递过程中完善 Agent 项目、框架二开以及高质量开源 Agent/论文学习。
- 方案原文：`C:/Users/dell/Downloads/MEDSEARCH_R1_MINI_PLAN.md`。该文件当前在仓库外，后续进入实现前应复制或整理到项目 `docs/`，以便纳入 Git 版本管理。

## MedicalGraphRAG：统一检索框架与已核验结果

- 已建立独立的 `MedicalGraphRAG/`，提供统一的 `cli run <retriever> --dataset <name>` 入口；支持 BM25、Dense、RRF Hybrid、Graph 与 Qwen3-Reranker Hybrid2。
- 数据、qrels、检索报告、评测、审计和哈希校验均落盘；检索层当前支持 PubMedQA、NFCorpus、SciFact、HotpotQA 的多相关 qrels。
- `pubmedqa_hard_v1` 是 **5,000 document 的封闭基准**，不代表全量 PubMed 检索能力。其旧版 abstract/chunk official test 中 Dense 最优：Recall@10 `0.994`、MRR@10 `0.977786`、nDCG@10 `0.981885`；RRF Hybrid 略低于 Dense，说明融合并非必然增益。
- Cross-encoder Hybrid2 在 NFCorpus（nDCG@10 `0.384`）、SciFact（Recall@10 `0.895`、MRR@10 `0.740`、nDCG@10 `0.772`）和 HotpotQA（`0.898`、`0.955`、`0.865`）胜出；HotpotQA 的 RRF Hybrid 也显著高于单路检索。
- BC5CDR Entity–Passage + PPR 图检索在四个已测基准均未胜出。当前证据指向：PubMedQA 短摘要和事实题中图传播是噪声；HotpotQA 等通用维基语料上医学 NER 实体稀疏。该负结果必须保留，不能以调参掩盖。
- 已对齐 LinearRAG 默认参数（`damping=0.5`、`passage_node_weight=0.05`、`passage_ratio=2`，并排除 `ORDINAL/CARDINAL`）；对齐后 Graph 的 Recall@10 为 PubMedQA `0.982`、SciFact `0.705`、HotpotQA `0.695`、NFCorpus `0.158`。
- 已完成接口重构（commit `5efb6e8`）和 README/简历版总结（commit `fe89a64`）；历史统计见各实验目录，不在本文件重复维护。

## MedicalGraphRAG 版本回退（2026-08-11）

- 后续 document-level v1/v2、MedRAG adapter、边策略与 reranker 扩展未达到预期，当前代码已恢复到 README 实验数据对应的 commit `fe89a64`。
- 被撤回内容仍保留在 Git 历史中；除非重新设计并独立验收，不再作为当前实现或实验结论。

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
| 后续 document-level 扩展不满意 | 已回退到 README 结果对应的 `fe89a64`；相关代码与结果只保留在 Git 历史。 |
| WSL 的 BC5CDR 测试失败 | 未解决；spaCy `3.8` 与模型 `3.7` 不兼容。Windows 针对性测试通过。 |
| Datasets 默认 cache 被历史 file lock 卡住 | 已定位为 `filelock.acquire()`；smoke 改用独立 `cache/sft_smoke_datasets`，不删除全局 cache。 |
| LinearRAG medical 无可靠 qrels | 已确认；只做定性验证，标准 IR 使用 HotpotQA 补充。 |
| Search-R1 官方配置超过单卡预算 | 未解决但预期正常；后续需设计 3B、小 batch、少 rollout 的单卡缩放配置。 |
| Hugging Face 直接 HTTPS 连接超时 | 改由官方 Git/LFS 仓库下载；当前后台传输中，完成前不运行任何处理脚本。 |
| RL mini plan 位于仓库外 | 已读取并用于本次路线更新，但尚未受 Git 管理；实现前需归档到项目 `docs/`。 |
| Wiki-18 + E5 与当前 Mini 环境不一致 | 新方案第一版改为 Medical Textbooks + CPU BM25；现有下载不再是当前训练前置条件，也不能因为已下载就强行纳入第一版。 |

# 下一步

**下一次唯一核心任务：你亲自阅读 Search-R1 多轮状态机并解释主干。**

只读 `generation.py` 的 `run_llm_loop()`、`execute_predictions()` 和 `postprocess_predictions()`。完成标准：能解释 `active_mask`、final rollout、`<information>` 的来源与 loss mask，并说明文本正则如何定义 Search/Answer 动作空间；本任务不安装环境、不启动训练。

后续严格按以下闸门推进，每次只解锁一个：

1. 数据隔离与代码入口映射；
2. Medical SFT 正式数据、训练和独立评测；
3. MedQA + Medical Textbooks BM25 的无 RL Search 环境；
4. 0/1/2-search rollout 与答案解析、reward 单元验证；
5. 单卡 3B 最小 GRPO；
6. Medical-SFT / Fixed RAG / MedSearch-R1 独立测试集对照；
7. 完成简历与第一轮投递；
8. 再评估 7B/9B、PiAgent 接入及额外 Retriever/Reward 实验。

# 待补知识

- QA Accuracy 与 Recall@k、MRR、nDCG 的边界，以及生成正确和检索命中的差异；
- scispaCy/医学 NER 与 BC5CDR 在医学、通用语料上的实体覆盖差异；
- Search-R1 rollout、retrieved-token mask、EM reward、GRPO 和检索服务之间的数据流；
- SFT 正式评测的格式遵循、医学问答质量和幻觉分析设计。
- MedQA 许可、版本、选项标签规范，以及与 Medical SFT 数据的题干/答案/语义近重复检测；
- 单卡 RTX 5090 上 veRL/vLLM 的 3B LoRA GRPO 资源边界，必须由真实 smoke 日志确认，不能从 7B/9B 或官方 8 卡配置外推；
- 仅使用 outcome reward 时，零方差 group、答案解析失败、无效搜索和 reward hacking 的处理方式。

# 实验结果总表

| 数据集 / 版本 | 当前最优或关键结论 | 结论边界 |
|---|---|---|
| PubMedQA abstract/chunk official test | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 document 封闭基准；不是全 PubMed。 |
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
