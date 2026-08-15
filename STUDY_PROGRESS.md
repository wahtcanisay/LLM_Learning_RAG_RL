# 当前阶段

**阶段 3～5 合并主线：MedSearch-R1 Mini（Medical SFT → Search-R1 → 医学搜索强化学习）。**

阶段 1 的 Medical GraphRAG 基线和阶段 2 的 LinearRAG 迁移已经完成并冻结。当前不再扩展检索器，优先完成 Qwen2.5-3B 的 Medical SFT、医学搜索环境、最小 GRPO 和独立评测闭环。

```text
Qwen2.5-3B Base
  → Medical LoRA SFT
  → MedQA + Medical Textbooks BM25
  → SEARCH / ANSWER 多轮 rollout（最多 2 次检索）
  → outcome-based GRPO
  → Medical-SFT / Fixed RAG / MedSearch-R1 对照评测
```

第一版固定范围：Qwen2.5-3B、LoRA、CPU BM25、`SEARCH`/`ANSWER` 两种动作、最多 2 次检索、可程序验证的最终答案 reward。暂不加入多工具、复杂过程奖励、LLM Judge、7B/9B 或新的 Retriever。

# 当前项目

**MedSearch-R1 Mini：面向医学问答的多轮搜索强化学习。**

当前代码学习位置：

- rollout 状态机：`Search-R1/search_r1/llm_agent/generation.py`
- reward 装配：`Search-R1/verl/trainer/main_ppo.py`
- EM reward：`Search-R1/verl/utils/reward_score/qa_em.py`
- reward → advantage 调度：`Search-R1/verl/trainer/ppo/ray_trainer.py`
- GRPO advantage / clipped loss：`Search-R1/verl/trainer/ppo/core_algos.py`
- actor 更新入口：`Search-R1/verl/workers/actor/dp_actor.py`
- 学习索引：`Search-R1/docs/source_code_learning_zh.md`

# 本周目标

1. 理解 Search-R1 的 rollout → reward → GRPO 主链，不安装重型依赖、不启动训练。
2. 明确 MedQA 数据隔离、Medical Textbooks BM25 接口和医学选择题答案解析的迁移点。
3. 每次只验收一个代码链路；能复述后再进入实现。

# 今日唯一任务

**2026-08-14：阅读 GRPO advantage 如何变成 actor policy loss。**

只读以下函数：

1. `ray_trainer.py::compute_advantage()` 的 GRPO 分支；
2. `core_algos.py::compute_grpo_outcome_advantage()`；
3. `dp_actor.py::DataParallelPPOActor.update_policy()` 中准备 `response_mask`、`old_log_prob`、`advantages` 的部分；
4. `core_algos.py::compute_policy_loss()`；
5. 回到 `dp_actor.py::update_policy()` 看 entropy、KL、backward 和 optimizer step。

`core_algos.py` 的两个核心函数已补充 `输入：`、`输出：`、`调用方式：` 注释。本任务不进入 Ray/FSDP/vLLM 底层，不启动训练。

# 完成标准

- 能用 `[1, 1, 0, 0]` 手算组内均值、标准差、正负 advantage，并解释 `[1, 1, 1, 1]` 为什么是零方差组。
- 能解释序列 advantage 为什么广播到每个有效 response token，但 information/padding token 不参与 policy loss。
- 能解释 `old_log_prob`、`log_prob` 和 `ratio = exp(log_prob - old_log_prob)` 分别代表什么。
- 能解释正/负 advantage 如何改变 token 概率，以及 clip 为什么限制一次更新幅度。
- 能区分 `pg_loss`、entropy bonus、reference-policy KL loss 和仅用于监控的 `ppo_kl`。
- 不把静态注释验证误报为模型、检索或训练已运行。

# 已完成

## 2026-08-14：Search-R1 reward → GRPO advantage 阅读验收

- 已能解释序列级 0/1 outcome reward 写在最后一个有效 response token，是因为结果只能在完整轨迹结束后判断，同时需适配 veRL 的 token-level reward 接口并避开 padding。
- 已区分 PPO/GAE 与 GRPO：GAE 使用 token value 从后向前传播 advantage；GRPO 不使用 value critic，而是对同题多条 rollout 的序列分数做组内标准化，再广播到有效 response token。
- 已正确判断全 1 或全 0 rollout group 的 advantage 为 0；进一步明确这表示该组没有策略梯度信号，是有效样本比例问题，不等于整个 GRPO 训练必然坍塌。
- 本次为源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-13：Search-R1 reward 学习注释准备

- 已为 5 个 reward 主链函数补充输入类型、输出类型、Tensor shape、上游调用位置和下游数据流。
- 验证命令：

```cmd
python Search-R1\scripts\verify_source_notes.py
python -m compileall -q Search-R1\verl\trainer\main_ppo.py Search-R1\verl\utils\reward_score\qa_em.py Search-R1\verl\trainer\ppo\ray_trainer.py
git diff --check
```

- 验证结果：8 个 Python 文件和 1 个 Bash 文件保持 comment-only；3 个 reward + 3 个 action parser 样例通过；Python 编译通过。
- 边界：注释未改变执行逻辑；reward 代码复述已于 2026-08-14 验收，但没有运行训练或产生训练指标。

## 2026-08-13：Search-R1 多轮 rollout 主干阅读

- 已理解 `active_mask` 控制仅对未结束样本生成；search 注入 `<information>` 后继续，answer 标记结束，非法动作按当前 parser 路径处理。
- 已区分 `rollings`（下一轮输入）与 `original_right_side`（最终训练轨迹）。
- 已理解 `max_turns` 后仍有一次 `do_search=False` 的 final rollout，以及 `attention_mask` 与用于排除环境 information token 的 `info_mask`。
- 本次只做源码阅读和协议检查，未运行模型、检索服务或训练。

## 2026-08-12：Search-R1 主干源码学习注释

- 已注释数据处理、dataset、generation、tensor helper、retrieval server、reward、训练入口和 trainer 主链。
- 已建立 `Search-R1/docs/source_code_learning_zh.md` 和标准库验证脚本 `Search-R1/scripts/verify_source_notes.py`。
- 官方 8 GPU 配置不能直接用于单张 RTX 5090；单卡配置必须以后通过真实 smoke 确认。

## 2026-08-11：MedSearch-R1 Mini 范围冻结

- 第一版动作空间、模型、检索后端、最大搜索轮数和 outcome reward 已冻结。
- 核心对照固定为 Medical-SFT、Medical-SFT + Fixed RAG、MedSearch-R1。
- PiAgent 是闭环完成后的产品接入层，不与当前 RL 主线并行开发。

## MedicalGPT SFT：已验证范围

- 已跑通 Qwen2.5-3B、BF16 LoRA 的 100 条 smoke：rank `8`、alpha `16`、batch `2`、gradient accumulation `4`、max length `1024`、1 epoch、12 steps、seed `20260809`。
- 真实结果：train loss `2.332629`、eval loss `2.008307`、perplexity `7.450692`、runtime `13.9074s`、整卡显存峰值 `17,113 MiB`。
- 结论边界：只证明训练链和 adapter 加载有效；3 条 validation 中 2 条输出变化，不代表医学能力提升。

## Medical GraphRAG / LinearRAG：冻结结论

- 已实现 BM25、Dense、RRF Hybrid、Graph、Qwen3-Reranker Hybrid2 的统一检索入口和四套真实基准评测。
- `MedicalGraphRAG/` 固定在 README 结果对应的 commit `fe89a64`；未验收扩展仅保留在 Git 历史。
- 图检索在四套基准均未胜出。该负结果保留，不再通过无依据调参追逐指标。

# 遇到的问题

| 问题 | 当前处理 |
|---|---|
| Search-R1 官方配置面向 8 GPU | 当前只读代码；以后单独设计 3B、小 batch、少 rollout 的单卡 smoke。 |
| MedQA 与 Medical SFT 可能数据泄漏 | 正式训练前做题干、答案和语义近重复检查，并隔离 train/validation/test。 |
| 开放域 `qa_em.py` 不适合直接评分 MedQA | 迁移时实现显式 A/B/C/D 解析并用单元样例验证，不沿用英文 alias EM 假设。 |
| 当前环境缺少 veRL 重型依赖 | 现阶段使用 AST、compileall 和纯协议样例验证；进入训练闸门后再建立隔离环境。 |
| 单卡 32GB 的 GRPO 资源边界未知 | 不做纸面外推；用真实日志记录 batch、rollout、长度、显存和耗时。 |

# 下一步

GRPO advantage → actor policy loss 阅读验收通过后，唯一下一任务是：**设计 MedQA 数据隔离与医学选择题 reward 的最小协议测试**。

后续闸门顺序：

1. MedQA 数据隔离和重复检查；
2. Medical SFT 正式训练与独立评测；
3. Medical Textbooks + CPU BM25 的无 RL Search 环境；
4. 0/1/2-search rollout、答案解析和 reward 单元验证；
5. 单卡 Qwen2.5-3B 最小 GRPO；
6. Medical-SFT / Fixed RAG / MedSearch-R1 独立测试集对照；
7. 基于真实结果整理简历，再评估规模扩展和 PiAgent 接入。

# 待补知识

- `token_level_scores`、KL 后的 `token_level_rewards` 与 GRPO `advantages` 的数据流。
- 零方差 rollout group、答案解析失败、无效搜索和 reward hacking 的处理方式。
- MedQA 许可、版本、选项标签规范及跨数据集近重复检测。
- 单卡 RTX 5090 上 veRL/vLLM + 3B LoRA GRPO 的真实资源边界。
- SFT 正式评测的格式遵循、医学问答质量和幻觉分析设计。

# 实验结果总表

| 数据集 / 实验 | 当前最优或关键结果 | 结论边界 |
|---|---|---|
| PubMedQA 封闭基准 | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 documents，不代表全量 PubMed。 |
| NFCorpus | Hybrid2 nDCG@10 `0.384` | 多相关 qrels；Graph 未胜出。 |
| SciFact | Hybrid2：R@10 `0.895`、MRR `0.740`、nDCG `0.772` | Graph 未胜出。 |
| HotpotQA | Hybrid2：R@10 `0.898`、MRR `0.955`、nDCG `0.865` | 每题 2 个 gold；Graph 未胜出。 |
| MedicalGPT 100 条 smoke | eval loss `2.008307`、PPL `7.450692` | 只验证训练管线。 |

# 失败案例

| 场景 | 原因 | 保留结论 |
|---|---|---|
| PubMedQA 上 RRF Hybrid 低于 Dense | BM25 高位错误稀释 Dense 信号 | 融合不是默认更好。 |
| Graph 在四套基准均未胜出 | 实体传播信号与任务/语料不匹配 | 保留负结果，不包装为提升。 |
| TREC-COVID 未接入 | 24.6% 文档正文为空，部分 qrels 指向空文本 | 不用标题回退伪造正文检索。 |
| MedicalGPT 内部显存统计路径停滞 | Trainer 路径未形成完整产物 | 该次运行不计入；已用外部只读采样完成 smoke。 |

> 更早的逐日过程、已撤回方案和详细实验配置由 Git 历史、各子项目 README 与 `experiments/` 保留；本文件只维护当前有效状态。
