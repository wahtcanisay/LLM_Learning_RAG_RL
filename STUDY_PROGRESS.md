# 当前阶段

**阶段 4：Search-R1 3B 搜索强化学习复现。**

阶段 1、阶段 2 已完成并冻结，阶段 3 已完成最小 SFT smoke。当前只学习和验证
Search-R1 原项目，不引入医学数据、领域 reward 或后续项目设计。

```text
NQ / HotpotQA Parquet
  → Search-R1 多轮 SEARCH / ANSWER rollout
  → 原项目 HTTP 检索服务
  → 开放域 QA Exact Match reward
  → outcome-based GRPO
  → actor policy update
```

当前学习范围固定为 Search-R1 官方数据协议、Agent loop、检索接口、规则 reward、
GRPO advantage 和 actor update；暂不讨论任何领域迁移。

# 当前项目

**Search-R1：通过强化学习学习多轮搜索与回答。**

当前代码学习位置：

- NQ 数据预处理：`Search-R1/scripts/data_process/nq_search.py`
- Parquet → batch：`Search-R1/verl/utils/dataset/rl_dataset.py`
- rollout 状态机：`Search-R1/search_r1/llm_agent/generation.py`
- reward 装配：`Search-R1/verl/trainer/main_ppo.py`
- EM reward：`Search-R1/verl/utils/reward_score/qa_em.py`
- reward → advantage 调度：`Search-R1/verl/trainer/ppo/ray_trainer.py`
- GRPO advantage / clipped loss：`Search-R1/verl/trainer/ppo/core_algos.py`
- actor 更新入口：`Search-R1/verl/workers/actor/dp_actor.py`
- 学习索引：`Search-R1/docs/source_code_learning_zh.md`

# 本周目标

1. 理解 Search-R1 的 rollout → reward → GRPO 主链，不安装重型依赖、不启动训练。
2. 理解原始 NQ 样本从 Parquet 字段、token batch 到规则 EM reward 的完整数据流。
3. 每次只验收一个代码链路；能复述后再进入实现。

# 今日唯一任务

**2026-08-18：验收 NQ Parquet → batch → GRPO uid 的数据流。**

本次只验收以下数据变化，不重复考察已经通过的 reward、advantage 和 policy loss：

1. `process_fn()` 的五个顶层字段及 prompt/gold 隔离；
2. `__getitem__()` 单样本 tensor 与 `collate_fn()` batch tensor 的 shape；
3. `extra_info.index → 顶层 index → DataProto non_tensor_batch → uid`；
4. `_read_files_and_tokenize()`、`filter_prompts=True`、`drop_last=True` 的真实行为。

验收已通过：已能区分 rollout 生成、uid 标签建立与 GRPO 分组统计的执行时序。
本次没有运行数据预处理、模型、检索服务或训练。

# 完成标准

- 能说明 gold 固定保存在 `reward_model.ground_truth.target`，且不能暴露在模型 prompt 中。
- 能区分 `__getitem__()` 的 `[max_prompt_length]` 与 collate 后的
  `[batch_size, max_prompt_length]`。
- 能用 `[7, 9]`、`n_agent=3` 推导出 interleave 后的
  `[7, 7, 7, 9, 9, 9]`，并按 uid 分成两个 GRPO group。
- 能区分 `index` 的数据集样本身份与 `uid` 的当前优化分组身份。
- 能解释 uid 不参与 rollout，只需在 `_balance_batch()` 和 `compute_advantage()` 前存在。
- 能准确说明 `drop_last=True` 只丢弃不足一个完整 batch 的尾部样本。

# 明日唯一任务（2026-08-19）

**只深入阅读 `Search-R1/verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit()`。**

进度纠正：此前只阅读了 `fit()` 中四个组件调用点，并分别学习了 rollout、reward、
advantage 和 actor loss；没有完整追踪一次 training step。因此启动脚本、Hydra 配置、
worker 初始化全部后移，明天只研究 `fit()` 如何串联这些组件。

阅读顺序：

```text
训练前验证 + GenerationManager
→ batch_dict 转 DataProto
→ n_agent 扩展同题轨迹
→ run_llm_loop
→ 重算 old_log_probs
→ uid + union + batch balance
→ ref_log_prob
→ rule reward
→ GRPO advantage
→ information loss mask
→ update_actor
→ validation / save / log / return
```

阅读时只走 `do_search=True + GRPO + use_critic=False + use_kl_loss=True` 的实际分支；
跳过普通单轮 rollout、GAE/critic、driver 侧 KL penalty 和所有被调函数内部。

详细的输入、输出、Tensor shape、字段消费者和注意点已更新到
`Search-R1/docs/source_code_learning_zh.md` 第 6 节。源码 `fit()` 的 docstring 与
关键 DataProto 变换点也已补充学习注释，未修改行为。

# 明日完成标准

- 能用 `B/G/N/Lp/Lr` 说明 batch size 和 tensor shape 怎样变化。
- 能逐阶段列出 `DataProto` 新增的字段。
- 能解释为什么多轮轨迹结束后必须重算 `old_log_probs`。
- 能区分 `old_log_probs` 与 `ref_log_prob` 的来源和用途。
- 能解释 `union()` 为什么是 metadata、完整轨迹与 old log-prob 汇合点。
- 能说明 reward 何时变成 advantage，advantage 何时被 actor 消费。
- 能说明 information token 如何参与 attention，却通过 `loss_mask` 排除出 policy loss。
- 能说明一个 training step 何时验证、保存、记录日志并退出。

检查问题：

> 如果删除 `batch = batch.union(final_gen_batch_output)`，后面的
> `RewardManager`、GRPO advantage 和 actor update 分别会缺少哪些数据？


# 已完成

## 2026-08-18：Search-R1 NQ 数据 → batch → uid 阅读验收

- 已读完 `nq_search.py`、`rl_dataset.py`、`ray_trainer.py`、`main_ppo.py`、`qa_em.py` 的指定代码段；已掌握的 reward/GRPO 重复段未重复阅读。
- 已能列出 Parquet 五个顶层字段，并解释 prompt 与 gold answer 隔离是字段协议与防止答案泄漏的共同要求。
- 已纠正 tensor shape：`__getitem__()` 返回单样本 `[max_prompt_length]`，`collate_fn()` 后为 `[batch_size, max_prompt_length]`，再按 `n_agent` 扩展 batch 维。
- 已理解 `n_agent` 在 rollout 前扩展轨迹，而 uid 不参与生成，只在后续 `compute_advantage()` 中提供分组标签。
- 已定位真正的分组代码：`compute_grpo_outcome_advantage()` 通过
  `id2score[index[i]].append(scores[i])` 按 uid 收集同题 rollout，并计算组内均值、标准差和 advantage。
- 已理解 `_read_files_and_tokenize()` 当前不 tokenize、`filter_prompts=True` 当前不生效，以及 `drop_last=True` 只丢弃不完整尾批次。
- 本次是源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-16：GRPO advantage → actor policy loss 阅读验收

- 已能解释 `old_log_prob` 来自 rollout 行为策略，`log_prob` 来自更新中的当前 actor；`ratio` 是逐 token 的新旧策略概率比，不是整条 trajectory 的单一偏好分数。
- 已理解正/负 advantage 分别提高/降低已采样 token 的概率；PPO clip 裁剪 surrogate objective 的有利更新方向，不是把模型真实概率硬锁在区间内。
- 已理解 information 是外部检索 observation：参与 attention 以指导后续生成，但通过 `loss_mask` 排除在 policy loss 之外。
- 已区分 entropy bonus、reference-policy KL 与 `ppo_kl` 监控量：entropy 鼓励分布保持探索性；reference KL 限制当前 actor 偏离冻结的 SFT/reference 模型；`ppo_kl` 监控 current actor 与 rollout old actor 的变化。
- 已通过最小 entropy 判断：分布越均匀 entropy 越高，越集中 entropy 越低。该项是 veRL PPO actor 原有的可选正则，官方快照默认 `entropy_coeff=0.001`，不是本项目新增算法。
- 本次只完成代码阅读和概念验收，没有运行模型或训练，没有新增实验指标。

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
| 当前环境缺少 veRL 重型依赖 | 现阶段使用 AST、compileall 和纯协议样例验证；进入训练闸门后再建立隔离环境。 |
| 单卡 32GB 的 GRPO 资源边界未知 | 不做纸面外推；用真实日志记录 batch、rollout、长度、显存和耗时。 |
| 自有 search engine 尚未接入 Search-R1 | 当前只冻结最小请求/响应协议；进入推理 smoke 前再实现适配层。 |

# 下一步

当前已读完 `fit()` 调用的各个核心组件，但尚未读完 `fit()` 本身的调度逻辑。
下一次唯一任务是：**沿一次 Search-R1 GRPO training step 完整阅读
`RayPPOTrainer.fit()`。**

只读一个函数，不展开其调用对象：

1. 训练前验证和 generation manager；
2. DataProto 构造、`n_agent` repeat 与 prompt/metadata 分离；
3. 多轮 rollout、`old_log_probs` 重算与 `union`；
4. reference、reward 和 GRPO advantage 的执行顺序；
5. information loss mask 与 actor update；
6. 验证、保存、日志和退出条件。

后续闸门顺序：

1. 验收 `fit()` 的完整单步数据流与字段增长；
2. 阅读 `RayPPOTrainer._validate()`，比较训练和验证 rollout；
3. 阅读训练启动、Hydra 配置合并和 worker 初始化；
4. 阅读 `infer.py`，区分训练、验证和交互推理；
5. 为自己的 search engine 做最小协议适配；
6. 运行一条无训练 Search-R1 推理；
7. 根据 RTX 5090 真实日志设计最小 GRPO smoke。


# 待补知识

- 答案解析失败、无效搜索和 reward hacking 的处理方式。
- 自有 search engine 的 Search-R1 协议适配、空结果和服务不可用处理。
- 单卡 RTX 5090 上 veRL/vLLM + 3B LoRA GRPO 的真实资源边界。

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
