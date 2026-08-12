# Search-R1 源码主干导读（中文学习版）

> 对应源码快照：官方 Search-R1 `598e61bd1d36895726d28a8d06b3a15bed19f5d3`。
> 本文只负责学习导航；源码中的中文注释不改变官方行为。

## 1. 先建立一张主干图

```text
train_grpo.sh（Hydra 参数覆盖）
  -> verl/trainer/main_ppo.py（组装 Ray workers、RewardManager、Trainer）
  -> verl/utils/dataset/rl_dataset.py（Parquet -> token batch + 题目元数据）
  -> verl/trainer/ppo/ray_trainer.py::fit（训练总循环）
       -> search_r1/llm_agent/generation.py::run_llm_loop
          模型生成 <search> / <answer>
          -> HTTP 检索器返回 <information>
          -> 继续生成，直到 answer 或达到轮数上限
       -> verl/utils/reward_score/qa_em.py（最终答案 EM reward）
       -> GRPO group advantage + KL/loss mask -> update_actor
```

理解 Search-R1 最重要的不是背 Ray/FSDP API，而是能沿这条链回答五个问题：

1. 一道题在 Parquet 里有哪些字段？
2. 模型如何用文本标签选择 Search 或 Answer？
3. 检索结果如何插回上下文，哪些 token 不参与 actor loss？
4. 最终答案怎样变成 0/1 reward？
5. 同一道题的多条 rollout 如何通过 GRPO 形成相对优势并更新 actor？

## 2. 文件优先级

### P0：第一轮必须精读

| 顺序 | 文件 | 只抓什么 |
|---:|---|---|
| 1 | `train_grpo.sh` | 实验参数入口；`n_agent`、`max_turns`、长度、KL、state masking 和 GPU 规模。 |
| 2 | `scripts/data_process/nq_search.py` | 原始 QA 如何变成 `prompt / reward_model / data_source / extra_info`。 |
| 3 | `verl/utils/dataset/rl_dataset.py` | Parquet 一行如何被 tokenizer 变成 `input_ids / attention_mask / position_ids`。 |
| 4 | `search_r1/llm_agent/generation.py` | Search-R1 最核心的多轮文本 Agent 状态机。 |
| 5 | `verl/utils/reward_score/qa_em.py` | `<answer>` 抽取、答案规范化和 0/1 EM reward。 |
| 6 | `verl/trainer/main_ppo.py` | 配置如何变成 workers、RewardManager 和 RayPPOTrainer。 |
| 7 | `verl/trainer/ppo/ray_trainer.py` 的 `fit()` | rollout -> reward -> advantage -> actor update 的训练主链。 |

### P1：P0 跑通后再读

| 文件 | 作用 |
|---|---|
| `search_r1/llm_agent/tensor_helper.py` | 多轮拼接时的 padding、attention mask、position id。 |
| `verl/trainer/ppo/core_algos.py` | GRPO advantage、KL penalty 和 PPO loss 的数学实现。 |
| `search_r1/search/retrieval_server.py` | BM25/Dense 检索器及 FastAPI `/retrieve` 接口。 |
| `verl/workers/fsdp_workers.py` | actor/rollout/reference worker 如何加载模型并执行 RPC。 |

### P2：当前先跳过

- `verl/third_party/vllm/`：多个 vLLM 版本的兼容层，不是 Search-R1 的算法创新。
- `verl/single_controller/`：Ray 分布式 worker 调度底座；知道“远程 worker group”即可。
- `verl/workers/megatron_*`、`verl/utils/megatron/`：大规模 Megatron 并行路径，单卡 Mini 第一版不使用。
- `example/multinode/`：32B/72B 多机脚本，不符合当前 RTX 5090 学习范围。
- `reward_model/` 与 critic 细节：官方 GRPO + 规则 EM 路径不启用 learned reward model，也不使用 critic。

## 3. 三个最容易误读的点

### 3.1 `max_turns=2` 后还有一次最终生成

`run_llm_loop()` 先最多执行两轮“允许搜索”的生成；如果轨迹仍未结束，再执行一次
`do_search=False` 的最终生成。因此它不是简单的“总共只生成两次”。最后一轮即使输出
`<search>` 也不会真的访问检索器。

### 3.2 information mask 屏蔽的是 loss，不是注意力

模型在下一轮必须读取 `<information>...</information>`，所以这些 token 的
`attention_mask` 仍为 1。`responses_with_info_mask` 把检索结果位置替换为 pad，随后形成
`info_mask/loss_mask`，使环境注入的文本不被当成“模型动作”计算 actor loss。

### 3.3 reward 解析依赖 prompt 中已有一个 answer 示例

`extract_solution()` 要求完整 `prompt + response` 中至少有两个 `<answer>...</answer>`，并取最后一个。
官方 prompt 内含格式示例，模型答案是第二个。迁移到 MedQA 时若删除该示例却不改解析器，所有奖励都会变成 0。

## 4. 外部包基础介绍（只讲本项目实际用法）

| 包 | 基础概念 | 在 Search-R1 中的用途 |
|---|---|---|
| PyTorch (`torch`) | tensor、自动求导和 GPU 计算框架。 | 保存 token batch/mask，计算 log-prob、reward、advantage 和 actor 梯度。 |
| Transformers | Hugging Face 模型与 tokenizer 接口。 | 加载 Qwen/Llama tokenizer，把 chat prompt 转为 token ids。 |
| vLLM | 面向高吞吐推理的 LLM engine，核心优化包括 continuous batching 与 KV cache 管理。 | rollout 阶段一次生成同题的多条候选轨迹；它不是负责反向传播的 trainer。 |
| Ray | Python 分布式任务与 actor 框架。 | 把 actor、rollout、reference policy 等 worker 放到 GPU 资源池，通过 RPC 调用。 |
| veRL | 火山引擎开源的 RLHF 训练框架。 | 提供 `DataProto`、PPO/GRPO trainer、FSDP workers 和 actor/rollout 混合引擎。 |
| Hydra + OmegaConf | 分层 YAML 配置和命令行覆盖系统。 | `train_grpo.sh` 中的 `a.b.c=value` 会覆盖 `ppo_trainer.yaml`，再传给 trainer。 |
| Pandas + PyArrow | 表格处理与 Parquet 序列化。 | 读取预处理后的训练/验证数据；Parquet 能保留嵌套 prompt 与 reward 字段。 |
| Requests | 同步 HTTP 客户端。 | rollout 进程把一批 query POST 到检索服务。当前代码没有 timeout/重试。 |
| FastAPI + Uvicorn | Python Web API 框架与 ASGI server。 | 暴露 `/retrieve`，把检索器与 RL rollout 进程解耦。 |
| FAISS | 高效向量近邻检索库。 | Dense 路径加载 embedding index 并做 top-k 搜索。BM25 路径不依赖向量。 |
| W&B (`wandb`) | 实验日志与曲线平台。 | 记录 reward、loss、KL、吞吐和验证指标；不参与训练算法。 |

## 5. 今天唯一任务（30～90 分钟）

只读 `generation.py` 的 `run_llm_loop()`、`execute_predictions()`、
`postprocess_predictions()`，画出一条题目从 prompt 到 search、information、answer 的状态变化。

完成标准：

1. 能解释 `active_mask` 为什么会逐轮变小；
2. 能解释允许搜索循环结束后为何还有一次 final rollout；
3. 能指出 `<information>` 是谁生成的、是否参与 actor loss；
4. 用自己的话回答：Search-R1 的动作空间为什么本质上是由文本正则表达式定义的？

在 `cmd.exe` 中查看关键位置：

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring"
rg -n "def run_llm_loop|def execute_predictions|def postprocess_predictions|def batch_search" Search-R1\search_r1\llm_agent\generation.py
```

本轮只做源码解释，不启动官方 `train_grpo.sh`。该脚本是 8 GPU 配置，不能作为单张 RTX 5090 的可运行承诺。
