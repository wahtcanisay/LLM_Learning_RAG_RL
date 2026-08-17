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
当前官方 prompt 内含第一个格式示例，模型最终答案是第二个；两处代码共同构成当前解析约定。

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

## 5. 数据到 reward 的逐函数阅读路线

整条链拆成两个检查点。不要一次通读五个大文件，也不要从头阅读 `fit()`；只看表中列出的函数或代码片段。

### 检查点 A：一行 Parquet 怎样变成一个 batch（今天唯一任务，45～75 分钟）

| 顺序 | 文件与函数 | 简单介绍 | 阅读时注意 |
|---:|---|---|---|
| 1 | `scripts/data_process/nq_search.py::make_prefix()` | 把题目拼进 Search-R1 的标签协议，返回一段 user prompt 字符串。 | 此处只拼文本，不做 tokenization；prompt 自己已经含有一个 `<answer> Beijing </answer>` 格式示例。 |
| 2 | 同文件 `make_map_fn(split)` | 创建并返回一个逐样本处理函数，同时用闭包记住当前是 train 还是 test。 | 真正处理样本的是内部 `process_fn()`；`split` 不是每次处理时重新传入。 |
| 3 | 同文件 `process_fn(example, idx)` | 把原始 NQ 行变成五个顶层字段，gold aliases 写进 `reward_model.ground_truth['target']`。 | `idx` 由 `Dataset.map(with_indices=True)` 传入；`extra_info.index` 后面会成为同题 rollout 的分组键。 |
| 4 | `verl/utils/dataset/rl_dataset.py::RLHFDataset.__init__()` | 保存 tokenizer/长度配置，定位 Parquet，并把文件读成 DataFrame。 | 构造阶段还没有逐条生成 token tensor；真正的逐条 tokenization 在 `__getitem__()`。 |
| 5 | 同文件 `_read_files_and_tokenize()` | 用 Pandas 读取并拼接所有 Parquet。 | 函数名容易误导：当前实现没有 tokenization，而且长度过滤代码已被注释。 |
| 6 | 同文件 `__getitem__(item)` | 取 DataFrame 一行，应用 chat template，并生成定长 `input_ids/attention_mask/position_ids`。 | prompt 使用左 padding；原始 `prompt` 列被弹出，其他 reward 元数据不参与 tokenization、仍保留在返回字典。 |
| 7 | 同文件 `collate_fn(data_list)` | 把多条 `__getitem__()` 结果合成一个 batch。 | tensor 用 `torch.stack` 增加 batch 维；`reward_model` 等 Python 对象变成 `np.ndarray(dtype=object)`，不是 tensor。 |

`_download()` 和 `__len__()` 本轮只需知道用途：前者把 HDFS/本地路径统一为本地路径，后者返回 DataFrame 行数；它们不改变样本字段，不需要精读。

检查点 A 完成标准：

1. 手写出 `process_fn()` 返回的五个顶层字段及类型；
2. 解释为什么 `reward_model` 不会经过 tokenizer；
3. 解释单样本 `[max_prompt_length]` 如何在 `collate_fn()` 后变成
   `[batch_size, max_prompt_length]`；
4. 能画出 `extra_info.index → __getitem__() 返回的顶层 index`，先停在这里。

### 检查点 B：batch 元数据怎样变成 0/1 reward（A 验收后再读）

| 顺序 | 文件与函数/片段 | 简单介绍 | 阅读时注意 |
|---:|---|---|---|
| 1 | `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer._create_dataloader()` | 用 `RLHFDataset + collate_fn` 创建 train/val DataLoader。 | train 和 val 都设置 `drop_last=True`，不足一个 batch 的尾部样本会被丢弃。 |
| 2 | 同文件 `fit()` 中 `DataProto.from_single_dict()`、`batch.repeat()`、`uid = index.copy()` 三处 | 把普通 batch 转成 veRL `DataProto`，为每题复制多条 rollout，并把题目 index 作为 GRPO uid。 | 不要通读整个 `fit()`；只追踪 metadata，`uid` 必须让同题的多条 rollout 保持相同。 |
| 3 | 同文件 `fit()` 中 `self.reward_fn(batch)` 到 `token_level_scores` | 调用规则 reward，并把返回 tensor 放入训练 batch。 | `token_level_scores` 还不是 advantage；后面才经过 KL 分支和组内标准化。 |
| 4 | `verl/trainer/main_ppo.py::RewardManager.__call__()` | 逐样本解码有效 `prompt + response`，取 `data_source` 和 ground truth，构造 token-level reward tensor。 | 序列分数只写到最后一个有效 response token；parser 看到的是完整 prompt 加 response。 |
| 5 | 同文件 `_select_rm_score_fn(data_source)` | 按数据来源选择规则评分函数。 | 当前列出的开放域 QA 数据集都进入同一个 `qa_em.compute_score_em()`。 |
| 6 | `verl/utils/reward_score/qa_em.py::compute_score_em()` | 串起“解析答案 → EM 比较”，返回默认 0/1 的序列分数。 | `format_score` 默认也是 0，所以“标签格式正确但答案错误”仍然是 0。 |
| 7 | 同文件 `extract_solution(solution_str)` | 用正则找出所有完整 answer 块，返回最后一个。 | 当前实现要求至少两个 answer 块；第一个来自 prompt 示例，第二个才是模型答案。 |
| 8 | 同文件 `em_check(prediction, golden_answers)` | 预测命中任意一个 gold alias 就返回 1。 | 比较的是规范化后的完整字符串，不是子串匹配。 |
| 9 | 同文件 `normalize_answer(s)` | 小写、删英文标点和冠词，再压缩空白。 | 这是英文开放域 QA 的表面规范化；它不理解答案语义。 |

检查点 B 完成标准：能从
`reward_model.ground_truth['target'] → RewardManager → compute_score_em → reward_tensor`
逐步说清每个中间值的类型，并说明为什么非零分数位于 response 的最后一个有效 token。

在 `cmd.exe` 中按检查点定位代码：

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring"
rg -n "def make_prefix|def make_map_fn|def process_fn" Search-R1\scripts\data_process\nq_search.py
rg -n "def collate_fn|class RLHFDataset|def __init__|def _read_files_and_tokenize|def __getitem__" Search-R1\verl\utils\dataset\rl_dataset.py
rg -n "def _create_dataloader|DataProto.from_single_dict|uid.*index|self.reward_fn" Search-R1\verl\trainer\ppo\ray_trainer.py
rg -n "def _select_rm_score_fn|class RewardManager|def __call__" Search-R1\verl\trainer\main_ppo.py
rg -n "def compute_score_em|def extract_solution|def em_check|def normalize_answer" Search-R1\verl\utils\reward_score\qa_em.py
```

本轮不下载数据、不启动 `train_grpo.sh`，也不阅读 Ray/FSDP/vLLM 底层。
