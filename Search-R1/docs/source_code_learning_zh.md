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

## 5. 数据到 reward 的文件与函数阅读路线

### 5.1 先看完整数据流向

阅读前先记住：同一条样本同时携带两类数据。模型输入会变成 tensor；答案、数据来源和题目编号保持为 Python 元数据，直到 reward 阶段才被读取。

```text
Hugging Face NQ 原始样本
  question: str
  golden_answers: list[str]
        │
        ▼
文件 1：scripts/data_process/nq_search.py
  make_prefix()
    → 把 question 写进带 think/search/information/answer 标签的文本 prompt
  make_map_fn() / process_fn()
    → 生成 data_source、prompt、ability、reward_model、extra_info
        │
        ▼ Dataset.to_parquet()
train.parquet / test.parquet
        │
        ▼
文件 2：verl/utils/dataset/rl_dataset.py
  RLHFDataset.__init__()
    → 读取 Parquet 为 pandas.DataFrame
  RLHFDataset.__getitem__()
    → prompt 变成 input_ids / attention_mask / position_ids
    → reward_model、data_source、extra_info 等仍是 Python 对象
  collate_fn()
    → 单样本 tensor [L] 变成 batch tensor [B, L]
    → Python 对象变成长度为 B 的 object 数组
        │
        ▼
文件 3：verl/trainer/ppo/ray_trainer.py
  _create_dataloader()
    → 组装 Dataset 与 DataLoader
  fit() 中 DataProto.from_single_dict()
    → tensor 字段进入 batch，Python 对象进入 non_tensor_batch
  fit() 中 batch.repeat() / uid = index.copy()
    → 一道题扩成多条 rollout，并保留同题分组标识
  fit() 中 self.reward_fn(batch)
    → 把完整 rollout 送入规则 reward
        │
        ▼
文件 4：verl/trainer/main_ppo.py
  RewardManager.__call__()
    → 解码有效 prompt + response
    → 读取 data_source 与 reward_model.ground_truth
  _select_rm_score_fn()
    → 根据 data_source 选择 qa_em.compute_score_em
        │
        ▼
文件 5：verl/utils/reward_score/qa_em.py
  compute_score_em()
    → extract_solution() 取最终 answer
    → em_check() 比较任意 gold alias
    → normalize_answer() 消除英文表面格式差异
        │
        ▼
标量 0/1 score
        │
        ▼ RewardManager 写入最后一个有效 response token
token_level_scores: torch.Tensor[float32], shape [B, response_len]
```

### 5.2 阅读顺序 1：数据预处理文件

#### 文件名：`Search-R1/scripts/data_process/nq_search.py`

这个文件负责定义训练数据协议。它不加载模型、不生成 token，也不计算 reward；它只决定一条 NQ 数据以什么字段和文本格式进入训练系统。

第一步读 `make_prefix(dp, template_type)`。

- 输入：包含 `question: str` 的原始样本和模板名称。
- 输出：一段完整的 user prompt 字符串。
- 设计作用：把“什么时候思考、什么时候搜索、什么时候作答”编码成文本标签协议，使 Base 模型通过生成 `<search>` 或 `<answer>` 表达动作。
- 调用位置：内部 `process_fn()` 在处理每条样本时调用。
- 注意：这里没有 tokenizer。prompt 中的 `<answer> Beijing </answer>` 是格式示例，后面的 answer parser 会把它算作第一个 answer 块。

第二步读 `make_map_fn(split)`。

- 输入：`split: str`，当前是 `train` 或 `test`。
- 输出：内部函数 `process_fn(example, idx)`。
- 设计作用：使用闭包保存 split，避免 Hugging Face `Dataset.map()` 每处理一行时都手工传 split。
- 调用位置：`train_dataset.map(make_map_fn('train'), with_indices=True)` 和 test 对应调用。
- 注意：这个函数本身不处理样本，真正的逐行转换发生在它返回的 `process_fn()` 中；而且它定义在 `if __name__ == '__main__'` 内，只服务于脚本执行。

第三步读 `process_fn(example, idx)`。

- 输入：原始 NQ 字典与 `Dataset.map(with_indices=True)` 自动提供的行号 `idx: int`。
- 输出：最终写入 Parquet 的一行字典。
- 设计作用：把原始字段转换成 Search-R1 与 veRL 约定的五个顶层字段：

```python
{
    "data_source": str,
    "prompt": list[dict[str, str]],
    "ability": str,
    "reward_model": {
        "style": str,
        "ground_truth": {"target": list[str]},
    },
    "extra_info": {"split": str, "index": int},
}
```

- 调用位置：Hugging Face `Dataset.map()` 对数据集每一行调用一次，返回结果最后由 `to_parquet()` 落盘。
- 注意：`golden_answers` 没有拼进模型 prompt，而是放在 `reward_model.ground_truth['target']`；否则模型在生成前就能看到答案。`extra_info.index` 是题目身份，不是 token 位置。

读完本文件后，应能回答：一条原始 NQ 样本怎样变成 Parquet 的五个字段，以及 prompt 与 gold 为什么必须分开保存。

### 5.3 阅读顺序 2：Dataset 与 batch 文件

#### 文件名：`Search-R1/verl/utils/dataset/rl_dataset.py`

这个文件连接“磁盘上的结构化样本”和“模型可以消费的 tensor”。它只对 prompt 做 tokenization，同时把 reward 所需元数据原样带到后续训练流程。

第一步读 `RLHFDataset.__init__(...)`。

- 输入：Parquet 路径、tokenizer、`max_prompt_length`、prompt 列名和截断策略等配置。
- 输出：无显式返回值；创建持有 `self.dataframe` 的 Dataset 实例。
- 设计作用：统一保存配置，先调用 `_download()` 解析文件路径，再调用 `_read_files_and_tokenize()` 读取数据。
- 调用位置：`RayPPOTrainer._create_dataloader()` 分别创建 train 和 validation Dataset。
- 注意：构造完成只代表 DataFrame 已加载，并不代表每行已经 tokenization。逐行 tensor 转换由 DataLoader 取样时触发。

第二步略读 `_download()`。

- 作用：把本地或 HDFS 配置路径统一转换成本进程可读的本地路径。
- 注意：本地学习时它通常只是路径准备，不改变样本内容，所以不需要追进 HDFS 工具内部。

第三步读 `_read_files_and_tokenize()`。

- 输入：实例中的本地 Parquet 路径列表。
- 输出：无显式返回值；设置 `self.dataframe: pandas.DataFrame`。
- 设计作用：读取一个或多个 Parquet，然后用 `pd.concat()` 合成一张表。
- 调用位置：只由 `RLHFDataset.__init__()` 调用。
- 注意：函数名容易误导。当前快照在这里没有 tokenization，长度过滤代码也已经注释，因此打印的 original len 与 filter len 通常相同。

第四步读 `RLHFDataset.__getitem__(item)`，这是本文件最重要的函数。

- 输入：DataLoader/Sampler 给出的单个行号 `item: int`。
- 输出：一个单样本字典，其中三类模型输入是长度为 `max_prompt_length` 的一维 tensor，其余仍是 Python 元数据。
- 设计作用：取出 `prompt`，有 chat template 时加入模型控制 token，然后调用 `tokenize_and_postprocess_data()` 完成 tokenization、定长处理和左 padding；最后计算 `position_ids`。
- 数据形状：

```text
input_ids:      torch.Tensor[int64], shape [max_prompt_length]
attention_mask: torch.Tensor,        shape [max_prompt_length]
position_ids:   torch.Tensor[int64], shape [max_prompt_length]
```

- 调用位置：DataLoader 迭代时自动逐样本调用，返回结果再交给 `collate_fn()`。
- 注意：`row_dict.pop(self.prompt_key)` 会把原始 prompt 从返回字典移除；模型得到的是 token tensor。`data_source/reward_model/extra_info` 没有经过 tokenizer，仍留在 `row_dict`。`extra_info.index` 还会被提升为顶层 `index`。

第五步读 `collate_fn(data_list)`。

- 输入：同一 mini-batch 内多个 `__getitem__()` 返回字典组成的列表。
- 输出：合并后的 batch 字典。
- 设计作用：按值类型分流。tensor 使用 `torch.stack(..., dim=0)`；嵌套字典、字符串和整数等使用 `np.array(..., dtype=object)` 保存。
- 数据形状变化：单样本 `[max_prompt_length]` 变成 `[batch_size, max_prompt_length]`。
- 调用位置：创建 DataLoader 时作为 `collate_fn=collate_fn` 传入，由 DataLoader 自动调用。
- 注意：`reward_model` 不是为了在 GPU 上做矩阵计算，所以不能强行转 tensor。它必须保持字典结构，供后面的 `RewardManager` 按样本读取 ground truth。

`__len__()` 只返回 DataFrame 行数，不改变数据，可以最后快速看一眼。

### 5.4 阅读顺序 3：Trainer 中的数据装配位置

#### 文件名：`Search-R1/verl/trainer/ppo/ray_trainer.py`

这个文件很大，本轮禁止从头通读。只看 `_create_dataloader()` 和 `fit()` 中四个与 metadata/reward 有关的片段。

第一步读 `RayPPOTrainer._create_dataloader()`。

- 输入：无显式参数；使用 trainer 的 config 和 tokenizer。
- 输出：无返回值；设置 train/val Dataset、DataLoader 和 `total_training_steps`。
- 设计作用：把前一个文件中的 `RLHFDataset` 与 `collate_fn` 真正接到训练器。
- 调用位置：`RayPPOTrainer.__init__()` 在构造 trainer 时调用。
- 注意：train 和 validation 都配置了 `drop_last=True`，不足一个 batch 的尾部样本会被丢弃；`filter_prompts=True` 在当前 Dataset 实现中并没有真正执行过滤。

第二步只读 `fit()` 中的以下四处，不读其他 Ray/FSDP 逻辑。

1. `DataProto.from_single_dict(batch_dict)`：把 collate 后的普通字典拆成 tensor `batch` 与 Python 对象 `non_tensor_batch`。
2. `batch.repeat(repeat_times=...n_agent, interleave=True)`：为同一道题复制多份输入，让模型生成多条独立 rollout。
3. `batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()`：让同题 rollout 共享相同 uid，之后 GRPO 才能按题分组。
4. `reward_tensor = self.reward_fn(batch)` 与 `batch.batch['token_level_scores'] = reward_tensor`：调用规则 reward 并保存结果。

注意：`token_level_scores` 只是规则打分结果，还不是最终 advantage；本轮追到这里即可，不继续进入 KL 和 GRPO 数学部分。

### 5.5 阅读顺序 4：RewardManager 文件

#### 文件名：`Search-R1/verl/trainer/main_ppo.py`

这个文件负责把“完整轨迹 + 样本元数据”装配成 token-level reward。

先读 `_select_rm_score_fn(data_source)`。

- 输入：来自 Parquet 的 `data_source: str`。
- 输出：一个规则打分函数。
- 设计作用：让不同数据集可以选择不同评分逻辑。
- 注意：当前列出的 NQ、TriviaQA、HotpotQA 等数据源最终都返回同一个 `qa_em.compute_score_em`；未知数据源直接抛出异常。

再读 `RewardManager.__call__(data)`。

- 输入：包含 rollout tensor 与 `non_tensor_batch` 元数据的 `DataProto`。
- 输出：`torch.Tensor[float32]`，shape 为 `[batch_size, response_len]`。
- 设计作用：逐样本裁掉 prompt 左 padding 和 response 右 padding，拼接有效 prompt 与 response 并解码；随后读取 `data_source` 和 `reward_model.ground_truth`，调用选中的评分函数。
- 调用位置：训练阶段由 `RayPPOTrainer.fit()` 中的 `self.reward_fn(batch)` 调用；验证阶段由 `self.val_reward_fn(test_batch)` 调用。
- 注意：评分函数返回的是整条轨迹的标量结果，但 veRL 接口需要 token 维度，因此代码只把该分数写到最后一个有效 response token，其余 token 保持 0。parser 接收的是完整 `prompt + response`，不是只有 response。

### 5.6 阅读顺序 5：Exact Match reward 文件

#### 文件名：`Search-R1/verl/utils/reward_score/qa_em.py`

建议按文件中的基础函数顺序阅读：`normalize_answer()` → `em_check()` → `extract_solution()` → `compute_score_em()`；实际运行调用方向则从 `compute_score_em()` 向下调用这些辅助函数。

先读 `normalize_answer(s)`。

- 输入与输出都是 `str`。
- 作用：转小写、删除 ASCII 标点、删除英文冠词 `a/an/the`、压缩空白。
- 注意：它只消除英文答案的表面差异，不理解同义词，也不做语义匹配。

再读 `em_check(prediction, golden_answers)`。

- 输入：一个预测字符串，以及一个 gold 字符串或多个 gold aliases。
- 输出：命中任意 alias 返回 `1`，否则返回 `0`。
- 作用：分别规范化 prediction 和 gold，再做完整字符串相等比较。
- 注意：它不是子串匹配；更宽松的 `subem_check()` 存在，但主训练入口默认不用。

然后读 `extract_solution(solution_str)`。

- 输入：解码后的完整 `prompt + response` 字符串。
- 输出：最后一个完整 answer 块中的文本，解析失败则为 `None`。
- 作用：用正则找到所有 `<answer>...</answer>`，取最后一个作为模型最终答案。
- 注意：当前实现要求至少出现两个 answer 块。prompt 内格式示例提供第一个，模型输出提供第二个；只有一个 answer 块也会返回 `None`。

最后读 `compute_score_em(solution_str, ground_truth, ...)`。

- 输入：完整轨迹字符串与 `{'target': str | list[str]}` ground truth。
- 输出：默认是 `0.0` 或 `1.0` 的序列级标量分数。
- 设计作用：先调用 `extract_solution()`，成功后调用 `em_check()`，把答案格式解析和答案正确性比较串起来。
- 调用位置：由 `RewardManager.__call__()` 通过 `_select_rm_score_fn()` 取得后调用。
- 注意：`format_score` 默认也是 0，因此“成功解析但答案错误”与“没有解析到答案”默认都得到 0；token 位置的写入不是本函数做的，而是 `RewardManager` 做的。

### 5.7 今天的停止位置与完成标准

今天只完成文件 1 和文件 2，即：

```text
Search-R1/scripts/data_process/nq_search.py
  → make_prefix()
  → make_map_fn()
  → process_fn()

Search-R1/verl/utils/dataset/rl_dataset.py
  → RLHFDataset.__init__()
  → _download()（略读）
  → _read_files_and_tokenize()
  → RLHFDataset.__getitem__()
  → collate_fn()
```

完成标准：

1. 能写出 `process_fn()` 返回的五个顶层字段和字段类型；
2. 能说明 prompt 为什么进入 tokenizer，而 `reward_model` 为什么不进入；
3. 能说明 `__getitem__()` 返回的三个 tensor 的类型与 shape；
4. 能说明 `collate_fn()` 如何分别处理 tensor 与 Python 对象；
5. 能追踪 `extra_info.index → __getitem__()` 返回字典的顶层 `index`。

完成上述复述后，再继续文件 3～5，追踪 ground truth 怎样成为 0/1 reward。

在 `cmd.exe` 中按文件定位代码：

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring"
rg -n "def make_prefix|def make_map_fn|def process_fn" Search-R1\scripts\data_process\nq_search.py
rg -n "def collate_fn|class RLHFDataset|def __init__|def _read_files_and_tokenize|def __getitem__" Search-R1\verl\utils\dataset\rl_dataset.py
rg -n "def _create_dataloader|DataProto.from_single_dict|uid.*index|self.reward_fn" Search-R1\verl\trainer\ppo\ray_trainer.py
rg -n "def _select_rm_score_fn|class RewardManager|def __call__" Search-R1\verl\trainer\main_ppo.py
rg -n "def compute_score_em|def extract_solution|def em_check|def normalize_answer" Search-R1\verl\utils\reward_score\qa_em.py
```

本轮不下载数据、不启动 `train_grpo.sh`，也不阅读 Ray/FSDP/vLLM 底层。
