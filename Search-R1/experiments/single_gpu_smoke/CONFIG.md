# Search-R1 单卡无训练多轮推理配置

日期：2026-08-20

## 固定数据

- 随机种子：`20260819`
- 来源：`data/nq_hotpotqa_train/{train,test}.parquet`
- 输出：`data/single_gpu_smoke/{train,test}.parquet`（Git 忽略）
- train：HotpotQA 原始行 `86094`
- test：2WikiMultihopQA 原始行 `41745`
- test 问题：`Where was the director of film Night Of The Scarecrow born?`
- gold aliases：`Aurora`、`Aurora, Ohio`

## 推理边界

- 模型：本地 `Qwen2.5-3B-Instruct`
- `trainer.val_only=true`：只执行 `_validate()`，不计算 advantage、不更新参数
- 单卡、train/val batch 均为 `1`
- `algorithm.adv_estimator=grpo`：不初始化 critic
- `max_turns=2`：该固定样本是双跳题，允许两次检索，之后执行 final rollout
- `max_start_length=512`
- `max_prompt_length=1024`
- `max_response_length=192`
- `max_obs_length=256`
- BM25：`http://127.0.0.1:8000/retrieve`，top-k `3`
- 实际通过后端：仓库原生 HF rollout、BF16
- vLLM 0.27.1：TP `1`、eager、GPU memory utilization `0.25`；通过 CPU state-dict
  交接、Actor 参数 offload、weights/KV 分阶段唤醒完成两轮真实生成
- logger：console；本次不创建 W&B run

## 验收

日志必须证明：

1. 没有进入 actor update；
2. 至少形成一条最终轨迹；
3. 若模型发出 `<search>`，则轨迹中出现对应 `<information>`；
4. 记录最终 `<answer>`、规则 reward、运行时间与 GPU 显存峰值；
5. 如果模型选择直接回答，诚实记录“没有触发检索”，不把它伪装成搜索成功。

## 实际结果

- 退出码：`0`
- 搜索次数：`2`
- 查询：`Director of the film "Night of the Scarecrow"` → `Jeff Burr birthplace`
- 最终答案：`<answer>Aurora, Ohio</answer>`
- 规则 reward：`1.0`
- 轨迹活跃数：`ACTIVE_TRAJ_NUM: [1, 1, 1, 0]`
- 日志：`experiments/single_gpu_smoke/inference.log`
- 显存峰值：本次 HF 通过运行未启动外部采样器，未知；不得用失败的 vLLM 初始化日志代替

## vLLM 单卡权重交换验证

- 日志：`experiments/single_gpu_smoke/vllm_weight_swap.log`
- 显存采样：`experiments/single_gpu_smoke/vllm_gpu_memory.csv`，间隔 `0.5s`，共 `777` 点
- 峰值：`32058 MiB`（2026-08-20 11:47:39）
- 交换次数：`2`；每次均按 `wake weights → load actor weights → wake KV → generate → level-2 sleep`
- 进程退出码：`0`；不再出现 `cumem_allocator.cpp:211` OOM
- 本次 `max_turns=1`，最终 rollout 再次请求搜索，EM reward `0.0`；只用于运行兼容验收，
  不作为模型效果结果
