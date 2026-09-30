#!/usr/bin/env bash
# V9 MedSearch-R1 / DAPO 训练唯一入口。
# 后续修改请只改本文件；先做 smoke，再启动全量训练。
set -euo pipefail  # 遇到错误、未定义变量或管道失败立即退出，避免“失败却继续训练”。

# ===== 项目与数据路径 =====
ROOT="/workspace/code_list/some tricks/LLMLeanring"  # 容器内项目根目录。
VERL_DIR="$ROOT/MedSearch/.deps/verl"  # 固定的 veRL 源码目录。
MODEL_DIR="$ROOT/models/Qwen2.5-3B-Instruct"  # V9 的基础模型；V9 从新 LoRA 初始化，不加载旧 RL 权重。
TRAIN_FILE="$ROOT/MedSearch/data/rl/train.parquet"  # 冻结的 RL 训练集。
DEV_FILE="$ROOT/MedSearch/data/rl/dev_fixed64_s42.parquet"  # 固定 64 条开发集，仅供 trainer 内部验证。
AGENT_CONFIG="$ROOT/MedSearch/configs/medsearch_agent_v9.yaml"  # 自定义 AgentLoop：BM25、top-k、工具轮数等。
REWARD_PATH="$ROOT/MedSearch/rl/reward.py"  # reward 回调；当前使用 3B verifier。
RUN_DIR="${MEDSEARCH_RUN_DIR:-$ROOT/MedSearch/outputs/dapo_v9_verifier_smoke}"  # 输出目录；环境变量可为每次运行指定新目录。
TRAINING_STEPS="${MEDSEARCH_TOTAL_TRAINING_STEPS:-1}"  # 总 optimizer step；默认 1 仅用于 smoke，全量运行必须显式覆盖。

# ===== 运行环境与兼容性 =====
export PYTHONPATH="$VERL_DIR:$ROOT"  # 让 Python 同时找到固定 veRL 与 MedSearch 自定义模块。
export VERL_USE_UV=0  # 不让 veRL 另建 uv 环境，复用已验证的容器依赖。
export VLLM_USE_V2_MODEL_RUNNER=0  # WSL/RTX 5090 兼容：关闭会触发 UVA 问题的 V2 model runner。
export VERL_FORCE_SHM_WEIGHT_TRANSFER=1  # WSL 兼容：actor→rollout 权重同步走 shared memory，不走 CUDA IPC。
export CUDA_VISIBLE_DEVICES=0  # 只使用第 0 张 GPU。
export WANDB_MODE=offline  # W&B 只写本地离线日志，不上传网络。
export HF_HUB_OFFLINE=1  # 训练时禁止 Hugging Face 下载，保证使用已冻结的本地模型。
export TRANSFORMERS_OFFLINE=1  # Transformers 同样禁止在线下载。
export MEDSEARCH_VERIFIER_URL=http://127.0.0.1:8002/verify  # 独立 medical-o1 3B verifier HTTP 地址。

mkdir -p "$RUN_DIR"  # 建立本次运行的独立输出目录。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 降低长时间训练中的 CUDA 内存碎片风险。

# ===== 可复现性记录 =====
{
  printf "started_utc="  # 记录启动时间。
  date -u +%FT%TZ
  printf "git_commit="  # 记录代码版本。
  git -C "$ROOT" rev-parse HEAD
  printf "training_steps=%s\n" "$TRAINING_STEPS"  # 记录实际 step 覆盖值。
  sha256sum "$0" "$AGENT_CONFIG" "$REWARD_PATH" "$TRAIN_FILE" "$DEV_FILE"  # 固化脚本、Agent、reward 与数据哈希。
} > "$RUN_DIR/run_manifest.txt"

# ===== GPU 显存监控 =====
(while true; do
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used --format=csv,noheader  # 每秒记录显存总量与已用量。
  sleep 1  # 监控采样间隔（秒）。
done) >> "$RUN_DIR/gpu_memory.csv" &
sampler_pid=$!  # 后台显存采样进程 PID。
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT INT TERM  # 训练退出/中断时关闭监控进程。

# ===== veRL / DAPO 主训练命令 =====
TRAIN_ARGS=(
  # RL 训练样本。
  data.train_files="$TRAIN_FILE"
  # 开发集；当前 val_before_train=false，不在训练前额外验证。
  data.val_files="$DEV_FILE"
  # 每个 optimizer step 采样的原问题数；结合 n=5 形成 80 条 rollout。
  data.train_batch_size=16
  # 初始 prompt 的最大 token 数，超出会截断。
  data.max_prompt_length=512
  # 每条完整多轮轨迹的最大 response token 数，含模型动作但不训练 observation。
  data.max_response_length=4096
  # 单进程读数据，避免 WSL DataLoader 子进程清理问题。
  data.dataloader_num_workers=0
  # 数据顺序与训练随机种子。
  data.seed=42
  # actor、rollout 与冻结 reference 共用的基础模型。
  actor_rollout_ref.model.path="$MODEL_DIR"
  # LoRA rank；越大可训练容量越强，也更占显存。
  actor_rollout_ref.model.lora_rank=8
  # LoRA 缩放；有效缩放为 alpha/rank=2。
  actor_rollout_ref.model.lora_alpha=16
  # 使用 PyTorch SDPA，绕过当前 FlashAttention ABI 兼容问题。
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  # 关闭依赖 FlashAttention bert_padding 的去 padding 路径。
  actor_rollout_ref.model.use_remove_padding=false
  # actor 用 FSDP 管理模型与梯度。
  actor_rollout_ref.actor.strategy=fsdp
  # 将部分 actor 参数状态卸载到 CPU，适配 32GB 显存。
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  # 一个 PPO/DAPO mini-batch 的 rollout 数。
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  # 单卡一次反传处理 1 条 rollout，控制显存峰值。
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  # reference log-prob 的单卡微批大小。
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  # LoRA actor 学习率。
  actor_rollout_ref.actor.optim.lr=1e-6
  # 启用 actor 相对冻结 reference 的 KL 正则。
  actor_rollout_ref.actor.use_kl_loss=true
  # KL 正则强度；过大限制学习，过小易偏离基础模型。
  actor_rollout_ref.actor.kl_loss_coef=0.001
  # 使用低方差 KL 估计。
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  # DAPO/PPO 概率比下侧裁剪阈值。
  actor_rollout_ref.actor.clip_ratio_low=0.2
  # DAPO/PPO 概率比上侧裁剪阈值。
  actor_rollout_ref.actor.clip_ratio_high=0.28
  # 对有效模型 token 取均值计算 policy loss。
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  # 使用 vLLM 作为 rollout 推理后端。
  actor_rollout_ref.rollout.name=vllm
  # 单卡推理，不做 tensor parallel。
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  # rollout 进程最多预留约 30% 显存，给 FSDP actor 留空间。
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30
  # rollout 最大上下文；512 prompt + 4096 response 的上界。
  actor_rollout_ref.rollout.max_model_len=4608
  # 每个问题采样 5 条轨迹，供 GRPO/DAPO 计算组内相对优势。
  actor_rollout_ref.rollout.n=5
  # rollout 采样温度；需要同组行为差异，不能设为 0。
  actor_rollout_ref.rollout.temperature=1
  # 验证时每题只生成 1 条，减少评测成本。
  actor_rollout_ref.rollout.val_kwargs.n=1
  # rollout log-prob 重算微批，控制显存。
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  # rollout 采样随机种子。
  actor_rollout_ref.rollout.seed=42
  # 自定义 AgentLoop worker 数；单卡/单检索服务下固定为 1。
  actor_rollout_ref.rollout.agent.num_workers=1
  # 指向 MedSearchAgentLoop 的 Hydra 配置。
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CONFIG"
  # 指定 batch 默认使用的注册 AgentLoop 名。
  actor_rollout_ref.rollout.agent.default_agent_loop=medsearch_agent
  # 自定义 reward 函数所在 Python 文件。
  reward.custom_reward_function.path="$REWARD_PATH"
  # V9 用 verifier 判分；EM 版本应改为 compute_outcome_reward。
  reward.custom_reward_function.name=compute_verifier_outcome_reward
  # reward worker 数；verifier 服务为串行 CPU 推理时避免并发挤占。
  reward.num_workers=1
  # 不启用 veRL 内建 reward model，避免与自定义 verifier 重复。
  reward.reward_model.enable=false
  # 使用 DAPO reward manager，对同组 rollout 过滤与汇总。
  reward.reward_manager.name=dapo
  # 每个节点使用 1 张 GPU。
  trainer.n_gpus_per_node=1
  # 单节点训练。
  trainer.nnodes=1
  # optimizer 更新总步数；由 MEDSEARCH_TOTAL_TRAINING_STEPS 覆盖。
  trainer.total_training_steps="$TRAINING_STEPS"
  # 不在训练前跑验证，避免额外模型加载与时间。
  trainer.val_before_train=false
  # 禁用训练中定期测试；dev 评测单独运行。
  trainer.test_freq=-1
  # 历史 V9 设置：禁用 checkpoint。全量重训前必须改为正数，至少保存最终 LoRA adapter。
  trainer.save_freq=-1
  # checkpoint 根目录；仅在 save_freq>0 时实际写入。
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  # 最多保留 2 个 actor checkpoint，控制磁盘占用。
  trainer.max_actor_ckpt_to_keep=2
  # 同时写终端日志与本地 W&B 离线日志。
  trainer.logger='["console","wandb"]'
  # W&B 项目名。
  trainer.project_name=medsearch_r1
  # W&B 实验名；全量重训时应改为新的、可追踪名称。
  trainer.experiment_name=dapo_v9_verifier_smoke
  # 不从旧 checkpoint 恢复，始终按当前模型路径重新开始。
  trainer.resume_mode=disable
  # 每个训练 step 的 rollout JSONL 保存位置。
  trainer.rollout_data_dir="$RUN_DIR/rollouts"
  # 验证轨迹和指标的保存位置。
  trainer.validation_data_dir="$RUN_DIR/validation"
  # 组相对优势估计器；同题多 rollout 做相对比较。
  algorithm.adv_estimator=grpo
  # 启用 DAPO 组过滤，去掉全对或全错等无学习信号组。
  algorithm.filter_groups.enable=true
  # 组过滤读取 reward 回调返回的 acc 字段。
  algorithm.filter_groups.metric=acc
  # 同时在飞的生成 batch 数，设为 1 降低显存和主机内存压力。
  algorithm.filter_groups.max_inflight_gen_batches=1
  # KL 作为 actor loss，不叠加到环境 reward。
  algorithm.use_kl_in_reward=false
  # WSL 主机内存保护：仅 1 个数据单元，避免 Ray 存储 OOM。
  transfer_queue.backend.SimpleStorage.num_data_storage_units=1
)

python -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/train.log"  # 同时显示并追加保存完整训练日志。
