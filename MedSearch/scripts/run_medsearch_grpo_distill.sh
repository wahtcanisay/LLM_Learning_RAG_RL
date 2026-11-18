#!/usr/bin/env bash
# MedSearch 新蒸馏实验的 GRPO 单步接线入口；在线 OPD 尚未接入。
set -euo pipefail  # 遇到错误、未定义变量或管道失败立即退出，避免“失败却继续训练”。

# ===== 项目与数据路径 =====
ROOT="/workspace/code_list/some tricks/LLMLeanring"  # 容器内项目根目录。
VERL_DIR="$ROOT/MedSearch/.deps/verl"  # 固定的 veRL 源码目录。
SFT_VARIANT="${MEDSEARCH_SFT_VARIANT:-top32}"  # top32=8轮，ce=3轮；两版使用同一入口。
INPUT_PATHS="$(python - "$ROOT" "$SFT_VARIANT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root=Path(sys.argv[1])
final=json.loads((root/'MedSearch/outputs/distillation/FINALIZED_SFT_CHECKPOINTS.json').read_text())
assert final['status']=='finalized'
variant=final['variants'][sys.argv[2]]
adapter=root/variant['checkpoint_path']
assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==variant['adapter_sha256']
release=json.loads((root/'MedSearch/data/cleaning_v2/CURRENT_RL_RELEASE.json').read_text())
assert release['training_ready'] is True
train=root/release['path']
assert hashlib.sha256(train.read_bytes()).hexdigest()==release['sha256']
for path in [root/final['base_model_path'],adapter,train]:
    print(path)
PY
)"
mapfile -t INPUTS <<< "$INPUT_PATHS"
MODEL_DIR="${INPUTS[0]}"  # 原始3B基座与训练好的adapter分开加载。
ADAPTER_DIR="${INPUTS[1]}"
TRAIN_FILE="${INPUTS[2]}"  # 读取当前正式RL发布，避免误用旧5756题训练池。
DEV_FILE="$ROOT/MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet"
AGENT_CONFIG="$ROOT/MedSearch/configs/medsearch_agent_v9.yaml"  # 自定义 AgentLoop：BM25、top-k、工具轮数等。
REWARD_PATH="$ROOT/MedSearch/rl/reward.py"  # reward 回调；当前使用 3B verifier。
RUN_DIR="${MEDSEARCH_RUN_DIR:-$ROOT/MedSearch/outputs/distillation/grpo_wiring_smoke}"
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
if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Student checkpoint directory not found: %s\n' "$MODEL_DIR" >&2
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 降低长时间训练中的 CUDA 内存碎片风险。

# ===== 可复现性记录 =====
{
  printf "started_utc="  # 记录启动时间。
  date -u +%FT%TZ
  printf "git_commit="  # 记录代码版本。
  git -C "$ROOT" rev-parse HEAD
  printf "training_steps=%s\nstudent_checkpoint=%s\n" "$TRAINING_STEPS" "$MODEL_DIR"
  printf "sft_variant=%s\nsft_adapter=%s\n" "$SFT_VARIANT" "$ADAPTER_DIR"
  sha256sum "$ADAPTER_DIR/adapter_model.safetensors" "$ROOT/MedSearch/data/cleaning_v2/CURRENT_RL_RELEASE.json"
  sha256sum "$0" "$AGENT_CONFIG" "$REWARD_PATH" "$TRAIN_FILE" "$DEV_FILE"  # 固化脚本、Agent、reward 与数据哈希。
} > "$RUN_DIR/run_manifest.txt"

# ===== GPU 显存监控 =====
(while true; do
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used --format=csv,noheader  # 每秒记录显存总量与已用量。
  sleep 1  # 监控采样间隔（秒）。
done) >> "$RUN_DIR/gpu_memory.csv" &
sampler_pid=$!  # 后台显存采样进程 PID。
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT INT TERM  # 训练退出/中断时关闭监控进程。

# ===== veRL V1 同步 GRPO 主训练命令 =====
TRAIN_ARGS=(
  # RL 训练样本。
  data.train_files="$TRAIN_FILE"
  # 开发集；当前 val_before_train=false，不在训练前额外验证。
  data.val_files="$DEV_FILE"
  # 单题双轨迹，只用于接线 smoke。
  data.train_batch_size=1
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
  # veRL FSDP原生加载训练好的LoRA，保留为可训练参数。
  actor_rollout_ref.model.lora_adapter_path="$ADAPTER_DIR"
  actor_rollout_ref.model.target_modules='[q_proj,v_proj]'
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
  # 一个mini-batch的题数；V1训练器会乘rollout.n，实际处理两条轨迹。
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  # 单卡一次反传处理 1 条 rollout，控制显存峰值。
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  # reference log-prob 的单卡微批大小。
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  # LoRA actor 学习率。
  actor_rollout_ref.actor.optim.lr=1e-6
  # 本轮基线关闭 reference KL，教师 OPD 留待独立 loss 接线。
  actor_rollout_ref.actor.use_kl_loss=false
  # DAPO/PPO 概率比下侧裁剪阈值。
  actor_rollout_ref.actor.clip_ratio_low=0.2
  # GRPO 起始配置采用对称裁剪。
  actor_rollout_ref.actor.clip_ratio_high=0.2
  # 对有效模型 token 取均值计算 policy loss。
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  # 使用 vLLM 作为 rollout 推理后端。
  actor_rollout_ref.rollout.name=vllm
  # 单卡推理，不做 tensor parallel。
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  # vLLM启动比例设置；下方显式KV大小决定缓存预算，并非进程显存硬上限。
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30
  # 单题双轨迹smoke无需默认1024并发；小prefill批次保留完整轨迹长度。
  actor_rollout_ref.rollout.max_num_seqs=2
  actor_rollout_ref.rollout.max_num_batched_tokens=2048
  actor_rollout_ref.rollout.enforce_eager=true
  # 多进程共卡时避免比例profile将actor占用扣入cache预算；512MiB用于本次双轨迹smoke。
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=536870912
  # rollout 最大上下文；512 prompt + 4096 response 的上界。
  actor_rollout_ref.rollout.max_model_len=4608
  # smoke 每题两条轨迹；正式预算待实际吞吐验收。
  actor_rollout_ref.rollout.n=2
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
  # 原生naive manager把AgentLoop extra_fields传给reward；组过滤另由algorithm配置。
  reward.reward_manager.name=naive
  # 每个节点使用 1 张 GPU。
  trainer.n_gpus_per_node=1
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  # 单节点训练。
  trainer.nnodes=1
  # optimizer 更新总步数；由 MEDSEARCH_TOTAL_TRAINING_STEPS 覆盖。
  trainer.total_training_steps="$TRAINING_STEPS"
  # 不在训练前跑验证，避免额外模型加载与时间。
  trainer.val_before_train=false
  # 禁用训练中定期测试；dev 评测单独运行。
  trainer.test_freq=-1
  # 单步 smoke 必须保存 checkpoint；重载与动作推理仍需单独验收。
  trainer.save_freq=1
  # checkpoint 根目录；仅在 save_freq>0 时实际写入。
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  # 最多保留 2 个 actor checkpoint，控制磁盘占用。
  trainer.max_actor_ckpt_to_keep=2
  # 同时写终端日志与本地 W&B 离线日志。
  trainer.logger='["console","wandb"]'
  # W&B 项目名。
  trainer.project_name=medsearch_r1
  # W&B 实验名；全量重训时应改为新的、可追踪名称。
  trainer.experiment_name=grpo_distill_wiring_smoke
  # 不从旧 checkpoint 恢复，始终按当前模型路径重新开始。
  trainer.resume_mode=disable
  # 每个训练 step 的 rollout JSONL 保存位置。
  trainer.rollout_data_dir="$RUN_DIR/rollouts"
  # 验证轨迹和指标的保存位置。
  trainer.validation_data_dir="$RUN_DIR/validation"
  # 组相对优势估计器；同题多 rollout 做相对比较。
  algorithm.adv_estimator=grpo
  # 失败 OPD 需要全错组进入训练 batch。
  algorithm.filter_groups.enable=false
  # 组过滤读取 reward 回调返回的 acc 字段。
  algorithm.filter_groups.metric=acc
  # 同时在飞的生成 batch 数，设为 1 降低显存和主机内存压力。
  algorithm.filter_groups.max_inflight_gen_batches=1
  # KL 作为 actor loss，不叠加到环境 reward。
  algorithm.use_kl_in_reward=false
  distillation.enabled=false
  # WSL 主机内存保护：仅 1 个数据单元，避免 Ray 存储 OOM。
  transfer_queue.backend.SimpleStorage.num_data_storage_units=1
)

python -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "$@" \
  2>&1 | tee -a "$RUN_DIR/train.log"  # 同时显示并追加保存完整训练日志。
