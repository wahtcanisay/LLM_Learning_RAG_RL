#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/code_list/some tricks/LLMLeanring"
VERL_DIR="$ROOT/MedSearch/.deps/verl"
MODEL_DIR="$ROOT/models/Qwen2.5-3B-Instruct"
TRAIN_FILE="$ROOT/MedSearch/data/rl/train.parquet"
DEV_FILE="$ROOT/MedSearch/data/rl/dev.parquet"
AGENT_CONFIG="$ROOT/MedSearch/configs/medsearch_agent.yaml"
REWARD_PATH="$ROOT/MedSearch/rl/reward.py"
RUN_DIR="$ROOT/MedSearch/outputs/dapo_v2_medrag_instruct_n8_b8_s42"

export PYTHONPATH="$VERL_DIR:$ROOT"
export VERL_USE_UV=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VERL_FORCE_SHM_WEIGHT_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RUN_DIR"

{
  printf "started_utc="
  date -u +%FT%TZ
  printf "git_commit="
  git -C "$ROOT" rev-parse HEAD
  sha256sum "$0" "$AGENT_CONFIG" "$REWARD_PATH" "$TRAIN_FILE" "$DEV_FILE"
} > "$RUN_DIR/run_manifest.txt"

(while true; do
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used --format=csv,noheader
  sleep 1
done) >> "$RUN_DIR/gpu_memory.csv" &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT INT TERM

python -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$DEV_FILE" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  data.dataloader_num_workers=0 \
  data.seed=42 \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CONFIG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=medsearch_agent \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_outcome_reward \
  reward.reward_manager.name=dapo \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=411 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=100 \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=medsearch_r1 \
  trainer.experiment_name=dapo_v2_medrag_instruct_n8_b8_s42 \
  trainer.resume_mode=auto \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  algorithm.adv_estimator=grpo \
  algorithm.filter_groups.enable=false \
  algorithm.use_kl_in_reward=false \
  2>&1 | tee -a "$RUN_DIR/train.log"
