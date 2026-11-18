#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
# CUDA IPC handles fail across the vLLM worker boundary under this WSL2/Docker
# stack. Current veRL's built-in CPU shared-memory transport preserves semantics.
export VERL_FORCE_SHM_WEIGHT_TRANSFER=1
export WANDB_MODE=online

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd -P)"
VERL_DIR="${PROJECT_ROOT}/.deps/verl"
MODEL_PATH="${WORKSPACE_ROOT}/models/Qwen2.5-3B-Instruct"
DATA_DIR="${PROJECT_ROOT}/data/single_gpu_smoke"
AGENT_CONFIG="${PROJECT_ROOT}/modern_verl/agent_loops.yaml"
REWARD_PATH="${PROJECT_ROOT}/modern_verl/search_r1_modern/reward.py"
LOG_PATH="${SCRIPT_DIR}/grpo_modern_1step.log"

test -f "${MODEL_PATH}/config.json"
test -f "${DATA_DIR}/train.parquet"
test -d "${VERL_DIR}/verl"

cd "${VERL_DIR}"
# The legacy trainer multiplies ppo_mini_batch_size by rollout.n, so 1 becomes
# the two-trajectory mini-batch used by this smoke run.
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=1 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1536 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.use_v1=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=Search-R1-local \
    trainer.experiment_name=qwen2.5-3b-search-r1-modern-verl-lora-1step \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    2>&1 | tee "${LOG_PATH}"
