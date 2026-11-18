#!/usr/bin/env bash
# 单卡 RTX 5090 上的最小真实 GRPO 更新：1 道题、2 条同题轨迹、1 个 optimizer step。
# 目的仅是验证 Search-R1 的完整训练链路并留下 W&B/本地日志，不作为性能实验。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VERL_PPO_LOGGING_LEVEL=DEBUG
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_MODE=online

MODEL_PATH='/workspace/code_list/some tricks/LLMLeanring/models/Qwen2.5-3B-Instruct'
DATA_DIR='/workspace/Search-R1/data/single_gpu_smoke'
EXPERIMENT_NAME='qwen2.5-3b-search-r1-grpo-1step'
LOG_PATH='/workspace/Search-R1/experiments/single_gpu_smoke/grpo_1step.log'

cd /workspace/Search-R1

python3 -m verl.trainer.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_data_num=1 \
    data.val_data_num=1 \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=768 \
    data.max_response_length=128 \
    data.max_start_length=512 \
    data.max_obs_length=192 \
    data.shuffle_train_dataloader=false \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1792 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.kv_cache_memory_bytes=134217728 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.logger="['console','wandb']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.project_name='Search-R1-local' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    max_turns=1 \
    retriever.url='http://127.0.0.1:8000/retrieve' \
    retriever.topk=3 \
    2>&1 | tee "${LOG_PATH}"
