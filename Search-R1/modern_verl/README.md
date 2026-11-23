# Search-R1 modern veRL fallback

This directory keeps the Search-R1 task protocol while moving the runtime to a
pinned current veRL checkout. It is the project's reproducible fallback, not a
claim that the original paper used these dependencies.

## Architecture boundary

- Qwen2.5-3B-Instruct with LoRA rank 16; no 1.5B fallback.
- Official veRL commit `e78efedbe56591cd53ac107150abb7565bb9b8b1`.
- veRL FSDP2 actor, async vLLM rollout, current `AgentLoopBase`, and custom
  reward-function interfaces.
- Search-R1's `<search>`, `<information>`, and `<answer>` protocol is implemented
  in `search_r1_modern/` without copying the old trainer.
- `trainer.use_v1=False` deliberately keeps the stable trainer orchestration for
  this fallback. The current V1 trainer additionally requires TransferQueue;
  learning and migrating to V1 is the next separate task.
- WSL2/Docker uses veRL's existing shared-memory weight transport because CUDA
  IPC handles fail across the spawned vLLM worker boundary. The setup script
  applies a small opt-in upstream patch controlled by
  `VERL_FORCE_SHM_WEIGHT_TRANSFER=1`.

## Reproduce in the existing container

Run these commands from Windows Command Prompt:

```cmd
docker exec c8f4518d842bbb796acecf2a7341df1554af86183ea121d910e759350e66b3c5 bash -lc "cd '/workspace/code_list/some tricks/LLMLeanring/Search-R1' && bash experiments/single_gpu_smoke/setup_modern_verl.sh"
docker exec c8f4518d842bbb796acecf2a7341df1554af86183ea121d910e759350e66b3c5 bash -lc "cd '/workspace/code_list/some tricks/LLMLeanring/Search-R1' && bash experiments/single_gpu_smoke/run_grpo_1step_modern_verl.sh"
```

The one-step gate passes only when the process exits with code 0 and the log
contains both `training/global_step:1` and `Training Progress: 100%`.

## Verified smoke result (2026-08-20)

- Final W&B run: `lx30qqpq`.
- Two rollouts, four turns each, one completed optimizer step.
- Step time: 21.549 s; actor update: 4.059 s; weight update: 3.271 s.
- External GPU peak: 23,117 MiB of 32,607 MiB.
- Both sampled answers scored 0, so group advantages, policy loss, and gradient
  norm were all 0. This validates the full training path but is not evidence of
  learning quality.

The complete console log and GPU samples are under
`experiments/single_gpu_smoke/`.

Current upstream veRL may print a W&B `Tracking.__del__` closed-transport
traceback during Ray teardown. It occurs after the step metrics and final
validation line; the verified process exit code remains 0. It is retained as a
known cleanup warning rather than patched into the training path.
