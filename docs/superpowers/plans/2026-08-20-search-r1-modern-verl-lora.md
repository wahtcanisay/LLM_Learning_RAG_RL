# Search-R1 Modern veRL LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one real Qwen2.5-3B Search-R1 GRPO optimizer step on a single RTX 5090 using current veRL's LoRA, FSDP and vLLM weight-sync path.

**Architecture:** Keep Search-R1's XML action protocol, HTTP retriever, EM reward and dataset. Replace the old vendored trainer execution path with a pinned upstream veRL checkout and a small custom `AgentLoopBase`; let upstream veRL own LoRA/FSDP/vLLM synchronization instead of backporting it.

**Tech Stack:** Python 3.12, PyTorch 2.13, veRL `e78efedbe56591cd53ac107150abb7565bb9b8b1`, vLLM 0.27.1, PEFT LoRA, Ray, W&B.

---

### Task 1: Remove the failed full-state CPU workaround

**Files:**
- Modify: `Search-R1/verl/workers/sharding_manager/fsdp_vllm.py`
- Modify: `Search-R1/tests/test_fsdp_vllm_weight_swap.py`

- [ ] Add/adjust the regression test so the legacy manager no longer materializes a second CPU copy of every tensor.
- [ ] Run the focused test and confirm it fails against the current workaround.
- [ ] Remove only the failed CPU-state-dict conversion; retain the independently verified vLLM scheduler/wake compatibility changes.
- [ ] Run the legacy compatibility tests and confirm they pass.

### Task 2: Add a modern Search-R1 AgentLoop

**Files:**
- Create: `Search-R1/search_r1/modern_verl/__init__.py`
- Create: `Search-R1/search_r1/modern_verl/search_agent_loop.py`
- Create: `Search-R1/search_r1/modern_verl/agent_loops.yaml`
- Create: `Search-R1/tests/test_modern_search_agent_loop.py`

- [ ] Write failing tests for XML action parsing, retrieval-result formatting and environment-token loss masking.
- [ ] Run the tests with the pinned upstream veRL on `PYTHONPATH` and verify the expected failures.
- [ ] Implement the smallest custom `AgentLoopBase` that generates `<search>`/`<answer>`, calls the existing `/retrieve` endpoint, inserts `<information>`, and returns `response_mask=0` for environment tokens.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Reuse Search-R1 EM reward through the modern API

**Files:**
- Create: `Search-R1/search_r1/modern_verl/reward.py`
- Create: `Search-R1/tests/test_modern_reward.py`

- [ ] Write failing tests for the current veRL reward signature and last-answer extraction.
- [ ] Verify the tests fail before implementation.
- [ ] Implement `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` by delegating to the existing Search-R1 EM scorer.
- [ ] Verify reward tests pass.

### Task 4: Add a pinned one-step LoRA GRPO launcher

**Files:**
- Create: `Search-R1/experiments/single_gpu_smoke/setup_modern_verl.sh`
- Create: `Search-R1/experiments/single_gpu_smoke/run_grpo_1step_modern_verl.sh`
- Create: `Search-R1/tests/test_modern_smoke_config.py`

- [ ] Write a failing static configuration test requiring 3B, positive LoRA rank, `load_format=safetensors`, `layered_summon=True`, GRPO, two rollouts, reference KL, one GPU and exactly one optimizer step.
- [ ] Verify the test fails before adding the launchers.
- [ ] Add a pinned upstream checkout/setup script and a single-GPU launcher using the existing model, Parquet subset, HTTP retriever and W&B login.
- [ ] Verify static tests and Python compilation pass.

### Task 5: Run the real training gate and record evidence

**Files:**
- Modify: `STUDY_PROGRESS.md`
- Create/update: `Search-R1/experiments/single_gpu_smoke/grpo_modern_1step.log`
- Create/update: `Search-R1/experiments/single_gpu_smoke/grpo_modern_gpu_memory.csv`

- [ ] Confirm the BM25 server responds and the 1.5B model directory is absent.
- [ ] Install the pinned upstream veRL checkout without replacing PyTorch/vLLM.
- [x] Run the focused/unit test suite inside Docker (`10 passed`).
- [x] Run one real optimizer step while sampling GPU memory.
- [x] Record exit code 0, global step 1, W&B run `lx30qqpq`, and measured peak VRAM. Actor loss/gradient were present but zero because both rollout rewards were zero.
- [x] Preserve failed-gate tracebacks in the iteration history; do not reduce the model size.
