# GRPO Advantage and Policy Loss Reading Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Search-R1 中从序列 reward 到 actor policy loss 的最小主链补充可学习、可核验的中文注释和阅读路线。

**Architecture:** 只注释 `core_algos.py` 中 GRPO advantage 与 clipped policy loss 两个核心函数。上游沿用已注释的 `ray_trainer.compute_advantage()`，下游只读取已有 `dp_actor.update_policy()`，不进入 Ray/FSDP/vLLM 实现。

**Tech Stack:** Python、PyTorch、veRL、GRPO/PPO clipped objective。

---

### Task 1: 注释核心数学函数

**Files:**
- Modify: `Search-R1/verl/trainer/ppo/core_algos.py`

- [x] **Step 1: 注释 GRPO advantage**

为 `compute_grpo_outcome_advantage()` 写明 `输入：`、`输出：`、`调用方式：`，解释 uid 分组、组内均值/标准差、零方差组和 token 广播。

- [x] **Step 2: 注释 clipped policy loss**

为 `compute_policy_loss()` 写明 old/current log probability、ratio、clip、mask 和三个标量输出，区分 `ppo_kl` 监控量与 reference-policy KL。

### Task 2: 更新验证与学习进度

**Files:**
- Modify: `Search-R1/scripts/verify_source_notes.py`
- Modify: `STUDY_PROGRESS.md`

- [x] **Step 1: 把 core_algos.py 纳入 comment-only 验证**

在 `PYTHON_FILES` 中加入该文件，使验证脚本把去除 docstring 后的 AST 与官方快照 `3d4832d` 比较。

- [x] **Step 2: 更新唯一学习任务**

记录 reward 主链问答已验收，并把今日任务更新为 GRPO advantage → clipped policy loss；不得写入训练、显存或效果指标。

- [x] **Step 3: 运行验证**

Run: `python Search-R1/scripts/verify_source_notes.py`

Run: `python -m compileall -q Search-R1/verl/trainer/ppo/core_algos.py`

Run: `git diff --check`

Expected: comment-only 检查、协议样例和 Python 编译均通过，补丁无空白错误。
