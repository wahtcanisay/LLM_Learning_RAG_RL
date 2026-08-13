# Search-R1 Reward Learning Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Search-R1 下一学习阶段的规则奖励主链补齐输入、输出和真实调用位置注释。

**Architecture:** 仅修改奖励入口与 EM 解析函数的 docstring，不改变可执行语句。调用关系以 `main_task → RayPPOTrainer.fit → RewardManager → compute_score_em → extract_solution → compute_advantage` 为边界。

**Tech Stack:** Python、PyTorch、veRL `DataProto`、Git。

---

### Task 1: 标注奖励主链

**Files:**
- Modify: `Search-R1/verl/trainer/main_ppo.py`
- Modify: `Search-R1/verl/utils/reward_score/qa_em.py`
- Modify: `Search-R1/verl/trainer/ppo/ray_trainer.py`

- [ ] **Step 1: 补充函数级学习注释**

为 `_select_rm_score_fn()`、`RewardManager.__call__()`、`extract_solution()`、`compute_score_em()`、`compute_advantage()` 写明 `输入：`、`输出：`、`调用方式：`，并标注关键 tensor shape 与非张量字段结构。

- [ ] **Step 2: 验证没有改变执行逻辑**

Run: `python Search-R1/scripts/verify_source_notes.py`

Expected: `Comment-only verification passed` 且 6 个协议样例通过。

- [ ] **Step 3: 验证 Python 语法与补丁格式**

Run: `python -m compileall -q Search-R1/verl/trainer/main_ppo.py Search-R1/verl/utils/reward_score/qa_em.py Search-R1/verl/trainer/ppo/ray_trainer.py`

Run: `git diff --check`

Expected: 两条命令退出码均为 0。

### Task 2: 记录、提交并推送

**Files:**
- Modify: `STUDY_PROGRESS.md`

- [ ] **Step 1: 记录注释范围和验证边界**

只记录源码阅读注释和静态验证，不写入训练、显存或效果指标。

- [ ] **Step 2: 精确暂存并提交**

Run: `git add Search-R1/verl/trainer/main_ppo.py Search-R1/verl/utils/reward_score/qa_em.py Search-R1/verl/trainer/ppo/ray_trainer.py STUDY_PROGRESS.md docs/superpowers/plans/2026-08-13-search-r1-reward-annotations.md`

Run: `git commit -m "docs: expand Search-R1 reward learning notes"`

- [ ] **Step 3: 推送当前 main**

Run: `git push origin main`

Expected: 远程 `main` 更新到新提交。
