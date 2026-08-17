# Search-R1 Data-to-Reward Reading Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Search-R1 的 NQ 数据到规则 reward 阅读路线细化为可逐函数执行的两个检查点，并补齐影响理解的源码注释。

**Architecture:** 不改运行逻辑。先核对真实调用链，再用“完整数据流 → 文件名 → 函数名 → 输入/输出 → 设计作用 → 调用位置 → 注意点”的层级重写中文导读；源码只补当前明显过短的 `_create_dataloader()` 注释，最后更新学习进度并执行 comment-only 验证。

**Tech Stack:** Python、Hugging Face Datasets、Pandas/Parquet、PyTorch DataLoader、veRL DataProto。

---

### Task 1: 核对并拆分真实调用链

**Files:**
- Read: `Search-R1/scripts/data_process/nq_search.py`
- Read: `Search-R1/verl/utils/dataset/rl_dataset.py`
- Read: `Search-R1/verl/trainer/ppo/ray_trainer.py`
- Read: `Search-R1/verl/trainer/main_ppo.py`
- Read: `Search-R1/verl/utils/reward_score/qa_em.py`

- [x] **Step 1: 核对 Parquet 到 batch 调用点**

确认 `Dataset.map → process_fn`、`DataLoader → __getitem__ → collate_fn`、`DataProto.from_single_dict` 的真实顺序。

- [x] **Step 2: 核对 batch 到 reward 调用点**

确认 `fit → RewardManager.__call__ → _select_rm_score_fn → compute_score_em → extract_solution/em_check/normalize_answer` 的真实顺序。

### Task 2: 补足阅读导引与必要源码注释

**Files:**
- Modify: `Search-R1/docs/source_code_learning_zh.md`
- Modify: `Search-R1/verl/trainer/ppo/ray_trainer.py`

- [x] **Step 1: 将路线改为数据流和文件级逐函数讲解**

先展示跨文件数据流，再以五个明确文件名作为标题，为必读函数分别解释输入、输出、设计作用、调用位置和注意点；今天只验收“Parquet 一行怎样变成一个 batch”。

- [x] **Step 2: 补充 `_create_dataloader()` 的输入、输出、调用方式和注意点**

说明该函数由 trainer 构造阶段调用，创建的对象保存在实例属性中，并指出 `drop_last=True` 与当前未启用长度过滤的实际影响。

### Task 3: 更新进度、验证并提交

**Files:**
- Modify: `STUDY_PROGRESS.md`

- [x] **Step 1: 更新 2026-08-17 唯一任务与完成标准**

- [x] **Step 2: 运行验证**

Run: `python Search-R1/scripts/verify_source_notes.py`

Run: `python -m compileall -q Search-R1/verl/trainer/ppo/ray_trainer.py`

Run: `git diff --check`

Expected: comment-only、行为样例、编译与补丁格式检查全部通过。

**Step 3: 提交并推送**

Run: `git add STUDY_PROGRESS.md Search-R1/docs/source_code_learning_zh.md Search-R1/verl/trainer/ppo/ray_trainer.py docs/superpowers/plans/2026-08-17-search-r1-data-reward-reading-guide.md`

Run: `git commit -m "docs: refine Search-R1 data reward reading guide"`

Run: `git push origin main`
