# Search-R1 Data-to-Reward Reading Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Search-R1 当前 NQ 样本从预处理、Dataset、DataLoader 到规则 reward 的真实调用链补充详细学习注释。

**Architecture:** 只解释现有代码行为，不讨论复用或迁移。注释集中在 `nq_search.py`、`rl_dataset.py`、`qa_em.py` 的函数输入、输出、调用位置和字段流转；阅读路线写入现有源码导读。

**Tech Stack:** Python、Hugging Face Datasets、Pandas/Parquet、PyTorch DataLoader、veRL DataProto。

---

### Task 1: 注释 NQ Parquet 预处理

**Files:**
- Modify: `Search-R1/scripts/data_process/nq_search.py`

- [x] **Step 1: 注释 `make_prefix()`**

写明输入样本字段、输出 prompt 字符串，以及它由 `process_fn()` 调用。

- [x] **Step 2: 注释 `make_map_fn()` 与 `process_fn()`**

写明 Hugging Face `Dataset.map(..., with_indices=True)` 如何传入样本和行号，以及输出行的 `prompt/data_source/reward_model/extra_info` 结构。

### Task 2: 注释 Parquet 到训练 batch

**Files:**
- Modify: `Search-R1/verl/utils/dataset/rl_dataset.py`

- [x] **Step 1: 注释 `collate_fn()`**

写明 tensor stack、对象数组保留方式和 DataLoader 调用位置。

- [x] **Step 2: 注释 `RLHFDataset` 生命周期**

为 `__init__()`、`_download()`、`_read_files_and_tokenize()`、`__len__()`、`__getitem__()` 写明输入、输出和调用方式。

### Task 3: 注释开放域 EM 基础函数

**Files:**
- Modify: `Search-R1/verl/utils/reward_score/qa_em.py`

- [x] **Step 1: 注释 `normalize_answer()` 与 `em_check()`**

写明字符串/alias 输入、规范化结果、0/1 输出，以及 `compute_score_em()` 的调用关系。

### Task 4: 更新导读、进度并验证

**Files:**
- Modify: `Search-R1/docs/source_code_learning_zh.md`
- Modify: `STUDY_PROGRESS.md`

- [x] **Step 1: 更新唯一阅读路线**

阅读顺序固定为 `nq_search.py → rl_dataset.py → ray_trainer._create_dataloader() → main_ppo.RewardManager → qa_em.py`，不讨论迁移。

- [x] **Step 2: 运行 comment-only 验证**

Run: `python Search-R1/scripts/verify_source_notes.py`

Run: `python -m compileall -q Search-R1/scripts/data_process/nq_search.py Search-R1/verl/utils/dataset/rl_dataset.py Search-R1/verl/utils/reward_score/qa_em.py`

Run: `git diff --check`

Expected: AST、协议样例、Python 编译和补丁格式均通过。

**Step 3: 提交并推送**

Run: `git add STUDY_PROGRESS.md Search-R1/docs/source_code_learning_zh.md Search-R1/scripts/data_process/nq_search.py Search-R1/verl/utils/dataset/rl_dataset.py Search-R1/verl/utils/reward_score/qa_em.py docs/superpowers/plans/2026-08-16-search-r1-data-reward-reading.md`

Run: `git commit -m "docs: annotate Search-R1 data reward pipeline"`

Run: `git push origin main`
