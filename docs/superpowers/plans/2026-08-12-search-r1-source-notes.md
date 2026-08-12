# Search-R1 Source Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Search-R1 官方源码增加不改变行为的中文学习注释，并提供一条按重要性排序、可从入口追到 GRPO 更新的阅读路线。

**Architecture:** 保留官方目录和实现，只在算法主干文件补充模块说明、数据流、张量字段与外部依赖用途。用单独的源码导读汇总阅读顺序和“暂时跳过”的框架底座，避免把通用 veRL/vLLM 兼容代码与 Search-R1 创新点混在一起。

**Tech Stack:** Python、PyTorch、veRL、Ray、Hydra/OmegaConf、Transformers、vLLM、FastAPI/Requests、Pandas/Parquet。

---

### Task 1: 建立主干源码地图

**Files:**
- Create: `Search-R1/docs/source_code_learning_zh.md`

- [x] **Step 1:** 按 P0/P1/P2 标出训练入口、数据协议、多轮 rollout、检索、reward、GRPO 主循环和暂缓阅读的框架底座。
- [x] **Step 2:** 为外部包写最小基础介绍，并说明它在本项目中的真实用途。
- [x] **Step 3:** 给出第一天唯一阅读任务、检查问题和可在 `cmd.exe` 执行的只读命令。

### Task 2: 注释数据、交互与奖励主干

**Files:**
- Modify: `Search-R1/scripts/data_process/nq_search.py`
- Modify: `Search-R1/verl/utils/dataset/rl_dataset.py`
- Modify: `Search-R1/search_r1/llm_agent/tensor_helper.py`
- Modify: `Search-R1/search_r1/llm_agent/generation.py`
- Modify: `Search-R1/search_r1/search/retrieval_server.py`
- Modify: `Search-R1/verl/utils/reward_score/qa_em.py`

- [x] **Step 1:** 解释 Parquet 行如何变成 tensor/non-tensor batch，以及 `prompt`、`reward_model`、`data_source`、`index` 的用途。
- [x] **Step 2:** 解释 left padding、position id、active mask 和 information mask。
- [x] **Step 3:** 完整标注 `<search>` / `<information>` / `<answer>` 多轮状态机、批量检索请求和停止规则。
- [x] **Step 4:** 标注 EM 规范化、答案抽取的“双 answer 标签”前提及终点 token reward。

### Task 3: 注释训练装配与 GRPO 主循环

**Files:**
- Modify: `Search-R1/train_grpo.sh`
- Modify: `Search-R1/verl/trainer/main_ppo.py`
- Modify: `Search-R1/verl/trainer/ppo/ray_trainer.py`

- [x] **Step 1:** 将官方 8 卡脚本按数据、rollout、优化、资源和日志分组解释，并明确不可直接用于单卡 5090。
- [x] **Step 2:** 标注 Hydra → Ray → worker → trainer 的装配流程和规则奖励管理器。
- [x] **Step 3:** 标注 rollout 重复采样、reward、GRPO advantage、reference KL、state masking、actor update 的主链。

### Task 4: 验证、提交与推送

**Files:**
- Create: `Search-R1/scripts/verify_source_notes.py`
- Modify: `STUDY_PROGRESS.md`（仅追加本轮可验证记录；保留已有未提交内容）

- [x] **Step 1:** 运行 `python -m compileall` 验证 Python 注释未破坏语法。
- [x] **Step 2:** 运行最小 reward 与 action parser 检查，确认原行为未变。
- [x] **Step 3:** 检查 `git diff --check`、变更文件列表和源码差异。
- [x] **Step 4:** 只暂存本轮源码导读、注释、计划和进度文件，不纳入 `.claude/`。
- [x] **Step 5:** 提交并推送 `codex/search-r1-source-notes` 到 `origin`。
