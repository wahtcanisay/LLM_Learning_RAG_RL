# MedSearch 医疗 RL 数据与 Search-R1 冒烟训练实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task by task with review checkpoints.

**Goal:** 从现有 MedQA/MedMCQA 官方训练候选中冻结一版无明显泄漏、可审计、能被当前 veRL AgentLoop 直接读取的医疗 MCQ RL 数据，并用 Stage-B adapter 完成一次真实检索的一步 GRPO 冒烟训练。

**Architecture:** 数据构建器只负责筛选、隔离、格式化和落盘；prompt 复用 Stage-B 的 `format_stage_b_prompt`，RL 运行时复用现有 `modern_verl` AgentLoop、规则奖励和 BM25 检索服务。第一轮不新增搜索质量奖励、不重写检索器、不合并 adapter，也不做正式 RL 训练。

**Tech Stack:** Python、PyArrow/Parquet、Transformers tokenizer、sentence-transformers、pytest、veRL、vLLM、PEFT、FastAPI、BM25、Docker（WSL 内）。

---

## 2026-09-02 决策更新（覆盖下文早期 MCQ 方案）

- 主数据源改为 Apache-2.0 的 `FreedomIntelligence/medical-o1-verifiable-problem`，不再从 raw MCQ 开始主实验；
- 已冻结 `2,048 train + 256 dev`，路径为 `MedSearch/data/rl/medical_verifiable/`；
- 已按规范化问题排除实际 Stage-A train/dev、Search100、protocol dev 和 MIRAGE 最终测试池；
- reward 主线改为 HuatuoGPT-o1 风格的最终医疗答案 verifier reward，不加入 query 关键词或 evidence shaping；
- normalized exact match 仅用于一步接口冒烟，正式 pilot 接入 medical verifier；
- 原始 MedQA/MedMCQA 只保留为来源追踪和后续 MCQ 对照，不与开放式主数据混合。

单卡 5090 的训练预算冻结为：

| 阶段 | 模型 | updates | prompts/update | rollouts/prompt | 最大搜索轮数 | 目的 |
|---|---:|---:|---:|---:|---:|---|
| 接口 smoke | 3B | 1 | 1 | 2 | 1 | 只验证闭环 |
| 短 pilot | 3B | 20 | 1 | 2 | 2 | 测速度、显存、reward 方差 |
| 主 pilot | 3B | 200 | 1 | 4；OOM 则 2 | 2 | 观察是否在 50–200 step 学习 |
| 迁移 smoke | 7B | 1/20 | 1 | 2 | 1/2 | 验证 7B 可行性 |
| 主 pilot | 7B | 100；有效再延长到 200 | 1 | 2 | 2 | 与 3B 使用同一冻结数据 |

checkpoint 固定在 20/50/100/150/200 step（不存在的阶段跳过）。任何扩大到 500 step 或更多数据的决定，都必须以 200-step 以内的真实 reward、答案正确率、搜索率、显存和墙钟结果为依据。

## 当前结论

RL 数据 **尚未整理完成**。

目前已有：

- `MedSearch/data/raw/medqa/train.parquet`：10,178 条原始候选；
- `MedSearch/data/raw/medmcqa/train.parquet`：182,822 条原始候选；
- `MedSearch/data/stage_b/search100_train.json`：100 条 Stage-B SFT 轨迹，禁止复用于 RL；
- `MedSearch/data/stage_b/stage_b_protocol_dev.json`：12 条协议开发集，禁止进入 RL；
- MIRAGE MedQA official test 与 MedMCQA official dev：最终测评集，禁止进入 RL；
- `MedSearch/data/stage_b/medtext_corpus.jsonl` 与 BM25 index：可直接作为 RL 搜索环境。

缺少的是：

- 冻结的 `MedSearch/data/rl/train.parquet`；
- 冻结的 `MedSearch/data/rl/dev.parquet`；
- 记录来源、过滤、去重、哈希和统计信息的 `manifest.json`；
- 使用医疗数据和 Stage-B adapter 的 Search-R1 一步训练入口。

## 第一版冻结决策

- 数据规模：128 条 train + 32 条 dev；
- 来源平衡：train 中 MedQA/MedMCQA 各 64 条，dev 中各 16 条；
- 固定随机种子：`20260902`；
- 任务类型：四选一、唯一正确答案；
- 答案奖励目标：选项字母 `A/B/C/D`；
- dev 仅用于 RL 开发期检查，不冒充最终 test；
- 冒烟训练从完整冻结文件取 `train_max_samples=1`，不再制造一套重复的 smoke 数据文件。

## 冻结后的 Parquet 契约

每行至少包含：

```python
{
    "id": "medqa:<source-id>",
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "golden_answers": ["B"],
    "data_source": "medsearch_mcq",
    "ability": "medical_mcq",
    "prompt": [{"role": "user", "content": "<Stage-B formatted prompt>"}],
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["B"]},
    },
    "extra_info": {"index": 0, "split": "train"},
}
```

`prompt` 必须调用现有 `format_stage_b_prompt(sample)` 生成，不能在 RL 目录维护第二份模板。

---

### Task 1：先用测试冻结数据契约

**Files:**

- Create: `MedSearch/docs/tests/test_rl_data_builder.py`
- Create: `MedSearch/tools/build_medical_verifiable_rl.py`
- Reuse: `MedSearch/src/medsearch_sft/data.py` 中现有 Stage-B prompt formatter 的实际位置

**Step 1: 写失败测试**

测试只覆盖可独立验证的行为：

- MedQA 与 MedMCQA 原始行都被统一成 A-D 单选样本；
- 缺题干、缺选项、重复选项、非法答案、多答案题被拒绝；
- 明确依赖不可见图片的题目被拒绝；
- `prompt` 只有一个 user message，且等于 `format_stage_b_prompt` 的输出；
- `reward_model.ground_truth.target` 是单个选项字母；
- 相同规范化题干不能同时进入 train/dev；
- Search100、protocol dev、Stage-A user 文本与最终测评池中的精确/近重复题被排除；
- tokenized prompt 不超过 768 tokens；
- 固定 seed 得到确定性相同的 ID 顺序。

**Step 2: 运行测试并确认先失败**

在 WSL 的既有 Docker 容器中运行：

```cmd
wsl.exe docker exec llm-pytorch bash -lc "cd '/workspace/code_list/some tricks/LLMLeanring' && pytest -q MedSearch/docs/tests/test_rl_data_builder.py"
```

预期：因为 builder 尚不存在或函数未实现而失败。不新建第二套环境。

**Step 3: 实现最小 builder**

只实现以下职责：

1. 读取两个原始 train parquet；
2. 统一字段与答案字母；
3. 应用质量过滤；
4. 排除禁用数据池；
5. 先做规范化精确去重，再对候选终选集做 lexical/semantic 近重复检查；
6. 使用真实 Qwen tokenizer 检查 prompt 长度；
7. 按来源分层、确定性抽取 128/32；
8. 写出两个 parquet 与 manifest。

明确的去重阈值沿用 protocol dev：token Jaccard `< 0.80`、本地 MPNet cosine `< 0.92`。先质量过滤和哈希去重，再只对终选候选做语义检查并从 reserve pool 补位，避免对 19 万条数据做全量两两 embedding。

图片依赖过滤只匹配明确的缺失视觉引用，例如 `shown in the image/figure`、`image below/above`、`following image`、`<img`、`.jpg/.png`，不能因为题目讨论医学影像就全部删除。

**Step 4: 运行单测**

```cmd
wsl.exe docker exec llm-pytorch bash -lc "cd '/workspace/code_list/some tricks/LLMLeanring' && pytest -q MedSearch/docs/tests/test_rl_data_builder.py"
```

预期：全部通过。

---

### Task 2：构建并独立验收冻结数据

**Files:**

- Create: `MedSearch/data/rl/train.parquet`
- Create: `MedSearch/data/rl/dev.parquet`
- Create: `MedSearch/data/rl/manifest.json`

**Step 1: 在 Docker 内执行 builder**

入口参数必须显式记录 seed、规模、tokenizer、本地 embedding 模型、输入与输出路径；模型和数据只从本地加载。

**Step 2: 独立读取产物验收**

验收标准：

- train 恰好 128 条、dev 恰好 32 条；
- 两个来源数量均按 64/64 与 16/16 平衡；
- schema 与上面的契约一致；
- ID、规范化 question 在 train/dev 间零重叠；
- 与 Search100、protocol dev、Stage-A train/dev、最终 benchmark 禁用池零越界；
- 每行 gold 都是 A-D，且对应选项存在；
- prompt 长度最大值不超过 768；
- manifest 包含原始文件 SHA256、输出 SHA256、每层过滤计数、答案分布、长度 p50/p95/max、最大近重复分数与最终 ID 列表；
- 人工抽查每个来源各 5 条，确认题干、选项和答案映射无错位。

只有这些条件全部满足，才把“RL 数据已整理好”写入 `STUDY_PROGRESS.md`。

---

### Task 3：验证医疗检索环境，不改检索实现

**Files:**

- Reuse: `Search-R1/search_r1/search/retrieval_server.py`
- Reuse: `MedSearch/data/stage_b/medtext_corpus.jsonl`
- Reuse: `MedSearch/data/stage_b/index/bm25`
- Reuse: `Search-R1/modern_verl/agent_loops.yaml`

**Step 1: 启动现有 BM25 服务**

沿用已验证入口：

```text
python search_r1/search/retrieval_server.py \
  --retriever_name bm25 \
  --index_path <MedSearch BM25 index> \
  --corpus_path <MedSearch medtext_corpus.jsonl> \
  --topk 3
```

**Step 2: 从冻结 train 中取一条真实医学问题发请求**

验收标准：HTTP 200、返回 3 篇文档、响应结构能被当前 `SearchR1AgentLoop` 解析。此处只验证“搜索环境通”，不评价 query 或证据质量。

---

### Task 4：新增医疗 RL 一步冒烟入口

**Files:**

- Create: `Search-R1/experiments/medical_rl_smoke/run_grpo_1step_modern_verl.sh`
- Reuse: `Search-R1/experiments/single_gpu_smoke/run_grpo_1step_modern_verl.sh`
- Reuse: `Search-R1/modern_verl/search_r1_modern/search_agent_loop.py`
- Reuse: `Search-R1/modern_verl/search_r1_modern/reward.py`

**Step 1: 写入口前置检查**

脚本在启动前检查：

- base model `config.json` 存在；
- Stage-B adapter 目录与 `adapter_config.json` 存在；
- RL train/dev parquet 存在；
- veRL 源码目录存在；
- 检索服务 `/retrieve` 可访问。

**Step 2: 从已跑通脚本做最小改动**

仅修改：

- data 指向 `MedSearch/data/rl/train.parquet` 与 `dev.parquet`；
- 设置 `data.train_max_samples=1`、`data.val_max_samples=1`；
- base model 仍为 Qwen2.5-3B-Instruct；
- 设置 `actor_rollout_ref.model.lora_adapter_path=<Stage-B formal adapter>`；
- LoRA rank 与已有 Stage-B adapter 保持 `r=8`，不叠加另一套 r=16 adapter；
- 项目名、实验名与日志路径改为 medical RL smoke；
- 保留已验证的 WSL2/Docker、vLLM、FSDP2、共享内存传输参数；
- 保持 `n=2`、`max_turns=1`、`total_training_steps=1`。

第一轮继续复用现有规则奖励：读取最后一个合法 `<answer>` 并与目标字母匹配。暂不加入证据命中奖励、搜索成本、重复 query 惩罚或多轮搜索。

**Step 3: 对脚本做静态检查**

```text
bash -n Search-R1/experiments/medical_rl_smoke/run_grpo_1step_modern_verl.sh
```

---

### Task 5：执行一次真实的一步 GRPO 冒烟

**Step 1: 保持检索服务运行并启动脚本**

训练必须在 WSL 内 Docker 环境执行，并记录完整日志、W&B run、GPU 峰值和 Git commit。

**Step 2: 验收运行链路**

通过标准：

- 进程 exit code 为 0；
- 日志确认加载的是 Stage-B adapter，而非新建随机 LoRA；
- 两条 rollout 至少完成一次合法的 `<search>query</search>`；
- 检索结果作为 `<information>` 返回 AgentLoop；
- 最终 reward 能按 A-D 正确计算；
- `global_step=1` 完成；
- 记录显存峰值、耗时、两条 reward、是否出现零方差 group。

如果两个 rollout reward 相同，仍可判定“训练接口冒烟通过”，但必须记录“该 step 的 GRPO advantage 为零，不能证明发生了有效学习”；换一个固定样本复测一次即可，不能据此扩大训练规模。

---

### Task 6：记录结论并决定是否进入小规模 RL

**Files:**

- Modify: `STUDY_PROGRESS.md`
- Modify: `MedSearch/README.md`（只补真实数据与运行入口，不写未执行结果）

记录：

- 数据来源、规模、隔离边界、SHA256 与 manifest 路径；
- 一步冒烟命令、日志、W&B run、显存、耗时与 reward；
- 当前结论仅为“医疗数据 → Stage-B adapter → Search → Information → Answer → Reward → 一步更新链路可运行”；
- 下一阶段是否进行 16/32 条小规模 RL，必须等这次真实结果后单独决定。

## 本轮明确不做

- 不使用 Search100 作为 RL 训练数据；
- 不使用 Stage-B protocol dev 或最终测试集训练；
- 不另写一套 prompt、AgentLoop、Retriever 或 reward 框架；
- 不先做 Stage-A/Base/Stage-B 大规模横向测评；
- 不做多轮搜索、证据质量奖励和正式 GRPO；
- 不合并 Stage-B adapter 到 base model；
- 不删除现有 smoke 与测试代码。

## 总体验收门槛

```text
RL 数据契约测试通过
  → 128/32 冻结数据与 manifest 验收通过
  → 医疗 BM25 HTTP 检索通过
  → Stage-B adapter 一步 GRPO exit 0
  → 再决定是否扩大 RL
```
