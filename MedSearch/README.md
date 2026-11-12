# MedSearch-R1

> **当前路线（2026-09-29）：**旧 Stage A/B SFT 已退役，相关编码、配置和专属训练数据已清理。当前冷启动数据入口为 `search_distill` 教师轨迹。下文 Stage A/B 的命令与结果仅是历史实验记录，不可作为当前运行指南；现行实现与任务以 `DISTILLATION_PLAN.md`、`DISTILLATION_ENGINEERING_PLAN.md` 和代码为准。

## 教师角色（每次开始工作前先读）

Codex 在本项目中是**教师、学习督导与技术研究助手**，不是替学生完成项目的外包开发者。

- 教师负责拆分任务、查找官方资料和可靠参考、讲解输入/输出/调用链、审查代码与真实日志；
- 教师负责搭建最小工程骨架，优先复用现有可靠实现，接好目录、依赖、配置、调用链和函数接口；
- 教师在需要学生掌握的关键函数中只保留明确的参数、返回值、契约和待实现位置，由学生根据接口填充核心 SFT/RL 逻辑并成为最终解释者；
- 2026-08-31 补充：后续正式 `main()` 的训练流程、调用顺序和分支由学生亲自编写。
  教师不得以“接线”为由代写完整训练编排；现有 main 只作为已验收的 smoke 入口，暂不重构。
- 后续逐项实现 QLoRA：4-bit 量化配置与加载、低比特训练准备、LoRA 挂载、Trainer 训练与
  保存/重载；学生完成关键适配并在 main 中组织流程，教师提供契约、讲解与隔离测试。
  当前 BF16 LoRA 不是 QLoRA，不能把本次验证记录成量化训练成功。
- 测试由教师全权负责，包括测试设计、测试代码、运行命令、失败定位和回归验证；学生需要理解测试对应的功能边界，但不承担测试代码编写；
- 新增的教学与训练测试统一放在 `MedSearch/docs/tests/`，不与学生需要阅读和填写的主代码并排；历史数据处理测试不为目录整洁而迁移；
- 数据下载、清洗、去重、风险分流和训练集构建是例外，由教师负责并留下可复现证据；
- 除非学生明确要求代为实现，否则教师不得填完预留给学生的核心函数，也不得一次性生成完整训练模块；
- 教师指导必须保留学生的思考空间：优先说明目标、数据流、约束和需要作出的判断，再用问题、
  伪代码或局部 API 提示推进；不得把待实现函数拆成一份按顺序照抄即可完成的答案；
- 医疗检索 Agent 在完成当前 SFT 与 RL 基础后单独建设；
- “代码运行”“训练有效”“指标提升”必须分别验证，所有结论只使用真实实验结果。

完整项目设计、冻结决策和阶段关口见 [PROJECT_BOOK.md](PROJECT_BOOK.md)。

### 主对话与侧边对话分工（长期约定）

学生无需在每次侧边对话中重复粘贴教师职责。进入 MedSearch 相关对话时，Codex 应先按本节
理解对话类型，再决定是否执行工程操作。

- **主对话**负责推进当前唯一任务：读取进度、确定接口、搭建最小骨架、维护隔离测试、运行
  Docker/WSL 验证、检查学生实现、更新 README 与 `STUDY_PROGRESS.md`，并决定何时进入下一步；
- **侧边对话**负责围绕学生选中的代码进行即时辅导：解释 Python/PyTorch/Transformers API、
  参数、返回值、数据流和常见错误，并通过追问帮助学生先形成自己的实现方案；
- 侧边对话默认只解释和只读检查，不替主对话继续实现、运行测试、更新进度或扩展任务；只有
  学生在该侧边对话中明确要求修改某个文件时，才做范围最小的修改；
- 侧边对话默认不提供能够直接粘贴进当前预留接口的完整代码，也不把实现步骤细化成逐行照抄的
  清单；应先给概念提示或伪代码，等待学生提交自己的版本后，再针对真实代码逐点审查；
- 只有学生明确追问某个独立 API 的语法时，侧边对话才给最小、脱离当前完整函数的局部示例；
  示例用于理解 API，不代替学生完成当前函数；
- 学生在侧边对话中只需选中代码并直接提问，例如“这行为什么这样写”或“给我这个函数的
  实现思路和 API 用法”，不必再次说明上述分工。

## 整体训练架构（2026-08-29 冻结）

```text
Qwen2.5-3B-Instruct
  -> Stage-A 医学对话 SFT
  -> Checkpoint A: Medical-SFT
  -> Stage-B Search100（是否加入 Stage-A replay 待实验决定）
  -> Checkpoint B: Medical-SFT+Search100
  -> Search-R1 在线检索环境 + LoRA-GRPO
  -> Checkpoint C: MedSearch-R1
```

Base、Checkpoint A、Checkpoint B 和 Checkpoint C 都必须保留并在同一冻结测试边界上评测，
分别衡量医学 SFT、搜索协议冷启动和 GRPO 带来的变化，不能只比较最终 checkpoint。

### RL 数据来源与候选池（2026-09-04）

首轮 RL 使用 Hugging Face 的
[`FreedomIntelligence/medical-o1-verifiable-problem`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem)，
对应论文 [HuatuoGPT-o1](https://arxiv.org/abs/2412.18925)，许可为 Apache-2.0。官方数据只有一个
train split，共 40,644 条英文开放式医学题；原始字段为 `Open-ended Verifiable Question` 和
`Ground-True Answer`，不含证据文档、难度或原始考试来源字段，因此本项目不得补造这些元数据。

本地原文件固定到 revision `0b347655bd332537e6814d53c6fe4bc03b555bdc`，SHA256 为
`f5998431acb0eb35e9365f81b6e434d67c44392a5667280068c63ec072ba53cd`，与该 revision 的官方文件
一致。派生记录在 `extra_info` 中保存来源数据集、revision、官方 train split 和原始行号；完整
来源、许可、字段、seed 与派生文件哈希见 `data/rl/medical_verifiable/manifest.json`。

当前候选池使用 seed `20260902`，包含 2,048 条 train 和 256 条 dev。构建时排除 Stage-A、
Search100、protocol dev、MIRAGE MedQA/MedMCQA 最终测试池中的精确规范化重复题，并在原始数据
内部精确去重；当前结果排除已见题 1,156 条、内部重复 342 条。`audit.json` 记录完整性和长度
统计，`manual_review.jsonl` 固定抽取 train 中全部答案长度不超过 2 或超过 200 字符的样本，再
按相同 seed 补足到 64 条供人工审核。答案长度只用于风险抽样，不作为自动删除规则。

这批数据目前仍是 candidate V1。使用本地 Qwen2.5-3B tokenizer 的检查中，train/dev prompt
最大长度分别为 262/265 tokens，超过 4,096 的样本均为 0；答案最大长度分别为 86/53 tokens。
token-set Jaccard 阈值 0.80 共发现 7 条近重复，人工核对后全部判定排除，其中 2 条命中最终
测试池、5 条命中 Stage-A。裁决分别保存在 `lexical_near_duplicates.jsonl` 和
`lexical_review_findings.jsonl`，尚未从 candidate V1 中直接删除，待 MPNet 审计结束后统一生成
版本化 V2。

语义检查复用 `MedicalGraphRAG/models/all-mpnet-base-v2`，不下载或复制第二份模型；模型权重
SHA256 为 `a8fd120b1a0032e70ff3d4b8ab8e46a6d01c2cb08ffe7c007a021c1788928146`。在归一化 embedding
cosine 阈值 `0.92` 下共命中 49 条：Stage-A 30 条、Search100 1 条、protocol dev 0 条、
MIRAGE 最终测试 18 条。逐条查看均为同题改写，报告见 `semantic_audit.json` 与
`semantic_near_duplicates.jsonl`；此前 7 条 Jaccard 命中包含在本轮语义风险中。

64 条人工队列中已优先核查 8 条高风险样本，确认存在历史指南答案错误、维生素答案错误、
折射率/屈光力混淆、妊娠期放射性碘场景错误以及缺失图片、表格、X 光或血气数值等问题；结果见
`manual_review_findings.jsonl`。其余 56 条医学正确性仍待审核，不能用这 8 条定向问题样本计算
总体错误率。其余人工审核与 no-search/BM25 搜索收益诊断尚未完成。在这些门槛完成前，
candidate V1 只能用于最小 RL 链路 smoke，不能作为正式训练数据或效果结论依据。

V2 的目标规模仍保持 2,048 train + 256 dev：按各 split 删除已裁决问题后，从 40,644 条原始
reserve pool 依次补位，每个补位样本重新经过精确、Jaccard、MPNet、长度和质量门槛。不能为了
保持原 optimizer-step 计划而保留污染样本；如果 reserve pool 无法提供足量合格样本，则缩小
数据集并重新计算训练步数，质量优先于固定数量。

候选 V2 位于 `data/rl/medical_verifiable_v2/`。构建以 V1 的 train/dev 原始行号为锚：保留
全部未被拒绝的原 split 成员，只从同一 seed 产生的固定 reserve 顺序补空位；不得在删除后重新
shuffle 全部候选。当前累计排除 74 条：V1 的 49 条语义重合与 14 条人工确认问题，以及 reserve
补位中的 3 条语义重合与 8 条人工确认问题。最终 train/dev 仍为 2,048/256，相对 V1 的
train/dev 分别替换 60/3 条，共形成 63 条当前补位人工队列，没有跨 split 移动。最终 MPNet
复审为 0 命中；
`manifest.json` 保存 V1 锚点 hash、五份裁决来源
hash、规范化 `applied_exclusions.jsonl` hash 与最终 parquet hash。此前“先删再 shuffle”产生的
全量漂移版本已判无效并覆盖，不计入实验结果。

V2 目前仍是正式训练候选而非完全冻结数据：自动隔离、长度和语义门槛已通过。网页端已完整
审核当前 63 条补位队列，裁决为 keep 39、reject 24，且 ID、字段和状态一致性检查通过；完整
报告与 24 条 rejects-only findings 保存在 V2 目录。项目决定不再为这 24 条补位，待最后一批
V1 完整 64 条审核返回后一次性过滤，因此预计最终规模不高于 2,280 条，并按实际 train 数重算
optimizer steps。最后一批网页输入为 `data/rl/medical_verifiable_v2/final_web_review_64.jsonl`；
它故意保留完整 64 条，不预先扣除本地已裁决项，便于网页端独立审核。

若要独立审核其余候选，使用 `data/rl/medical_verifiable_v2/remaining_web_review_batches/`：当前
V2 共 2,304 条，63 条补位审核记录全部仍在 V2，V1 的 64 条队列中有 49 条仍在 V2、15 条已被
此前裁决或语义隔离删除，因此当前已覆盖 112 条，剩余 2,192 条。目录包含一份完整 master 和
22 个按当前 train→dev 稳定顺序切分的批次；前 21 批各 100 条，最后一批 92 条。建议网页端
每个新对话只审核一个 batch，并原样保留 ID、review_status 与四个布尔审核字段，避免单对话
一次处理两千余条造成漏项。

V1 完整 64 条网页复核已回收：35 keep、29 reject，64/64 ID 完整且审核字段一致。其中 14 条
reject 已在当前 V2 之外，15 条会在最终合并时新增删除；另有 1 条网页判 keep，但它因先前语义
隔离而继续排除。完整报告为 `final_web_review_64_audited.jsonl`，rejects-only 文件为
`v1_web_review_rejects.jsonl`。当前 256 条 dev 已另行导出为 `dev_web_review_256.jsonl`，通用标准见
`WEB_REVIEW_STANDARD.md`。dev 中 253 条也存在于 2,192 条未审 master，3 条已在补位队列审过；
最终合并一律按 ID 去重，重复审核不会重复计数。

2,192 条剩余批次的第一轮网页审核现已全部回收并归档到
`data/rl/medical_verifiable_v2/quality_audit/`。与此前 112 条审核合并后，桌面原判为
1,700 keep / 604 reject；二次审计恢复了 specificity 计算唯一明确的 `22697`，并清理 10 条
重复、答案冲突或明显依赖旧处置口径的通过项。当前临时候选为 1,691 条（train 1,494、dev
197），清洗清单为 613 条；完整规则、覆盖关系和 override ID 见 `quality_audit/DECISION.md`、
`summary.json` 与两份 candidate JSONL。该版本仍等待独立的 256-dev 审核，不覆盖原 parquet。

后续实验使用独立的 `data/rl/train.parquet`（3,283 条）和 `data/rl/dev.parquet`（396 条）：
二者均为质量审核后的 veRL 八列 schema，ID 不重叠。2026-09-09 已将 1,789 条 audited
text-only expansion train 与 199 条 dev 合并进正式数据；合并前 parquet 已备份到
`data/rl/pre_expansion_backup_20260909/`，记录见 `data/rl/MERGE_MANIFEST_20260909.json`。
来源、去重、隔离、审查和使用边界见 `data/rl/MEDICAL_VERIFIABLE_RL_CLEANING.md` 以及
`data/rl/medical_verifiable_expansion_v1/MEDICAL_AUDIT.md`。train 只用于 GRPO，dev 只用于
开发期诊断；最终评测仍仅使用冻结的 MIRAGE、MedQA、MedMCQA 测试集。

检索需求与医学质量分开管理：已确认 `moderate_or_high` 467 条、`low` 205 条、尚未进行检索需求
复审 1,019 条。low 样本保留为 no-search/control 数据资产，不作为质量 reject；unknown 样本也
不冒充 search-positive。首轮 GRPO 已决定直接使用全部 1,494 条质量合格 train；467 条严格搜索
子集、no-search/BM25 收益诊断和检索需求分层均后移为消融实验，不再阻塞首轮训练。扩充合并后不
立即继续加入更多题目，先用 20–50 steps 检查 reward 方差、有效组比例和 dev-32。首轮结果只
回答完整训练闭环和 outcome reward 是否有效，不能据此声称模型已经学会最优搜索或停止策略。

### Stage-A dev 评测与人工抽样

`sft/evaluate_stage_a.py` 只读取冻结的 328 条 Stage-A dev，不读取 MedQA/MedMCQA 最终测试集。
它在完整 dev 上计算只覆盖 assistant labels 的 token-weighted NLL/PPL，并按 single、multi、
reasoning、general 分组；随后每组固定抽样并生成：

- `metrics.json`：dev loss/PPL 与 `<think>`/`<answer>` 标签率；
- `generations.jsonl`：问题、参考回答、模型回答和自动标签检查；
- `review.md`：留有医学正确性、相关性、幻觉、安全风险与标签行为栏的人工审阅表。

训练结束且 GPU 空闲后，在仓库根目录 CMD 中分别运行 Base 和 Stage-A adapter；除 variant、
adapter 与输出目录外，两条命令的 seed、抽样数和生成参数必须相同：

```cmd
wsl.exe docker exec -w "/workspace/code_list/some tricks/LLMLeanring" -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 llm-pytorch python -m MedSearch.sft.evaluate_stage_a --config MedSearch/configs/stage_a.yaml --data MedSearch/data/processed/sft_stage_a/stage_a_dev.jsonl --variant base --output-dir MedSearch/outputs/eval/base --per-group 8 --seed 20260823
wsl.exe docker exec -w "/workspace/code_list/some tricks/LLMLeanring" -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 llm-pytorch python -m MedSearch.sft.evaluate_stage_a --config MedSearch/configs/stage_a.yaml --data MedSearch/data/processed/sft_stage_a/stage_a_dev.jsonl --variant stage-a-final --adapter-path MedSearch/outputs/stage_a/adapter --output-dir MedSearch/outputs/eval/stage-a-final --per-group 8 --seed 20260823
```

默认 greedy 生成衡量跨题标签成功率。若要检查同一道题在随机生成下是否稳定，给 Base 与 SFT
都使用 `--repeats 3 --temperature 0.7 --top-p 0.9` 并写入新的输出目录。报告同时区分：

- `tag_signature_consistency_rate`：多次输出的标签形态是否一致，一致地犯错也会计入；
- `stable_expected_behavior_rate`：reasoning 组每次都严格输出完整 think+answer，普通组每次都不
  非预期输出 think，只有这一项才表示稳定符合当前 Stage-A 数据协议。
- `max_token_truncation_rate`：生成撞到 token 上限且未以 EOS 结束的比例；这类样本不能直接归因
  为标签闭合能力失败，必须在人工抽样时单独查看。

### 训练中 dev loss 接口（已接通）

Stage-A 训练中的 validation 固定使用 `data/processed/sft_stage_a/stage_a_dev.jsonl`，不得读取
MedQA/MedMCQA 最终测试资产。教师已给 `build_dataloader(..., split="train|eval")` 与
`build_trainer(..., eval_dataloader=...)` 增加接口契约；学生只需完成路径分派、eval 禁止 shuffle、
TrainingArguments 原生 eval 字段透传、Trainer eval_dataset 接线及 main 编排。不新增 Dataset、
callback 或手写 evaluation loop。

`docs/tests/test_sft_eval_wiring.py` 的五项契约均已转绿，当前完整 SFT 回归为
`67 passed, 18 warnings`。Stage-A 使用 step evaluation；首轮正式 Stage-B 只在 epoch 末评估
一次，避免 100 条训练数据每一步都重复遍历 328 条 dev。这只记录 validation loss；最终医学
能力仍由冻结测试集 Accuracy 验收。

Stage-B 不用 Search100 轨迹做 token-level validation loss。它通过 `eval_stage: stage_a` 复用
同一份 328 条 Stage-A dev，只监控医学对话能力是否遗忘；该 loss 不用于判断 search query
是否优质，也不用于选择 Stage-B checkpoint。搜索协议能力由独立 protocol dev 的合法 action、
Evidence Hit@k、闭环完成率与最终答案准确率评测。

2026-08-31 调整：暂缓 Stage-B replay 数据组合，不再将其作为接入 Trainer 的前置条件。
当前 Stage-B 入口保持只读取 Search100；这不意味着已证明“不混合更好”。Stage-A 已冻结的
65/20/10/5 数据配方不变，本次不重新清洗、合并或导出任何数据。

### 首轮正式 Stage-B 与 protocol dev（2026-09-01）

首轮正式配置为 `configs/stage_b_formal.yaml`：Search100 共 100 条、batch size `2`、梯度累积
`4`、max length `4096`、1 epoch。100 条对应 50 个 micro-batch 和 13 个 optimizer step，
不是只训练了 13 条。真实结果为 train loss `1.5409584`、runtime `912.2916s`；epoch 末在
328 条 Stage-A dev 上得到 Trainer eval loss `0.9218875`。W&B run ID 为 `krv3cmlr`。

`data/stage_b/stage_b_protocol_dev.json` 冻结 12 条未进入 Search100 的 official-train 题：8 条
一轮搜索、4 条两轮搜索，MedQA/MedMCQA 各 6 条。构建时排除了 Search100、30,073 个
Stage-A user turn 和 MIRAGE 的 5,456 条 MedQA/MedMCQA 最终测试题；本地 MPNet 语义相似度
拒绝阈值为 `0.92`，实际最高值为 `0.781795`。这些 ID 同时预留为未来医学 RL 排除项。
该集合只用于开发期行为验收，不计算 token-level loss，也不作为最终 test。

### Stage-B 首轮协议门槛（已接通）

`sft/evaluate_stage_b.py` 只生成每条 protocol dev 的第一轮，首次出现完整
`</search>` 或 `</answer>` 即停止。它不调用 BM25、不注入 `<information>`、不评判 query
相关性；用途是确认 RL 之前的协议冷启动，而不是提前做 Agent 评测。

2026-09-02 使用相同 12 条、greedy、seed `20260823`、`max_new_tokens=256` 的真实结果：

| checkpoint | 合法 search | 非空 query | 直接 answer | 畸形 action |
|---|---:|---:|---:|---:|
| Base | 1/12 | 1/12 | 3/12 | 8/12 |
| Stage-A | 0/12 | 0/12 | 0/12 | 12/12 |
| Stage-B | 12/12 | 12/12 | 0/12 | 0/12 |

同一轮还以 `evaluate_stage_a.py` 在冻结的 328 条 Stage-A dev 上检查保留能力：Base、Stage-A、
Stage-B 的 overall NLL 分别为 `1.3803`、`1.0014`、`1.0384`。Stage-B 相比 Stage-A 有约
`3.7%` 的 NLL 回升，尚非灾难性遗忘；32 条固定普通/推理生成中没有 `<search>` 泄漏，但普通
对话非预期 `<think>` 为 `4/24`。因此当前仅通过“进入小规模 RL smoke”的协议门槛，尚不能以
此宣称无过拟合或批准正式 RL 实验。固定随机采样的 protocol dev 稳定性检查保留为后续环境
接入后的回归项，不阻塞当前 RL smoke；正式 RL 前仍须预先写明 Stage-A dev 的可接受退化阈值。

### 职责边界

- **MedSearch**：拥有 Stage-A/Stage-B 数据语义、Search-R1 序列格式、loss mask、训练配置和统一评测；
- **MedicalGPT**：提供已经验证过的 Qwen 模型加载、LoRA/QLoRA、Trainer、checkpoint 和日志能力；
- **Search-R1/veRL**：在后续 RL 阶段提供在线搜索循环、rollout、reward 和 LoRA-GRPO；
- 当前不建设名为 `agent/` 的模块；医疗检索 Agent 是完成 SFT 与 RL 基础后的后续工作。

### RL 引擎策略（2026-09-04）

决定采用锁定版本的原生新版 veRL，旧 `trainer.use_v1=False` 调度只保留为历史参考，不再作为 MedSearch-R1 训练路线。实现使用一个薄的 `MedSearchAgentLoop`，只迁移已验证的搜索协议语义，不搬运旧版 trainer、rollout 或 WSL 专用兼容层。

不变量：策略模型原生生成搜索动作；模型生成 token 的 `response_mask=1`，工具返回 token 的 `response_mask=0`；数据、BM25 语料、奖励和评测保持一致。

本次医学 parquet 在旧路径上的实测表明：1,494 条 train 与 197 条 dev 可被 veRL 正确加载，但异步 vLLM engine 在同卡 FSDP actor 共存时无法分配任何 KV cache block；将显存比例从 0.25 提至 0.35 并缩短上下文后仍失败。故该路径不再投入兼容性修复；日志保存在 `docs/run_logs/modern_verl_fullpool_1step*.log`，不得将此前 23,117 MiB 的旧 smoke 写成医学数据训练成功。

#### 原生新版 veRL 实施计划与隐藏风险

采用最小扩展路线：veRL 作为只读依赖，MedSearch 只实现搜索循环、检索客户端、协议和奖励；不
fork trainer，不把项目代码写进 veRL 源码，也不为单个实现增加 factory 或抽象基类。

2026-09-05 已从官方仓库 `https://github.com/verl-project/verl.git` 拉取新版 `main`，并固定为
commit `23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`。本地目录是 `MedSearch/.deps/verl`，由根
`.gitignore` 排除。学习与扩展以官方 [Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、
[扩展指南](https://verl.readthedocs.io/en/latest/extend_guide.html) 和
[安装说明](https://verl.readthedocs.io/en/latest/start/install.html) 为准；Search-R1 旧 fork 只用于理解
BM25 服务语义，不作为新版 veRL API 的依据。

P6 的 V1 教师评分接线以独立补丁提交。固定 commit 上的应用命令、已验证范围和剩余训练验收见
[V1 教师评分接线验收](docs/audits/v1_teacher_step_acceptance_20261109.md)。

计划文件边界：

```text
MedSearch/
├─ .deps/verl/                    # 干净的固定 commit，只作本地依赖，不提交
├─ rl/
│  ├─ protocol.py                # 严格解析 search/answer，一次只接受一个 action
│  ├─ retrieval.py               # Search-R1 /retrieve 客户端与响应校验
│  ├─ medsearch_agent_loop.py    # 原生 AgentLoopBase 多轮循环
│  └─ reward.py                  # 格式与医学答案 outcome reward
├─ configs/rl_3b_single_gpu.yaml # 唯一单卡配方
└─ tests/
   ├─ test_rl_protocol.py
   ├─ test_rl_retrieval.py
   ├─ test_rl_agent_loop.py
   └─ test_rl_reward.py
```

必须先解决的隐藏风险：

1. **干净依赖。** 现有 `Search-R1/.deps/verl` 位于旧 commit 且工作树已有本地补丁，不能复制后
   宣称为原生版本。新版官方 checkout 已固定到上述 commit，当前工作树必须保持 clean；升级 commit
   必须作为单独变量实验。
2. **V1 依赖门槛。** 当前新版默认 `trainer.use_v1=True`，会直接导入并初始化 TransferQueue。写
   AgentLoop 前必须先验证 `import transfer_queue`、V1 配置展开和 upstream single-turn dry-run；只启用
   单机 `SimpleStorage`，不安装或配置 Mooncake/RDMA。
3. **环境锁定。** Git commit 不能替代运行环境。该 checkout 锁定 PyTorch 2.11.0 与 vLLM 0.24.0，
   现有 `llm-pytorch` 则是 Python 3.12.3、PyTorch 2.13.0、vLLM 0.27.1、veRL 0.10.0.dev0。
   考虑磁盘约束，2026-09-05 决定首轮不运行完整 `uv sync`，避免重复安装
   PyTorch 与 vLLM；改用 `VERL_USE_UV=0` 复用现有大包，并以 `PYTHONPATH` 指向新版 veRL 源码。
   仅按官方 `uv.lock` 的精确 commit 安装 TransferQueue `0.1.9.dev0`，未改动 PyTorch
   `2.13.0+cu130` 或 vLLM `0.27.1`。这是有意的兼容性试验，不得宣称为官方锁定环境；若后续出现
   明确版本 API 冲突，再决定是否付出额外磁盘建立完整 uv 环境。实测 veRL 官方为 PyTorch 2.11
   构建的 `flash-attn==2.8.3` wheel 在 PyTorch 2.13 下导入即出现缺失 C++ 符号，已卸载；该结果是
   ABI 不兼容，不应通过忽略依赖或继续启动训练绕过。后续须在当前 PyTorch/CUDA 上源码编译并通过
   import、SM120 前向和反向三项验证，或让 actor 明确使用 SDPA。当前已选择后者：actor 使用
   `sdpa` 且 `use_remove_padding=false`，现有环境不再安装 FlashAttention；只有该路径不可用或实测
   性能成为瓶颈时，才另建完整 uv 环境。
4. **WSL 权重同步边界。** SDPA 已能完成 Qwen2.5-3B/FSDP actor 初始化。vLLM 0.27.1 在当前 WSL
   环境还需设置 `VLLM_USE_V2_MODEL_RUNNER=0` 以避开要求 UVA 的 V2 runner；随后服务可完整启动，
   但 CUDA IPC 权重同步实测报 `invalid resource handle`。veRL 已有 shared-memory 权重传输实现，后续
   兼容项使用显式 opt-in 的 `VERL_FORCE_SHM_WEIGHT_TRANSFER=1`：默认 native Linux 行为不变，开启后
   复用上游已有 SHM transport。测试 `14 passed`，真实 smoke 已越过权重同步并进入 rollout/actor
   update；后续 OOM 来自 smoke 未启用 LoRA 而创建全参数 Adam state，不应归因于 SHM，也不应通过
   盲目降低 batch/rollout 参数掩盖。可复现补丁脚本为 `scripts/patch_verl_for_wsl.py`，传入固定的
   veRL checkout 根目录即可；重复执行是幂等的。
   Stage-B LoRA 的一优化步 smoke 已通过，固定脚本为 `scripts/run_verl_v1_lora_smoke.sh`，完整日志
   位于 `outputs/rl_smoke/run.log`。该次两条 rollout reward 相同且均为 0，所以零 advantage/grad
   不能作为学习效果结论；它只验收运行闭环。
5. **Checkpoint-B 必须是真实初始策略。** Stage-B adapter 是 rank 8、alpha 16、七类 Qwen linear
   modules；RL 不得悄悄新建 rank 16 adapter。actor 和 rollout 在第 0 步的输出必须来自
   Base + Checkpoint-B，并通过固定 prompt 的 logits/生成对照验证。
6. **KL reference 不能错。** veRL 的 LoRA 优化可能把“关闭 adapter 的 Base”当 reference；但我们的
   初始策略是 Checkpoint-B。若不能证明 reference 等于冻结的 Base + Checkpoint-B，首轮配置将 KL
   关闭，而不是对错误的 Base reference 计算 KL；后续再单独实现或验证正确 reference。
7. **token-in/token-out。** vLLM 返回的 `token_ids` 原样进入 `ct_merge_assistant_token()`；decode 只
   用于解析和日志，不能截取文本后重新 encode。工具 observation 通过 veRL continuous-token helper
   合并，模型 token mask 为 1，环境 token mask 为 0。
8. **协议必须严格。** 空 query、双 action、标签未闭合、answer 后继续输出均标记为 malformed；
   observation 截断时必须先为正文留预算并完整保留 `</information>`，不能从整体右侧硬切。
9. **环境错误不等于策略错误。** BM25 timeout、HTTP 错误和 schema 错误记录
   `environment_failed=1`，不能直接作为模型负奖励；该 group 从有效学习信号统计中单列。非法 action
   才属于策略错误。
10. **BM25 只复用服务边界。** 第一轮可用 Search-R1 Wikipedia BM25 验证 HTTP 闭环；正式医学结果
   必须切到 MedRAG 医学语料和对应索引。BM25 独立启动并先过 health check，AgentLoop 不负责偷偷
   拉起后台进程。
11. **GRPO 的 batch 概念分开。** `train_batch_size` 是每步问题组数量，`rollout.n` 是每题轨迹数，
    actor micro/mini batch 决定反向显存；不能把 SFT 的 gradient accumulation 直接套用。首个 GPU
    gate 固定 `train_batch_size=1`、`rollout.n=2`，完成后才单变量增加到 4。
12. **零方差组必须记录。** 两条 rollout 同奖时 GRPO advantage 为 0，这是有效数据统计而非程序
    故障。每次实验记录 mixed-reward group 比例、全零/全一组比例、grad norm 和有效更新组数量。
13. **dev 分两级。** 396 条 dev 是开发评测，不是最终测试；频繁验证先固定抽取 32 条，396 条只在
    milestone 跑，冻结 MIRAGE MedQA/MedMCQA 继续承担最终报告，避免每个短实验都产生昂贵 rollout。
14. **保存与恢复。** 正式 pilot 前必须验证一次 adapter-only checkpoint 保存、optimizer/global step
    恢复和重载后输出一致；否则 3,283 条训练不能启动。

### DAPO V1 outcome reward（2026-09-08 冻结）

- 优化配方使用 DAPO-style：GRPO advantage + Dynamic Sampling + Clip-Higher
  (`0.2/0.28`) + `token-mean`；Overlong Reward Shaping 暂时关闭，只记录长度与截断率。
- outcome reward 完全复用 Search-R1 normalized exact match：从 response 取最后一个完整闭合的
  `<answer>...</answer>`，规范化后与任一 gold alias 完整相等返回 `acc=1`，其余返回 `acc=0`；
  `outcome_reward=2*acc-1`，即正确 `+1`、错误 `-1`；训练 `score=outcome_reward`，DAPO group
  filtering 仍读取 `acc`，不另设 `is_correct()` 或语义 verifier。
- 未闭合、空 answer、没有 answer 或截断导致闭标签缺失均为 `0`，不猜测、不修复、不提供格式分。
- 训练 prompt 沿用 Search-R1 的闭合格式说明和 `<answer> Beijing </answer>` 示例；动作 parser 仍只
  接受恰好一个完整闭合、非空且动作后无额外文本的 search/answer。
- DAPO group filtering 只读取 `acc`。格式、非法调用、搜索次数、重复查询、证据和成本在 V1 只作为
  metric；环境失败单列，不当作模型错误。
- 答案解析与 normalization 参考 `Search-R1/modern_verl/search_r1_modern/reward.py`；自定义函数返回
  `{"score": outcome_reward, "acc": acc}`。V1 不增加 observation 扫描或转义设计，只有真实
  rollout 出现协议标签污染时再处理。

按以下顺序教学和实现，每个门槛只推进一个变量：

开始编写 MedSearch 代码前，必须显式完成以下五个学习门槛：

1. 看懂 veRL 的五个角色：trainer、actor、rollout、reward、reference，并能说明各自保存或计算什么。
2. 看懂一次 GRPO 数据流：一个问题如何生成同组多条 rollout、计算组内相对 advantage，再更新 actor。
3. 用官方配置启动单轮最小任务，理解配置文件、Hydra 命令行覆盖和最终展开配置之间的关系。
4. 看懂 `AgentLoopBase.run()` 的输入、`AgentLoopOutput` 的输出，以及 `response_mask` 为什么只训练模型
   生成 token、不训练工具 observation token。
5. 上述门槛通过后，由用户从测试开始编写 `MedSearch/rl/protocol.py`，再进入 BM25 与 AgentLoop。

学习方法固定为“文档—源码—最小验证”三步：官方文档用于理解公开契约；只读官方
`single_turn_agent_loop.py`、`agent_loop.py` 和最小 GRPO 配置确认真实调用；最后用一条数据或一个小测试
验证理解。暂不通读完整 trainer，不以 Search-R1 旧 fork 代替新版官方文档，也不在阅读阶段改源码。

- [ ] **任务 1：建立干净依赖门槛。** 获取固定 veRL commit，记录完整版本矩阵；运行
  `python -c "import verl, transfer_queue; print(verl.__file__)"`，输出必须指向
  `MedSearch/.deps/verl`，且 upstream V1 配置可展开。
- [x] **任务 2：先跑 upstream 原生单轮。** 不加载 MedSearch AgentLoop 代码、不调用 BM25，用 1 条数据确认
  V1 trainer、TransferQueue、FSDP2 与 rollout engine 能在单卡初始化。该门槛失败时只修环境，不写
  AgentLoop。
- [ ] **任务 3：TDD 实现协议。** 先在 `test_rl_protocol.py` 写 search、answer、空 query、双 action、
  未闭合标签五类失败/成功断言，再实现 `rl/protocol.py`；命令
  `pytest MedSearch/tests/test_rl_protocol.py -q` 必须全通过。
- [ ] **任务 4：TDD 接入 BM25。** 先用本地假 HTTP 响应覆盖成功、timeout 和坏 schema，再实现
  `rl/retrieval.py`；测试不得加载真实 14 GB corpus。通过后另跑一次真实 `/retrieve` health check。
- [ ] **任务 5：TDD 实现 MedSearchAgentLoop。** 用 fake server manager 固定返回 token IDs，验证
  search→information→answer 顺序、原始 token 保留、mask、轮数上限和 environment failure；随后
  才实现最小 `run()`。
- [ ] **任务 6：接入最小 reward。** 复用 modern Search-R1 response-only normalized EM 计算
  `acc=0|1`，再计算 `outcome_reward=2*acc-1`；返回 `score=outcome_reward` 与 `acc`。闭合命中、
  闭合未命中、未闭合、空 answer、多个 answer 取最后一个五类测试通过。格式、非法 action、证据、
  搜索成本和置信度只记录，不进入 V1 reward。
- [ ] **任务 7：Checkpoint-B rollout-only。** 固定 1 条题、2 条 rollout，不反向；证明 actor/vLLM
  均加载 rank-8 Checkpoint-B，至少一条真实 BM25 调用能形成完整闭合轨迹。
- [ ] **任务 8：单步 GRPO。** 仅在前七项通过后运行 1 optimizer step；验收 global step、reward
  分量、response mask、advantage、grad norm、显存峰值及环境失败数，不以“进程退出 0”代替验收。
- [ ] **任务 9：逐级放大。** 固定配置依次运行 20 steps、50 steps、200 steps，再运行完整 3,283 条；
  每阶段先比较 dev-32，只有 milestone 才跑 dev-396。`rollout.n=4` 是单独实验，不能与其他参数同改。

明确延后：Evidence reward、搜索成本、置信度、4 轮搜索、Dense/Hybrid/LinearRAG、7B 模型以及
自动检索器路由。只有 3B + BM25 + 2 rollout 的原生 V1 链路稳定且产生真实有效组后再讨论。

### SFT 数据流

不引入带 `trainable` 字段的统一中间结构，也不为两种数据强建抽象基类。第一版保留两个
直接编码函数：

```text
Stage-A sample -> encode_stage_a() -> input_ids / attention_mask / labels
Stage-B sample -> encode_stage_b() -> input_ids / attention_mask / labels
```

两种数据共用一个 `sft/train.py` 训练入口，通过 `stage_a.yaml` 与 `stage_b.yaml` 切换数据、
初始 checkpoint 和训练参数。Stage-B 必须从 Checkpoint A 初始化，Stage-A 从 Base 初始化。

统一 loss 边界如下：

| 内容 | 参与 attention | 参与 SFT loss |
|---|---:|---:|
| system / user / question / options | 是 | 否，labels 为 `-100` |
| Stage-A assistant 回复 | 是 | 是 |
| Stage-B `<think>` / `<search>` / `<answer>` | 是 | 是 |
| 环境 `<information>` | 是 | 否，labels 为 `-100` |
| padding | 否 | 否，labels 为 `-100` |

### Stage-B 序列必须严格模拟 Search-R1

Stage-B 不是普通的多轮 ChatML 对话。Qwen system/user prompt 只在开头渲染一次，后续搜索、
环境 observation 和最终答案组成同一条滚动上下文：

冻结 JSON 继续分别保存 `question` 与结构化 `options`；教师维护的
`format_stage_b_prompt(sample)` 在运行时将 Search-R1 指令、题干和 A/B/C/D 选项拼成唯一的
user prompt。SFT 和后续 RL 必须复用该函数，不修改冻结 JSON，也不从 prompt 反向解析选项。

```text
Qwen system/user chat prompt
<|im_start|>assistant
<think>...</think>
<search>...</search>

<information>...</information>

<think>...</think>
<answer>B</answer>
<|im_end|>
```

必须满足：

- `<information>` 原样插入上下文，不包装成新的 Qwen `user` 消息；
- `</search>` 后不插入 EOS，因为在线 Search-R1 还会继续注入检索结果；
- `<information>` token 全部使用 `-100` label，但仍参与 attention；
- 最终 `</answer>` 后才结束整个序列；
- Stage-B SFT 读取 Search100 中冻结的 observation，不在训练数据加载时实时执行检索。
- Stage-A 使用 `max_length=1024`，Stage-B 使用 `max_length=4096`；超长样本报错，不静默截断。

### 规划目录

以下是职责规划，不要求当前一次性创建全部目录和空文件：

```text
MedSearch/
├── data/                    # 已冻结的 Stage-A、Search100、评测资产和检索资产
├── docs/tests/              # 教师维护的隔离测试，学生无需修改
├── sft/
│   ├── data.py              # 读取 Stage-A / Stage-B，保留两个直接编码入口
│   ├── preprocess.py        # Qwen2.5 渲染、tokenization 和 loss mask
│   └── train.py             # Stage-A / Stage-B 共用的唯一 SFT 入口
├── rl/                      # 到 GRPO 阶段再创建，不提前搭空框架
│   ├── search_loop.py       # Search-R1 在线滚动搜索环境
│   ├── reward.py            # 答案、格式、重复搜索和成本奖励
│   └── train.py             # GRPO 入口
├── evaluation/              # Base / A / B / C 的统一评测
└── configs/
    ├── stage_a.yaml
    ├── stage_b.yaml
    └── grpo.yaml
```

当前只实现 `sft/` 所需的最小文件；`rl/`、`evaluation/` 和医疗检索 Agent 均按阶段关口延后。

## 当前唯一任务

**2026-09-04：收拢 Medical Verifiable Problems V2 全量质量审核，并等待 dev 独立复核。**

当前不启动正式 RL。先核对 `quality_audit/quality_candidate_keep.jsonl` 的 1,691 条候选及独立
256-dev 审核意见；随后冻结质量集，并对 retrieval-unknown reserve 决定是否补做检索需求标签。

以下保留此前已验收实现与历史命令。

实测固定8条Stage-A、global_step1、loss1.1438109874725342、grad_norm0.8728765249252319；
252个LoRA张量更新，434个基座参数无梯度且无非预期数值变化。整卡采样峰值15,285MiB（含背景）。
命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --qlora`。
[完整单步报告](docs/run_logs/3b-smoke-SQf36DFi/SUMMARY.md)及产物outputs/smoke/stage_a-_y_9hw2d/。
最新CPU回归48 passed、16 warnings（16.11s）。未保存adapter，不构成正式Checkpoint A。
允许进入代码整合讨论：main保留配置、数据/模型装配、Trainer训练与保存编排；教师诊断留docs。
不为减少行数堆叠新封装，不删除运行元数据/测试；本轮没有删除或迁移代码。

当前入口复查已通过：学生已修正attach_lora返回缩进；dtype映射、prepare_qlora_model调用及
普通LoRA兼容均通过CPU检查。最新回归48 passed、16 warnings（19.09s），无xfail。
上述CPU入口检查之后，真实QLoRA Trainer单步也已通过，详见本节最新报告。

参数骨架已补充：`--qlora` 使用 argparse 的 store_true；不传时 args.qlora=False，传入时为True。
学生现已接入模型分支并通过真实Trainer单步；参数解析本身与训练验收分别留有记录。
教师未代写训练编排。参数测试在docs/tests/test_sft_cli.py；最新回归44 passed、16 warnings（11.01s）。

2026-08-31 GPU实测：固定1条Stage-A、515 tokens、loss1.9581148624420166，504个LoRA张量
梯度有限、252个B梯度非零，基座无梯度且数值保留。108个冻结bias发生库内部dtype转换，
并非优化器更新；详细证据见[梯度联检报告](docs/run_logs/qlora-backward-VKJlYdcw/SUMMARY.md)。
命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward`。
仅反向传播，没有optimizer.step、Trainer训练或保存；教师未修改学生函数/main/YAML。
最新CPU回归42 passed、16 warnings（12.09s），无xfail。

```python
def prepare_qlora_model(model, use_gradient_checkpointing: bool):
```

- 输入：已经加载的 4-bit 基座（尚未挂 LoRA），以及梯度检查点布尔开关；不是路径或配置字典。
- 输出：准备后的模型本身；PEFT 原地调整模型，不返回 dict 或 adapter。
- 复用已导入的 `prepare_model_for_kbit_training`，布尔开关原样传递；
  `gradient_checkpointing_kwargs={"use_reentrant": False}` 与现有 Trainer 配置保持一致。
- 参考 MedicalGPT/training/supervised_finetuning.py:920-923；它在量化条件下准备基座，
  随后在940行创建新LoRA。我们不复制其整套训练流程，也不手写冻结/dtype/hook逻辑。
- 学生已实现原生准备调用与返回，教师未修改其函数。非量化调用方跳过此接口；use_cache 仍交已有 Trainer 路径。
  本轮不修改 main、加载函数、attach_lora 或 YAML，不启动训练。
- 教师测试：docs/tests/test_sft_kbit_preparation.py；命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
  撤去占位标记后回归 `42 passed, 16 warnings in 11.11s`，无 xfail；警告为已知第三方弃用提示。
  CPU 检查复用 tiny 模型及真实 PEFT，但用量化标记触发准备分支，不代表真实4-bit训练通过。

学生已实现可选 `quantization_config=None` 分支，非量化路径保留；
CUDA 提前检查、单卡映射 `{"": 0}`、量化对象原样传递和本地加载参数均通过。
教师没有改学生实现、main 或 YAML；加载阶段不做训练准备、LoRA 挂载或训练。
完整契约、容器安装位置和可见检查命令见 [QLORA_ENVIRONMENT.md](docs/QLORA_ENVIRONMENT.md)。
量化加载验收时的测试在 docs/tests/test_sft_loading.py；当时回归 `40 passed, 16 warnings in 15.08s`，无 xfail。
真实 Qwen2.5-3B 的 252 个 NF4 双重量化层、CUDA 单卡放置和单次前向也已通过，见
[量化加载报告](docs/run_logs/qlora-load-jxVC6GU3/SUMMARY.md)。这不代表 QLoRA 训练通过。
低比特准备接口已由学生实现并通过 CPU 验收，正式 main 仍归学生。

以下保留已验收的量化配置接口（位于 `sft/train.py`）：

```python
def build_quantization_config(use_qlora: bool, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
```

- 输入：是否启用 QLoRA，以及已解析的 PyTorch dtype（如 `torch.bfloat16`），不是 YAML 字典或字符串。
- 输出：关闭时为 `None`；开启时为原生 `BitsAndBytesConfig`，指定 4-bit NF4、double quant，
  计算 dtype 使用传入值，不开启 8-bit。只构造配置，不加载模型或准备训练。
- 参考 MedicalGPT 的 QLoRA 配置分支（740～765 行）。这属于原有 LoRA/QLoRA 学习路线，
  不是医学任务专有设计。学生已完成该函数；教师没有修改其实现、main、load_base_model 或 YAML。
- 教师测试在 `docs/tests/test_sft_quantization.py`；撤去待实现标记后完整回归
  `37 passed in 15.64s`，无警告、无 xfail。关闭分支与 BF16/FP16/FP32 配置传递已通过，
  结论仅限配置对象，不是实际量化加载或训练验收。
- Docker `llm-pytorch` 已新增 bitsandbytes 0.50.2，没有升级既有训练依赖；NF4 微型层前向已通过。
  真实 Qwen 4-bit 加载现已通过独立验收；低比特训练准备与 QLoRA 训练仍待验收。

后续按 MedicalGPT 逐步适配 QLoRA：先量化配置，再量化加载和低比特训练准备，最后由学生
亲自编排正式 main。当前不一次性补全流程。参考源码为
`MedicalGPT/training/supervised_finetuning.py` 的 740～765 行（配置）与 923 行附近（训练准备）。
以下保留已验收保存接口的边界：

学生已实现 `sft/train.py` 中的接口：

```python
def save_adapter(model: PeftModel, tokenizer, output_dir: str | Path) -> None:
```

- 输入：已有 LoRA 模型、配套 tokenizer，以及教师入口已创建的全新空目录。
- 输出：返回 `None`。文件必须包含 `adapter_config.json`、`adapter_model.safetensors`
  及 tokenizer 原生保存文件（含 chat template 和特殊 token 配置）；不自行规定 tokenizer
  的全部文件名，不返回 metrics 或文件路径。
- 学生职责：复用 PEFT/HF `save_pretrained`；使用 `safe_serialization=True` 与
  `save_embedding_layers=False`。当前词表未扩充、embedding 冻结，不合并 LoRA、不保存底模。
- 参考：`MedicalGPT/training/supervised_finetuning.py` 的 `save_model()`；当前单卡无需复制
  分布式包装或新增 Trainer 子类。`run_training()` 的接口和实现保持不变。
- 教师已接好：只有 `--train-smoke --save-adapter` 才在单步成功后调用保存，目标为该次
  `outputs/smoke/stage_a-*/adapter/`，不覆盖正式目录；默认各 smoke 模式行为不变。
- 完成标准：adapter 权重逐项一致，tokenizer/chat template 保留，在相同底模、相同输入、
  `eval()` 下重载前后 logits 一致；复用 `attach_lora()` 验证可训练加载分支。
- 测试位于 `docs/tests/test_sft_save.py`；学生只填函数，验收命令与测试维护均交给教师。
  本次 CPU 回归 `31 passed in 11.76s`，无警告、无 xfail。保存文件、adapter 权重、tokenizer
  和同底模重载 logits 均通过；教师未改动学生实现。随后真实 3B 保存/重载也已通过，见下方报告。
- 边界：此产物不含 optimizer/scheduler/step，不能用于完整断点恢复，也不构成正式 Checkpoint A。
  上次 3B 单步没有保存权重，不能从已有 metrics 反推出 adapter；后续真实验证需要重跑单步。

Stage-A/Stage-B 编码、数据入口及 Qwen2.5-3B + LoRA 装配已验收。教师复用 MedicalGPT 的
Trainer 接线方式，提供 `build_training_arguments()`、两个 YAML 的 `training` 字段和
`--trainer-smoke` 入口；学生已实现 `build_trainer()`，CPU 验收为 `22 passed in 12.31s`。
Stage-A/B、梯度检查点开/关、mask 原样保留与参数冻结均已检查；Stage-A 真实 3B 已完成单步。

- 输入：config、已挂载 LoRA 的 model、tokenizer、已有 dataloader；输出：HF Trainer。
- Trainer 接收现有 dataloader 的 **dataset 和 collate_fn**，自己创建训练 DataLoader；
  `shuffle` 仅控制独立数据 smoke，Trainer 训练采样器默认打乱。
- 沿用已有 labels 和动态 padding，不重新编码、不覆盖 loss mask。
- 梯度检查点使用 `use_reentrant=False`，由 Trainer 在 train 时启用；学生需处理 KV cache 边界。
- 装配模式不调用 `train()`，不创建 optimizer、不评估、不保存 checkpoint；`max_steps=1` 仅是
  smoke 的安全默认，学习率、epoch 和有效 batch 尚不是经过验证的正式实验配方。
- 本轮 eval/save 策略固定为 `no`。dev 评估、adapter 保存/完整断点恢复必须在正式训练前另行接入验收。
- Stage-B 仍须真实 Checkpoint A；不伪造 adapter，也不把 tiny adapter 验证当成真实 3B 续训。

教师测试入口（从仓库根目录的 CMD 调用）：

```cmd
wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh
```

全部已验收接口（含 `save_adapter`）的 xfail 标记已撤去。保存重载已通过 tiny CPU 和真实 3B 验证。
教师原有“关闭 checkpointing 后 cache 必须保持原值”的断言
已按本地 Transformers 5.10.4 源码修正：Trainer 初始化时会写入 `args.use_cache`（默认 False）。
学生代码没有因此修改；`--trainer-smoke` 只做装配，真正单步训练使用 `--train-smoke`。

教师已接 `--train-smoke`、`prepare_train_smoke()`、隔离输出、初始化前随机种子和测试。
学生负责最小训练调用及结果返回，不手写 optimizer/backward 循环。先验证 tiny 模型，再用少量
Stage-A 数据做 3B 单步 smoke；
检查 global_step=1、有限 loss/梯度、LoRA 参数更新和基座冻结，并记录真实显存与耗时。
当前单卡 batch=2、gradient_accumulation_steps=4，对应完整累积时有效 batch=8；
max_steps=1 指一次 optimizer update，不是一整个 epoch。

### 单步训练接口边界

- `run_training()` 接收已装配的 Trainer，委托其执行一次配置好的训练，返回新的 metrics 字典，
  并从真实 TrainOutput 补入 `global_step`；不捕获并吞掉训练异常，不执行评估或保存 checkpoint。
- 默认数据 smoke、`--model-smoke`、`--trainer-smoke` 不训练；只有互斥的 `--train-smoke`
  进入新路径。该模式要求 Stage-A、max_steps=1、单进程/单设备；保持 Stage-B replay 暂缓。
- 固定 seed 抽取 batch_size × gradient_accumulation_steps 条不同样本，不修改冻结文件。
- 每次新建 `MedSearch/outputs/smoke/stage_a-*`；记录 `run_config.json`（有效配置、Git HEAD、
  当前三份 SFT 源码哈希、完整数据文件哈希、样本 ID），成功后记录 `train_metrics.json` 和
  `trainer_log_history.json`。不会覆盖正式 `outputs/stage_a`，不会生成可交付 Checkpoint A。
- run_training 已由学生实现并验收，完整 CPU 回归为 `27 passed in 12.94s`，无 xfail。
  tiny 模型真实完成一次更新，loss/梯度有限、LoRA 更新、基座不变，返回指标完整。
- 指标保留以实际 TrainOutput.metrics 为准，不要求补造日志中存在但返回对象没有的字段；
  教师测试已直接观察真实返回对象，修正 total_flos 的日志/返回值混淆。
- 2026-08-31 真实3B单步通过：8条固定样本，global_step=1，loss=1.173087，grad_norm=0.814825，
  Trainer训练耗时6.3281秒；252个LoRA张量变化，434个冻结基座张量全量哈希无变化。
- 整卡采样峰值18,139 MiB（含背景占用）；实际最长样本842 tokens，不能外推到4096。
  没有保存adapter，不构成正式Checkpoint A。详细配置、命令、日志和限制见
  [本次报告](docs/run_logs/3b-smoke-eqK4l3I8/SUMMARY.md)。
- 2026-08-31真实3B保存/重载通过：固定8条、global_step1、loss1.173087，504个adapter张量、
  434个基座哈希及tokenizer一致；统一BF16 autocast后最后位置全词表logits最大差0。
  产物在 `outputs/smoke/stage_a-kv1h2i46/adapter/`，仍非正式Checkpoint A，也不是QLoRA。
  教师探针两次失败与修正均保留，完整证据见[保存重载报告](docs/run_logs/3b-smoke-SrvIu01k/SUMMARY.md)。
- 最新CPU回归42 passed、16条第三方警告，无xfail；真实4-bit梯度联检通过，下一项由学生编排QLoRA调用路径以做Trainer单步验收。正式main由学生编写；
  dev评估、完整断点恢复仍须在正式训练前验收。
