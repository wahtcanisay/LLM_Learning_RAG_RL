# MedSearch-R1 项目书

## 一、教师角色声明（每次开始工作前先读）

我在本项目中的角色是**教师、学习督导与技术研究助手**，不是替学生完成项目的外包开发者。

我的首要职责是帮助学生建立能够解释、能够验证、能够迁移的知识体系，而不只是让仓库暂时跑通。具体职责如下：

1. 根据当前进度，把项目拆成一次只解决一个核心问题的学习任务，并明确完成标准。
2. 在学生编码前补齐所需知识，包括但不限于：论文原理、官方文档、官方源码、维护者示例、可靠技术博客，以及成熟 GitHub 项目的代码组织和实现风格。
3. 搜集参考代码时，说明来源、适用版本、输入、输出、调用方式、可复用部分和需要改造的部分；不能只扔链接，也不能把不兼容代码直接拼进项目。
4. 结合 MedSearch-R1 当前代码解释知识，不做脱离项目的百科式讲解。解释函数时优先说明：
   - 输入：字段、数据类型、shape 和数据来源；
   - 输出：字段、数据类型、shape 和下游用途；
   - 调用方式：谁调用它、它又调用谁；
   - 设计原因：为什么需要这一层，删掉后会发生什么；
   - 验证方式：用什么最小样例、日志或测试证明理解正确。
5. 具体代码的编写、改写和参考代码的复制粘贴原则上由学生亲自完成。我负责给出小而明确的实现目标、必要的参考片段、接口约束和审查意见。
   数据下载、清洗、去重、风险分流和训练集构建是本项目的例外：由教师代理完成并保留可复现
   规则与报告；学生只需理解最终字段和过滤边界，把主要学习时间用于训练管线与实验。
6. 学生提交代码后，我必须检查真实文件、配置、命令和日志，再判断是否完成；不能仅凭“已经跑通”的口头描述下结论。
7. 当学生理解有误时，先指出与当前代码对应的具体证据，再引导学生修正；当学生已经掌握时，减少重复讲解并推进实验。
8. 不擅自大规模重构，不一次性生成整个模块，不为了跑通而删除奖励、评测或关键训练逻辑。只有学生明确要求我代为实现时，才进入直接编码角色，并且仍需说明改动原因和影响。
9. 所有实验结论必须来自真实运行结果。不得编造准确率、reward、训练时间、显存占用或性能提升。
10. 持续维护本项目书：记录已经确定的决策、尚未解决的问题、学习证据和实验结果，剪除已经失效或与当前项目无关的内容。

学生是本项目的**主要实现者与最终解释者**。学生需要亲自完成关键代码、运行实验，并能够用自己的话解释数据怎样流动、模型为什么这样训练、结果为什么可信。

---

## 二、项目定位

项目名称：**MedSearch-R1：基于领域微调与强化学习的医学证据搜索 Agent**

本项目处于总体学习路线的第五阶段，计划复用前面阶段已经掌握或验证的能力：

```text
医学领域 SFT 初始化
        ↓
医学问题与工具调用格式
        ↓
Search-R1 Search / Answer 多轮交互（Inspect 是否保留待决）
        ↓
医学检索后端返回证据
        ↓
规则奖励与 GRPO 训练
        ↓
独立测试集上的答案、证据与搜索成本评测
```

项目用于学习和研究，不提供真实临床诊断建议。

## 三、项目目标

最终目标不是只得到一个可以调用搜索工具的 Demo，而是构建一套可复现、可比较、可解释的医学搜索 Agent 实验：

- 使用医学 SFT 建立领域问答和工具格式基础；
- 复用已经验证的医学检索器，而不是重新制作不可比较的检索系统；
- 让模型能够根据问题和已有证据决定继续搜索或提交答案；
- 使用可程序化验证的奖励完成小规模 RL 训练；
- 分别评估答案质量、证据质量、工具行为和搜索成本；
- 保留 Base、SFT、固定 RAG、无 RL Agent 和 RL Agent 的可比基线；
- 保存代码版本、数据边界、配置、日志、失败轨迹和真实指标。

## 四、当前边界

### 现在确定的内容

- 第一项工作从 SFT 开始讨论和学习。
- 学生负责编写或复制粘贴具体代码，教师提供资料、讲解、接口约束与审查。
- SFT 使用两阶段设计：阶段 A 以医学单轮/多轮对话为主体，并混入少量带可见医学解释的
  supervised rationale 样本，保存 `Medical-SFT`；阶段 B 使用约 100 条真实检索支持的
  精品轨迹，并回放阶段 A 的一部分普通医学对话，保存 `Medical-SFT+Search100`。两个
  checkpoint 都保留用于消融。
- Search100 默认执行，不再以基础模型协议测试失败作为启动条件；其目标仅限于教会
  Search-R1 标签、基本 query 构造和一至两轮闭环，不承担大规模策略模仿。
- MedQA/MedMCQA 不再作为阶段 A 普通医学 SFT 的主体。它们的官方 train 只作为后续
  Search100/RL 候选题池；本地 MIRAGE 的 MedQA 1,273 条 test 和 MedMCQA 4,183 条
  dev 继续冻结为评测资产，不进入训练。
- 第一版 SFT 训练入口沿用已经跑通过的 MedicalGPT LoRA/QLoRA 管线；RL 框架在后续
  阶段单独选择，不能因此把 veRL 的 rollout 数据结构硬塞进 SFT 数据层。
- SFT 冷启动不保留 Qwen/OpenAI 风格的通用 Function Calling 协议；模型直接学习
  Search-R1 的 ``<search>...</search>``、环境插入的
  ``<information>...</information>`` 和最终 ``<answer>...</answer>`` 标记。
- 普通对话只对 assistant 消息计算 token-level cross-entropy，不强行添加
  ``<think>``。少量推理样本训练的是数据中可见、可审计的 supervised rationale，不能
  称为模型真实的内部思维。Search100 中的问题与检索返回的 ``information`` 只参与
  attention、不参与 loss；模型生成的 ``think/search/query/answer/EOS`` token 参与
  loss。这里不是 RL entropy bonus；具体 labels 必须通过解码样本审计确认。
- 阶段 A 与阶段 B 使用不同且明确的 system/task prompt：普通医学对话允许直接回答，
  Agent 样本才要求 Search-R1 标签，避免同一输入对应互相冲突的训练目标。
- 第一轮只做能够验证完整数据流的最小实验，再决定是否扩大数据和训练规模。
- 使用公开、可验证的医学问答数据；训练集、RL 问题和最终测试集必须隔离。
- 后续检索后端优先复用已有 MedRAG/Medical GraphRAG 能力。
- 所有指标必须来自真实脚本输出。

### 现在尚未决定的内容

以下问题必须经过资料核对和讨论后再写入“已确认决策”，不能由教师默认决定：

- 阶段 B 训练时从阶段 A 回放多少普通对话；不能把阶段 A 全量数据直接拼回阶段 B，
  否则 100 条搜索信号会被淹没。该值仍是待实验超参数，不能伪装成已验证最优比例；
- 旧设计中的 ``Inspect`` 是否从第一版删除；若严格沿用 Search-R1 原协议，第一版只有
  Search 和 Answer，Submit 由 ``<answer>`` 表示；
- 使用全参数、LoRA 还是 QLoRA；
- 基座模型的准确版本与 checkpoint；
- Chat Template、最大序列长度和 loss mask 规则；
- 第一版评测集、指标和数据去重方法；
- SFT checkpoint 怎样进入后续 Agentic RL。

### 已冻结的 Stage-A 与 Stage-B 数据组成

#### Stage-A：医学能力与正常对话 SFT

Stage-A 共 16,405 条，固定 seed `20260823`，其中 train 16,077、dev 328，规范化
user 文本的 train/dev 重叠为 0。四个 mixture group 的首版配方如下：

| 组成 | 数据来源 | 选中 | train | dev | 占 Stage-A |
|---|---|---:|---:|---:|---:|
| 医学单轮 | Meerkat MedInstruct-52K、LiveQA、MedicationQA | 10,663 | 10,450 | 213 | 65% |
| 医学多轮 | Meerkat MedQA-dialog | 3,281 | 3,215 | 66 | 20% |
| 可见推理 | medical-o1-reasoning-SFT English | 1,640 | 1,607 | 33 | 10% |
| 通用对话 replay | UltraChat 200k `train_sft`，revision `8049631...` | 821 | 805 | 16 | 5% |
| 合计 |  | 16,405 | 16,077 | 328 | 100% |

按原始数据集统计：MedInstruct-52K 10,462 条（63.77%）、MedQA-dialog 3,281 条
（20.00%）、medical-o1 1,640 条（10.00%）、UltraChat 821 条（5.00%）、
MedicationQA 107 条（0.65%）、LiveQA 94 条（0.57%）。65/20/10/5 是首版训练配方，
不是性能最优结论。

正式资产：

- `MedSearch/data/processed/sft_stage_a/stage_a_train.jsonl`
- `MedSearch/data/processed/sft_stage_a/stage_a_dev.jsonl`
- `MedSearch/data/processed/sft_stage_a/stage_a_manifest.json`

#### Stage-B：Search-R1 协议与真实检索轨迹

Stage-B Search100 共 100 条，全部来自 MedQA/MedMCQA official train，query 与 rollout
阶段隐藏 gold，information 来自本地 MedText BM25 真实检索。独立二次复审替换了 9 条带明显
OCR/拼写噪声的样本，最终 100/100 通过答案、gold、证据、引用、题目质量和泄漏检查。

| 维度 | 组成 | 数量 | 占 Stage-B Search100 |
|---|---|---:|---:|
| 题目来源 | MedQA official train | 72 | 72% |
| 题目来源 | MedMCQA official train | 28 | 28% |
| 搜索轮数 | 一轮搜索 | 80 | 80% |
| 搜索轮数 | 两轮搜索 | 20 | 20% |

正式训练 JSON 每条只保留 `id/question/options/trajectory`；trajectory 仅包含
`search → information → answer`。Gold、document ID、检索配置和复核结论只保存在审计文件。

正式资产：

- `MedSearch/data/stage_b/search100_training.json`
- `MedSearch/archive/data_preparation/data/search100_audit.json`
- `MedSearch/archive/data_preparation/data/search100_failure_archive.json`
- `MedSearch/data/stage_b/stage_b_protocol_dev.json`
- `MedSearch/data/stage_b/stage_b_protocol_dev_manifest.json`

首版 protocol dev 共 12 条，全部来自未进入 Search100 的 MedQA/MedMCQA official train：
一轮搜索 8 条、两轮搜索 4 条，两数据源各 6 条。它只用于开发期搜索格式、证据命中、闭环完成
与最终答案评测，不提供唯一参考 query，不计算 CE loss，也不称为最终 test。构建时对 Search100、
Stage-A user turn 和 MIRAGE MedQA/MedMCQA 最终测试题执行精确、词汇与本地 MPNet 语义近重检查；
未来医学 RL 数据必须继续排除这 12 个 ID。

2026-08-31：阶段 B 是否混入 Stage-A 正常对话 replay 改为待讨论的实验选择，暂不实现数据组合，
也不将其作为 Trainer 接入的前置条件。当前输入仍只有 Search100；若后续启用 replay，候选池
限于 Stage-A 的 single、multi、general，不包含 reasoning。是否混合及具体条数均需单变量比较，
不预先宣称混合或不混合更优。Stage-A 自身的 65/20/10/5 配方不受影响。

## 五、教师提供资料的标准

资料优先级如下：

```text
官方文档 / 官方源码 / 论文
        ↓
项目维护者提供的示例
        ↓
活跃且可复现的开源项目
        ↓
解释清楚并给出代码依据的技术博客
```

每一份推荐资料至少记录：

| 字段 | 要求 |
|---|---|
| 知识问题 | 它解决当前哪个具体疑问 |
| 来源 | 文档、论文、博客或 GitHub 的链接 |
| 版本 | 框架、模型、仓库 commit 或发布日期 |
| 建议阅读位置 | 具体章节、文件、类或函数 |
| 可借鉴内容 | 接口、数据处理、配置或代码风格 |
| 不可直接照搬之处 | 版本差异、任务差异或硬件差异 |
| 许可证 | 是否允许复制、修改和发布 |
| 学习检查 | 学生读完后需要回答或实现什么 |

推荐第三方代码时，教师必须区分“学习其写法”和“直接复用其实现”。如果需要复制片段，必须保留来源与许可证信息。

## 六、单次学习与实现流程

每次只推进一个核心任务，按以下顺序进行：

1. **定位问题**：说明本次任务在完整数据流中的位置。
2. **补充知识**：指定少量且具体的官方资料、源码函数或优质参考实现。
3. **阅读检查**：要求学生解释关键输入、输出、调用链和设计目的。
4. **学生实现**：学生亲自编写、改写或复制并适配代码。
5. **教师审查**：检查代码是否符合接口、版本、数据边界和实验目标。
6. **最小验证**：运行最小样例，保存命令、日志、输出和显存数据。
7. **结果复盘**：区分代码跑通、训练有效和指标提升三个不同结论。
8. **更新项目书**：记录证据、问题与下一项唯一任务。

教师给出的编码任务应采用下面的格式：

```text
任务：
所在文件：
需要实现的函数：
输入：
输出：
调用方式：
可参考的官方/开源实现：
需要自己完成的部分：
完成标准：
验证命令：
```

## 七、阶段检查框架

这里仅定义检查关口，不提前锁定具体实现方案。

### 关口 A：SFT 目标和数据边界

- [ ] 能解释这次 SFT 究竟要让模型学会什么。
- [ ] 数据来源、许可证、字段和样本规模已经核验。
- [ ] SFT、RL 和测试数据的隔离规则已经确定。
- [ ] 最小评测能比较 Base 与 SFT，而不只是比较训练 loss。

### 关口 B：最小 SFT 数据流

- [ ] 能从一条原始样本追踪到 tokenized batch。
- [ ] 能解释 Chat Template、labels 和 loss mask。
- [ ] 单 batch 前向与反向传播通过。
- [ ] checkpoint 能保存、加载并完成确定性推理。

### 关口 C：SFT 小规模实验

- [ ] 训练命令、配置、随机种子和 Git commit 已记录。
- [ ] W&B 或本地日志能看到真实训练/验证曲线。
- [ ] 有 Base 与 SFT 的定量结果和具体输出案例。
- [ ] 能分析过拟合、格式学习与医学能力变化。

### 关口 D：工具与检索环境

- [ ] ``<search> / <information> / <answer>`` 的输入输出协议确定；若保留 Inspect，另行定义且不得破坏 Search-R1 主协议。
- [ ] 检索服务可以独立启动并通过最小请求验证。
- [ ] Agent 与检索器仅通过稳定接口交互。
- [ ] 非法调用、重复查询和强制停止能够被记录。

### 关口 E：RL smoke test

- [ ] 一条问题的多条 rollout 可以被正确分组。
- [ ] reward 各分量能够单独检查。
- [ ] old log-prob、advantage、ratio 和 policy loss 的数据流完整。
- [ ] 至少完成一个真实优化 step，并检查参数或 loss 确实发生变化。

### 关口 F：对比实验与项目总结

- [ ] Base、SFT、固定 RAG、Agent 和 RL Agent 使用同一测试边界。
- [ ] 记录答案、证据、工具行为、成本和显存指标。
- [ ] 保留成功案例、失败案例和 reward hacking 轨迹。
- [ ] README、环境、命令、结果表和非临床声明齐全。

## 八、已确认决策记录

| 日期 | 决策 | 原因 | 证据位置 | 影响范围 |
|---|---|---|---|---|
| 2026-08-21 | 教师负责规划、资料检索、讲解与代码审查；学生负责关键代码的编写和适配 | 保证项目能够被学生独立解释和迁移 | 本项目书“教师角色声明” | 全项目 |
| 2026-08-21 | 先讨论 SFT，不立即生成完整训练工程 | SFT 的目标、数据和框架尚未共同确定 | 本项目书“当前边界” | SFT 阶段 |
| 2026-08-21 | SFT 不训练通用 Function Calling，直接使用 Search-R1 的 search/information/answer 标签协议 | 保证 SFT 与后续 Search-R1 风格 rollout 的动作空间和文本协议一致 | 本项目书“当前边界” | SFT 数据、loss mask、推理与 RL 接口 |
| 2026-08-21 | 采用 `Medical-SFT → Medical-SFT+Search100` 两阶段 SFT，并默认制作约 100 条精品真实检索轨迹 | 同时保留医学 QA 基线与协议冷启动效果，避免把 Search100 的作用混入领域 SFT | 本项目书“数据组成” | SFT 数据、checkpoint 与消融 |
| 2026-08-21 | 只使用 MedQA/MedMCQA 官方 train 训练；MIRAGE 中 MedQA test 与 MedMCQA dev 冻结 | 本地实物分别为 1,273 条 test 和 4,183 条 dev，混入会造成评测泄漏 | 本项目书“数据组成” | 数据隔离与评测 |
| 2026-08-21 | SFT 沿用 MedicalGPT 管线；`information` 仅参与 attention，labels 为 -100 | 已有 MedicalGPT 3B LoRA smoke，且环境 observation 不是模型生成目标 | 本项目书“当前边界” | SFT 实现 |
| 2026-08-23 | 阶段 A 改为“多数医学单轮/多轮对话 + 少量 supervised rationale”；阶段 B 使用 Search100 + 阶段 A 普通对话 replay | 选择题不适合作为通用对话 SFT 主体；replay 用于保持正常回答能力并抑制逢题必搜和协议过拟合 | 本项目书“数据组成” | SFT 数据配方、prompt、checkpoint 与消融 |
| 2026-08-23 | 第一版停用 ChatDoctor 与 MedBooks；单轮使用 MedInstruct/LiveQA/MedicationQA，推理只使用 medical-o1 English | 风险抽检发现多轮数据含缺图、答案机械确认和安全边界问题，最终按清洗后的真实有效量构建 | Stage-A manifest | 阶段 A 数据选择、清洗、长度和消融 |
| 2026-08-23 | 数据清洗改由教师代理负责；首版 Stage-A 使用 65% 医学单轮、20% 医学多轮、10% reasoning、5% UltraChat 通用 replay | Qwen2.5 Instruct 已具备通用指令能力，少量 replay 只用于缓解领域 SFT 遗忘；比例仍须消融 | Stage-A manifest | 数据职责、训练输入与首轮消融基线 |
| 2026-08-23 | 9,572 条 review 由三个 `gpt-5.6-sol` 一次性二元终审，不交给学生或人工 | 用户不具备医学英文审核条件；强模型只判断 SFT 适用性，无法确认时保守拒绝 | 本项目书保留结论；一次性审计文件已清理 | 风险边界 |
| 2026-08-23 | 通用 replay 固定 UltraChat 200k `train_sft` revision `8049631...`，经主题/质量规则、强模型抽检和 1024-token 闸门后占 Stage-A 5% | 防止医学 SFT 覆盖通用对话能力，同时避免重复灌入过多通用数据 | Stage-A manifest | Stage-A 混合比例与数据版本 |
| 2026-08-23 | Stage-B Search100 冻结为 MedQA/MedMCQA official train 的 72/28，搜索轮数为一轮/两轮 80/20 | 保留真实医学检索闭环，同时剔除错答、弱证据、缺图、过时/歧义题和明显文本噪声 | `MedSearch/data/stage_b/search100_training.json` | Stage-B SFT 数据、协议学习与后续消融 |
| 2026-08-23 | Stage-B 二次复审 100/100 通过；9 条明显 OCR/拼写噪声样本由强审通过的备用轨迹替换 | 精品 SFT 不仅要求答案正确，也要求题干可读、证据直接 | 本项目书保留验收结论 | Stage-B 数据验收 |
| 2026-08-29 | Stage-A 最终化：1,640 条 reasoning 使用 `medical_o1_{provenance.index}` 稳定 ID；删除2条上游已含 `U+FFFD` 的 UltraChat 样本并用同 revision 的2条合格样本替换 | 保证样本可追踪、无不可恢复乱码，同时维持16,405总量和65/20/10/5配方 | Stage-A manifest、`MedSearch/tools/finalize_stage_a.py` | Stage-A 训练输入与实验追踪 |

## 九、资料与参考实现登记表

2026-08-31 决策更新：按学生要求暂缓 Stage-B replay，直接进入 Trainer 骨架；教师只负责
参数转换、入口和测试，`build_trainer()` 由学生填写。此决策替代上表 2026-08-23 关于
Stage-B 必须加入 replay 的约定，不改动已冻结数据资产。

| 状态 | 知识问题 | 来源与版本 | 阅读位置 | 借鉴内容 | 风险/差异 | 学习检查 |
|---|---|---|---|---|---|---|
| 已核对 | MedQA 官方 split 与字段 | MedQA 官方 README | Data 章节 | official train/dev/test、US 4-options | 本地 MIRAGE 只有 test | 解释为何不能用于 SFT |
| 已核对 | MedMCQA split、答案编码与解释空值 | MedMCQA 官方仓库、论文与本地 `build.py` | dataset split / `cop` | `cop` 为 1-4；train/dev/test 边界 | HF 转换可能显示 0-based class label，必须保留 raw 值校验 | 手工转换 3 条样本 |
| 已核对 | 对话/instruction/reasoning 候选实物质量 | Meerkat `f0621a0...`、medical-o1 `fc2c9e8...` | 八类数据各前 10 条 | 字段、轮数、target 语义、明显质量风险 | 抽样只用于筛选候选源；一次性样本和审计文件已清理 | 以冻结后的 Stage-A manifest 为准 |

## 十、实验记录表

| 实验 ID | 日期 | Git commit | 模型 | 数据及样本数 | 主要变量 | 随机种子 | 指标 | 显存峰值 | 日志位置 | 结论 |
|---|---|---|---|---|---|---:|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |

## 十一、问题与失败记录

| 日期 | 问题或失败现象 | 已核验证据 | 原因 | 处理办法 | 是否解决 |
|---|---|---|---|---|---|
| 2026-08-29 | Stage-A reasoning 顶层 ID 全部缺失；2条 general 含 `U+FFFD` | 全量扫描：missing ID 1,640、损坏文本2条；原始 UltraChat 同样损坏 | 从 provenance 生成稳定 ID；整条替换损坏样本；固定脚本可幂等复现 | 是 |

## 十二、当前状态与下一步

2026-08-31 教学边界补充：后续正式 main 的训练编排由学生编写，教师只提供接口、必要骨架
和 docs 内测试，不代填完整调用流程。后续按步骤实现 QLoRA，包括 4-bit 加载、低比特训练
准备、LoRA 挂载及训练/保存重载；本轮仍验证 BF16 LoRA，不混淆两者的实验结论。

当前状态：Stage-A 与 Stage-B 数据均已冻结。Stage-A 为 16,405 条 65/20/10/5 配方，
train 16,077、dev 328；Stage-B Search100 为 100 条，MedQA/MedMCQA 72/28，
一轮/两轮搜索 80/20。Stage-B 已完成独立二次复审，100 条正式记录均为 official train、
答案与 gold 一致、引用存在且证据可支持答案。Stage-A 全部样本已有唯一稳定 ID，最终数据
不含 `U+FFFD`；两条损坏的上游 UltraChat 记录已整条替换。尚未启动本轮 SFT。

编码、Dataset/DataLoader、LoRA 与 Trainer 装配均已验收：学生已实现 `build_trainer()`，
2026-08-31 CPU 回归 `22 passed in 12.31s`，不再保留 xfail。教师按本地 Transformers 5.10.4
源码修正 use_cache 测试假设，没有改动学生核心实现。此结论不包含真实 3B Trainer 或 backward。

学生 run_training 已完成：返回完整真实 metrics 及 global_step，tiny CPU 真实单步、
有限 loss/梯度、LoRA 更新和基座不变均通过；最终完整回归 `27 passed in 12.94s`。
教师修正日志与返回指标混淆的测试假设，不要求学生补造 total_flos，未修改学生实现。

2026-08-31真实3B单步已通过：8条Stage-A、seed20260831、global_step1、loss1.173087、
grad_norm0.814825、Trainer训练耗时6.3281秒，整卡采样峰值18,139 MiB。
252个LoRA张量变化，434个冻结基座张量全量哈希不变。最长样本842，不能外推Stage-B4096。
详细证据见 `docs/run_logs/3b-smoke-eqK4l3I8/SUMMARY.md`。未保存adapter，不能称为Checkpoint A。

2026-08-31 保存接口已由学生实现：`save_adapter(model, tokenizer, output_dir) -> None`；
教师接好 `--train-smoke --save-adapter`，仅保存至本次全新 smoke 目录的 adapter 子目录。
契约只保存 PEFT adapter 与原生 tokenizer，不保存整套基座或 optimizer/scheduler/step。
测试位于 `docs/tests/test_sft_save.py`，使用非零 tiny adapter 与相同底模副本验证重载；
学生显式设置 `save_embedding_layers=False` 后，权重/tokenizer/固定输入 logits 的 tiny CPU
round-trip 已通过。教师未修改学生实现；撤去 xfail 后完整回归 `31 passed in 11.76s`，
无警告、无 xfail。本轮未重新启动真实 3B，未生成正式 Checkpoint A。

后续真实3B验证已通过：固定8条、global_step1、loss1.173087，504个adapter张量、434个
基座哈希和tokenizer一致；统一BF16 autocast下最后位置全词表logits最大差0。
最新CPU回归33 passed，实测产物 `outputs/smoke/stage_a-kv1h2i46/adapter/`。
教师输入格式/推理精度的两次失败均保留，详见 `docs/run_logs/3b-smoke-SrvIu01k/SUMMARY.md`。
没有修改学生主训练代码；这不是QLoRA、正式Checkpoint A或效果提升。

QLoRA 配置已由学生实现：`build_quantization_config(use_qlora, compute_dtype)`，
关闭返回 None，开启返回 4-bit NF4/double quant 的原生 BitsAndBytesConfig，计算 dtype 原样传递。
测试位于 `docs/tests/test_sft_quantization.py`；最新回归 `37 passed in 15.64s`，无警告、无 xfail。
关闭分支与 BF16/FP16/FP32 计算 dtype 均通过；教师未改学生实现、main/load_base_model/YAML，也未安装依赖。
Docker 已新增 bitsandbytes0.50.2，原训练依赖不变；微型NF4双重量化GPU前向通过。
环境位置、安装命令及范围见 `docs/QLORA_ENVIRONMENT.md`，微型层验证与下述真实加载分别记录。

学生量化加载已验收：load_base_model 的 CUDA 检查、原生量化加载和非量化兼容均通过，
教师未改学生实现或 main/YAML。最新 CPU 回归 40 passed、16 条第三方弃用警告，无 xfail。
真实 Qwen2.5-3B 252 个 NF4 双重量化层均通过检查，所有参数在 cuda:0，单次前向 logits 有限。
加载耗时27.8416秒；PyTorch加载+前向峰值allocated2,149,335,040 bytes，非训练/整卡显存。
报告见 `docs/run_logs/qlora-load-jxVC6GU3/SUMMARY.md`；教师报告属性误用的首次失败已保留。

准备接口已留出：sft/train.py::prepare_qlora_model(model, use_gradient_checkpointing)，
仅含契约和NotImplementedError。参考MedicalGPT:920-923，复用PEFT准备，并与Trainer统一
gradient_checkpointing_kwargs={"use_reentrant": False}；非量化调用方跳过。未改main/YAML。
docs/tests/test_sft_kbit_preparation.py先验证缺少接口的2项失败；骨架回归
40 passed、2 xfailed、16 warnings（11.35s），占位仅允许NotImplementedError。
CPU测试不代表真实4-bit准备或QLoRA反向传播已通过。

学生现已完成prepare_qlora_model，CPU准备开关True/False、冻结与精度/数值边界均通过；
教师未修改学生实现。撤去两项xfail后默认回归42 passed、16 warnings in 11.11s，无xfail。
本轮未运行GPU、反向传播或训练；真实4-bit权重保留与LoRA梯度仍待联检。

真实4-bit梯度联检已通过：Stage-A单条515tokens、loss1.9581148624420166，504个LoRA梯度有限，
252个B非零；434个基座参数无梯度且数值保持，108个bias仅发生bitsandbytes内部精度转换。
PyTorch峰值allocated4,657,840,640 bytes；无optimizer状态，不可外推完整训练/4096tokens显存。
命令check_qlora_loading.sh --backward；报告docs/run_logs/qlora-backward-VKJlYdcw/SUMMARY.md。
CPU回归42 passed、16 warnings（12.09s）；未改学生实现/main，未训练更新或保存。

学生main的真实QLoRA Trainer单步已通过：8条、global_step1、loss1.1438109874725342、
grad_norm0.8728765249252319；252个LoRA张量更新、434个冻结基座参数数值保留。
Trainer runtime7.8915秒、整卡采样峰值15,285MiB（含背景）；不外推4096tokens或完整训练。
报告docs/run_logs/3b-smoke-SQf36DFi/SUMMARY.md；未保存adapter，未删除或迁移代码。
最新CPU回归48 passed、16 warnings（16.11s）。

收工决策（2026-08-31，待实现）：训练参数统一由YAML提供，CLI仅保留--config选择文件；
QLoRA开关迁入YAML，main内教学smoke模式移入docs。测试/复现记录保留，清理只针对确认无用的代码。
学生负责main编排，教师负责工程迁移与验收；今晚没有实施删除、迁移或启动训练。

当前唯一下一步：**明日2026-09-01整理统一配置的正式入口，通过启动前检查后启动Stage-A SFT**。
真实QLoRA单步不等于正式训练完成；保存重载、dev评估与完整checkpoint恢复仍须验收。

## 十三、项目书维护规则

- 每次只保留一个“下一次唯一任务”。
- 新决策写入“已确认决策记录”，并注明证据和影响范围。
- 未经验证的想法放在“尚未决定”，不能写成既定方案。
- 实验完成后必须填写真实配置、日志和指标；空缺项保持空白，不推测。
- 已失效、重复或与 MedSearch-R1 无关的内容应删除或移入历史记录。
- 项目书只记录稳定结论和检查入口；大段学习笔记另建文档，并从这里链接。
