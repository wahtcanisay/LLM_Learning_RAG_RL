# MedSearch-R1

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

2026-08-31 调整：暂缓 Stage-B replay 数据组合，不再将其作为接入 Trainer 的前置条件。
当前 Stage-B 入口保持只读取 Search100；这不意味着已证明“不混合更好”。Stage-A 已冻结的
65/20/10/5 数据配方不变，本次不重新清洗、合并或导出任何数据。

### 职责边界

- **MedSearch**：拥有 Stage-A/Stage-B 数据语义、Search-R1 序列格式、loss mask、训练配置和统一评测；
- **MedicalGPT**：提供已经验证过的 Qwen 模型加载、LoRA/QLoRA、Trainer、checkpoint 和日志能力；
- **Search-R1/veRL**：在后续 RL 阶段提供在线搜索循环、rollout、reward 和 LoRA-GRPO；
- 当前不建设名为 `agent/` 的模块；医疗检索 Agent 是完成 SFT 与 RL 基础后的后续工作。

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

**2026-08-31收工；明日目标为统一配置、整理正式入口，在启动前检查通过后启动Stage-A SFT。**

收工决策（待明日实现，今晚未删除代码或启动训练）：

- 训练参数统一放YAML，包括是否使用QLoRA；命令行只保留--config选择配置文件，不另设训练参数覆盖来源。
- 从正式main移出--model-smoke、--trainer-smoke、--train-smoke等测试模式；测试及诊断保留docs。
  保存行为由正式配置/流程管理，不作为教学smoke开关；保留运行配置、种子、指标和失败证据。
- max_steps、batch size、学习率等仍是合法训练参数；去掉smoke专用约束，不盲删训练能力。
- 学生负责main核心编排，教师负责工程迁移及回归；清理前核对具体对象，不删除数据、模型或有效实验记录。
- QLoRA保存重载、dev评估与完整断点恢复尚未完成，明日启动前仍须核对相关关口。

以下保留已验收实现与历史命令；其中CLI开关尚未迁移，不代表上述新方案已经落地。

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
