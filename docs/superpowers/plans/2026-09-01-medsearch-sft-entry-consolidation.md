# MedSearch SFT 正式入口整理实施计划

> **执行边界：** 学生编写正式 `main()` 核心编排；教师负责 YAML 工程骨架、`docs/tests` 测试迁移、失败定位和最终验收。步骤使用复选框跟踪。

**目标：** 训练行为统一由 YAML 配置，正式 CLI 只保留 `--config`，在可保存、可恢复且记录完整的前提下启动 Stage-A QLoRA SFT。

**架构：** `MedSearch/sft/train.py` 保留数据、量化、模型装配、Trainer、训练与 adapter 保存接口；正式 `main()` 只按顺序调用这些接口。所有 smoke 子集、参数哈希、显存采样和诊断逻辑继续保留在 `MedSearch/docs/tests/`，不进入生产入口。

**技术栈：** Python、PyTorch、Transformers Trainer、PEFT、bitsandbytes、YAML、pytest；运行路径为 Windows CMD → WSL → Docker `llm-pytorch`。

---

## 文件职责与边界

- 修改 `MedSearch/configs/stage_a.yaml`：Stage-A 正式 QLoRA 训练配方，是今日唯一启动配置。
- 修改 `MedSearch/configs/stage_b.yaml`：仅补齐统一 schema；保持未启动、未验证的 Stage-B 配方状态。
- 修改 `MedSearch/sft/train.py`：正式 CLI、配置驱动的模型路径、训练、恢复和最终 adapter 保存编排。
- 修改 `MedSearch/docs/tests/test_sft_cli.py`：验证正式 CLI 只接受 `--config`，以及 YAML 控制 QLoRA。
- 修改 `MedSearch/docs/tests/test_sft_step.py`：删除对生产 `prepare_train_smoke` 的依赖；保留真实 Trainer 单步检查。
- 修改 `MedSearch/docs/tests/test_sft_train.py`：批次描述在测试内直接断言，不要求生产诊断函数。
- 修改 `MedSearch/docs/tests/test_sft_save.py`：移除旧 `--save-adapter` CLI 检查；继续验证 adapter 原生保存/重载。
- 修改 `MedSearch/docs/tests/run_sft_3b_smoke.py`：在教师脚本内构造8条、`max_steps=1`、临时输出目录；调用正式入口时仅传 `--config`。
- 保留其余 `MedSearch/docs/tests/`：量化、加载、准备、编码和参数更新证据不能因“剪枝”删除。
- 不修改冻结数据，不删除现有 `MedSearch/outputs/smoke/` 和 `run_logs/` 证据。

## 统一配置决策

- 正式 CLI：仅 `--config PATH`。
- YAML 顶层新增 `use_qlora: true`；不再提供 `--qlora` 覆盖。
- YAML 顶层新增 `resume_from_checkpoint: null`；仅传给训练调用，不传给 `TrainingArguments`。
- Stage-A 正式训练：16,077条，1 epoch，batch size 2，梯度累积4，有效batch 8，约2,010次更新。
- Stage-A 使用 `max_steps: -1`，不能继续使用 smoke 的1步限制。
- Stage-A checkpoint：按steps保存、每250步、最多保留2个；必须包含optimizer/scheduler/step，最终另存PEFT adapter与tokenizer。
- Stage-A 输出使用新的未占用目录，不能覆盖现有smoke和未来正式结果。
- 当前不在训练中做dev评估；328条dev在训练后统一评测。README明确这不是省略评测。
- Stage-B 只补 `use_qlora`/`resume_from_checkpoint` schema，本日不得启动或擅自确定Search100 epoch配方。

## Task 1：教师写入口新契约测试（RED）

- [x] 修改 `MedSearch/docs/tests/test_sft_cli.py`，真实解析只允许 `--config`；`--qlora`、`--model-smoke`、`--trainer-smoke`、`--train-smoke`、`--save-adapter` 都应被 argparse 拒绝。
- [x] 在同一测试中使用CPU假对象隔离昂贵加载，验证 `use_qlora=true` 的顺序为 load → prepare → attach → trainer → train → save；`false` 跳过prepare。真实量化继续由Docker smoke验收。
- [x] 修改 `test_sft_train.py`，要求两份受检YAML都具有布尔 `use_qlora` 和 `resume_from_checkpoint`。
- [x] 运行：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh MedSearch/docs/tests/test_sft_cli.py MedSearch/docs/tests/test_sft_train.py MedSearch/docs/tests/test_sft_step.py`。
- [x] 结果：12个预期RED、9个通过；失败仅来自旧CLI/旧main、缺YAML字段和缺恢复参数，无fixture或导入错误。

## Task 2：教师补YAML骨架，学生整理main（GREEN）

- [x] 教师给两份YAML补统一字段，并把Stage-A改为正式1-epoch、checkpoint可恢复的配置；Stage-B保持未启动配方说明。
- [ ] 学生让 `main()` 只解析 `--config`，并从 `config["use_qlora"]` 决定量化、准备和LoRA路径。
- [ ] 学生复用现有 `build_dataloader`、`build_quantization_config`、`load_base_model`、`prepare_qlora_model`、`attach_lora`、`build_trainer`、`run_training`、`save_adapter`；不重写底层实现。
- [ ] 学生让 `resume_from_checkpoint` 进入真实 `trainer.train(...)` 调用边界；为null时从头训练，有路径时恢复。
- [ ] 学生在训练结束后保存最终adapter/tokenizer，并写出实际配置、命令行、Git HEAD、数据/源码哈希和训练metrics。
- [ ] 不把smoke抽样、临时目录、参数哈希或显存采样搬进新的生产helper。

## Task 3：教师迁移smoke与清理生产代码

- [ ] 将8条抽样、临时输出目录和`max_steps=1`覆盖移入 `run_sft_3b_smoke.py`，教师脚本通过临时YAML调用正式 `main --config`。
- [ ] 将 `describe_model`/`describe_batch` 的断言放进教师脚本或测试，不要求生产入口打印诊断。
- [ ] 从 `train.py` 删除只被旧smoke使用的 `prepare_train_smoke`、`describe_model`、`describe_batch` 及对应未使用import。
- [ ] 更新 `test_sft_step.py`、`test_sft_save.py` 和 `run_sft_3b_smoke.py`，保留原有单步、冻结基座、LoRA更新与保存重载能力。
- [ ] 运行：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`；预期全部通过，第三方已知弃用警告单独记录。

## Task 4：启动前关口

- [ ] 对正式Stage-A配置做只读检查：16,077条、输出目录未被占用、`max_steps=-1`、1 epoch、有效batch8、QLoRA启用、checkpoint策略有效。
- [ ] 运行8条真实 QLoRA smoke，并保存/reload adapter；检查global_step1、loss/grad有限、LoRA更新、基座不变、重载一致。
- [ ] 用tiny CPU checkpoint执行“训练1步→释放→恢复到下一步”，证明optimizer/scheduler/global_step恢复，不仅adapter重载。
- [ ] 保存启动前报告；任何一项失败都不得启动正式训练。

## Task 5：启动Stage-A SFT

- [ ] CMD命令只使用 `wsl.exe` 调用Docker内正式入口，并只传 `--config MedSearch/configs/stage_a.yaml`。
- [ ] 启动后确认实际读取 `use_qlora=true`、样本16,077、预计更新约2,010步、输出目录和Git/配置/数据哈希正确。
- [ ] 记录初始loss、grad norm、显存峰值和checkpoint产生情况；启动成功不等于训练完成或效果提升。
- [ ] 训练中断时使用YAML中的 `resume_from_checkpoint` 恢复，不添加临时CLI覆盖。

## 自检

- 配置只有一个训练真源；CLI无训练行为开关。
- smoke能力迁移而非删除；正式代码不承担测试诊断。
- 正式训练有周期checkpoint和最终adapter，两者用途明确不同。
- 未决定的Stage-B replay和epoch配方没有被本计划偷偷固定。
- 学生仍是正式main的实现者；教师只搭骨架、写测试和验收。
