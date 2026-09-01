# 当前阶段

**阶段 5：MedSearch-R1 医学证据搜索 Agent——两阶段 SFT 数据实现。**

阶段 1、阶段 2 已完成并冻结，阶段 3 已完成最小 MedicalGPT SFT smoke，阶段 4 已完成
Search-R1 3B 单步 GRPO 保底架构。MedSearch-R1 的 SFT 数据与隔离边界已经冻结，Stage-A/
Stage-B tokenization、统一数据入口与 Qwen2.5-3B + LoRA 模型装配均已通过验证。2026-08-31
按学生要求暂缓 Stage-B replay；学生 `build_trainer()` 已通过 CPU 验收，完整回归 22 passed。
学生 `run_training()` 已验收：返回完整真实 metrics 与顶层 global_step；tiny CPU 单步、
有限 loss/梯度、LoRA 更新和基座不变均通过。最新完整回归 27 passed；真实3B也已完成
Stage-A 8条单步验证，loss=1.173087、整卡峰值18,139 MiB。保存接口与测试骨架已搭建，
学生 `save_adapter()` 已通过 tiny CPU 保存/重载验收；最新完整回归 31 passed，无 xfail。
真实 3B 单步保存/重载也已通过：504 个 adapter 张量与 434 个基座张量一致，固定输入最后
位置的全词表 logits 最大差 0。学生 `build_quantization_config()` 已验收，最新 CPU 回归
37 passed，无 xfail。随后已在 Docker 安装 bitsandbytes 0.50.2 并通过微型 NF4 GPU 前向；
学生量化加载分支现已通过：CPU 回归 40 passed、16 条第三方弃用警告，无 xfail；
真实 Qwen2.5-3B 的 252 个 NF4 双重量化层及单次前向通过。尚未验证 QLoRA 训练。
学生 prepare_qlora_model 已通过CPU验收；最新回归42 passed、16条警告，无xfail。
真实4-bit准备与LoRA单样本反向传播已通过：515tokens、loss1.9581148624420166，
504个LoRA张量梯度有限；无optimizer更新，不能称完整QLoRA训练通过。
2026-09-01 正式入口与恢复分支已验收，Stage-A QLoRA 已在 Docker 后台启动并接入W&B；
当前run `gufa7zwo`，首次观测至少17/2010步，训练尚未完成，不能声称效果提升。

```text
NQ / HotpotQA Parquet
  → Search-R1 多轮 SEARCH / ANSWER rollout
  → 原项目 HTTP 检索服务
  → 开放域 QA Exact Match reward
  → outcome-based GRPO
  → actor policy update
```

当前学习范围固定为 Search-R1 官方数据协议、Agent loop、检索接口、规则 reward、
GRPO advantage 和 actor update；暂不讨论任何领域迁移。

# 当前项目

**MedSearch-R1：基于领域微调与强化学习的医学证据搜索 Agent。**

当前设计入口：

- 项目书：`MedSearch/PROJECT_BOOK.md`
- SFT 训练管线参考：`MedicalGPT/`
- 后续多轮协议参考：`Search-R1/search_r1/llm_agent/generation.py`

# 本周目标

1. 推进“对话型 Medical-SFT → Search100”两阶段 SFT，Stage-B 是否加入 replay 暂缓讨论。
2. 审计候选单轮、多轮和 supervised rationale 数据的真实样本。
3. 建立 SFT、Search100、RL 与最终测试集的严格隔离表。

# 今日唯一任务

**2026-09-01：持续观察正式 Stage-A QLoRA 训练，确认首个 checkpoint 与最终产物。**

学生正式main与恢复分支已通过；教师已迁移冒烟到临时YAML、完成QLoRA保存重载和tiny断点恢复。
正式训练PID7819，W&B run `gufa7zwo`；当前只观察真实曲线、checkpoint与最终adapter，
不改训练参数、不启动Stage-B、不把进程运行包装成模型效果提升。

# 完成标准

- YAML作为训练参数唯一来源；--config仅选择文件，不重复覆盖训练参数；实际QLoRA设置写入运行记录；
- 正式main不携带smoke教学模式，复用已有接口；测试移出入口但不删除，回归行为保持；
- 启动前核对训练规模、输出目录、种子、配置记录及保存方式，不把smoke的8条/max_steps1误作正式训练设置；
- QLoRA保存重载、dev评估和完整断点恢复仍是待办，不因整合而删除验证关口；
- 明日先完成相应启动前检查，再启动Stage-A SFT；不将“进程启动”视为训练效果通过。

# 已完成

## 2026-09-01：Stage-A正式QLoRA启动并接入W&B（运行中）

- 学生将量化开关职责放在main：量化配置函数只接收dtype，模型加载显式接收已解析dtype；
  教师旧测试已按新接口迁移，未要求学生退回旧设计。
- 完整CPU回归：53 passed、18 warnings；真实QLoRA单步保存/释放/4-bit重载通过，
  adapter张量一致，固定输入logits最大差0.0；tiny checkpoint恢复optimizer/scheduler/global_step并继续到第2步。
- 正式配置：16,077条、max_length1024、batch2、累积4、有效batch8、1 epoch、预计2010步，
  QLoRA NF4双重量化/BF16、LoRA r8/alpha16/dropout0.05、lr2e-4、seed20260831。
- Docker后台PID7819；W&B project `MedSearch-R1`，run ID `gufa7zwo`。
  首次记录step10：loss1.174、grad_norm0.4244、lr0.0001991；观察时至少运行到step17。
- 运行清单：`MedSearch/outputs/stage_a/run_manifest.json`；训练日志：
  `MedSearch/outputs/stage_a/train.log`。训练尚未结束，尚无最终adapter/dev指标或能力结论。

## 2026-09-01：正式SFT入口教师骨架与RED契约

- 实施计划：`docs/superpowers/plans/2026-09-01-medsearch-sft-entry-consolidation.md`。
- Stage-A YAML改为正式QLoRA配方：16,077条、1 epoch、max_steps=-1、batch2、累积4、
  每250步保存、最多2个checkpoint；`resume_from_checkpoint` 初始为null。
- Stage-B仅补统一schema，adapter路径指向`MedSearch/outputs/stage_a/adapter`；未启动、未决定replay。
- 教师测试锁定：CLI只允许--config、YAML控制QLoRA、正式训练后保存adapter、恢复路径原样转发。
- 聚焦回归：19 passed、9 failed、14条第三方警告；9个失败全部属于学生尚未实现的旧main或恢复分支，
  无fixture、导入、配置保存策略故障。尚未运行GPU或启动完整训练。

## 2026-08-31：学生main真实QLoRA Trainer单步通过

- 命令 `wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --qlora`，退出码0；
  教师仅扩展docs探针并调用学生main，未改学生函数/main/YAML或数据，未保存adapter。
- Qwen2.5-3B NF4/double quant/BF16，r8/alpha16/dropout0.05，seed20260831；
  固定8条Stage-A、batch2、梯度累积4、max_steps1、max_length1024，最长实际842tokens。
- global_step1、loss1.1438109874725342、grad_norm0.8728765249252319、Trainer runtime7.8915秒。
  252/504个LoRA张量更新；434个基座参数无梯度且数值保留，108个bias仅库内部dtype转换。
- PyTorch训练峰值allocated7,081,850,880 bytes，reserved12,297,699,328 bytes；
  整卡200ms采样250点峰值15,285MiB（含背景），非4096tokens可行性或性能对比结论。
- CPU回归48 passed、16 warnings（16.11s）；报告MedSearch/docs/run_logs/3b-smoke-SQf36DFi/SUMMARY.md；
  产物MedSearch/outputs/smoke/stage_a-_y_9hw2d/，含配置、实际QLoRA参数、指标、全量参数检查。
- GPU进程已退出；未保存更新后权重，不构成Checkpoint A。QLoRA保存重载/dev评估/断点恢复待验收。
- 可开始代码整合：学生负责正式main编排，教师处理docs中的检查逻辑；本轮未删除或迁移代码。

## 2026-08-31：学生修复LoRA返回缩进，主入口CPU验收通过

- attach_lora的return已由学生退到if/else之后，教师未修改学生代码。
- `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：48 passed、16 warnings in 19.09s，无xfail。
  新增入口检查覆盖QLoRA开/关与梯度检查点开/关；先前None返回导致的连锁失败已消失。
- CPU使用tiny加载替身，不等于真实QLoRA Trainer单步；下一项运行学生入口的GPU单步更新验收。

## 2026-08-31：QLoRA主入口检查发现LoRA返回缩进回归（待学生修复）

- 学生main已正确映射dtype并调用prepare_qlora_model；未代改学生实现。
- 新增docs/tests/test_sft_cli.py主入口开/关QLoRA、开/关检查点的4项CPU测试，昂贵加载以tiny替代，
  其余准备/挂载/data路径真实执行，不代表GPU训练验收。
- 完整回归35 passed、13 failed、16 warnings（17.31s）：attach_lora的return peft_model
  缩进到else内，新建adapter分支隐式返回None，引发入口/Trainer/保存等连锁失败。
- 当前先由学生将返回语句退一级至if/else之后，再重跑；暂停GPU Trainer单步验收。

## 2026-08-31：QLoRA命令行参数骨架

- 学生表示不会引入参数，教师仅添加argparse的--qlora/store_true，默认False。
  args.qlora已可读取，量化模型分支仍未接入，留学生填写；不代写训练编排。
- 参数测试先出现2项缺少参数的失败；添加后完整CPU回归
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：44 passed、16 warnings in 11.01s。
- 未启动GPU训练或更改其他函数；当前唯一任务仍是学生编排QLoRA调用路径。

## 2026-08-31：真实4-bit准备与LoRA梯度联检通过

- 命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward`，退出码0。
  复用教师加载检查新增可选backward模式，不改学生函数/main/YAML或数据。
- Qwen2.5-3B、NF4/double quant/BF16、LoRA r8/alpha16/dropout0.05、seed20260831；
  Stage-A第0条medqa-dialog_4790，batch1×515，281个监督token，loss1.9581148624420166。
- 252个量化层保留；504个LoRA张量梯度有限，252个B非零、A首次为零。
  434个冻结基座参数无梯度且数值保留；108个bias仅发生库内部FP32→BF16转换。
- 首次教师全量字节哈希断言失败；核对bitsandbytes源码并记录变化参数后，只允许指定bias
  精确数值不变的dtype转换，其他326个基座参数仍字节一致。失败日志保留，未改学生实现。
- forward+backward0.9319077680120245秒；PyTorch峰值allocated4,657,840,640 bytes，
  reserved4,752,146,432 bytes；无optimizer状态，不是整卡或完整训练显存，不外推4096tokens。
- CPU命令 `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：42 passed、16 warnings in 12.09s。
- 报告：MedSearch/docs/run_logs/qlora-backward-VKJlYdcw/SUMMARY.md。
  未运行optimizer更新/Trainer训练/保存；下一项调用编排归学生，教师继续负责验收。

## 2026-08-31：学生低比特准备接口 CPU 验收通过

- prepare_qlora_model 复用PEFT一次，梯度检查点True/False均原样传递，kwargs固定use_reentrant=False，
  返回原模型；tiny CPU检查基座冻结、非量化参数转FP32、数值/配置/设备保留均通过。
- `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh MedSearch/docs/tests --runxfail`：
  42 passed、16 warnings in 14.78s。移除两项占位标记后默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：42 passed、16 warnings in 11.11s，无xfail。
- 未修改学生函数或main，未启动GPU、训练或保存。测试仍在MedSearch/docs/tests/；
  CPU tiny量化标记只触发PEFT准备分支，不能证明实际Params4bit保留或QLoRA梯度正确。
- 下一项复用现有加载/准备/attach_lora做真实4-bit准备与LoRA梯度联检，不代写正式main。

## 2026-08-31：低比特训练准备教学接口（待学生填写）

- 已核对MedicalGPT/training/supervised_finetuning.py:920-923与当前PEFT0.20.0源码。
  MedicalGPT在创建新LoRA前准备量化基座；本项目额外保持既有non-reentrant检查点设置一致。
- 新增prepare_qlora_model(model, use_gradient_checkpointing)，只留契约和NotImplementedError；
  导入PEFT原生函数，不改加载/挂载/Trainer实现或main/YAML，不安装依赖、不运行GPU。
- 教师测试先出现2项缺少接口的真实失败；骨架后命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：
  `40 passed, 2 xfailed, 16 warnings in 11.35s`。预期失败仅限占位NotImplementedError。
- 测试位置MedSearch/docs/tests/test_sft_kbit_preparation.py，覆盖开/关、原生调用、返回对象、
  基座冻结与精度/数值边界。tiny CPU量化标记是测试替身，不冒充4-bit GPU准备或反向传播。

## 2026-08-31：学生量化加载分支验收通过

- CUDA 提前检查、单卡 device_map、量化对象原样传递和非量化兼容均通过；未修改学生代码。
- 默认命令 `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：
  `40 passed, 16 warnings in 15.08s`，已撤去两项占位 xfail，警告为已知第三方弃用提示。
- 真实加载命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh`，退出码 0。
  Qwen2.5-3B-Instruct、seed20260831、NF4/double quant/BF16，252 个 Linear4bit 层；
  全部参数在 cuda:0，35-token 单次前向 logits 有限。加载耗时 27.8416 秒。
- PyTorch 加载及单次前向峰值 allocated 2,149,335,040 bytes，reserved 2,176,843,776 bytes；
  不是整卡占用或训练显存，不能据此外推 4096-token QLoRA 训练。GPU 验证进程已退出。
- 报告：`MedSearch/docs/run_logs/qlora-load-jxVC6GU3/SUMMARY.md`，含版本、源码哈希和原始日志。
  首次教师报告引用缺失的 hf_device_map 属性而失败，保留在 qlora-load-Gd9jRIwx；
  改为报告实际参数设备后重跑通过，学生无需为教师报告增加属性。
- 不挂 LoRA、不 backward、不训练、不保存模型。下一项为低比特训练准备，正式 main 仍归学生。

## 2026-08-31：量化加载骨架与 Docker 环境微型检查

- Docker `llm-pytorch` 内新增 bitsandbytes0.50.2，使用 --no-deps 保留原 torch/transformers。
  PEP668保护在明确工具批准后仅对容器安装覆盖，主机Python未修改；版本固定在requirements-qlora.txt。
- 真实RTX5090/CC12.0/CUDA13.0微型NF4+double-quant层前向通过，非Qwen加载或训练。
  命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_environment.sh`。
  日志：`MedSearch/docs/run_logs/qlora-env-2p8oGYHS/check.log`。
- load_base_model新增仅限关键字quantization_config=None；保留非量化实现，量化分支仅留
  NotImplementedError，CUDA检查/加载调用由学生填写。未改main/YAML或已验收配置函数。
- 新测试初始1 passed、2 failed（缺少接口参数）；骨架后完整回归
  `38 passed, 2 xfailed, 16 warnings in 13.06s`。两项占位待验收，第三方弃用警告未屏蔽。
- 参数传递CPU测试不等于GPU实际量化；学生实现后再验收真实加载。
- 安装命令、wheel哈希、依赖版本及容器持久性边界详见`MedSearch/docs/QLORA_ENVIRONMENT.md`。

## 2026-08-31：QLoRA 量化配置验收通过

- 学生实现 `build_quantization_config()`：关闭返回 None，开启返回原生 BitsAndBytesConfig，
  4-bit/NF4/double quant 正确，计算 dtype 按调用方传入，不同时启用 8-bit。
- `--runxfail` 完整回归 37 passed in 14.38s；教师撤去待实现标记后，默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh` 得到 `37 passed in 15.64s`，无警告、无 xfail。
- 配置检查覆盖关闭分支及 BF16/FP16/FP32 三种 compute_dtype；未修改学生核心实现或 main。
- 本轮没有安装 bitsandbytes、加载量化模型或运行 GPU 训练。结论仅为配置对象正确，
  不能称为 QLoRA 训练跑通。下一项为量化加载，依赖先由教师核验，关键适配仍交学生。

## 2026-08-31：QLoRA 量化配置教学骨架（待学生填写）

- 新增 `build_quantization_config(use_qlora: bool, compute_dtype: torch.dtype)` 的签名和契约，
  函数体保留 NotImplementedError，返回类型为 BitsAndBytesConfig 或 None。
- 复用 MedicalGPT 的 NF4/double quant 配置思路；未修改 main、load_base_model、YAML 或既有
  学生实现。QLoRA 属于既有学习路线，不是医学任务额外的自定义量化算法。
- 教师测试 `MedSearch/docs/tests/test_sft_quantization.py` 先出现 4 个接口缺失红灯；
  骨架后完整回归 `33 passed, 4 xfailed in 10.87s`。预期失败限 NotImplementedError，
  学生完成后由教师撤去标记、真实检查配置对象。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- Docker 中 bitsandbytes 尚未安装；本地 Transformers 5.10.4 配置构造没有要求该包，
  不阻碍纯配置任务。真实量化加载前须核验依赖与硬件兼容。本轮未安装依赖或启动 GPU。

## 2026-08-31：真实 3B BF16 LoRA 保存/重载通过

- 最终命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --save-roundtrip`，退出码 0；
  最新 CPU 回归 `33 passed in 10.22s`。
- 固定 8 条 Stage-A、seed20260831、global_step1、loss1.173087239265442、grad_norm0.8145956993，
  Trainer runtime7.3378 秒。252 个 LoRA 张量更新，434 个冻结基座张量哈希不变。
- 保存后释放训练模型，再重载相同底模与 adapter；504 个 adapter 张量、434 个基座哈希、
  tokenizer 均一致。同一 BF16 autocast/SDPA/eval 下最后位置全词表 logits 最大差0。
- adapter 权重59,933,632 bytes；整卡400个采样点峰值18,062 MiB（含背景）。GPU 进程已退出。
- 成功报告：`MedSearch/docs/run_logs/3b-smoke-SrvIu01k/SUMMARY.md`；产物：
  `MedSearch/outputs/smoke/stage_a-kv1h2i46/`，含 adapter、配置、指标及前后 logits 文件。
- 此前两次教师探针失败均保留：Q9PJhbEt 为 chat template 默认返回格式误用；8byqSe5Q 为
  保存前后 autocast 不一致。仅修正 docs 内脚本并补测试，未改学生训练实现或 main。
- 本轮不是 QLoRA、正式 Checkpoint A 或医学效果提升。后续正式 main 由学生编写；
  下一项唯一任务改为 QLoRA 4-bit 配置，不提前代写完整流程。

## 2026-08-31：save_adapter CPU 验收通过

- 学生增加 `save_embedding_layers=False`；原生模型/tokenizer 保存与隐式 None 返回符合契约。
  教师未修改学生实现，也未要求删除无害的重复 mkdir 或显式重写默认 safe_serialization=True。
- 修正前专项测试 3 passed、1 failed：tiny 与引用配置词表大小不同，auto 分支额外保存 embedding；
  修正后全量 `--runxfail` 回归 31 passed in 13.87s。撤去待实现标记后，默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh` 得到 `31 passed in 11.76s`，无警告、无 xfail。
- 验证 safetensors 仅含 LoRA、adapter 权重逐项一致、tokenizer 词表/特殊 token/chat template 保留、
  同底模副本重载后的固定输入 logits 一致，以及 attach_lora 可训练加载分支。
- tiny adapter 人工设为非零，仅用于序列化验收；不宣称真实 3B 保存已验证或医学效果提升。
  未启动 GPU 训练、未生成正式 Checkpoint A；下一唯一任务为真实 3B 单步保存/重载验证。

## 2026-08-31：adapter 保存教学骨架（学生实现待验收）

- `sft/train.py` 新增 `save_adapter(model, tokenizer, output_dir) -> None`，函数体保留
  NotImplementedError；教师接好仅限 `--train-smoke` 的 `--save-adapter` 参数与独立 adapter 子目录。
- 参考 MedicalGPT 原生保存方法；契约要求只保存 LoRA/config/tokenizer，不合并底模，
  不保存 embedding，不改已验收的 run_training 返回值，不做 optimizer/step 恢复。
- 新增教师测试 `MedSearch/docs/tests/test_sft_save.py`：非法模式提前拒绝，以及 adapter
  权重/tokenizer/同底模固定输入 logits 的 round-trip 验收。tiny adapter 人工设为非零仅用于
  序列化检查，不冒充训练结果；重载底模使用原底模副本，避免随机初始化差异。
- 初始红灯为 4 failed（接口与参数未创建）；保存重载测试保留严格 NotImplementedError xfail，
  等学生实现后再撤去，不能将预期失败算作保存功能通过。本轮未运行真实 3B 或产生正式 adapter。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 最新 CPU 回归：`30 passed, 1 xfailed in 13.41s`，无警告；仅学生保存实现待验收。

## 2026-08-31：真实 Qwen2.5-3B Stage-A 单步 smoke 通过

- CPU回归27 passed in 14.26s；GPU运行退出码0，复用学生run_training和唯一main入口。
- BF16 LoRA rank8/alpha16/dropout0.05、学习率0.0002、batch2×累积4、max_steps1、seed20260831。
  固定8条样本，实际长度143～842，配置上限1024。
- 真实global_step=1、loss=1.173087239265442、grad_norm=0.8148247599601746、
  Trainer训练耗时6.3281s；252/504个LoRA张量变化，434个冻结基座张量全量哈希变化为0。
- 整卡显存306点采样峰值18,139 MiB/32,607 MiB；PyTorch peak allocated 9.134 GiB、
  reserved 14.113 GiB。整卡包含背景占用；真实训练进程已结束，显存释放。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh`。
  报告：`MedSearch/docs/run_logs/3b-smoke-eqK4l3I8/SUMMARY.md`，同目录保存原始日志和显存CSV。
  有效配置、代码/数据哈希、metrics与参数验证保存在`MedSearch/outputs/smoke/stage_a-_rs5vew5/`。
- 没有改动学生核心实现或冻结数据；只增加docs内教师观察脚本。无OOM；初始化较慢后自然完成，
  未禁用监控或修改训练逻辑。tokenizer特殊token提示为Trainer配置对齐，不是训练失败。
- 不宣称医学效果提升，不外推Stage-B4096或全量训练；没有保存adapter，本次不是Checkpoint A。

## 2026-08-31：run_training 最终验收通过

- 学生通过字典展开保留完整 TrainOutput.metrics，并加入真实 global_step；新字典没有修改
  原 metrics。教师未修改学生实现，不再增加该接口的要求。
- 教师修正新增测试的错误假设：当前 Trainer 回调可能从返回 metrics 移除 total_flos，不能
  将 log_history 作为返回字段的真值。此前关于“必须补回 total_flos”的判断撤回；现在直接
  观察真实 TrainOutput，逐字段验证返回值一致，真实训练没有被 mock 替代。
- 撤去 xfail 后完整回归 `27 passed in 12.94s`，命令：
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 验证真实 tiny 训练调用恰好一次、global_step=1、有限 loss/非零有限梯度、LoRA 参数更新、
  基座逐项不变及完整返回指标。未运行真实 3B，未生成训练 checkpoint。
- 沟通改进：今后预留接口应明确返回结构示例，区分“消费端读取哪些字段”和“返回时保留哪些
  字段”，不再让学生因教师说明或测试假设不清反复返工。

## 2026-08-31：run_training 扁平结构复验（指标完整性待修正）

- 学生已修复返回层级；tiny CPU 实际通过 global_step=1、有限 loss/非零有限梯度、LoRA
  参数变化、基座参数逐项不变检查，未生成 checkpoint。
- 教师补上此前遗漏的“保留全部原有 metrics”断言；实际缺失 total_flos、
  train_samples_per_second、train_steps_per_second。未修改学生实现。
- 相同限定目录命令复验结果 `26 passed, 1 failed in 13.48s`。当前只要求完整保留指标，
  不扩展训练功能；未启动真实 3B。

## 2026-08-31：run_training 首次验收未通过

- 保存后的实现正确调用一次 trainer.train() 并读取真实 global_step，但返回
  `{metrics: {...}, global_step: ...}`，不符合下游直接读取顶层 train_loss 的接口契约。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh MedSearch/docs/tests --runxfail`。
  结果 `26 passed, 1 failed in 13.54s`，失败为 KeyError('train_loss')。
- tiny CPU 真实执行至 global_step=1，日志显示 loss 约4.136、grad_norm 约0.5355；这是合成
  token 小模型的管线检查，不是医学模型指标。参数更新/基座不变断言位于失败之后，尚未执行。
- 未修改学生代码；当前等待学生将返回结构改为扁平字典，未启动真实 3B 或正式训练。

## 2026-08-31：单步训练教学骨架（run_training 待学生实现）

- 教师新增 `prepare_train_smoke()` 和互斥 `--train-smoke`，只允许 Stage-A、max_steps=1、
  单进程/单设备；每次新建 smoke 输出目录，原配置和正式 Checkpoint A 目录不受影响。
- 模型/LoRA 初始化前 set_seed，固定抽取一次完整梯度累积的样本数；入口记录有效配置、
  样本 ID、Git HEAD、SFT 源码/数据哈希；训练成功后记录真实 metrics 和 Trainer log history。
- 学生 `run_training()` 仅预留 Trainer 训练调用与真实结果转换契约，未替学生完成实现。
- 新增红灯检查因接口缺失产生 `5 failed`；教师保护实现后，完整 CPU 回归
  `26 passed, 1 xfailed in 12.92s`。唯一预期失败为学生训练接口，未执行 backward。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。新增测试在
  `MedSearch/docs/tests/test_sft_step.py`，待学生完成后检查有限 loss/梯度、LoRA 更新与基座不变。
- 本轮未跑真实 3B、完整 CLI 训练路径或 GPU 显存测量，未产生训练 checkpoint；不能将骨架
  回归通过写成单步训练通过。

## 2026-08-31：学生 Trainer 接口验收

- 学生正确实现 `build_trainer()`，原样复用模型、tokenizer、Dataset/collator，返回 HF Trainer；
  不手写 DataLoader/optimizer，不调用 train，不改 labels 或解冻基座。
- 首次限定目录回归为 `19 passed, 1 failed`：教师断言错误地要求关闭 checkpointing 时保留
  原 cache。核对本地 Transformers 5.10.4 `trainer.py:608`，Trainer 会写入 args.use_cache，
  其默认值 False；修正测试与契约说明，未改学生执行逻辑。
- 撤去 xfail，补验 Stage-B 和完整 batch 三张量对比；CPU fixture 禁用无意义的 pin_memory。
  最终干净结果：`22 passed in 12.31s`，无跳过项或预期失败项。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`；位置：`MedSearch/docs/tests/`。
- 只测试 tiny LoRA 模型构造与真实 Stage-A/B 样本组 batch；global_step=0、optimizer/scheduler
  未创建、可训练参数仅 LoRA。未运行真实 3B Trainer、backward、训练指标或显存测量。

## 2026-08-31：Trainer 教学骨架（学生接口待实现）

- 按学生决定暂缓 Stage-B replay，当前保持只读取 Search100；Stage-A 冻结配方和全部数据未改动。
- 教师提供 `build_training_arguments()`、两个 YAML 的 `training` 配置及 `--trainer-smoke` 接线；
  `build_trainer()` 只留契约和 `NotImplementedError`，未替学生填写。
- 参考本地 MedicalGPT `training/supervised_finetuning.py` 的 Trainer 接线，复用 HF 原生类；
  配置非重入式 gradient checkpointing、保留 labels、禁用评估/保存/外部日志；本轮没有训练循环。
- 红灯验证：新测试因两个接口不存在产生 `3 failed`。最终 CPU 回归真实结果
  `18 passed, 2 xfailed in 12.25s`；两项 xfailed 仅代表学生接口未实现，不是验收通过。
- 运行命令（仓库根目录 CMD）：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 环境：现有 `llm-pytorch` 容器，Transformers 5.10.4、PEFT 0.20.0、Accelerate 1.14.0；
  测试禁用 CUDA 和联网，只验证 CPU 接线。没有真实 3B Trainer、backward、显存或训练指标。
- 学生填写后需移除严格 xfail 标记，再验收真实 tiny LoRA Trainer 与 batch；后续正式训练前
  仍需接好 dev 评估、保存/恢复和真实单步检查。

## 2026-08-30：Qwen2.5-3B + LoRA 模型装配验收

- 教师参照 MedicalGPT 已成功 SFT adapter，固定 BF16、Qwen 7 类投影模块、rank 8、alpha 16、
  dropout 0.05，并接好本地底模加载、参数边界检查与统一入口 `--model-smoke`；
- 学生完成 `attach_lora()`：Stage-A 用 `LoraConfig/get_peft_model` 新建 adapter，Stage-B 用
  `PeftModel.from_pretrained(..., is_trainable=True)` 续载，并对缺失 adapter 路径明确报错；
- tiny Qwen 真实完成新建、保存、续载和缺失路径测试；encoder、Dataset、入口与模型完整回归
  为干净的 `15 passed in 6.32s`；
- 真实 Qwen2.5-3B-Instruct BF16 CPU smoke 正常加载434组权重；LoRA 可训练参数
  `14,966,784`、总参数 `3,100,905,472`、占比 `0.4826585%`，没有非 LoRA 可训练参数；
- Stage-B 真实 adapter 尚不存在，因此未伪造 Checkpoint A，也未声称 Stage-B 3B 续训加载通过；
  当前只有 tiny adapter 续载边界通过。

## 2026-08-30：单一 SFT 数据入口 smoke 验收

- 教师参照 MedicalGPT 的参数入口、AutoTokenizer 与 collator 调用链，创建单一
  `MedSearch/sft/train.py`、`stage_a.yaml`、`stage_b.yaml` 和隔离测试；没有复制完整 Trainer 参数层；
- 学生完成 `build_dataloader()`：解析相对/绝对数据路径，复用 `load_samples()`、`SFTDataset`
  与 `build_data_collator()`，并把 batch size、shuffle、worker 数从配置传入 PyTorch DataLoader；
- encoder、Dataset 与统一入口完整回归真实结果为 `12 passed in 10.48s`；
- 正式 CLI smoke 共用同一入口：Stage-A 读取 16,077 条并输出 batch `[2, 683]`、上限 1,024；
  Stage-B 读取 100 条并输出 batch `[2, 1254]`、上限 4,096；两者均正常退出；
- 本轮只加载 tokenizer 和一个 CPU batch，没有加载模型、创建 LoRA/Trainer 或占用 GPU。

## 2026-08-30：统一 SFT Dataset 与动态 padding 验收

- 教师先在 `MedSearch/docs/tests/test_sft_data.py` 建立红灯测试和 `MedSearch/sft/data.py`
  接口骨架；学生完成 `.jsonl/.json` 读取与 Stage-A/Stage-B encoder 分派；
- 动态 padding 未手写重复循环，直接复用 MedicalGPT 已验证的 Transformers
  `DataCollatorForSeq2Seq`：input_ids 使用 tokenizer pad id、attention mask 补0、labels 补 `-100`；
- Dataset/encoder 隔离回归真实结果为 `9 passed in 8.01s`；
- 正式数据 smoke 读取 Stage-A train 16,077 条和 Search100 100 条；抽取首尾样本组成真实 batch，
  Stage-A 长度为 515/409、batch shape `(2, 515)`，Stage-B 长度为 1005/1258、batch shape
  `(2, 1258)`，三张量 shape 一致；
- 当前采用按 `__getitem__` 即时编码的最小实现；只有真实 profiling 证明 tokenizer 成为训练
  瓶颈时才增加预编码缓存，不提前复杂化。

## 2026-08-29：Stage-B Search-R1 编码与长度边界验收

- 学生完成 `MedSearch/sft/preprocess.py::encode_stage_b()`：开头只渲染一次 Qwen user prompt，
  `think/search/answer/EOS` 参与 loss，环境 `information` 保留在 attention 中但 labels 为 `-100`；
- 修复 `step[type]` 将 Python 内置 `type` 误作字典键的问题，并为未知 trajectory type 增加
  明确 `ValueError`；保留 answer 后 `break`，不为已冻结合法轨迹增加重复结构校验；
- 本地 Qwen2.5-3B-Instruct 原生 `max_position_embeddings=32768`；Search100 编码最长约
  3,070 tokens，因此 Stage-B 固定 `max_length=4096`，超长直接报错而不静默截断；
- 教师修正隐藏测试中 prompt 示例 `<answer>B</answer>` 导致的首次匹配误判；Docker 最终结果
  为 `5 passed in 4.88s`，其中包含正式 100 条 Search100 全量 4,096-token 闸门；
- 后续 RL 训练策略确定为 LoRA-GRPO；训练速度、显存和有效 reward 组比例仍必须由真实 smoke
  记录，不能根据模型规模纸面宣称可行。

## 2026-08-29：Stage-A 对话编码与 loss mask 验收

- 学生完成 `MedSearch/sft/preprocess.py::encode_stage_a()`：直接读取冻结数据的 `messages`，
  复用 Qwen2.5 `apply_chat_template`，system/user labels 为 `-100`，assistant 内容和每轮 EOS
  参与 loss；没有引入额外中间 schema、Dataset 或 Trainer；
- 教师维护的隔离测试位于 `MedSearch/docs/tests/`，真实容器运行结果为 `1 passed in 8.28s`；
- 使用本地 Qwen2.5-3B-Instruct tokenizer 全量编码 Stage-A train/dev 共 16,405 条：最短
  48 tokens、最长 1,022 tokens、超 1,024 为0、空 target 为0，三数组长度全部一致；
- 学生能解释多轮中 assistant 必须进入 `accumulated`，否则 `previous_rendered` 会缺失历史并
  破坏下一轮增量切片；能解释 causal LM 在 `forward(labels=...)` 内部对 logits/labels 做移位，
  因此数据预处理不手动右移 labels。严格说该移位由模型 loss 实现，不是 Trainer 本身完成。

## 2026-08-29：Stage-A 最终质量修复

- 全量复核发现 reasoning 1,640 条缺顶层 `id`，以及2条 UltraChat general 样本含原始数据
  自带的 `U+FFFD`；其中一条还包含明显事实性幻觉，不能用标点替换修复；
- reasoning ID 已确定性补为 `medical_o1_{provenance.index}`；两条损坏 general 已整条删除，
  并从同一固定 UltraChat revision 补入2条经结构、主题和 Qwen 1,024-token 检查的样本；
- 总量和配方保持不变：train 16,077、dev 328，single/multi/reasoning/general 为
  10,663/3,281/1,640/821；最终 missing ID、重复 ID、`U+FFFD` 均为0；
- 新增可幂等运行的 `MedSearch/tools/finalize_stage_a.py` 和最小回归测试；
- `MedSearch/README.md` 明确教师/学生职责，下一步正式进入学生编写 SFT 数据入口。

## 2026-08-24：MedSearch SFT 工作区瘦身

- 当前活动根目录只保留 `.gitignore`、`PROJECT_BOOK.md`、正式 `data/` 和一个可恢复的
  `archive/`；不再并列堆放审计说明、抽样目录和一次性脚本；
- Stage-A 活动数据只保留 `stage_a_train.jsonl`、`stage_a_dev.jsonl`、
  `stage_a_manifest.json`；Stage-B 活动数据只保留正式 Search100、医学检索语料与 BM25 索引；
- 清洗、复核、构造、验证脚本及其测试，一次性样本、审计 Markdown、候选判定与失败档案统一
  收进 `MedSearch/archive/data_preparation/`，不参与当前 SFT 编码与训练；
- `PROJECT_BOOK.md` 是当前唯一项目说明来源；旧审计文档不再被活动文档引用。

## 2026-08-23：Stage-B Search100 精品检索轨迹完成

- 从 MedQA/MedMCQA official train 构造 gold-blind 候选；query、检索和最终决策阶段均不读取 gold；
- 真实调用 Search-R1 BM25 医学服务，语料为 MedRAG Textbooks + StatPearls；检索 observation 和稳定
  `document_id` 全部保存在本地审计侧文件；
- 生成 190 条候选（含最初 10 条 pilot），由 `gpt-5.6-sol/xhigh` 按答案、证据、引用和题目质量终审；
- 独立二次复审替换 9 条医学答案正确但存在明显 OCR/拼写噪声的样本，最终 100/100 通过；
- 正式训练集精确保留 100 条：80 条一轮搜索、20 条两轮搜索；顶层字段仅为
  `id/question/options/trajectory`，trajectory 仅含 `search/information/answer`；
- 正式训练文件为 `MedSearch/data/stage_b/search100_training.json`；一次性审计与失败候选
  已在冻结结论和哈希后清理，不进入学生的 SFT 工作区；
- 正式文件 652,019 bytes，SHA-256
  `e9796395e17f6a9c29520cb48758ed9cdbb083c1d231f9562e4d2ef2b25725fb`；
- 清理前自动验证：100 个唯一 ID、字段无审计信息泄漏、answer 均属于 options、轨迹顺序合法；
  当时 53 项测试通过。一次性构建、审计与测试脚本随后按学习边界清理。

## 2026-08-23：Stage-A 医学三池全量清洗

- 数据清洗职责改由教师代理负责，学生从训练搭建开始学习；
- 77,846 条 raw 经确定性规则得到 drop 3,638、review 9,572、keep 候选 64,636；
- 三个 `gpt-5.6-sol` 对 review 做一次性二元终审：accept 7,500、reject 2,072，不留人工队列；
- 目标 Qwen2.5 tokenizer 长度闸门排除 2,494 条，最终 eligible pool 为 69,642 条；
- 三个低成本子智能体完成跨域盲审，修复了多轮 answer-confirmation 过度删除和单轮 false keep；
- 医学三池清洗结果作为下方 65/20/10/5 当前配方的输入；已剪除被替代的 70/20/10
  构建结果，避免学生误用旧 manifest；
- 当前有效统计、字段与冻结配方统一见 `MedSearch/PROJECT_BOOK.md` 和 Stage-A manifest。

## 2026-08-23：Stage-A 加入 5% 通用对话 replay

- 固定 UltraChat 200k 官方 revision `8049631c405ae6576f93f445c6b8166f76f5505a`，只下载并
  校验 `train_sft` 三个 Parquet 分片，共 207,865 条；行数、字节数和官方 SHA-256 一致；
- 确定性规则排除医学、代码、数学、工具动作、时效问法、模板拒答、截断和伪亲历，得到
  121,509 条候选；强模型分层抽检 240 条，accept 151、reject 89，并将 reject ID 固定排除；
- Qwen2.5 真实 chat template 长度闸门保留 59,133 条，排除 62,376 条超 1,024-token 样本；
- 固定 seed `20260823` 重建 65/20/10/5 Stage-A：single 10,663、multi 3,281、reasoning
  1,640、general 821；train 16,077、dev 328，user 重叠 0，强审 reject 命中 0；
- Qwen2.5 Instruct 已有通用指令能力，因此通用数据只作为 5% replay；0%/5%/10% 留作后续
  单变量消融，不把 5% 写成已验证最优值；
- 原 70/20/10 版本已被本配方替代，不再作为当前训练输入。

## 2026-08-23：全量下载验证与首轮风险抽检

- 五个入选文件已全量下载并完成 JSON 解析、记录数与 SHA-256 检查，共 77,846 条记录；
- 固定 seed `20260823` 生成单轮 90 条、多轮 40 条、medical-o1 40 条风险审计清单；
- 三个低成本子智能体只读初筛，共标记 `DROP 21`、`REVIEW 46`、
  `provisional KEEP 103`；该审计集刻意富集风险样本，比例不能外推全库；
- 多轮样本发现缺图和答案机械确认，MedQA-dialog 的 30%～40% 改为待验证上限；
- 当前有效的数据边界和最终选择统一见 `MedSearch/PROJECT_BOOK.md`。

## 2026-08-23：四类 SFT 候选数据各 10 条实物审计

- 固定 Meerkat revision `f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` 与 medical-o1
  revision `fc2c9e8a37b38f38da6d449564a8c350b244aef4`，通过镜像范围读取原仓库文件；
  没有下载全量数据。
- 四类候选各 10 条已完成抽查；冻结数据来源后，一次性样本和独立审计说明已清理。
- ChatDoctor-cleaned 抽样出现 C. difficile 被误判为 ulcerative colitis、持续胸痛被直接
  归为 gastritis 等安全问题，不能全量直接训练。
- MTS-dialog 的现有 target 是 clinical note generation，不是医生多轮回复；第一版排除。
- MedQA-dialog 是 5～17 messages 的 GPT-3.5 合成多轮考试对话，条件保留为小比例数据；
  缺图、答案确认模板和 MedQA 测试污染必须过滤。
- medical-o1 English 条件保留为少量 supervised rationale；需过滤缺图、异常题面，并
  避免把重复的 `Complex_CoT` 与 `Response` 无设计地串接。
- 当前尚未找到足够可靠的普通医学对话主数据源，因此不启动全量清洗或训练。

## 2026-08-23：Meerkat 第二批四个子集各 10 条审计

- 从同一固定 revision 取得 LiveQA、MedicationQA、MedInstruct-52K 和 MedBooks-18-CoT
  各 10 条；宿主机镜像连接超时后，使用现有 `llm-pytorch` 容器访问完全相同的固定 URL，
  文件写回同一工作区，没有改变数据版本。
- LiveQA 适合作为小型单轮质量锚点，但重复问题对应多个答案，需按规范化问题分组；
  MedicationQA 有过短和表格残片 target，只能过滤后保留。
- MedInstruct-52K 是多任务单轮 instruction，不是真实对话；可作为普通医学 assistant
  主体候选，但需过滤 `As an AI`、时效问题、模板答案和输入输出近复制。
- MedBooks-18-CoT 结构完整且论文有 CoT/textbook augmentation 消融支持，推荐作为第一版
  少量 rationale；它不承担普通对话，也不证明临床安全。
- 当前推荐配方为普通 instruction 75%～85%、多轮 5%～10%、rationale 10%～20%；
  该初步配方已被同日最终决策替换。

## 2026-08-23：阶段 A 数据配方冻结

- 第一版停用 ChatDoctor-cleaned、MTS-dialog 和 MedBooks-18-CoT；ChatDoctor 的医学标签
  与分诊噪声不依赖后续 RAG 修复。
- 普通单轮使用过滤后的 MedInstruct-52K 与去重后的 LiveQA/MedicationQA，目标占比
  45%～55%；多轮 MedQA-dialog 提高到 30%～40%。
- reasoning 只使用 medical-o1 English，目标占比 10%～15%。它的 `Complex_CoT` 作为
  SFT target 时仍属于 supervised rationale，不能写成无监督推理。
- medical-o1 第一版候选长度为 CoT 不超过 512 tokens、整条不超过 1024 tokens；使用
  Qwen2.5 tokenizer 统计后再冻结。超长样本整条过滤，不从中间截断推理。
- 总规模目标约 10K；清洗后不足时按真实数量训练，不用已过滤数据补齐。

## 2026-08-23：SFT 数据职责重新冻结

- 阶段 A 改为多数医学单轮/多轮对话加少量 supervised rationale，不再以
  MedQA/MedMCQA 选择题作为普通医学 SFT 主体。
- 阶段 B 固定为约 100 条真实检索支持的 Search100 加普通医学对话 replay；保留 replay
  是为了维持正常回答能力并降低协议过拟合与逢题必搜风险，但具体比例仍是实验超参数。
- 普通对话不强行添加 `<think>`；reasoning 子集只训练可见、可审计的医学推理说明，
  不把它描述为模型真实的内部思维。普通对话与 Agent 样本通过 system/task prompt 区分。
- MedQA/MedMCQA 官方 train 改作 Search100/RL 候选题池；冻结评测资产边界不变。
- 本次只更新项目书与数据审计，没有下载、转换、tokenization 或训练数据。

## 2026-08-21：MedSearch-R1 两阶段 SFT 与数据实物审计（阶段 A 配方已于 2026-08-23 替换）

- 当时先冻结两个 checkpoint 与 Search100 冷启动框架；其中“医学选择题 SFT + 医学 QA
  replay”的旧配方已被 2026-08-23 的“对话型 SFT + 普通医学对话 replay”替换。
- Search100 改为默认冷启动资产，不再以基础模型协议测试失败为前提；一轮/两轮/直答
  比例仍是待实验超参数。
- 本地 MIRAGE MedQA 文件真实为 1,273 条官方 test；MedMCQA 文件真实为 4,183 条
  official dev。二者均冻结为评测，不进入 SFT、Search100 或 RL train。
- MedMCQA dev 的 `cop` 为 1-4；`exp` 有 1,977 条为空，`topic_name` 有 3,760 条为空。
  因此解释和 topic 都不能设为必需字段，也不能为缺失解释伪造真值。
- 第一版统一字段、target 规则、loss mask 和隔离表已合并进
  `MedSearch/PROJECT_BOOK.md`。本次未下载、转换或训练数据，没有实验指标。

## 2026-08-21：MedQA/MedMCQA 官方 train 固定镜像下载与字段打印

- 固定 `openlifescienceai/medqa@153e61cdd129eb79d3c27f82cdf3bc5e018c11b0` 与
  `openlifescienceai/medmcqa@91c6572c454088bf71b679ad90aa8dffcd0d5868`；直连
  Hugging Face 超时后，通过镜像站下载相同仓库、revision 和路径。
- MedQA 文件 5,679,350 bytes、SHA-256
  `d7bac59f5cee804a006172d44f675559dc5160bdc84dbae05aeed9d5aa1b56f3`，PyArrow 读取
  10,178 行；字段为 `id/data/subject_name`，答案字段嵌套在 `data`。
- MedMCQA 文件 85,899,025 bytes、SHA-256
  `b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5`，PyArrow 读取
  182,822 行；Parquet 已把 `cop` 转成 0-3，因此映射是 `"ABCD"[cop]`。
- 两个数据集各 10 条完整字段已检查；一次性打印样本现已清理。MedQA 有 2 个
  精确重复题干且抽样存在缺失图像引用；MedMCQA 有 21,953 条空解释，抽样可见 OCR、
  页码和文本粘连噪声。
- `MedSearch/data/raw/` 已加入局部 `.gitignore`，原始外部数据不会被 `git add -A`
  提交。本次没有转换训练数据或启动训练，没有实验指标。

## 2026-08-20：最新版 veRL/FSDP2 的 3B GRPO 保底架构

- 固定官方 veRL commit `e78efedbe56591cd53ac107150abb7565bb9b8b1`，新增独立
  `Search-R1/modern_verl/` 适配包；Search-R1 XML action、HTTP retrieve、环境 token
  loss mask 和开放域 EM reward 均接入当前 `AgentLoopBase` / custom reward 接口。
- 单卡使用 Qwen2.5-3B-Instruct、LoRA rank `16`、FSDP2、异步 vLLM `0.27.1`、同题
  rollout `2`；没有下载或回退到 1.5B。
- WSL2/Docker 的 vLLM spawn worker 无法重建 CUDA IPC handle；保留 veRL 原有语义，
  通过可复现补丁启用其 CPU shared-memory weight transport。实际日志确认进入 fallback。
- 最终复跑退出码 `0`，完成 `training/global_step:1`：两条轨迹均为 4 turns，step
  `21.549 s`，生成 `13.102 s`，actor update `4.059 s`，权重同步 `3.271 s`。
- 外部 `nvidia-smi` 共采样 `5,420` 点，峰值 `23,117 MiB / 32,607 MiB`；actor 日志
  内部峰值 allocated `14.031 GiB`、reserved `14.789 GiB`。
- 最终 W&B run 为 `lx30qqpq`。两条 rollout 的 EM reward 都是 `0`，因此组内 advantage、
  policy loss、grad norm 都为 `0`；这证明完整训练路径可执行，不证明模型发生有效学习。
- 当前最新版 veRL 的 V1 trainer 还要求额外 TransferQueue 依赖；保底脚本明确使用
  `trainer.use_v1=False` 的稳定调度，但 actor/rollout/AgentLoop/reward 均采用当前接口。
- `data.dataloader_num_workers=0` 已消除 smoke 退出时 DataLoader worker 被杀的噪声；
  上游 W&B `Tracking.__del__` 仍可能在 Ray teardown 后打印 closed-transport traceback，
  发生于 step 和 final validation 之后且进程退出码为 `0`，记录为非训练路径清理告警。

## 2026-08-20：vLLM 单卡 hybrid-engine 权重交换修复

- 根因：旧 sharding manager 在 GPU 上物化 FSDP `state_dict` 后一次性唤醒 vLLM 的
  weights 与 KV cache，导致 Actor、交接张量、vLLM 权重在同一卡上重叠并 OOM。
- 修复后的时序：CPU state-dict → Actor 参数移出 GPU → vLLM 只唤醒 weights → 写入
  最新 Actor 权重 → 唤醒 KV → 生成 → vLLM level-2 sleep → Actor 参数恢复到 GPU。
- 真实 Ray/vLLM 运行连续完成两次交换和两次生成，退出码 `0`；不再出现
  `cumem_allocator.cpp:211`。
- `nvidia-smi` 每 `0.5s` 采样 `777` 点，峰值为 `32058 MiB`；原始记录位于
  `Search-R1/experiments/single_gpu_smoke/vllm_gpu_memory.csv`。
- 权重交换与已有兼容回归共 `10/10` 通过。该方案以 CPU 传输延迟换取单卡显存可行性，
  不声称具备多卡 NCCL 方案的吞吐。

## 2026-08-20：固定双跳样本无训练 Agent 闭环

- 固定子集：train 为 HotpotQA 原始行 `86094`，test 为 2WikiMultihopQA 原始行
  `41745`；随机种子 `20260819`。
- 使用本地 `Qwen2.5-3B-Instruct`、仓库原生 HF rollout、`max_turns=2`、BM25
  top-k `3` 完成无参数更新验证，进程退出码 `0`。
- 真实轨迹依次检索电影导演与 `Jeff Burr birthplace`，最终输出
  `<answer>Aurora, Ohio</answer>`，规则 EM reward 为 `1.0`。
- `ACTIVE_TRAJ_NUM: [1, 1, 1, 0]` 表明轨迹在最终回答后结束；完整证据保存在
  `Search-R1/experiments/single_gpu_smoke/inference.log`。
- 两次 observation 分别从 `471`、`514` tokens 截到 `256`；本题仍答对，但后续实验
  必须记录该截断，不能把一次成功外推为检索信息总是充分。
- 本次未启动外部显存采样器，因此 HF 运行的显存峰值记为“未知”，不以失败运行或
  纸面估算补填。

## 2026-08-20：官方 wiki-18 BM25 检索服务闸门

- 下载并校验官方 `PeterJinGo/wiki-18-bm25-index`：28 个索引文件，合计
  `2,297,611,339` bytes。
- 下载并校验官方 `PeterJinGo/wiki-18-corpus`：压缩文件 `5,123,307,260` bytes，
  SHA-256 `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`
  与 Hugging Face LFS oid 一致，`gzip -t` 通过。
- 发现上游文件虽命名为 `wiki-18.jsonl.gz`，实际是 tar.gz；正确提取出的
  `wiki_dump.jsonl` 为 `14,393,573,105` bytes、`21,015,324` 篇文档，字段为
  `{id, contents}`。
- 修复全局升级后的实际兼容问题：Pyserini 会连带导入 FAISS，旧 `faiss-cpu 1.8.0`
  与 NumPy `2.3.5` ABI 不兼容；已升级到 `faiss-cpu 1.15.0`。
- 作者原有 CPU BM25 server 已启动；真实 `POST /retrieve` 返回 HTTP 200、3 条带正文
  和分数的结果。请求与响应保存在 `Search-R1/experiments/single_gpu_smoke/`。
- 一次混用 Xet 分块缓存与普通 Range 续传产生了哈希不匹配文件，已隔离为
  `.corrupt-90124a`，未解压、未用于服务；待正确资产和服务验收后清理。

## 2026-08-20：vLLM 0.27.1 兼容迁移与训练资产下载

- 保持 Search-R1/veRL 上层同步调用方式不变，新增版本专用兼容层，将旧的
  `generate()`、`sync_model_weights()`、sleep/wake 和 cache 方法映射到 vLLM `0.27.1`。
- 该兼容层明确限定单卡 `TP=1`；由于 WSL2 不支持当前 vLLM V2 runner 所需的 CUDA
  UVA，并且跨进程同卡 CUDA IPC/NCCL 路径不可用，改用 vLLM 官方 public `LLM` 的
  单进程 V1 runner。未修改 Search-R1 的 trainer、rollout 或 GRPO 算法调用方式。
- 兼容层单元测试 `2/2` 通过，Python `compileall` 与 `git diff --check` 通过。
- RTX 5090 真实 smoke 通过并以退出码 `0` 输出 `SMOKE_OK`：初始生成是
  `Paris. The capital of Germany is Berlin`；将内存 actor 的末层 norm 权重置零并同步后，
  输出变为 `!!!!`，证明 actor 权重确实进入 vLLM，而非只验证导入。
- `RotaryEmbedding: Failed to load weights` 已定位为 vLLM 自建的
  `persistent=False` cos/sin 位置缓存不属于 HF `state_dict`；reload 代码保留原缓存，
  不是 actor 参数漏同步。
- Qwen2.5-3B-Instruct 两个 safetensors 分片已完整下载：`3,968,658,944` 与
  `2,203,268,048` bytes；模型位于工作区忽略目录 `models/Qwen2.5-3B-Instruct/`。
- 官方 `PeterJinGo/nq_hotpotqa_train` 数据已完整下载：train `169,615` 行、
  `355,663,891` bytes；test `51,713` 行、`70,370,337` bytes。文件位于
  `Search-R1/data/nq_hotpotqa_train/`，后续 smoke 会从全量文件抽取小子集。
- 没有执行仓库 `scripts/download.sh`：脚本硬编码作者目录、参数名与当前 Python CLI
  不一致，而且默认下载约 64.6 GB Dense Flat 索引；这与当前固定的 CPU BM25 路线不符。

## 2026-08-19：按用户决定直接升级 Docker 全局训练环境

- 容器 `llm-pytorch` 已直接升级为 PyTorch `2.13.0+cu130`、vLLM `0.27.1`、
  Transformers `5.15.0`、Ray `2.57.0`、W&B `0.28.2`、TensorDict `0.14.0`。
- NVIDIA driver `591.74` 声明支持 CUDA 13.1；真实 CUDA tensor 测试通过：RTX 5090、
  compute capability `(12, 0)`、`torch.ones(..., device='cuda') == 1.0`。
- Search-R1 已通过 editable/no-deps 安装；`verl.__file__` 指向工作区
  `/workspace/code_list/some tricks/LLMLeanring/Search-R1/verl/__init__.py`，确认使用本地源码。
- `RayPPOTrainer` 与 `RewardManager` 能导入；`ActorRolloutRefWorker` 在本地
  `verl/third_party/vllm/__init__.py` 主动拒绝 vLLM `0.27.1`：该兼容层只允许
  `0.3.1/0.4.2/0.5.4/0.6.3`。不得只删除版本检查后假定 API 兼容。
- vLLM `0.27.1` 要求 PyTorch `2.13.0`，但其元数据仍安装 TorchAudio `2.11.0`
  CUDA 12.8 build；因 Search-R1 是纯文本任务且二进制不兼容，已卸载 TorchAudio。
- 直接全局升级已如预期破坏旧项目依赖闭包：MedicalGraphRAG、sentence-transformers、
  LangChain 等要求旧 Transformers/NumPy/AnyIO/Packaging。当前阶段接受该代价，不回滚。
- 训练数据没有下载：`Search-R1/data` 中不存在 Parquet。wiki-18 只有约 2.08 GB 的
  `part_aa.incomplete`，而该分片完整大小约 42.9 GB，明显不是“只差一点”；不再续传
  64.6 GB Dense Flat 索引，训练 smoke 只准备少量 NQ 样本。

## 2026-08-19：Docker 与 Search-R1 运行环境审计

- 复用容器：`llm-pytorch`（容器 ID 前缀 `c8f4518d842b`），Linux/WSL2，项目在容器内
  映射为 `/workspace/code_list/some tricks/LLMLeanring/Search-R1`。
- GPU 透传正常：RTX 5090，32 GB；PyTorch `2.11.0+cu128`、CUDA 12.8、BF16 可用；
  根文件系统约 1007 GB，检查时约 912 GB 可用。
- 已有 `transformers 4.44.2`、`datasets 2.16.1`、`pyserini 0.22.1`、
  `faiss-cpu 1.8.0`；缺少 `vllm`、`ray`、`hydra-core`、`omegaconf`、`wandb`。
- 当前 Python 为 `/usr/bin/python` 3.12，并受 PEP 668 保护；直接全局 `pip install`
  被拒绝。后续必须在容器内创建隔离环境，不能使用 `--break-system-packages` 污染 MedRAG。
- 官方旧环境固定 Python 3.9、PyTorch 2.4/CUDA 12.1、vLLM 0.6.3；vLLM 0.6.3
  早于 RTX 5090/SM120 支持，不能在当前硬件上原样运行。不得直接执行旧 requirements。
- `vllm==0.9.0` 是首批可支持 RTX 5090 的版本，但依赖 dry-run 显示它会固定
  `torch==2.7.0`、`transformers>=4.51.1`、`ray>=2.43.0` 等，与容器当前全局环境及
  vendored veRL 的旧接口均不相同；只能放入隔离环境并先做兼容性闸门测试。
- Search-R1 官方的新版 veRL/SGLang 集成仍可作为参考，但当前 veRL 文档注明内置
  `SearchTool` 已移除；不能直接追随 latest，必须固定到仍包含该工具的官方快照。
- 官方 Dense Flat 索引为 64.6 GB，语料压缩包约 5.12 GB；当前 `data/wiki-18`
  只有约 2 GB 的未完成下载，没有可用索引或解压语料。官方 BM25 索引约 2.3 GB，
  更适合作为单卡训练时的独立 CPU 检索服务。
- `Qwen/Qwen2.5-3B-Instruct`、NQ/HotpotQA Parquet、W&B 登录信息均尚未就绪；
  当前没有产生训练 step、W&B run 或实验指标。

## 2026-08-19：Search-R1 源码阅读边界收束

- `fsdp_workers.py` 确认为 veRL 执行基础设施，不继续阅读 device mesh、FSDP、offload、
  vLLM 构建或 Ray RPC 细节。
- 当前掌握标准固定为：知道 actor+rollout worker 负责生成/更新，独立 reference 提供
  KL 对照，GRPO 不创建 critic；能够使用这些接口即可。
- Search-R1 特有的 data、Agent loop、reward、GRPO advantage、actor loss 与 `fit()`
  调度已经形成完整理解链，源码阅读阶段在此收束。
- 后续仅在真实运行出现明确错误时，沿堆栈按需查看 veRL 内部。

## 2026-08-19：路线剪枝——跳过 `_validate()` 深入阅读

- `_validate()` 使用独立 `val_files`；只有配置 `val_data_num` 时才从验证文件内部抽样，
  不是从当前训练 batch 临时切出测试样本。
- 它只生成验证轨迹、计算平均规则 reward 并写日志，不计算 advantage 或更新 actor；
  当前阶段不把它作为 Search-R1 核心机制继续阅读。
- 已保留一个运行阶段待核实问题：多轮路径创建 active 子批次时丢失 meta_info，静态代码
  显示 `do_sample/recompute_log_prob` 会回退到默认 `True`。
- 下一学习点直接进入 actor/rollout/reference worker 的角色与初始化边界。

## 2026-08-19：`RayPPOTrainer.fit()` 完整调度链阅读验收

- 已完整追踪一批题目从 `DataProto`、`n_agent` rollout、old/reference log-prob、
  rule reward、GRPO advantage、information loss mask 到 actor update 的执行顺序。
- 已能解释 `batch.union(final_gen_batch_output)` 是题目 metadata 与完整轨迹 tensor 的
  汇合点；删除后最先在 `_balance_batch()` 读取 `attention_mask` 时失败。
- 已能说明后续 `RewardManager` 缺少 prompts/responses/mask，reward 无法产生，继而
  advantage、ratio、loss 和反向更新均无法执行。
- 已进一步区分直接由 rollout/union 带回的字段与后续派生字段：完整轨迹、mask、
  `old_log_probs` 是直接输入；`ref_log_prob`、scores/rewards、advantages/returns 和
  `loss_mask` 是合并后逐阶段计算得到。
- 本次为源码阅读验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-18：Search-R1 NQ 数据 → batch → uid 阅读验收

- 已读完 `nq_search.py`、`rl_dataset.py`、`ray_trainer.py`、`main_ppo.py`、`qa_em.py` 的指定代码段；已掌握的 reward/GRPO 重复段未重复阅读。
- 已能列出 Parquet 五个顶层字段，并解释 prompt 与 gold answer 隔离是字段协议与防止答案泄漏的共同要求。
- 已纠正 tensor shape：`__getitem__()` 返回单样本 `[max_prompt_length]`，`collate_fn()` 后为 `[batch_size, max_prompt_length]`，再按 `n_agent` 扩展 batch 维。
- 已理解 `n_agent` 在 rollout 前扩展轨迹，而 uid 不参与生成，只在后续 `compute_advantage()` 中提供分组标签。
- 已定位真正的分组代码：`compute_grpo_outcome_advantage()` 通过
  `id2score[index[i]].append(scores[i])` 按 uid 收集同题 rollout，并计算组内均值、标准差和 advantage。
- 已理解 `_read_files_and_tokenize()` 当前不 tokenize、`filter_prompts=True` 当前不生效，以及 `drop_last=True` 只丢弃不完整尾批次。
- 本次是源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-16：GRPO advantage → actor policy loss 阅读验收

- 已能解释 `old_log_prob` 来自 rollout 行为策略，`log_prob` 来自更新中的当前 actor；`ratio` 是逐 token 的新旧策略概率比，不是整条 trajectory 的单一偏好分数。
- 已理解正/负 advantage 分别提高/降低已采样 token 的概率；PPO clip 裁剪 surrogate objective 的有利更新方向，不是把模型真实概率硬锁在区间内。
- 已理解 information 是外部检索 observation：参与 attention 以指导后续生成，但通过 `loss_mask` 排除在 policy loss 之外。
- 已区分 entropy bonus、reference-policy KL 与 `ppo_kl` 监控量：entropy 鼓励分布保持探索性；reference KL 限制当前 actor 偏离冻结的 SFT/reference 模型；`ppo_kl` 监控 current actor 与 rollout old actor 的变化。
- 已通过最小 entropy 判断：分布越均匀 entropy 越高，越集中 entropy 越低。该项是 veRL PPO actor 原有的可选正则，官方快照默认 `entropy_coeff=0.001`，不是本项目新增算法。
- 本次只完成代码阅读和概念验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-14：Search-R1 reward → GRPO advantage 阅读验收

- 已能解释序列级 0/1 outcome reward 写在最后一个有效 response token，是因为结果只能在完整轨迹结束后判断，同时需适配 veRL 的 token-level reward 接口并避开 padding。
- 已区分 PPO/GAE 与 GRPO：GAE 使用 token value 从后向前传播 advantage；GRPO 不使用 value critic，而是对同题多条 rollout 的序列分数做组内标准化，再广播到有效 response token。
- 已正确判断全 1 或全 0 rollout group 的 advantage 为 0；进一步明确这表示该组没有策略梯度信号，是有效样本比例问题，不等于整个 GRPO 训练必然坍塌。
- 本次为源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-13：Search-R1 reward 学习注释准备

- 已为 5 个 reward 主链函数补充输入类型、输出类型、Tensor shape、上游调用位置和下游数据流。
- 验证命令：

```cmd
python Search-R1\scripts\verify_source_notes.py
python -m compileall -q Search-R1\verl\trainer\main_ppo.py Search-R1\verl\utils\reward_score\qa_em.py Search-R1\verl\trainer\ppo\ray_trainer.py
git diff --check
```

- 验证结果：8 个 Python 文件和 1 个 Bash 文件保持 comment-only；3 个 reward + 3 个 action parser 样例通过；Python 编译通过。
- 边界：注释未改变执行逻辑；reward 代码复述已于 2026-08-14 验收，但没有运行训练或产生训练指标。

## 2026-08-13：Search-R1 多轮 rollout 主干阅读

- 已理解 `active_mask` 控制仅对未结束样本生成；search 注入 `<information>` 后继续，answer 标记结束，非法动作按当前 parser 路径处理。
- 已区分 `rollings`（下一轮输入）与 `original_right_side`（最终训练轨迹）。
- 已理解 `max_turns` 后仍有一次 `do_search=False` 的 final rollout，以及 `attention_mask` 与用于排除环境 information token 的 `info_mask`。
- 本次只做源码阅读和协议检查，未运行模型、检索服务或训练。

## 2026-08-12：Search-R1 主干源码学习注释

- 已注释数据处理、dataset、generation、tensor helper、retrieval server、reward、训练入口和 trainer 主链。
- 已建立 `Search-R1/docs/source_code_learning_zh.md` 和标准库验证脚本 `Search-R1/scripts/verify_source_notes.py`。
- 官方 8 GPU 配置不能直接用于单张 RTX 5090；单卡配置必须以后通过真实 smoke 确认。

## MedicalGPT SFT：已验证范围

- 已跑通 Qwen2.5-3B、BF16 LoRA 的 100 条 smoke：rank `8`、alpha `16`、batch `2`、gradient accumulation `4`、max length `1024`、1 epoch、12 steps、seed `20260809`。
- 真实结果：train loss `2.332629`、eval loss `2.008307`、perplexity `7.450692`、runtime `13.9074s`、整卡显存峰值 `17,113 MiB`。
- 结论边界：只证明训练链和 adapter 加载有效；3 条 validation 中 2 条输出变化，不代表医学能力提升。

## Medical GraphRAG / LinearRAG：冻结结论

- 已实现 BM25、Dense、RRF Hybrid、Graph、Qwen3-Reranker Hybrid2 的统一检索入口和四套真实基准评测。
- `MedicalGraphRAG/` 固定在 README 结果对应的 commit `fe89a64`；未验收扩展仅保留在 Git 历史。
- 图检索在四套基准均未胜出。该负结果保留，不再通过无依据调参追逐指标。

# 遇到的问题

| 问题 | 当前处理 |
|---|---|
| Search-R1 官方配置面向 8 GPU | 单卡 smoke 使用 3B、小 batch、2 条 rollout、短序列；参数只能通过真实运行确认。 |
| vendored Search-R1 兼容层只接受 vLLM ≤0.6.3 | 已新增 vLLM 0.27.1、单卡 TP=1 兼容层，并通过 CPU 交接与分阶段 wake 修复完整 Ray 路径的瞬时 OOM；真实两轮生成退出码为 `0`。 |
| 全局升级造成旧 MedRAG/GraphRAG 依赖冲突 | 用户明确选择不隔离；保留 `pip check` 结果，当前不回滚。 |
| 单卡 32GB 的 GRPO 资源边界未知 | 不做纸面外推；用真实日志记录 batch、rollout、长度、显存和耗时。 |
| 官方检索 observation 超过 `max_obs_length=256` | 当前按原逻辑截断；固定双跳样本仍得到 reward `1.0`，后续只把长度作为实验配置记录，不研究 retriever 内部。 |
| 多轮验证没有传递 `do_sample=False/recompute_log_prob=False` | 静态路径显示 active 子批次丢失 meta_info；当前先记录，不修改行为，运行阶段再实测。 |
| WSL2/Docker 跨 vLLM spawn 进程的 CUDA IPC 报 `invalid resource handle` | 为当前 veRL 增加环境变量控制的 shared-memory fallback 补丁；默认 native Linux 行为不变。 |
| smoke 的轨迹数小于默认 8 个 AgentLoop workers | 固定 `agent.num_workers=2`，与两条 rollout 等分。 |
| 两条 smoke trajectory 的 reward 都为 0 | advantage/loss/grad_norm 均为 0；保留为 GRPO 同组无差异信号的真实案例，不包装成训练提升。 |

# 下一步

- 观察W&B曲线与本地日志；首先确认`checkpoint-250`包含adapter、optimizer、scheduler和trainer state。
- 训练完成后验收最终`adapter/`与`train_metrics.json`，再使用冻结的328条dev执行独立评测。
- 正式训练期间不改YAML和训练源码；生产测试helper剪枝放到训练完成后的独立步骤。
- Stage-B replay仍为待讨论项，不阻塞Stage-A；本日不启动Stage-B。


# 待补知识

- MedQA-dialog 清洗后的实际保留量能否支持 30%～40% 多轮比例。
- medical-o1 在 512-token CoT 阈值下的保留率。
- Search100 一轮/两轮/直答比例和普通医学对话 replay 比例。
- Search100 是否需要过采样，才能在保留 replay 时学会稳定协议。

# 实验结果总表

| 数据集 / 实验 | 当前最优或关键结果 | 结论边界 |
|---|---|---|
| PubMedQA 封闭基准 | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 documents，不代表全量 PubMed。 |
| NFCorpus | Hybrid2 nDCG@10 `0.384` | 多相关 qrels；Graph 未胜出。 |
| SciFact | Hybrid2：R@10 `0.895`、MRR `0.740`、nDCG `0.772` | Graph 未胜出。 |
| HotpotQA | Hybrid2：R@10 `0.898`、MRR `0.955`、nDCG `0.865` | 每题 2 个 gold；Graph 未胜出。 |
| MedicalGPT 100 条 smoke | eval loss `2.008307`、PPL `7.450692` | 只验证训练管线。 |
| Search-R1 3B GRPO 单步 | step `21.549 s`；GPU peak `23,117 MiB`；W&B `lx30qqpq` | 两条 reward 均为 0，验证训练链路，不代表学习提升。 |

# 失败案例

| 场景 | 原因 | 保留结论 |
|---|---|---|
| PubMedQA 上 RRF Hybrid 低于 Dense | BM25 高位错误稀释 Dense 信号 | 融合不是默认更好。 |
| Graph 在四套基准均未胜出 | 实体传播信号与任务/语料不匹配 | 保留负结果，不包装为提升。 |
| TREC-COVID 未接入 | 24.6% 文档正文为空，部分 qrels 指向空文本 | 不用标题回退伪造正文检索。 |
| MedicalGPT 内部显存统计路径停滞 | Trainer 路径未形成完整产物 | 该次运行不计入；已用外部只读采样完成 smoke。 |
| Search-R1 单步两条 rollout 均答错 | 同题组 reward 全 0，GRPO advantage 与 gradient 为 0 | 保留为“同组无差异则无学习信号”的真实失败样本；扩大实验时监控有效组比例。 |

> 更早的逐日过程、已撤回方案和详细实验配置由 Git 历史、各子项目 README 与 `experiments/` 保留；本文件只维护当前有效状态。
