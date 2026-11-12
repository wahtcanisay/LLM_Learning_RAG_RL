# 学生主入口 QLoRA Trainer 单步验收（2026-08-31）

结论：通过。教师脚本调用学生main的--train-smoke --qlora路径，真实完成一次优化器更新。
未修改学生main、已验收函数、YAML或数据；本次没有保存adapter。

```cmd
wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --qlora
wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh
```

GPU退出码0；CPU回归48 passed, 16 warnings in 16.11s，无xfail。
原始输出：[run.log](run.log)；整卡采样：[gpu_memory.csv](gpu_memory.csv)。
产物目录：`MedSearch/outputs/smoke/stage_a-_y_9hw2d/`。

## 配置与结果

- WSL → Docker llm-pytorch → RTX5090；Qwen2.5-3B-Instruct，252个NF4/double-quant层，BF16计算。
- LoRA r8、alpha16、dropout0.05；q/k/v/o/gate/up/down七类模块；基座冻结。
- seed20260831，Stage-A固定8条，batch2、梯度累积4、max_steps1、max_length1024；
  实际长度[842,265,143,148,661,587,339,348]；AdamW学习率0.0002，non-reentrant梯度检查点。
- global_step=1；train_loss=1.1438109874725342；grad_norm=0.8728765249252319。
- Trainer runtime7.8915秒，教师run_training外围计时8.579441050998867秒。
- 504个LoRA参数张量中252个发生更新，值全部有限。
- 全量检查434个冻结基座参数：无梯度、无非预期数值变化。
  108个q/k/v bias有bitsandbytes内部FP32→BF16存储转换，转回FP32后数值精确一致；
  其他参数字节哈希不变，量化层保持uint8/NF4/nested。
- PyTorch训练区间峰值allocated7,081,850,880 bytes，reserved12,297,699,328 bytes。
  整卡200ms采样250个点，峰值15,285 MiB，含背景和探针开销；不能外推4096-token训练，
  不据此宣称相对其他运行的显存优化比例或速度提升。
- GPU任务结束后未见运行中的GPU进程，整卡背景占用2949MiB。

## 复现记录与限制

- `run_config.json`包含有效配置、8条样本ID、数据SHA256、Git HEAD和学生源码SHA256。
- `teacher_execution.json`额外记录真实argv、qlora=true、量化配置及探针SHA256；
  当前main的run_config还没有记录命令行QLoRA开关，整合时应纳入正式运行元数据。
- `parameter_verification.json`记录实际更新名单、精度转换名单、参数检查及版本；
  torch2.13.0+cu130/transformers5.10.4/peft0.20.0/bitsandbytes0.50.2/accelerate1.14.0。
- `train_metrics.json`、`trainer_log_history.json`保存Trainer真实输出。
- Trainer输出的epoch=1仅指8条smoke子集一轮，不是完整Stage-A训练一轮。
- tokenizer与模型BOS/PAD配置对齐提示保留于run.log，是Trainer运行时的配置对齐，未扩词表。
- 模型展示total_parameters来自打包后的numel，不等同原始3B逻辑参数总数。
- 未保存adapter或完整checkpoint，不能恢复本次已退出进程里的更新后权重；本次不是Checkpoint A。
  QLoRA保存重载、dev评估和完整断点恢复仍待验收。

## 下一唯一代码任务

可以开始整理正式训练入口，不再添加训练算法模块：学生负责main中的配置/数据/模型/训练保存编排；
教师负责将smoke专用诊断、参数哈希检查等保留在docs中，并验证整理前后行为一致。
正式运行仍应保留配置、种子、实际QLoRA开关、指标和保存边界。此次未删除或迁移代码。
