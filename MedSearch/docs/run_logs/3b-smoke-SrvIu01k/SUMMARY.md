# 2026-08-31：真实 Qwen2.5-3B BF16 LoRA 保存与重载通过

## 命令与范围

```cmd
wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh
wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --save-roundtrip
```

CPU 回归 `33 passed in 10.22s`；GPU 验证退出码 0。
本轮只修改 docs 内教师验证脚本、测试和文档，没有修改学生 `sft/train.py` 的实现或 main。
当前不是 QLoRA，不是正式 Checkpoint A，也不代表医学问答效果提升。

## 配置与真实结果

- 本地 Qwen2.5-3B-Instruct，BF16 LoRA；rank=8、alpha=16、dropout=0.05，7 类投影模块。
- Stage-A 固定 8 条样本，seed=20260831，batch=2、gradient_accumulation_steps=4，
  max_steps=1、max_length=1024，lr=0.0002，adamw_torch，非重入梯度检查点。
- 实际长度 `[842, 265, 143, 148, 661, 587, 339, 348]`；不外推 Stage-B 4096 可行性。
- global_step=1；train_loss=1.173087239265442；grad_norm=0.8145956993103027。
- Trainer 报告 train_runtime=7.3378 秒；探针 perf_counter 范围 run_training=7.006019 秒。
  两种原始计时均保留，不是包括模型加载和重载的总耗时，也不据此比较训练性能。
- 504 个 LoRA 张量中 252 个更新；434 个冻结基座张量全量 SHA256 无变化。
- PyTorch 训练范围峰值 allocated=9,807,748,096 bytes、reserved=15,154,020,352 bytes。
- 整卡 400 个采样点（目标间隔 200ms）峰值 18,062 MiB / 32,607 MiB，包含背景占用。
  运行结束后无训练进程，整卡回落至 2,929 MiB。

## 保存/重载检查

产物根目录：`MedSearch/outputs/smoke/stage_a-kv1h2i46/`。

- adapter 位于 `adapter/`；adapter_model.safetensors 为 59,933,632 bytes。
- 同目录含 adapter_config.json、tokenizer.json、tokenizer_config.json、chat_template.jinja
  及 PEFT 原生 README；不保存整套基座、embedding、optimizer/scheduler/step 状态。
- 训练模型释放后，从相同本地基座重新加载 adapter；复用学生 `attach_lora` 的可训练分支。
- 504 个 adapter 张量逐项相等，434 个基座哈希相等；tokenizer 词表、特殊 token、padding_side、
  chat template、固定 prompt 编码全部一致。
- 固定 prompt：`What does a thermometer measure?`。前后均 `eval()`、`use_cache=False`、
  SDPA、BF16 autocast，比较最后输入位置的全词表 logits（不是多轮生成或医学评估）。
- logits 最大绝对差为 **0.0**，逐值完全相等；既定验收阈值 atol=rtol=1e-5 未放宽。

可复查证据：

- run_config.json：有效配置、Git HEAD、学生源码 SHA256、冻结数据哈希与样本 ID。
- train_metrics.json、trainer_log_history.json：真实 Trainer 返回与日志。
- parameter_verification.json：梯度、更新张量、基座检查、显存、版本、教师探针哈希。
- adapter_roundtrip_verification.json：文件大小、比较范围、权重/tokenizer/输出结论。
- roundtrip_reference.safetensors：固定输入和保存前 logits。
- roundtrip_reloaded_logits.safetensors：重载后 logits。
- 本报告同目录 run.log 与 gpu_memory.csv。

版本：torch 2.13.0+cu130、Transformers 5.10.4、PEFT 0.20.0。
Git HEAD 为 `8cccacd57afb66018aa0cc43725fea2706793dbf`，工作区有改动，不能仅凭 HEAD 复现实验；
源码/数据/探针哈希见上述 JSON。学生 train.py SHA256：
`3d0ac0b3dfd40176301b75dabb3befb3c06f3bab4fa1c4402d744fd0d8e2385c`。

## 失败记录与教学边界

同一任务此前两次失败均保留：

1. `../3b-smoke-Q9PJhbEt/`：教师误用 chat template 的默认字典返回值，保存前失败。
2. `../3b-smoke-8byqSe5Q/`：保存/权重重载通过，但教师前后推理 autocast 不一致；logits
   最大误差 0.625。核对 Accelerate 源码、统一计算精度并补 CPU 回归后，本次通过。

没有让学生为这两处教师探针错误改写已通过的核心函数。所有失败产物保留，不只展示成功运行。

后续正式 main 训练步骤由学生编写。下一项唯一任务是 QLoRA 的 4-bit 量化配置，参考
MedicalGPT/training/supervised_finetuning.py:740；随后逐项准备低比特训练、挂 LoRA，
由学生组织正式训练流程。教师继续承担测试，不提前代写完整 QLoRA/main。
