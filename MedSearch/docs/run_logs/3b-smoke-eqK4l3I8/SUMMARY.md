# Stage-A Qwen2.5-3B 单步 smoke：通过

日期：2026-08-31。进程退出码 0，仅一次 optimizer update，无正式 checkpoint。

## 可复现入口

仓库根目录 CMD：

```cmd
wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh
```

教师检查器委托现有 `MedSearch.sft.train.main` 和学生 `run_training` 执行，没有替换训练实现。
新增检查只读取训练前后参数，并记录日志。原始记录：本目录 `run.log`、`gpu_memory.csv`。
训练产物：`MedSearch/outputs/smoke/stage_a-_rs5vew5/` 中的 `run_config.json`、
`train_metrics.json`、`trainer_log_history.json`、`parameter_verification.json`。

## 固定配置与版本

- 基座：本地 Qwen2.5-3B-Instruct；BF16；LoRA rank 8、alpha 16、dropout 0.05，七类投影模块。
- 学习率 0.0002，adamw_torch，batch 2、梯度累积4、max_steps=1，非重入 gradient checkpointing。
- seed 20260831；从 Stage-A train 16,077 条固定抽8条，不改动冻结文件。
- 实际长度：842、265、143、148、661、587、339、348；配置上限1024。
- Git HEAD `8cccacd57afb66018aa0cc43725fea2706793dbf`，工作区有未提交代码，
  因此以 manifest 中三份 SFT 源码 SHA-256 补充精确记录，不能只靠 HEAD 复现。
- 数据 SHA-256 `c63ab881635ece1a35f2f3a1413c19f23b863d4532b634332ae6a8153fe06014`。
- 实际环境：torch 2.13.0+cu130、transformers 5.10.4、peft 0.20.0；RTX 5090。

## 真实结果

| 项目 | 结果 |
|---|---:|
| global_step | 1 |
| train_loss | 1.173087239265442 |
| grad_norm | 0.8148247599601746 |
| Trainer train_runtime | 6.3281 秒 |
| run_training 外部计时 | 6.890461 秒 |
| 可训练参数 | 14,966,784 / 3,100,905,472（0.4826585%） |
| 检查的 LoRA 张量 | 504 |
| 变化的 LoRA 张量 | 252（均为 lora_B） |
| 全量检查的冻结基座张量 | 434 |
| 基座哈希变化 | 0 |
| PyTorch peak allocated | 9,807,748,096 bytes（9.134 GiB） |
| PyTorch peak reserved | 15,154,020,352 bytes（14.113 GiB） |
| nvidia-smi 整卡采样峰值 | 18,139 MiB / 32,607 MiB |

GPU CSV 共306条数据，名义间隔200ms；整卡值包含桌面等背景占用，不能与 PyTorch allocated
混为一谈。采样时间为容器 UTC 03:25:05.029 至03:26:12.079，包含加载与参数校验；6.3281秒
只表示 Trainer 训练耗时，不是整次启动用时。

初始化耗时较长但随后自然完成，未修改或禁用内存监控。tokenizer 对齐提示将模型的
bos_token_id 对齐为 None、pad_token_id 对齐为151643，属于 Trainer 配置对齐，不是样本被修改。

## 结论边界与下一步

这里只证明8条数据上前向、反向、一次更新与参数冻结成立；loss不是独立测试集质量指标，
不能据此宣称医学能力提升。日志 epoch=1 仅指遍历这8条子集，不是完成16,077条训练集。
未测试1024满长压力、Stage-B 4096、dev评估或完整训练；未保存 adapter，不能将本次目录
当成 Checkpoint A。GPU进程已结束。

下一唯一任务：adapter 保存与重新加载一致性验证，随后再安排dev评估和完整断点恢复。
