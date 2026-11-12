# Minimal Trainer Skeleton Implementation Plan

**Goal:** 延后 Stage-B replay 争议，给学生留下唯一的 `build_trainer()` 接线任务。

**Architecture:** 沿用 `sft/train.py` 单入口和已有 DataLoader 的 dataset/collate_fn，复用
Transformers TrainingArguments/Trainer。不写训练循环、不启动 train、不改冻结数据。

**Tech Stack:** 现有 WSL/Docker、Transformers 5.10.4、PEFT 0.20.0、PyTorch、pytest。

教师边界优先：计划不提供学生函数完整答案，不分派代理，不提交用户的未提交改动。

## 本轮步骤

- [x] 在 `MedSearch/docs/tests/test_sft_trainer.py` 写参数映射、真实 tiny LoRA Trainer 构造测试；缺失接口产生 `3 failed`。
- [x] 在 `sft/train.py` 实现教师负责的 `build_training_arguments()`，复用配置中的 training 字段，固定 mask/日志/无评测无保存边界。
- [x] 在同一文件留下 `build_trainer(config, model, tokenizer, dataloader, project_root)` 契约与 `NotImplementedError`；接入互斥 `--trainer-smoke` 开关，只构造并检查 batch，不 train。
- [x] 给两个 YAML 增加最小 training 配置；max_steps=1 只是后续 smoke 安全默认，不是正式实验配方。
- [x] 未实现学生函数的测试只允许因 `NotImplementedError` 预期失败，其他错误必须暴露；学生填写后撤去预期失败标记再验收。
- [x] 运行 `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`，最终 `18 passed, 2 xfailed in 12.25s`；更新 README、PROJECT_BOOK、STUDY_PROGRESS 的当前有效决策。

## 交付边界

学生需能解释 Trainer 为什么接收 Dataset 而不是 DataLoader、loss mask 为什么不能被覆盖、
gradient checkpointing 与 use_cache 的关系，以及构造 Trainer 不等于执行训练。
真实 backward、adapter 保存/恢复、dev 评估和正式训练另设任务，不能据本轮测试宣称已完成。
