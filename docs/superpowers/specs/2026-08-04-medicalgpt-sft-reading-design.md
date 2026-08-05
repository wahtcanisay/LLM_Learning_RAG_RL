# MedicalGPT SFT 源码注释设计

## 目标

在不改变官方训练行为的前提下，为 MedicalGPT 的 SFT 最小调用链加入初学者可读的中文注释，使学习者能够追踪“启动参数 → 数据预处理 → loss mask → QLoRA/LoRA → Trainer → 保存”的完整数据流。

## 范围

- 注释 `MedicalGPT/scripts/run_sft.sh`：解释启动方式和本次最重要的参数组。
- 注释 `MedicalGPT/training/supervised_finetuning.py`：只标记关键控制点，不逐行翻译库样板代码。
- 在 `STUDY_PROGRESS.md` 记录阶段 3 临时预习，以及 MedicalGPT GRPO 与 Search-R1 的边界。

## 明确不做

- 不下载模型或外部训练数据，不启动训练。
- 不修改默认参数、分支条件、函数签名或数据格式。
- 不阅读或注释 PPO、DPO、ORPO、OPD 的完整实现。
- 不把 MedicalGPT 的 GSM8K GRPO 描述为搜索强化学习。

## 验证

使用 `python -m py_compile` 检查 Python 语法；若环境存在 Bash，使用 `bash -n` 检查启动脚本；最后审查差异，确认源码变化只有注释。
