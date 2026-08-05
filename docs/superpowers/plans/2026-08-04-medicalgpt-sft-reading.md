# MedicalGPT SFT Reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MedicalGPT SFT 最小调用链加入不改变行为的中文学习注释。

**Architecture:** 保持官方文件结构与控制流不变，只在 Shell 参数组和 Python 关键数据流节点前增加解释性注释。学习记录单独说明本次临时切换和 Search RL 边界。

**Tech Stack:** Bash、Python、Hugging Face Transformers、Datasets、PEFT/LoRA、bitsandbytes/QLoRA。

---

### Task 1: 注释 SFT 启动参数

**Files:**
- Modify: `MedicalGPT/scripts/run_sft.sh`

- [ ] 在命令前说明该脚本当前是双卡示例，以及单卡学习时需要调整的入口参数。
- [ ] 按模型/数据、训练规模、优化、LoRA、精度与日志分组解释关键参数，不修改参数值。

### Task 2: 注释 SFT Python 主调用链

**Files:**
- Modify: `MedicalGPT/training/supervised_finetuning.py`

- [ ] 标注四组 dataclass 参数如何接收 Shell 参数。
- [ ] 标注 tokenizer/chat template 与本地 JSONL 数据加载路径。
- [ ] 标注 `preprocess_function` 如何构造 `input_ids`、`attention_mask`、`labels`。
- [ ] 解释 `train_on_inputs=False` 如何用 `IGNORE_INDEX` 屏蔽用户输入的 loss。
- [ ] 标注 4-bit NF4 QLoRA、k-bit 准备、LoRA target modules 与可训练参数。
- [ ] 标注 collator、Trainer、checkpoint 恢复、adapter/模型保存与 eval perplexity。

### Task 3: 记录学习边界并验证

**Files:**
- Modify: `STUDY_PROGRESS.md`

- [ ] 记录阶段 3 为临时预习，不覆盖阶段 1/2 未完成项。
- [ ] 记录 MedicalGPT GRPO 默认使用 GSM8K、正确性/格式 reward，没有 Search 工具环境。
- [ ] 运行 `python -m py_compile MedicalGPT/training/supervised_finetuning.py`，预期退出码为 0。
- [ ] 若 Bash 可用，运行 `bash -n MedicalGPT/scripts/run_sft.sh`，预期退出码为 0；否则明确记录未执行原因。
- [ ] 审查源码差异，确认只增加注释。
