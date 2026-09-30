# Search-R1 Single-GPU Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 在现有 `llm-pytorch` Docker 容器和单张 RTX 5090 上，用
`Qwen/Qwen2.5-3B-Instruct` 完成一个可在 W&B 查看的一步 Search-R1 GRPO smoke。

**Architecture:** 按用户决定直接升级 MedRAG 容器的全局 PyTorch/vLLM，veRL 继续从
Search-R1 工作区 editable 加载。先迁移本地 veRL 的 vLLM wrapper/worker，再准备
少量 NQ smoke 数据。检索服务使用官方 BM25 索引并在 CPU 独立运行。

**Tech Stack:** Docker Linux、RTX 5090、Qwen2.5-3B-Instruct、veRL、vLLM/SGLang、
Ray、Pyserini BM25、W&B。

---

## Task 1: 验证全局最新版运行栈

**状态：已完成（2026-08-20）。**

**Files:**

- Modify: `STUDY_PROGRESS.md`
- Create: `Search-R1/experiments/single_gpu_smoke/ENVIRONMENT.md`

1. 安装并记录 PyTorch、vLLM、Transformers、Ray、W&B 与 TensorDict 版本。
2. 运行 RTX 5090 CUDA tensor 测试。
3. editable/no-deps 安装工作区 Search-R1，确认 `verl.__file__` 指向本地源码。
4. 保存 `pip check` 冲突与 worker 导入结果。

## Task 2: 迁移本地 veRL 到 vLLM 0.27.1

**状态：已完成（2026-08-20）。兼容层、完整 Ray worker、两轮单卡权重交换与真实
vLLM 生成通过；峰值显存 `32058 MiB`。**

**Files:**

- Modify: `Search-R1/experiments/single_gpu_smoke/ENVIRONMENT.md`

1. 追踪 `verl/third_party/vllm` 与 rollout worker 实际使用的旧 vLLM API。
2. 只迁移 Search-R1 训练路径需要的 wrapper，不扩展无关 backend。
3. 导入 `ActorRolloutRefWorker` 并运行最小构造检查。
4. 加载 `Qwen/Qwen2.5-3B-Instruct` 完成一次短文本生成，记录显存峰值与输出。

## Task 3: 启动官方 BM25 检索服务

**状态：已完成（2026-08-20）。官方索引/corpus 完整，CPU 服务和真实 HTTP 响应通过。**

**Files:**

- Create: `Search-R1/experiments/single_gpu_smoke/retrieval_request.json`
- Create: `Search-R1/experiments/single_gpu_smoke/retrieval_response.json`

1. 下载官方 `PeterJinGo/wiki-18-bm25-index` 与 `wiki-18-corpus`，校验下载完整性。
2. 使用现有 Search-R1 retrieval server，以 BM25 模式在 CPU 启动服务。
3. 向 `/retrieve` 发送一个真实问题，保存请求和未经改写的响应。
4. 验收响应含 `result[*][*].document.contents`，且至少返回一条正文。

## Task 4: 准备最小 NQ 数据并做无训练推理

**状态：已完成（2026-08-20）。HF 完整双跳轨迹得到 EM reward `1.0`；vLLM 路径
另以一次搜索完成运行兼容闸门。**

**Files:**

- Create: `Search-R1/experiments/single_gpu_smoke/CONFIG.md`
- Create: `Search-R1/experiments/single_gpu_smoke/inference.log`

1. 从已下载的官方全量 Parquet 固定一个小子集与随机种子 `20260819`。
2. 配置 `Qwen/Qwen2.5-3B-Instruct`、官方检索 URL、短上下文和最多一次搜索。
3. 跑一条多轮推理，保存原始轨迹、搜索次数、最终答案、耗时与显存峰值。
4. 确认 `<search>`、`<information>`、`<answer>` 数据流完整后才进入训练。

## Task 5: 运行一个 optimizer step 的 GRPO smoke

**Files:**

- Create: `Search-R1/experiments/single_gpu_smoke/run_grpo_1step.sh`
- Create: `Search-R1/experiments/single_gpu_smoke/train.log`
- Modify: `STUDY_PROGRESS.md`

1. 使用 train batch `1`、同题 rollout `2`、短序列、一次搜索，并关闭训练前验证。
2. 将 `trainer.total_training_steps` 固定为 `1`；W&B project 使用
   `Search-R1-local`，run name 包含模型、算法、步数和日期。
3. 启动训练并保存完整 stdout/stderr、`nvidia-smi` 峰值、配置和 Git commit。
4. 验收 W&B 至少出现一步 reward、advantage/有效组统计、policy loss、KL、entropy
   和学习率；缺失项必须在报告中明确记录，不能补造。
5. 若 OOM，依次缩短序列、保持 rollout=2、减小 micro batch、开启 offload；不删除
   reward、GRPO 分组、reference KL 或搜索调用。

## Task 6: 对照官方结果并收束

**Files:**

- Create: `Search-R1/experiments/single_gpu_smoke/RESULTS.md`
- Modify: `STUDY_PROGRESS.md`

1. 链接本地 W&B run，并列出实际 step 数、reward、loss、KL、显存和耗时。
2. 只把本次结果称为“管线 smoke”，不与论文最终准确率做性能优劣结论。
3. 将论文中 Qwen2.5-3B-Instruct 的最终结果作为背景基线，注明训练规模不可比。
4. 记录失败轨迹与下一次唯一变量，不同时修改检索器、奖励、数据与模型。
