# Search-R1 单卡 smoke 环境记录

记录日期：2026-08-20

## 容器与硬件

- Docker 容器：`llm-pytorch`（ID 前缀 `c8f4518d842b`）
- GPU：NVIDIA RTX 5090，32 GB，compute capability `12.0`
- 工作区挂载：`/workspace/code_list/some tricks/LLMLeanring/Search-R1`
- 本地 veRL：editable/no-deps 加载，`verl.__file__` 指向上述工作区源码

## 关键依赖

- PyTorch：`2.13.0+cu130`
- vLLM：`0.27.1`
- Transformers：`5.15.0`
- Ray：`2.57.0`
- W&B：`0.28.2`
- TensorDict：`0.14.0`

这是用户选择的容器内全局升级环境，不与此前 MedRAG 环境隔离。旧项目的依赖冲突
已经接受，但不能把此环境称为可复现的 MedRAG 环境。

## vLLM 0.27.1 迁移边界

- 保留 Search-R1/veRL 上层的同步 rollout 接口，仅在
  `verl/third_party/vllm/vllm_v_0_27_1/` 内转换新版 vLLM API。
- 当前只支持单卡 `tensor_parallel_size=1`。
- WSL2 环境关闭 V2 runner、FlashInfer sampler 和 V1 multiprocessing：V2 runner
  需要当前 WSL2 不提供的 CUDA UVA；另两条跨进程/SM120 路径已在真实运行中失败。
- 使用 vLLM public `LLM` 的进程内 V1 engine；actor state dict 直接重载到 vLLM worker，
  Search-R1 trainer 与 rollout 调用方式不变。

## 验证结果

```text
兼容层 unittest: 2 tests, OK
compileall: OK
git diff --check: OK
真实 GPU smoke exit code: 0
初始生成: Paris. The capital of Germany is Berlin
actor 权重同步后: !!!!
SYNC_CHANGED_OUTPUT: True
SMOKE_OK
```

`RotaryEmbedding: Failed to load weights` 是 vLLM 自建的非持久化 cos/sin cache 不在
HF state dict 中；reload fallback 会恢复已有 cache。它不是 actor 参数同步失败。

独立 smoke 脚本结束时出现 `destroy_process_group()` 提示；训练 worker 中 NCCL 进程组
需要持续存在，因此兼容层不负责销毁它。完整 Ray/GRPO 仍需后续实测。

## 已下载资产

- 模型：`models/Qwen2.5-3B-Instruct/`
  - shard 1：`3,968,658,944` bytes
  - shard 2：`2,203,268,048` bytes
- 官方训练数据：`Search-R1/data/nq_hotpotqa_train/`
  - train：`169,615` 行，`355,663,891` bytes
  - test：`51,713` 行，`70,370,337` bytes

模型和数据目录均被 Git 忽略，不随源码提交。后续只从全量 Parquet 抽取小规模 smoke
子集；当前尚未产生 W&B run。

## BM25 检索服务

- 官方预构建索引：`PeterJinGo/wiki-18-bm25-index`，28/28 文件完整，
  `2,297,611,339` bytes。
- 官方 corpus 压缩包：`5,123,307,260` bytes，SHA-256
  `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`，
  与 Hugging Face LFS oid 一致；`gzip -t` 通过。
- 该文件实际是 tar.gz，而不是直接 gzip JSONL；提取出的 `wiki_dump.jsonl` 为
  `14,393,573,105` bytes，包含 `21,015,324` 篇 `{id, contents}` 文档。
- vLLM 升级带来的 NumPy `2.3.5` 与旧 `faiss-cpu 1.8.0` ABI 冲突会阻止
  Pyserini 导入；已将 faiss-cpu 升级为 `1.15.0`。
- 作者原有 BM25 server 已在 CPU 启动；`POST /retrieve` 返回 HTTP 200、3 条正文和分数。
  请求与原始响应保存在同目录的 `retrieval_request.json`、`retrieval_response.json`。
