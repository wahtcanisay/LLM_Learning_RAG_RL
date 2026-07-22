# R2RAG 中文学习译注

> 本文件是对官方 `README.md` 的学习版翻译和代码导读，不逐句复制原文。官方仓库：<https://github.com/rmit-ir/NeurIPS-MMU-RAG>；对应论文：<https://arxiv.org/abs/2602.20735>。

## 1. 项目定位

这个仓库是 RMIT IR 团队参加 NeurIPS 2025 MMU-RAG 挑战赛的完整工程。它提供三类服务入口：

1. `/run`：面向动态评测的流式问答接口；
2. `/evaluate`：面向批量评测的请求接口；
3. OpenAI 兼容接口：便于接入 ASE 2.0、OpenWebUI 或其他客户端。

论文中的 R2RAG 不是一个单独的检索器，而是一个“先判断问题复杂度，再选择检索路径”的系统：

```text
问题
  ↓
QueryComplexityLLM
  ├─ simple  → VanillaRAG：查询变体 → 搜索 → 重排 → 生成
  └─ complex → VanillaAgent：多轮查询 → 搜索 → 重排 → 证据审查
                                      ↓
                         证据不足则改写查询并继续
                         证据足够/达到预算则生成
```

官方系统使用 Qwen3-4B 处理查询分类、查询改写、文档审查和答案生成，使用 Qwen3-Reranker-0.6B 进行点式重排。论文报告的配置面向单张消费级 GPU；这不代表我们现在已经在本机跑出了相同结果。

## 2. README 快速开始的中文解释

### 环境与安装

官方推荐使用 `uv` 管理 Python 环境，项目要求 Python 3.11 或更高版本。基本流程是：

```bash
git clone https://github.com/rmit-ir/NeurIPS-MMU-RAG
cd NeurIPS-MMU-RAG
uv sync
```

仓库的 `pyproject.toml` 将依赖分成基础依赖和可选组：

- 基础组：FastAPI、LangGraph、OpenAI 客户端、Transformers 等；
- `local_llm`：vLLM、Sentence Transformers、tiktoken 等本地推理依赖；
- `classifier`：spaCy、scikit-learn 等查询复杂度分类依赖；
- `eval`：评测器和数据集依赖；
- `crawl`、`proxy`、`runpod`：特定部署或搜索路线的附加依赖。

我们当前只下载和阅读代码，不立即安装全部可选依赖，也不启动远程搜索或大模型服务。

### 启动接口

```bash
# MMU-RAG 挑战赛接口
uv run fastapi run src/apis/mmu_rag_router.py

# OpenAI 兼容接口
uv run fastapi run src/apis/openai_router.py
```

默认端口是 8000。前者暴露 `/run` 和 `/evaluate`，后者暴露 `/v1/chat/completions` 等兼容接口。开发时可将 `run` 换成 `dev` 获得自动重载。

官方 Docker 说明要求设置 `CLUEWEB_API_KEY`，并指出本地推理至少需要约 24GB 显存和网络访问。这个 API key 属于比赛搜索服务，不是 MedRAG 本地语料；没有它时我们只做离线代码阅读和 toy 适配。

### 测试与评测

仓库提供 `local_test.py` 检查接口是否满足动态/静态评测协议：

```bash
python local_test.py --base-url http://localhost:8000
python local_test.py --base-url http://localhost:8000 --test-mode run
python local_test.py --base-url http://localhost:8000 --test-mode evaluate
```

动态 `/run` 返回 SSE 流，事件可能包含中间步骤、最终报告、引用和完成标记；静态 `/evaluate` 接收 `query` 与 `iid`，返回 `query_id`、生成答案、引用和上下文。

## 3. 代码目录怎么读

建议按下面顺序阅读，不要从部署脚本开始：

1. `src/systems/rag_interface.py`：统一接口和请求/响应数据结构，先理解 `run_streaming()` 与 `evaluate()` 的契约；
2. `src/systems/rag_router/rag_router_llm.py`：R2RAG 的路由核心，`simple` 走 `VanillaRAG`，`complex` 走 `VanillaAgent`；
3. `src/tools/classifiers/llm_query_complexity.py`：查询复杂度 LLM 分类器的输入输出；
4. `src/systems/vanilla_agent/vanilla_rag.py`：单轮路径，生成查询变体、搜索、重排、截断上下文、生成答案；
5. `src/systems/vanilla_agent/vanilla_agent.py`：复杂问题路径的循环；重点看 `tries`、`query_history`、`review_documents()`、`max_tries` 和上下文 token 上限；
6. `src/systems/vanilla_agent/rag_util_fn.py`：查询变体、搜索和 prompt 构造的公共函数；
7. `src/tools/web_search.py`、`src/tools/reranker_vllm.py`：外部搜索结果格式和 Qwen 重排 API；
8. `src/apis/mmu_rag_router.py`、`src/apis/openai_router.py`：最后再看 HTTP/SSE 如何把系统接出去。

## 4. 与 MedRAG 的对应关系

MedRAG 解决的是“给定检索器如何召回证据”；R2RAG 在其上增加了控制层：

| MedRAG 基础 | R2RAG 对应概念 |
|---|---|
| BM25/Dense/Hybrid 检索 | `search_w_qv()` 背后的搜索工具 |
| Top-k 候选 | `VanillaRAG`/`VanillaAgent` 的候选文档 |
| Reranker | `GeneralReranker` |
| context 截断 | `truncate_docs()` / `atruncate_docs()` |
| 一次检索后生成 | `VanillaRAG` |
| 多轮检索与停止 | `VanillaAgent` |
| `medrag_answer()` 调用链 | `/run` → `RAGRouterLLM` → 选定 RAG 系统 |

注意：官方 R2RAG 面向 Web 搜索，论文描述的是整篇文档/网页检索，不等同于我们 MedRAG 的医学 JSONL chunk 索引。后续迁移时，应该让 R2RAG 控制器调用 MedRAG 的统一 `Retriever.search()` 接口，而不是直接复制外部 ClueWeb API。

## 5. 我们的学习边界

- 先读路由、单轮 RAG、迭代 Agent 和停止条件；
- 暂不下载 R2RAG 的比赛数据，不把 `data/`、`models/`、`logs/` 提交到主仓库；
- 暂不启动 vLLM，也不声称已经完成官方复现；
- 第一版迁移目标是：用一个假的 `search()` 返回 toy 文档，验证 simple/complex 两条路径和停止逻辑；
- 之后再把搜索后端替换成 MedRAG 的 BM25/Dense/Hybrid。

## 6. 今天阅读后的检查问题

1. `RAGRouterLLM` 为什么不能简单地对所有问题都调用 `VanillaAgent`？它节省了什么成本？
2. `VanillaRAG` 和 `VanillaAgent` 的共同步骤有哪些，真正新增的状态是什么？
3. `review_documents()` 返回 `is_sufficient` 和 `new_query` 后，主循环分别如何处理？
4. `max_tries`、上下文 token 上限和“证据足够”三者分别防止什么问题？
5. 如果把 MedRAG 的 `RetrievalSystem.retrieve()` 接到这里，最小适配层需要统一哪些字段？

