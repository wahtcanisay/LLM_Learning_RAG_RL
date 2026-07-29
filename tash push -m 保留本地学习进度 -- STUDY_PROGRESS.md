warning: in the working copy of 'STUDY_PROGRESS.md', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/STUDY_PROGRESS.md b/STUDY_PROGRESS.md[m
[1mindex 576b25c..3f0b909 100644[m
[1m--- a/STUDY_PROGRESS.md[m
[1m+++ b/STUDY_PROGRESS.md[m
[36m@@ -1,306 +1,172 @@[m
 # 当前阶段[m
 [m
[31m-阶段 1：MedRAG 基础检索学习与语料准备已完成；原生完整建库、生成和测评按用户决策暂缓。[m
[31m-阶段 2：R2RAG 已完成核心思想学习；按用户决策停止继续阅读外围工程代码。[m
[31m-阶段 3：LinearRAG 图结构检索与医学迁移（当前阶段）[m
[32m+[m[32m- 阶段 1：MedRAG 基础代码、toy BM25/Dense/RRF 和四套语料核验已完成；正式全量索引、生成与评测暂缓。[m
[32m+[m[32m- 阶段 2：R2RAG 核心动态路由思想已学习；外围 API、服务和完整复现停止推进。[m
[32m+[m[32m- 阶段 3：LinearRAG 图结构检索与医学迁移（当前阶段）。[m
 [m
 # 当前项目[m
 [m
[31m-LinearRAG 官方源码学习与最小医学数据复现。MedRAG 保留为后续医学 chunk 语料和 Dense/Hybrid 基线；R2RAG 只保留动态路由、查询改写、证据充分性和停止控制思想，不再继续阅读其 API、服务与流式展示代码。[m
[32m+[m[32m**Medical Routing GraphRAG**：MedRAG 提供医学语料与基础检索基线，LinearRAG 提供关系无关图检索，R2RAG 后续只作为动态路由与停止控制层。[m
 [m
[31m-LinearRAG 第一轮只阅读关系无关 Tri-Graph、实体抽取、语义桥接、段落打分和 PPR 两阶段检索。先使用官方 medical 小数据形成可验证结果，再迁移 MedRAG 的 Textbooks/StatPearls 小子集。[m
[32m+[m[32m当前只学习和最小复现 LinearRAG。先使用官方 medical 小数据验证，再考虑迁移 MedRAG 的 Textbooks/StatPearls 子集。[m
 [m
[31m-# 已确认的正式路线[m
[32m+[m[32m## LinearRAG 真实学习边界[m
 [m
[31m-- 日期：2026-07-18[m
[31m-- 决策：阶段 6 使用 **MedSearch-R1：基于领域微调与成本感知强化学习的医学证据搜索 Agent**，替换原阶段 6 计划。[m
[31m-- 衔接关系：MedRAG/R2RAG/LinearRAG 提供医学检索与动态控制基础，MedicalGPT 提供医学领域 SFT，Search-R1 提供多轮工具调用、rollout、reward 和 GRPO，LA-CDM 仅提供假设驱动、置信度校准和成本感知思想。[m
[31m-- 数据边界：不依赖 MIMIC-CDM，不模拟或编造患者临床检查；第一版使用公开、可验证的医学选择题或医学问答以及独立医学检索语料。[m
[31m-- 实施顺序：当前进入阶段 3 LinearRAG；R2RAG 不做完整服务复现，待 LinearRAG 和医学基线产生真实指标后，再决定是否把其控制思想接回统一 Retriever。[m
[31m-- 设计文档：`docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`[m
[32m+[m[32m- 已亲自阅读：`LinearRAG/readme.md`、`LinearRAG/run.py`、`src/config.py`、`src/embedding_store.py`、`src/utils.py::compute_mdhash_id()`、`src/LinearRAG.py` 的 `__init__()` 与 `load_embedding_store()`。[m
[32m+[m[32m- 尚未阅读：`src/ner.py`、`src/LinearRAG.py` 的离线建图与在线检索部分、`src/evaluate.py`、`src/utils.py` 的其余部分。[m
[32m+[m[32m- 助手添加注释、执行语法检查或检查调用链，只算材料准备，不算学习者完成。[m
[32m+[m[32m- LinearRAG 官方源码版本：`bcc94e66c221f798801255efba09311d6fbcd8d6`。[m
 [m
[31m-# 本周目标[m
[31m-[m
[31m-完成 LinearRAG 核心代码第一轮阅读：理解实体、句子语义桥和 passage 三层结构，区分离线构图与在线两阶段检索，并确定官方 medical 数据到 MedRAG chunk 的适配边界。[m
[31m-[m
[31m-# 今日唯一任务[m
[31m-[m
[31m-获取 LinearRAG 官方源码并确认 README、`run.py` 和 `src` 下 6 个核心 Python 文件的职责与阅读顺序；不下载数据、不安装依赖、不运行模型。[m
[31m-[m
[31m-# 完成标准[m
[31m-[m
[31m-- 官方源码位于 `LinearRAG/`，且没有嵌套 `.git`；[m
[31m-- 记录官方 commit SHA、文件数量和核心入口；[m
[31m-- 能复述 `load_dataset → index → qa/retrieve → evaluate` 顶层调用链；[m
[31m-- 不下载 official medical 数据、Embedding 模型或 SciSpacy 模型，不产生虚构指标。[m
[31m-[m
[31m-# 已完成[m
[32m+[m[32m## 已确认的路线与边界[m
 [m
[31m-- 日期：2026-07-16[m
[31m-- 完成内容：确认 MedRAG 仓库、容器基础环境、Git LFS 和仓库安全配置；确认不把旧依赖版本当作唯一目标。[m
[31m-- 运行命令：容器内完成 Java、Python、GPU、Git、Git LFS 和仓库来源检查。[m
[31m-- 结果与指标：仓库来源为 `https://github.com/gzxiong/MedRAG.git`，当前提交为 `7599a72`；RTX 5090 可见；暂无 RAG 实验指标。[m
[31m-- 代码或日志位置：`D:\code_list\some tricks\LLMLeanring\MedRAG`[m
[31m-[m
[31m-- 日期：2026-07-17[m
[31m-- 完成内容：完成 Day 1 第一部分阅读：`template.py`、`MedRAG.__init__`、`medrag_answer`；在 `src/medrag.py`、`src/utils.py`、`src/template.py` 的关键位置加入中文学习注释；完成 Python 语法检查。[m
[31m-- 运行命令：本地无字节码语法检查：`python -c "compile(...)"`（三个目标文件均返回 `SYNTAX_OK`）。[m
[31m-- 结果与指标：代码阅读完成到 `medrag_answer`；学习者已用自己的话回答模板职责、`rag` 分支、三种证据来源和上下文截断；`rrf_k` 已补充解释；未下载语料、未运行生成实验；无 QA 或召回指标。[m
[31m-- 显存峰值：本次代码阅读与语法检查未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/src/medrag.py`、`MedRAG/src/utils.py`、`MedRAG/src/template.py`。[m
[31m-[m
[31m-- 日期：2026-07-18[m
[31m-- 完成内容：完成 `Retriever`、`get_relevant_documents`、`RetrievalSystem.retrieve`、`RetrievalSystem.merge` 的代码阅读和检查题回答。[m
[31m-- 运行命令：本次以代码阅读和口头解释为主，未下载语料、未建立 BM25/FAISS 索引。[m
[31m-- 结果与指标：已理解 BM25/Dense 分支、`k` 与 `rrf_k`、RRF 按文档 ID 去重累加、Retriever 初始化的下载/建索引路径；暂无检索或 QA 指标。[m
[31m-- 显存峰值：未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/src/utils.py` 中 `Retriever`、`get_relevant_documents`、`RetrievalSystem.retrieve`、`merge`。[m
[31m-[m
[31m-- 日期：2026-07-19[m
[31m-- 完成内容：完成 `src/data/textbooks.py` 与 `src/data/pubmed.py` 阅读；理解 JSONL 数据契约、教材切块、PubMed 摘要解析和 `idx2txt()` 的 source/index 映射。[m
[31m-- 运行命令：本次以代码阅读和 toy 数据结构复述为主，未下载语料、未建立 BM25/FAISS 索引。[m
[31m-- 结果与指标：已解释 `id/title/content/contents`、`chunk_size=1000`、`chunk_overlap=200`、chunk 粒度差异及无摘要文章跳过逻辑；暂无检索或 QA 指标。[m
[31m-- 显存峰值：未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/src/data/textbooks.py`、`MedRAG/src/data/pubmed.py`、`MedRAG/src/medrag.py` 的 `Retriever.idx2txt()`。[m
[31m-[m
[31m-- 日期：2026-07-20[m
[31m-- 完成内容：完成 toy BM25/Lucene 建库与查询闭环；从 3 条 JSONL 文档建立索引，输入 `facial nerve`，核对 docid、score 和 `source/index` 到 chunk 的映射。[m
[31m-- 运行命令：`python -m pyserini.index.lucene --collection JsonCollection --input docs/task4_toy_collection --index /tmp/medrag_task4_index --generator DefaultLuceneDocumentGenerator --threads 1`；随后使用 `LuceneSearcher.search("facial nerve", k=3)` 查询。[m
[31m-- 结果与指标：3 个文档全部成功索引，`unindexable=0`、`empty=0`、`skipped=0`、`errors=0`；Top-1 为 `docid=toy_0`、`score=1.1835999488830566`、`source=toy`、`index=0`、`title=Neurology`。[m
[31m-- 显存峰值：未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/docs/task4_toy_collection/toy.jsonl`、`MedRAG/docs/task4_toy_bm25.md`、容器终端建库与查询日志。[m
[31m-[m
[31m-- 日期：2026-07-20[m
[31m-- 完成内容：完成 BM25/Lucene 与 Dense/FAISS 索引构建、加载和查询路径阅读；理解 `embed()`、`construct_index()`、FAISS `indices`、`scores/distances`、`metadatas.jsonl` 以及 HNSW 的作用。[m
[31m-- 运行命令：本次以代码阅读和检查题回答为主，未下载语料、未建立真实 BM25/FAISS 索引。[m
[31m-- 结果与指标：已能解释 BM25 倒排索引、Dense 向量索引、L2/IP 排序方向、metadata 映射、RRF 分数不能直接相加，以及 Retriever 初始化的条件性建库副作用；暂无检索或 QA 指标。[m
[31m-- 显存峰值：未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/src/utils.py` 的 `embed()`、`construct_index()`、`Retriever.__init__()`、`get_relevant_documents()`。[m
[31m-[m
[31m-- 日期：2026-07-20[m
[31m-- 完成内容：使用 `all-MiniLM-L6-v2` 在同一批 3 条 toy JSONL 上完成 Dense Embedding、FAISS `IndexFlatIP` 建库、query 检索和 metadata 映射。[m
[31m-- 运行命令：`python docs/task5_toy_dense.py`。[m
[31m-- 结果与指标：Top-1 为 `faiss_position=0`、`score=0.7347357273101807`、`source=toy`、`index=0`、`id=toy_0`、`title=Neurology`；Top-2 为 `toy_2`（0.11410441994667053），Top-3 为 `toy_1`（0.09615442156791687）。[m
[31m-- 显存峰值：未记录 GPU 峰值；本次 toy 编码未作为 GPU 实验统计。[m
[31m-- 代码或日志位置：`MedRAG/docs/task5_toy_dense.py`、`/tmp/medrag_task5_dense/faiss.index`、`embeddings.npy`、`metadatas.jsonl`、容器终端日志。[m
[31m-[m
[31m-- 日期：2026-07-20[m
[31m-- 完成内容：在同一 query 和 toy 文档上运行 BM25、Dense，并直接调用 MedRAG `RetrievalSystem.merge()` 完成 RRF 融合。[m
[31m-- 运行命令：`python docs/task6_toy_rrf.py`。[m
[31m-- 结果与指标：BM25 返回 `toy_0`（1.1835999488830566）；Dense 顺序为 `toy_0`（0.7347357273101807）、`toy_2`（0.11410441994667053）、`toy_1`（0.09615442156791687）；RRF 顺序为 `toy_0`（0.019801980198019802）、`toy_2`（0.00980392156862745）、`toy_1`（0.009708737864077669）。`toy_0` 的分数等于 `1/101 + 1/101`，验证了两路排名贡献累加。[m
[31m-- 结果边界：Dense 仍使用 Task 5 的 MiniLM toy 向量，不是 MedCPT 正式基线；本次验证的是 RRF 逻辑和数据接口。[m
[31m-- 显存峰值：未记录 GPU 峰值；本次 toy 实验未作为 GPU 实验统计。[m
[31m-- 代码或日志位置：`MedRAG/docs/task6_toy_rrf.py`、容器终端日志。[m
[31m-[m
[31m-- 日期：2026-07-23[m
[31m-- 完成内容：完成 Task 7 四套官方 chunk 的最终核验和冗余空间清理。逐文件检查 LFS pointer，抽查首/中/末 JSONL；Textbooks、PubMed、Wikipedia 的 LFS 对象通过 `git lfs fsck`，其中 Wikipedia 的通过记录来自容器终端，Textbooks 与 PubMed 在本次检查中返回 `Git LFS fsck OK`。随后使用 `git lfs prune --force` 删除三个数据仓库中已展开到工作区、可从远端重新下载的 LFS 缓存；删除 StatPearls 已生成 chunk 后可重新下载/生成的原始压缩包和解压目录。[m
[31m-- 运行命令：`git lfs fsck`；`git lfs prune --dry-run --force`；`git lfs prune --force`。JSONL 与磁盘检查由 UTF-8 Node 脚本直接读取宿主机挂载目录完成，未使用 PowerShell 读写文件。[m
[31m-- 结果与指标：Textbooks 18 个 JSONL、201.76 MiB；StatPearls 9646 个 JSONL、441.54 MiB；PubMed 1166 个 JSONL、65.20 GiB；Wikipedia 646 个 JSONL、42.54 GiB。四套语料均无 LFS pointer；PubMed 的 `pubmed23n0654.jsonl` 是官方仓库中原本就存在的唯一零字节分片，不是下载失败，其他抽样分片均可解析。LFS 缓存清理后均为 0 B；StatPearls 的 1.76 GiB 压缩包和 2.35 GiB 解压原料已删除。磁盘可用空间从 123.55 GiB 增至 235.64 GiB，实际释放 112.09 GiB。[m
[31m-- 结果边界：删除 LFS 缓存不会影响当前 `chunk/` 读取，但以后若执行会重建工作区文件的 `git checkout/reset`，Git LFS 需要重新从 Hugging Face 下载对象；删除 StatPearls 原料后，若要重新运行切块脚本，需要重新下载并解压 NCBI 源文件。[m
[31m-- 显存峰值：本次数据核验与清理未使用 GPU。[m
[31m-- 代码或日志位置：`MedRAG/corpus/{textbooks,statpearls,pubmed,wikipedia}/chunk`；本文件记录最终核验结果，语料目录由根仓库 `.gitignore` 忽略。[m
[31m-[m
[31m-- 日期：2026-07-23[m
[31m-- 完成内容：路线调整。MedRAG 完成基础代码阅读、toy BM25/Dense/RRF 闭环、四套医学 chunk 下载核验和冗余清理；不再继续 MedRAG 原生完整索引、生成和测评。进入 R2RAG 学习，并计划把可共享的 Embedding 调用抽象为统一接口。[m
[31m-- 决策依据：R2RAG 主检索默认使用 FineWeb/ClueWeb 等搜索服务；`sentence-transformers/all-MiniLM-L6-v2` 出现在 GRAG/LocalGRAG 的候选重排和 RAGAS 评测中。复杂度分类器使用训练时绑定的 BERT 特征模型，Qwen3-Reranker 是独立的重排模型，不能全部替换成一个 Embedding 模型。[m
[31m-- 下一步：阅读已加注释的 `R2RAG/src/systems/rag_interface.py`、`rag_router/rag_router_llm.py`、`rag_router/llm_query_complexity.py`、`vanilla_agent/vanilla_rag.py`，先完成调用链解释，再设计 `EmbeddingProvider`，不启动 LLM 或外部搜索服务。[m
[31m-[m
[31m-前三个 RAG 项目的融合定位（2026-07-23）：MedRAG 作为医学语料、chunk 数据契约和基础检索底座；R2RAG 作为控制层，决定单轮/多轮、查询改写、证据审查和停止；LinearRAG 作为可插拔的结构化检索后端，通过关系无关的 Tri-Graph、实体激活、语义桥接和段落重要性聚合处理跨实体/多跳问题。目标系统暂定为 `R2-Linear-MedRAG`：R2RAG 控制器只调用统一 `Retriever.search(query, top_k)`，后端可选择 `HybridRetriever` 或 `LinearGraphRetriever`，最终统一返回 `id/source/index/title/content/score` 等字段。[m
[31m-[m
[31m-融合顺序：先用 toy 文档验证统一 Retriever 接口；再把 MedRAG chunk 接入 BM25/Dense/Hybrid 适配器；然后在相同 chunk 上构建 LinearRAG 图索引并验证单轮图检索；最后让 R2RAG 的 simple/complex 控制器调用这些后端。不要一开始同时改变路由、图构建、Embedding、重排和生成模型。[m
[31m-[m
[31m-- 日期：2026-07-24[m
[31m-- 完成内容：为 `vanilla_agent.py` 的 `QueryHistoryItem`、`VanillaAgent`、`review_documents()` 和 `run_streaming()` 主循环补充中文学习注释；明确本轮候选、跨轮证据池、查询历史、查询改写、上下文预算和语义停止的关系。外围 `agent_tools.py`、`pre_flight_models()`、流式展示细节和 `__main__` 手工入口暂时跳过。[m
[31m-- 运行命令：`python -m py_compile R2RAG/src/systems/vanilla_agent/vanilla_agent.py`；`git diff --check`。[m
[31m-- 结果与指标：语法检查通过，差异无空白错误；本次只增加注释和文档字符串，未改变代码行为，未调用 LLM、vLLM 或外部搜索，暂无实验指标。[m
[31m-- 显存峰值：未使用 GPU。[m
[31m-- 代码或日志位置：`R2RAG/src/systems/vanilla_agent/vanilla_agent.py`。[m
[32m+[m[32m- 顺序保持为：MedRAG → R2RAG → LinearRAG → MedicalGPT → Search-R1 → MedSearch-R1。[m
[32m+[m[32m- 阶段 6 使用 `MedSearch-R1`；设计文档位于 `docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`。[m
[32m+[m[32m- 不依赖 MIMIC-CDM，不模拟真实临床检查，不声称提供临床诊断。[m
[32m+[m[32m- 暂缓：MedRAG 原生全量索引/生成/评测、R2RAG 外围工程、PageIndex 候选项目。[m
[32m+[m[32m- 不同时修改路由、图构建、Embedding、重排和生成模型；每次只验证一个主要变量。[m
 [m
[31m-# 遇到的问题[m
[32m+[m[32m# 本周目标[m
 [m
[31m-- 问题：`VanillaAgent` 在审查前通过 `base_count` 给候选文档的 `sid` 做跨轮偏移，但该计数按 `usef