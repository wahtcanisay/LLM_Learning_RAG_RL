# 当前阶段

阶段 1：MedRAG 基础检索学习与语料准备已完成；原生完整建库、生成和测评按用户决策暂缓。
阶段 2：R2RAG 已完成核心思想学习；按用户决策停止继续阅读外围工程代码。
阶段 3：LinearRAG 图结构检索与医学迁移（当前阶段）

# 当前项目

LinearRAG 官方源码学习与最小医学数据复现。MedRAG 保留为后续医学 chunk 语料和 Dense/Hybrid 基线；R2RAG 只保留动态路由、查询改写、证据充分性和停止控制思想，不再继续阅读其 API、服务与流式展示代码。

LinearRAG 第一轮只阅读关系无关 Tri-Graph、实体抽取、语义桥接、段落打分和 PPR 两阶段检索。先使用官方 medical 小数据形成可验证结果，再迁移 MedRAG 的 Textbooks/StatPearls 小子集。

# 已确认的正式路线

- 日期：2026-07-18
- 决策：阶段 6 使用 **MedSearch-R1：基于领域微调与成本感知强化学习的医学证据搜索 Agent**，替换原阶段 6 计划。
- 衔接关系：MedRAG/R2RAG/LinearRAG 提供医学检索与动态控制基础，MedicalGPT 提供医学领域 SFT，Search-R1 提供多轮工具调用、rollout、reward 和 GRPO，LA-CDM 仅提供假设驱动、置信度校准和成本感知思想。
- 数据边界：不依赖 MIMIC-CDM，不模拟或编造患者临床检查；第一版使用公开、可验证的医学选择题或医学问答以及独立医学检索语料。
- 实施顺序：当前进入阶段 3 LinearRAG；R2RAG 不做完整服务复现，待 LinearRAG 和医学基线产生真实指标后，再决定是否把其控制思想接回统一 Retriever。
- 设计文档：`docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`

# 本周目标

完成 LinearRAG 核心代码第一轮阅读：理解实体、句子语义桥和 passage 三层结构，区分离线构图与在线两阶段检索，并确定官方 medical 数据到 MedRAG chunk 的适配边界。

# 今日唯一任务

获取 LinearRAG 官方源码并确认 README、`run.py` 和 `src` 下 6 个核心 Python 文件的职责与阅读顺序；不下载数据、不安装依赖、不运行模型。

# 完成标准

- 官方源码位于 `LinearRAG/`，且没有嵌套 `.git`；
- 记录官方 commit SHA、文件数量和核心入口；
- 能复述 `load_dataset → index → qa/retrieve → evaluate` 顶层调用链；
- 不下载 official medical 数据、Embedding 模型或 SciSpacy 模型，不产生虚构指标。

# 已完成

- 日期：2026-07-16
- 完成内容：确认 MedRAG 仓库、容器基础环境、Git LFS 和仓库安全配置；确认不把旧依赖版本当作唯一目标。
- 运行命令：容器内完成 Java、Python、GPU、Git、Git LFS 和仓库来源检查。
- 结果与指标：仓库来源为 `https://github.com/gzxiong/MedRAG.git`，当前提交为 `7599a72`；RTX 5090 可见；暂无 RAG 实验指标。
- 代码或日志位置：`D:\code_list\some tricks\LLMLeanring\MedRAG`

- 日期：2026-07-17
- 完成内容：完成 Day 1 第一部分阅读：`template.py`、`MedRAG.__init__`、`medrag_answer`；在 `src/medrag.py`、`src/utils.py`、`src/template.py` 的关键位置加入中文学习注释；完成 Python 语法检查。
- 运行命令：本地无字节码语法检查：`python -c "compile(...)"`（三个目标文件均返回 `SYNTAX_OK`）。
- 结果与指标：代码阅读完成到 `medrag_answer`；学习者已用自己的话回答模板职责、`rag` 分支、三种证据来源和上下文截断；`rrf_k` 已补充解释；未下载语料、未运行生成实验；无 QA 或召回指标。
- 显存峰值：本次代码阅读与语法检查未使用 GPU。
- 代码或日志位置：`MedRAG/src/medrag.py`、`MedRAG/src/utils.py`、`MedRAG/src/template.py`。

- 日期：2026-07-18
- 完成内容：完成 `Retriever`、`get_relevant_documents`、`RetrievalSystem.retrieve`、`RetrievalSystem.merge` 的代码阅读和检查题回答。
- 运行命令：本次以代码阅读和口头解释为主，未下载语料、未建立 BM25/FAISS 索引。
- 结果与指标：已理解 BM25/Dense 分支、`k` 与 `rrf_k`、RRF 按文档 ID 去重累加、Retriever 初始化的下载/建索引路径；暂无检索或 QA 指标。
- 显存峰值：未使用 GPU。
- 代码或日志位置：`MedRAG/src/utils.py` 中 `Retriever`、`get_relevant_documents`、`RetrievalSystem.retrieve`、`merge`。

- 日期：2026-07-19
- 完成内容：完成 `src/data/textbooks.py` 与 `src/data/pubmed.py` 阅读；理解 JSONL 数据契约、教材切块、PubMed 摘要解析和 `idx2txt()` 的 source/index 映射。
- 运行命令：本次以代码阅读和 toy 数据结构复述为主，未下载语料、未建立 BM25/FAISS 索引。
- 结果与指标：已解释 `id/title/content/contents`、`chunk_size=1000`、`chunk_overlap=200`、chunk 粒度差异及无摘要文章跳过逻辑；暂无检索或 QA 指标。
- 显存峰值：未使用 GPU。
- 代码或日志位置：`MedRAG/src/data/textbooks.py`、`MedRAG/src/data/pubmed.py`、`MedRAG/src/medrag.py` 的 `Retriever.idx2txt()`。

- 日期：2026-07-20
- 完成内容：完成 toy BM25/Lucene 建库与查询闭环；从 3 条 JSONL 文档建立索引，输入 `facial nerve`，核对 docid、score 和 `source/index` 到 chunk 的映射。
- 运行命令：`python -m pyserini.index.lucene --collection JsonCollection --input docs/task4_toy_collection --index /tmp/medrag_task4_index --generator DefaultLuceneDocumentGenerator --threads 1`；随后使用 `LuceneSearcher.search("facial nerve", k=3)` 查询。
- 结果与指标：3 个文档全部成功索引，`unindexable=0`、`empty=0`、`skipped=0`、`errors=0`；Top-1 为 `docid=toy_0`、`score=1.1835999488830566`、`source=toy`、`index=0`、`title=Neurology`。
- 显存峰值：未使用 GPU。
- 代码或日志位置：`MedRAG/docs/task4_toy_collection/toy.jsonl`、`MedRAG/docs/task4_toy_bm25.md`、容器终端建库与查询日志。

- 日期：2026-07-20
- 完成内容：完成 BM25/Lucene 与 Dense/FAISS 索引构建、加载和查询路径阅读；理解 `embed()`、`construct_index()`、FAISS `indices`、`scores/distances`、`metadatas.jsonl` 以及 HNSW 的作用。
- 运行命令：本次以代码阅读和检查题回答为主，未下载语料、未建立真实 BM25/FAISS 索引。
- 结果与指标：已能解释 BM25 倒排索引、Dense 向量索引、L2/IP 排序方向、metadata 映射、RRF 分数不能直接相加，以及 Retriever 初始化的条件性建库副作用；暂无检索或 QA 指标。
- 显存峰值：未使用 GPU。
- 代码或日志位置：`MedRAG/src/utils.py` 的 `embed()`、`construct_index()`、`Retriever.__init__()`、`get_relevant_documents()`。

- 日期：2026-07-20
- 完成内容：使用 `all-MiniLM-L6-v2` 在同一批 3 条 toy JSONL 上完成 Dense Embedding、FAISS `IndexFlatIP` 建库、query 检索和 metadata 映射。
- 运行命令：`python docs/task5_toy_dense.py`。
- 结果与指标：Top-1 为 `faiss_position=0`、`score=0.7347357273101807`、`source=toy`、`index=0`、`id=toy_0`、`title=Neurology`；Top-2 为 `toy_2`（0.11410441994667053），Top-3 为 `toy_1`（0.09615442156791687）。
- 显存峰值：未记录 GPU 峰值；本次 toy 编码未作为 GPU 实验统计。
- 代码或日志位置：`MedRAG/docs/task5_toy_dense.py`、`/tmp/medrag_task5_dense/faiss.index`、`embeddings.npy`、`metadatas.jsonl`、容器终端日志。

- 日期：2026-07-20
- 完成内容：在同一 query 和 toy 文档上运行 BM25、Dense，并直接调用 MedRAG `RetrievalSystem.merge()` 完成 RRF 融合。
- 运行命令：`python docs/task6_toy_rrf.py`。
- 结果与指标：BM25 返回 `toy_0`（1.1835999488830566）；Dense 顺序为 `toy_0`（0.7347357273101807）、`toy_2`（0.11410441994667053）、`toy_1`（0.09615442156791687）；RRF 顺序为 `toy_0`（0.019801980198019802）、`toy_2`（0.00980392156862745）、`toy_1`（0.009708737864077669）。`toy_0` 的分数等于 `1/101 + 1/101`，验证了两路排名贡献累加。
- 结果边界：Dense 仍使用 Task 5 的 MiniLM toy 向量，不是 MedCPT 正式基线；本次验证的是 RRF 逻辑和数据接口。
- 显存峰值：未记录 GPU 峰值；本次 toy 实验未作为 GPU 实验统计。
- 代码或日志位置：`MedRAG/docs/task6_toy_rrf.py`、容器终端日志。

- 日期：2026-07-23
- 完成内容：完成 Task 7 四套官方 chunk 的最终核验和冗余空间清理。逐文件检查 LFS pointer，抽查首/中/末 JSONL；Textbooks、PubMed、Wikipedia 的 LFS 对象通过 `git lfs fsck`，其中 Wikipedia 的通过记录来自容器终端，Textbooks 与 PubMed 在本次检查中返回 `Git LFS fsck OK`。随后使用 `git lfs prune --force` 删除三个数据仓库中已展开到工作区、可从远端重新下载的 LFS 缓存；删除 StatPearls 已生成 chunk 后可重新下载/生成的原始压缩包和解压目录。
- 运行命令：`git lfs fsck`；`git lfs prune --dry-run --force`；`git lfs prune --force`。JSONL 与磁盘检查由 UTF-8 Node 脚本直接读取宿主机挂载目录完成，未使用 PowerShell 读写文件。
- 结果与指标：Textbooks 18 个 JSONL、201.76 MiB；StatPearls 9646 个 JSONL、441.54 MiB；PubMed 1166 个 JSONL、65.20 GiB；Wikipedia 646 个 JSONL、42.54 GiB。四套语料均无 LFS pointer；PubMed 的 `pubmed23n0654.jsonl` 是官方仓库中原本就存在的唯一零字节分片，不是下载失败，其他抽样分片均可解析。LFS 缓存清理后均为 0 B；StatPearls 的 1.76 GiB 压缩包和 2.35 GiB 解压原料已删除。磁盘可用空间从 123.55 GiB 增至 235.64 GiB，实际释放 112.09 GiB。
- 结果边界：删除 LFS 缓存不会影响当前 `chunk/` 读取，但以后若执行会重建工作区文件的 `git checkout/reset`，Git LFS 需要重新从 Hugging Face 下载对象；删除 StatPearls 原料后，若要重新运行切块脚本，需要重新下载并解压 NCBI 源文件。
- 显存峰值：本次数据核验与清理未使用 GPU。
- 代码或日志位置：`MedRAG/corpus/{textbooks,statpearls,pubmed,wikipedia}/chunk`；本文件记录最终核验结果，语料目录由根仓库 `.gitignore` 忽略。

- 日期：2026-07-23
- 完成内容：路线调整。MedRAG 完成基础代码阅读、toy BM25/Dense/RRF 闭环、四套医学 chunk 下载核验和冗余清理；不再继续 MedRAG 原生完整索引、生成和测评。进入 R2RAG 学习，并计划把可共享的 Embedding 调用抽象为统一接口。
- 决策依据：R2RAG 主检索默认使用 FineWeb/ClueWeb 等搜索服务；`sentence-transformers/all-MiniLM-L6-v2` 出现在 GRAG/LocalGRAG 的候选重排和 RAGAS 评测中。复杂度分类器使用训练时绑定的 BERT 特征模型，Qwen3-Reranker 是独立的重排模型，不能全部替换成一个 Embedding 模型。
- 下一步：阅读已加注释的 `R2RAG/src/systems/rag_interface.py`、`rag_router/rag_router_llm.py`、`rag_router/llm_query_complexity.py`、`vanilla_agent/vanilla_rag.py`，先完成调用链解释，再设计 `EmbeddingProvider`，不启动 LLM 或外部搜索服务。

前三个 RAG 项目的融合定位（2026-07-23）：MedRAG 作为医学语料、chunk 数据契约和基础检索底座；R2RAG 作为控制层，决定单轮/多轮、查询改写、证据审查和停止；LinearRAG 作为可插拔的结构化检索后端，通过关系无关的 Tri-Graph、实体激活、语义桥接和段落重要性聚合处理跨实体/多跳问题。目标系统暂定为 `R2-Linear-MedRAG`：R2RAG 控制器只调用统一 `Retriever.search(query, top_k)`，后端可选择 `HybridRetriever` 或 `LinearGraphRetriever`，最终统一返回 `id/source/index/title/content/score` 等字段。

融合顺序：先用 toy 文档验证统一 Retriever 接口；再把 MedRAG chunk 接入 BM25/Dense/Hybrid 适配器；然后在相同 chunk 上构建 LinearRAG 图索引并验证单轮图检索；最后让 R2RAG 的 simple/complex 控制器调用这些后端。不要一开始同时改变路由、图构建、Embedding、重排和生成模型。

- 日期：2026-07-24
- 完成内容：为 `vanilla_agent.py` 的 `QueryHistoryItem`、`VanillaAgent`、`review_documents()` 和 `run_streaming()` 主循环补充中文学习注释；明确本轮候选、跨轮证据池、查询历史、查询改写、上下文预算和语义停止的关系。外围 `agent_tools.py`、`pre_flight_models()`、流式展示细节和 `__main__` 手工入口暂时跳过。
- 运行命令：`python -m py_compile R2RAG/src/systems/vanilla_agent/vanilla_agent.py`；`git diff --check`。
- 结果与指标：语法检查通过，差异无空白错误；本次只增加注释和文档字符串，未改变代码行为，未调用 LLM、vLLM 或外部搜索，暂无实验指标。
- 显存峰值：未使用 GPU。
- 代码或日志位置：`R2RAG/src/systems/vanilla_agent/vanilla_agent.py`。

# 遇到的问题

- 问题：`VanillaAgent` 在审查前通过 `base_count` 给候选文档的 `sid` 做跨轮偏移，但该计数按 `useful_docs` 数量推进；因此不能未经验证地把历史 `sid` 视为全局唯一。
- 原因：`update_docs_sids()` 按当前候选列表重新编号，而累积池只统计被选中的文档；两套计数粒度不同。
- 解决办法：本次只补充严谨注释，不改变原始逻辑；后续若迁移到统一 Retriever，再用 toy 多轮测试专门验证 ID、去重和引用映射。

- 问题：初次回答时把 `rrf_k` 说成检索器参数，并把 Retriever 初始化概括成“先检查索引”。
- 原因：尚未区分候选排名融合和原始检索分数，也没有按 `Retriever.__init__` 的实际顺序追踪副作用。
- 解决办法：已纠正：`rrf_k` 是 RRF 排名平滑常数；初始化顺序是先检查 chunk/下载语料，再检查并构建具体索引。

- 问题：第一次执行 `docs/task6_toy_rrf.py` 报 `ModuleNotFoundError: No module named 'src'`。
- 原因：直接执行 docs 下的脚本时，Python 首要模块路径是 `docs/`，项目根目录没有自动加入导入路径。
- 解决办法：脚本根据 `__file__` 自动把 MedRAG 根目录加入 `sys.path`；已通过 Python 语法检查，待重新运行 RRF 实验。

# 下一步

## 2026-07-21：Task 7——下载并核验全部独立语料源

今天唯一核心任务：在已配置的容器中下载官方 `MedRAG` 的四个独立语料源 `textbooks`、`statpearls`、`pubmed`、`wikipedia` 的可检索 chunk，检查目录、JSONL 文件和字段契约；暂不调用 LLM。

不单独克隆 `MedCorp`：代码中的 `MedCorp` 只是组合上述四个目录。优先只拉取 `chunk/**`，不额外拉取 PubMed 原始 `baseline/**`，避免无必要的重复占用；不实例化会自动建库的 `Retriever`，不建立 BM25/FAISS 索引。

完成标准：目录下载成功；不存在 LFS pointer 残留；能统计 JSONL 文件数和磁盘占用；能读取至少一行并确认 `id/title/content/contents` 四个字段。

推荐阅读顺序：

1. 先检查容器磁盘空间和 Git LFS 状态。
2. 克隆 `https://huggingface.co/datasets/MedRAG/textbooks` 到 `corpus/textbooks`。
3. 检查 `corpus/textbooks/chunk` 是否存在以及 JSONL 文件数量。
4. 读取一条 JSONL，核对四个字段和 `id` 是否稳定。
5. 记录语料规模和磁盘占用，暂不建立索引。

Task 7 的生成链路学习顺延到语料核验完成后。

当前状态：用户反馈四个独立语料源已经下载完成；本地进度文件尚未看到对应的磁盘、文件数和字段核验输出，因此 Task 7 暂不标记为完成。

核验结果（2026-07-21）：`textbooks` 正常，18 个 JSONL、约 404M，首条记录包含 `id/title/content/contents` 且未发现 LFS pointer；`statpearls` 缺少 `chunk` 目录，仅约 60K；`pubmed` 有 1166 个文件但总大小约 372K，首条内容无法解析为 JSON；`wikipedia` 有 646 个文件但总大小约 236K。后两者规模明显异常，四个目录也不是 Git 仓库，不能用 `git -C ... lfs pull` 修复，需先检查文件头并重新用正确方式下载。

当前进展：StatPearls 原始压缩包 `corpus/statpearls/statpearls_NBK430685.tar.gz` 正在从 NCBI 下载；文件总大小约 1.8G，用户日志显示已完成约 587.61M（32%）。下载完成前不能解压或运行 chunk 生成脚本。

补充核验：用户贴出的 StatPearls 输出是 tar 解压产生的 `.nxml`、图片和视频文件列表，只能证明原始目录正在/已经展开，尚未证明 `corpus/statpearls/chunk/*.jsonl` 已生成；需要单独检查并运行 `src/data/statpearls.py`。

StatPearls 最新结果（2026-07-22）：已处理 9648 个 `.nxml` 文件，生成 9646 个 `chunk/*.jsonl` 文件；脚本进度 `9648/9648` 完成。少出的 2 个文件符合 `statpearls.py` 对空 `saved_text` 文章跳过写出的逻辑，仍需抽样核验 JSONL 字段。

Wikipedia 完成结果（2026-07-23）：Git LFS 断点续传期间多次在 Hugging Face Batch API 出现 `EOF`，降低批量/并发压力后完成下载。`git lfs fsck` 返回 `Git LFS fsck OK`；`corpus/wikipedia/chunk` 中共有 646 个 JSONL，工作区 chunk 占用 43G，`.git/lfs/objects` 本地缓存另占 43G，`du -sh corpus/wikipedia` 实测仓库总占用 86G。两者分别是可直接读取的工作区语料和 Git LFS 内容寻址缓存，不是两套不同 Wikipedia 数据。

Wikipedia JSONL 抽样结果（2026-07-23）：首文件为 `corpus/wikipedia/chunk/wiki20220301en000.jsonl`，首条记录可按 UTF-8/JSON 正常解析，不是 LFS pointer；字段为 `content/contents/id/title/wiki_id`，其中 `id=wiki20220301en000_0`、`title=Anarchism`、`content_length=559`，脚本输出 `WIKIPEDIA_VALIDATION_OK`。Wikipedia 语料下载与基础字段核验正式标记完成。

Task 7 最终状态（2026-07-23）：Textbooks、StatPearls、PubMed、Wikipedia 均已有有效 chunk，并通过文件数、pointer、UTF-8/JSON 字段和磁盘占用核验。PubMed 的 LFS 哈希检查返回 `Git LFS fsck OK`，因此 Task 7 正式完成。冗余 LFS 缓存和 StatPearls 原始中间文件已经清理，完整数据结果与恢复边界见“已完成”中的 2026-07-23 记录。

当前下一次唯一核心任务：按 R2RAG 核心代码顺序完成四个文件的阅读和问题回答；MedRAG 的完整索引与 Recall@k/MRR/QA 测评暂时不安排，待统一检索接口确定后再决定是否迁移到医学语料。

明天要回答：

1. 四个独立语料源的目录分别是什么？`MedCorp` 为什么不需要单独下载？
2. 下载后的真实 JSONL 是否仍遵循 `id/title/content/contents` 契约？
3. 为什么 PubMed 先只拉取 `chunk/**`，而不是原始 `baseline/**`？
4. 为什么下载语料时不能直接实例化 `Retriever`？
5. 每个语料源的 JSONL 文件数、chunk 数量和磁盘占用如何记录？

明天暂不下载完整语料、不运行完整 QA 生成；只使用已经生成的 toy 结果进行分析。

## 2026-07-22：并行准备阶段 2——下载并阅读 R2RAG

用户决定暂缓剩余 PubMed/Wikipedia 大型 LFS 语料的断点续传，先下载下一阶段 R2RAG 源码。此举是代码学习准备，不代表阶段 1 的 MedRAG 语料核验已经完成，也不提前运行 R2RAG 正式实验。

- 官方仓库：`https://github.com/rmit-ir/NeurIPS-MMU-RAG`
- 本地路径：`R2RAG/`
- 下载提交：源码已克隆后移除嵌套 `.git` 元数据，作为主学习仓库中的普通代码目录保存。
- 当前源码体量：约 67.5 MiB（233 个文件；不含嵌套 Git 元数据）。
- 中文学习译注：`R2RAG/README_zh.md`
- 核心入口：`src/systems/rag_router/rag_router_llm.py`、`src/systems/vanilla_agent/vanilla_rag.py`、`src/systems/vanilla_agent/vanilla_agent.py`、`src/systems/rag_interface.py`。
- 代码理解：`RAGRouterLLM` 用 LLM 判断 simple/complex；简单问题进入单轮 `VanillaRAG`，复杂问题进入带查询改写、证据审查、去重、token 上限和 `max_tries` 的 `VanillaAgent`。
- 数据边界：R2RAG 的 `data/`、`models/`、`logs/` 与 MedRAG 的 `corpus/` 均加入忽略规则，不进入主仓库提交。

R2RAG 第一轮学习顺序：`RAGInterface` → `RAGRouterLLM` → `QueryComplexityLLM` → `VanillaRAG` → `VanillaAgent` 主循环 → `search_w_qv`/`GeneralReranker` → API 路由。完成代码解释后，再做不调用 LLM 的 toy 路由测试。

R2RAG 核心注释准备（2026-07-22）：已为下一次阅读的四个文件补充分层中文学习注释：`rag_interface.py`、`rag_router_llm.py`、`llm_query_complexity.py`、`vanilla_rag.py`。注释覆盖统一流式协议、静态评测如何消费异步流、simple/complex 路由、LLM 分类器的延迟初始化与标签含义、单轮查询扩展、检索、重排、证据截断、流式生成和引用终止事件。

- 验证命令：使用 UTF-8 读取四个文件；使用 Python `compile()` 做无导入语法检查；对 `HEAD` 与工作区源码去除 docstring 后比较 AST；运行 `git diff --check`。
- 验证结果：`R2RAG_CORE_COMMENTS_SYNTAX_OK`；四个文件均返回 `COMMENT_ONLY_AST_OK`；`git diff --check` 无空白错误。
- 环境边界：宿主机当前没有 `uv`，因此本次仅做不导入依赖的等价 Python 语法检查；未启动 vLLM、未调用搜索服务或 LLM、未产生实验指标。
- 下一次唯一任务：按 `RAGInterface` → `RAGRouterLLM` → `QueryComplexityLLM` → `VanillaRAG` 阅读并回答 `R2RAG/README_zh.md` 中的前四个检查问题；`VanillaAgent` 留到下一轮。

## 当前（2026-07-24）

- `VanillaAgent` 主循环已完成注释和语法验证，不再继续阅读 R2RAG 的外围服务代码。
- 下一步唯一任务：用自己的话复述一次“简单问题单轮、复杂问题多轮”的完整调用链，并指出 `review_documents()` 的四个返回值如何改变控制状态；完成后进入 LinearRAG 的官方索引和两阶段检索核心。

# 待补知识

- `Retriever.__init__` 的下载和索引构建副作用；
- BM25 的 Pyserini/Lucene 路径；
- Dense Retrieval 的 Embedding、FAISS 和 metadata 映射；
- `RetrievalSystem.retrieve` 的候选组织；
- RRF 分数、排序、去重和 Top-k 截断；
- 检索命中与最终问答正确不是同一个指标。

# 实验结果总表

| 方法 | Recall@5 | Recall@10 | MRR | QA 准确率 | 平均延迟 | 显存峰值 |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| BM25 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| Dense | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| Hybrid | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| Hybrid + Reranker | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

# 失败案例

尚无正式实验失败案例。环境配置过程中的 Git/Git LFS 问题已定位并解决，后续保留真实的下载、检索、生成和评测失败记录。

- 日期：2026-07-20
- 任务：Task 4 toy BM25/Lucene 建库。
- 报错：`pyjnius` 报 `Exception: Unable to find javac`，随后 Pyserini 在导入 Lucene 模块时失败。
- 初步根因：容器中有 Java runtime，但当前环境找不到 `javac`；此前安装的 `openjdk-17-jre-headless` 只提供 JRE，不保证提供 JDK 编译器。
- 下一步：在容器中检查 `java`、`javac`、`JAVA_HOME` 和实际 JDK 路径；安装与当前 Java 主版本匹配的 headless JDK 后，再重跑 toy 建库。

## 2026-07-24：暂停推进并完成新 RAG 项目选型调研

- 完成内容：暂停 R2RAG 后续代码学习；筛查 GitHub 新近热门 RAG 仓库，并检查 Papernotes 的 ICLR 2026 Information Retrieval/RAG 分类页。该页面共 81 篇条目，本次深查 11 个有代表性的论文仓库。
- 核心结论：不更换既定 LinearRAG 方向；将项目收敛为“LinearRAG 核心复现 + GraphRAG-Benchmark 医学评测”。PageIndex 作为工业侧独立小项目候选，Reranker-Guided Search 作为可选检索增强模块。
- 跳过内容：R2RAG 的 API、服务启动和流式展示等外围工程；Youtu-GraphRAG、DeepRAG、HiPRAG、FrugalRAG、Q-RAG 等当前过重项目；GRO-RAG、SmartChunk 等尚无可核验官方实现的项目。
- 验证依据：核对候选官方 GitHub、README、核心源码目录、OpenReview、提交活跃度、许可证和最小运行入口；未克隆新仓库、未运行模型、未产生或引用为自己的实验指标。
- 工具问题：本机 `gh 2.96.0` 已安装，但 `gh auth status` 显示账号 `wahtcanisay` 的 token 无效；本次改用公开网页和 GitHub API 完成只读调研。后续如需用 `gh` 克隆或查询，应先运行 `gh auth login -h github.com`。
- 详细报告：`docs/rag-project-scout-2026-07-24.md`
- 下一步：用户确认选型后，只阅读 LinearRAG README 和 `src` 下 6 个 Python 文件，画出构图与两阶段检索调用链；不下载大数据、不运行生成模型。

### GitHub 热门侧补充核验

- 补充原因：上一轮最终结论以 ICLR 2026 论文仓库为主，没有单独展示 GitHub 热门项目的完整筛选结果。
- 查询方式：使用 GitHub 官方公开 REST API，组合 `topic:rag`、`topic:retrieval-augmented-generation`、`rag in:name`、`graphrag`、2025 年后创建和 2026-04 后仍更新等条件，按 stars 排序并去重；随后读取候选 README 和递归源码树。
- 新结论：PageIndex 是 GitHub 热门侧最符合“新、完整、核心代码小”的项目；LinearRAG 是学术侧最符合要求的项目；ViDoRAG 是多模态方向代码较简洁的备选。RAG-Anything、PixelRAG、LEANN、Memvid、Local Deep Research 和 DeepSearcher 因框架、基础设施或服务代码偏重而不作为当前主项目。
- 当前边界：仅完成公开元数据与源码结构核验，未克隆、未安装、未运行任何候选项目，未产生实验指标。
- 下一步候选任务：只读 PageIndex 的 README、`pageindex/page_index.py` 和 `pageindex/retrieve.py`，判断其语义树检索是否值得做成独立医学长文档 RAG 小项目。

### 语料适配后的执行顺序

- 决策：先学习并最小复现 LinearRAG，PageIndex 调整为第二个独立小项目。
- 原因：现有 MedRAG 是 `id/title/content` chunk 语料，能直接适配 LinearRAG；LinearRAG 官方发布 `medical/chunks.json` 和 `medical/questions.json`，并提供 biomedical SciSpacy 配置，可以较快形成 evidence 命中和 QA 指标。PageIndex 依赖长 PDF 的页码、章节与层级，直接输入已经切平的 MedRAG JSONL 会损失其核心优势，而且官方没有配套医疗 benchmark。
- 当前唯一下一步：只读 LinearRAG README 和 `src` 下 6 个核心 Python 文件，画出实体抽取、Tri-Graph 构建、查询激活和两阶段检索调用链；暂不下载数据、不运行模型。

## 2026-07-25：正式切换到 LinearRAG

- 完成内容：停止继续学习 R2RAG；下载 LinearRAG 官方 `main` 分支源码压缩包并解压到 `LinearRAG/`，未保留嵌套 `.git`。确认顶层入口为 `run.py`，核心目录包含 `config.py`、`embedding_store.py`、`evaluate.py`、`LinearRAG.py`、`ner.py`、`utils.py`。
- 源码版本：`bcc94e66c221f798801255efba09311d6fbcd8d6`，官方仓库 `https://github.com/DEEP-PolyU/LinearRAG`。
- 代码规模：全仓 7 个 Python 文件；`src/LinearRAG.py` 672 行，其余 5 个 `src` 文件合计 335 行，顶层 `run.py` 75 行。
- 已确认调用链：`run.py::load_dataset()` 读取 `questions.json/chunks.json` → `LinearRAG.index()` 构建 embedding、NER 结果和图 → `LinearRAG.qa()` 调用 `retrieve()` 并生成答案 → `Evaluator.evaluate()` 评测。
- 发现的问题：`run.py` 硬编码 `CUDA_VISIBLE_DEVICES="4"`，单卡机器正式运行前必须删除或改为参数；本次只记录，不修改官方行为。
- 验证边界：未下载 official medical 数据、Embedding、SciSpacy 或 LLM；未安装依赖、未使用 GPU、未产生检索或 QA 指标。
- 下一步唯一任务：从 `LinearRAG/src/ner.py` 和 `LinearRAG.index()` 开始，画出实体、句子桥接和 passage 图的离线构建链路。
