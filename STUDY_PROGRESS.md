# 当前阶段

阶段 1：MedRAG 基础检索复现（Task 1～Task 6 已完成，Task 7 待开始）

# 当前项目

MedRAG 医学 RAG 基线。当前目标是先理解代码调用链，再运行最小样例；不提前下载大规模语料或模型。

# 已确认的正式路线

- 日期：2026-07-18
- 决策：阶段 6 使用 **MedSearch-R1：基于领域微调与成本感知强化学习的医学证据搜索 Agent**，替换原阶段 6 计划。
- 衔接关系：MedRAG/R2RAG/LinearRAG 提供医学检索与动态控制基础，MedicalGPT 提供医学领域 SFT，Search-R1 提供多轮工具调用、rollout、reward 和 GRPO，LA-CDM 仅提供假设驱动、置信度校准和成本感知思想。
- 数据边界：不依赖 MIMIC-CDM，不模拟或编造患者临床检查；第一版使用公开、可验证的医学选择题或医学问答以及独立医学检索语料。
- 实施顺序：当前仍处于阶段 1 MedRAG，必须完成前置阶段及真实基线后再进入组合项目，不改变今天的 Retriever 代码阅读任务。
- 设计文档：`docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`

# 本周目标

完成 MedRAG 官方代码的第一轮只读调研，明确入口、检索器、语料、索引、上下文构造、生成和评测如何衔接。

# 今日唯一任务

阅读并理解：

- `MedRAG/src/template.py`；
- `MedRAG/src/medrag.py` 中的 `MedRAG.__init__`；
- `MedRAG/src/medrag.py` 中的 `medrag_answer`。

今晚已完成代码阅读，检查题和调用链复述留到明天，今天不运行语料下载或大规模实验。

# 完成标准

- 能解释 `rag=False` 与 `rag=True` 的区别；
- 能指出 `MedRAG.__init__` 中可能触发检索器、语料或索引初始化的位置；
- 能解释 `medrag_answer` 中 `snippets`、`snippets_ids` 和实际检索三种证据来源；
- 能解释检索结果如何变成 `context`，以及为什么要做上下文截断；
- 能解释 `general_cot` 与 `general_medrag` 模板及其占位符；
- 能用自己的话复述目前已经读到的调用链，但不要求今晚完成。

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

# 遇到的问题

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

明天要回答：

1. 四个独立语料源的目录分别是什么？`MedCorp` 为什么不需要单独下载？
2. 下载后的真实 JSONL 是否仍遵循 `id/title/content/contents` 契约？
3. 为什么 PubMed 先只拉取 `chunk/**`，而不是原始 `baseline/**`？
4. 为什么下载语料时不能直接实例化 `Retriever`？
5. 每个语料源的 JSONL 文件数、chunk 数量和磁盘占用如何记录？

明天暂不下载完整语料、不运行完整 QA 生成；只使用已经生成的 toy 结果进行分析。

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
