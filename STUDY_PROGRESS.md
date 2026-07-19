# 当前阶段

阶段 1：MedRAG 基础检索复现（Task 1、Task 2 已完成，Task 3 待开始）

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

# 遇到的问题

- 问题：初次回答时把 `rrf_k` 说成检索器参数，并把 Retriever 初始化概括成“先检查索引”。
- 原因：尚未区分候选排名融合和原始检索分数，也没有按 `Retriever.__init__` 的实际顺序追踪副作用。
- 解决办法：已纠正：`rrf_k` 是 RRF 排名平滑常数；初始化顺序是先检查 chunk/下载语料，再检查并构建具体索引。

# 下一步

## 2026-07-20：Task 3——学习索引构建与检索后端

明天唯一核心任务：阅读并追踪 BM25/Lucene 与 Dense/FAISS 索引的构建、加载和 query 检索路径，不下载完整医学语料。

已在 `src/utils.py` 的 `embed()`、`construct_index()`、`Retriever.__init__()` 和 Dense query 路径补充学习注释；注释只说明第三方库的调用契约、索引持久化和 metadata 映射，不改变运行逻辑。

明天推荐阅读顺序：

1. 先看 `src/utils.py` 中 BM25 索引存在性检查和 Pyserini/Lucene 初始化。
2. 再看 Dense 分支中的 query encoder、FAISS index 加载和向量检索返回值。
3. 回到 `get_relevant_documents()`，串起 query、index、indices、scores 和 chunk 映射。
4. 最后对比单路检索与 `RetrievalSystem.merge()` 的 RRF 输入输出。

明天完成标准：能画出 BM25/Dense 两条索引路径；能解释索引位置如何映射回 JSONL chunk；能说明为什么不把 BM25 原始分数和 Dense 原始分数直接相加。

明天要回答：

1. BM25 建库时保存了什么信息？Dense 建库时保存了什么信息？
2. Dense query 为什么必须经过和文档一致的 Embedding 编码器？
3. FAISS 返回的 `indices` 和 `scores/distances` 分别代表什么？
4. 为什么检索索引位置还需要通过 `source/index` 找回 JSONL 文本？
5. 为什么 BM25 和 Dense 的原始 score 不适合直接相加？

明天暂不下载完整语料、不运行完整 QA 生成；只做代码追踪，必要时使用 toy 数据说明索引映射。

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
