# 当前阶段

阶段 1：MedRAG 基础检索复现（Day 1 进行中）

# 当前项目

MedRAG 医学 RAG 基线。当前目标是先理解代码调用链，再运行最小样例；不提前下载大规模语料或模型。

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

# 遇到的问题

- 问题：完整调用链的 `Retriever` 内部和 `RetrievalSystem.merge` 还没有阅读；`rrf_k` 初次不清楚。
- 原因：今天先完成 `template.py`、`MedRAG.__init__` 和 `medrag_answer`，检索实现留到下一步。
- 解决办法：已确认 `rrf_k` 是 RRF 融合常数；下一步阅读 `utils.py`，补齐 Retriever → retrieve → merge 的链路。

# 下一步

## 2026-07-18：Day 1 收尾与 Retriever 代码

明天唯一核心任务：补充第 5 题的完整调用链，再阅读 `utils.py` 中的 `Retriever`、`RetrievalSystem.retrieve` 和 `RetrievalSystem.merge`，补齐“检索器 → RRF → 返回 snippets”的调用链。

明天要回答：

1. `template.py` 为什么只负责 Prompt 结构，不负责检索和生成？`{{context}}`、`{{question}}`、`{{options}}` 分别从哪里来？
2. `MedRAG.__init__` 中 `rag=False` 和 `rag=True` 分别初始化了什么？为什么 `rag=True` 可能触发下载或建索引？
3. `medrag_answer` 中直接传入 `snippets`、传入 `snippets_ids`、调用 `retrieval_system.retrieve()` 有什么区别？
4. 检索结果怎样被拼成 `context`，为什么要按 tokenizer 和 `context_length` 截断？
5. 从 `MedRAG(...)` 到最终 `answer`，目前已经读到的完整调用链是什么？

明天暂不下载语料、不建立 BM25/FAISS 索引、不运行大模型生成。

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
