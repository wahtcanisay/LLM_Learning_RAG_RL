# 维护记录

- 2026-07-30 及之前：由 GPT 维护本学习进度。
- 2026-07-31：由 DeepSeek 接续维护，并完成图节点/边阅读记录与路线调整。
- 2026-08-01 起：GPT 根据 `STUDY_PROGRESS_deepseek.md` 同步最新进度并继续维护；根目录 `STUDY_PROGRESS.md` 恢复为主进度文件。

# 当前阶段

- 阶段 1：MedRAG 基础代码、toy BM25/Dense/RRF 和四套语料核验已完成；正式全量索引、生成与评测暂缓。
- 阶段 2：LinearRAG 图结构检索与医学迁移（当前阶段）。
- 阶段 3：MedicalGPT LoRA/QLoRA SFT。
- 阶段 4：Search-R1 搜索强化学习。
- 阶段 5：MedSearch-R1 医学证据搜索 Agent。
- R2RAG 已于 2026-07-31 从路线剔除，详见“路线变更记录”。

# 当前项目

**Medical GraphRAG + Agent**：MedRAG 提供医学语料与基础检索基线，LinearRAG 提供关系无关图检索，MedicalGPT/Search-R1 提供领域 SFT 与搜索 RL，MedSearch-R1 组合为医学搜索 Agent。

当前只学习和最小复现 LinearRAG。先使用官方 medical 小数据验证，再考虑迁移 MedRAG 的 Textbooks/StatPearls 子集。

## LinearRAG 真实学习边界

- 已亲自阅读：`LinearRAG/readme.md`、`LinearRAG/run.py`、`src/config.py`、`src/embedding_store.py`、`src/ner.py`、`src/utils.py::compute_mdhash_id()`，以及 `src/LinearRAG.py` 中从初始化到离线 `index()` 的缓存、NER、Entity↔Sentence 映射和正式图构建（`add_entity_to_passage_edges() → add_adjacent_passage_edges() → augment_graph() → add_nodes() → add_edges()`）流程。
- 尚未阅读：`src/LinearRAG.py` 的在线检索部分（`retrieve()`、实体传播、PPR）、`src/evaluate.py`、`src/utils.py` 的其余部分。
- 2026-07-31 正式图节点/边任务已完成（学习者决定不再复述）。记录：Q1/Q4 确认为笔误（Entity 误写 Sentence）；Q2 正确表述为“边权是归一化比例（0～1），不是原始次数”；Q3 术语为“正则表达式”。
- 助手添加注释、执行语法检查或检查调用链，只算材料准备，不算学习者完成。
- LinearRAG 官方源码版本：`bcc94e66c221f798801255efba09311d6fbcd8d6`。

## 已确认的路线与边界

- 顺序保持为：MedRAG → LinearRAG → MedicalGPT → Search-R1 → MedSearch-R1。
- 最终阶段使用 `MedSearch-R1`；设计文档位于 `docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`。
- 不依赖 MIMIC-CDM，不模拟真实临床检查，不声称提供临床诊断。
- 暂缓：MedRAG 原生全量索引/生成/评测、PageIndex 候选项目。
- 不同时修改图构建、Embedding、重排和生成模型；每次只验证一个主要变量。

# 本周目标

完成 LinearRAG 核心代码第一轮阅读，能够区分：

1. 对象初始化与 Embedding 缓存；
2. NER 和离线建图；
3. 在线实体传播、Passage 打分和 PPR；
4. 生成指标与检索指标。

## LinearRAG 阅读顺序

1. 对象初始化与 Embedding 缓存

   `config.py → LinearRAG.__init__() → load_embedding_store() → embedding_store.py → compute_mdhash_id()`。
2. NER 数据契约

   `SpacyNER.__init__() → batch_ner() → extract_entities_sentences() → question_ner()`。
3. 离线索引总流程

   `index() → load_existing_data() → merge_ner_results() → save_ner_results() → extract_nodes_and_edges()`。
4. 正式图节点和边

   `add_entity_to_passage_edges() → add_adjacent_passage_edges() → augment_graph() → add_nodes() → add_edges()`。
5. 在线入口与 Seed Entity

   `retrieve() → get_seed_entities()`，第一遍只读默认非向量化分支。
6. 实体传播、Passage 先验和 PPR

   `calculate_entity_scores() → dense_passage_retrieval() → calculate_passage_scores() → run_ppr()`。
7. 生成与评测

   `qa() → evaluate.py → normalize_answer()`。

前一项未完成复述和检查题，不进入下一项。

# 今日唯一任务

2026-08-02：Seed Entity 的 BFS-style 实体传播（已完成）：

```text
graph_search_with_seed_entities()
→ calculate_entity_scores()
```

只读默认非向量化分支；向量化实现、Passage 权重和 PPR 留到后续任务。

# 完成标准

- 能解释 `actived_entities`、`current_entities`、`new_entities` 和 `entity_weights` 的职责；
- 能复述 `Seed Entity → Sentence Top-k → 下一层 Entity` 的传播流程；
- 能解释 Sentence 去重、两次阈值剪枝及多轮终止条件；
- 能区分 `entity_weights` 的分数累加与 `new_entities` 的字典覆盖；
- 不运行检索、不使用 GPU、不产生实验指标。

## 今日检查题

1. Seed Entity 初始化时进入哪些状态结构？
2. 当前 Entity 如何找到、筛选并去重桥接 Sentence？
3. 新 Entity 的传播分数如何计算，两次阈值剪枝在哪里？
4. 多轮传播何时停止，同一 Entity 被多条路径找到时如何处理？

# 已完成

## MedRAG（2026-07-16 至 2026-07-23）

- 仓库来源：`https://github.com/gzxiong/MedRAG.git`；记录版本：`7599a72`。
- 已阅读：生成入口、模板、Retriever、BM25/Dense 分支、RRF 融合、教材与 PubMed 数据处理。
- 已解释：`id/title/content/contents` 契约、BM25 与 Dense 的区别、FAISS metadata 映射、RRF 去重与排名累加。
- 已完成 3 条 toy 文档上的 BM25、Dense 和 RRF 闭环；这些结果只验证代码与接口，不代表正式医学基线。
- 已核验 Textbooks、StatPearls、PubMed、Wikipedia 的 chunk；无未展开 LFS pointer。
- 已清理可恢复的 LFS 缓存和 StatPearls 原始中间文件，释放约 112.09 GiB。

### MedRAG 语料核验

| 语料 | JSONL 文件数 | 工作区大小 | 状态 |
|---|---:|---:|---|
| Textbooks | 18 | 201.76 MiB | 字段与 LFS 核验通过 |
| StatPearls | 9646 | 441.54 MiB | 字段核验通过 |
| PubMed | 1166 | 65.20 GiB | LFS 核验通过；官方含一个零字节分片 |
| Wikipedia | 646 | 42.54 GiB | 字段与 LFS 核验通过 |

## 路线变更记录

**2026-07-31：R2RAG 从学习路线剔除。**

- 官方仓库：`https://github.com/rmit-ir/NeurIPS-MMU-RAG`；源码保留在 `R2RAG/`（2026-07-22 至 2026-07-24 曾学习其核心思想，未运行 vLLM、外部搜索或正式实验）。
- 剔除判断：Router 与 `VanillaAgent` 的停止/改写决策主要依赖 prompt 输出和字符串匹配，无阈值、概率校准或可学习控制；后续 Search-R1/MedSearch-R1 将用 GRPO 学习搜索与停止策略，继续复现会重复且偏弱。
- 保留给 MedSearch-R1 的参考点：查询变体与融合召回、压缩轮次历史避免上下文膨胀、纯 prompt 停止控制的脆弱性。

## LinearRAG（2026-07-25 至今）

- 官方仓库：`https://github.com/DEEP-PolyU/LinearRAG`。
- 已下载源码并确认顶层调用链：`load_dataset → index → qa/retrieve → evaluate`。
- 已发现 `run.py` 硬编码 `CUDA_VISIBLE_DEVICES="4"`，正式运行前必须改为单卡可配置方式。
- 已完成 README、`run.py`、配置对象、初始化链和 `EmbeddingStore` 阅读。
- 2026-07-28：完成 10 道初始化与缓存检查题及补答；纠正了“Parquet 会保存映射字典”和“对象类型等于 Python 数据类型”的混淆。
- 2026-07-29：完成 Passage NER 与 Question NER 阅读。
- 2026-07-30：阅读 `index()` 的缓存复用、NER 合并、五个节点/映射返回值和文本→hash 映射主流程；缓存覆盖、set 顺序和无显式返回等边界题按学习者决定跳过，不计为已掌握。
- 2026-07-29：完成 `src/ner.py` 第一轮阅读，能够解释 `SpacyNER.__init__()`、`batch_ner()`、`extract_entities_sentences()` 和 `question_ner()` 的输入、输出、去重、大小写与离线/在线用途。
- 2026-07-29：确认 `doc.ents` 的元素是实体 `Span`，不是字典；Passage→Entity 和 Sentence→Entity 都按实体文本去重，同一实体重复出现只保留一个映射项，但 Entity–Passage 边权后续会重新统计原文出现次数。
- 2026-07-29：确认 Passage NER 保留原始大小写，Question NER 转为小写；问题无实体时 `question_ner()` 直接返回空集合 `set()`。
- 2026-07-29：确认 `max_workers` 当前没有通过 `n_process` 启用 spaCy 多进程，只参与 `batch_size` 计算；当 Passage 数少于 `max_workers` 时存在 `batch_size == 0` 的待验证风险。
- 2026-07-29：只读核验 MedRAG 的 Textbooks 和 StatPearls：两者均已切成带 `id/title/content/contents` 的 JSONL chunk；迁移 LinearRAG 需要转换为 `chunks.json`，并保留文档元数据以避免跨文档错误相邻边。
- 2026-07-31：正式图节点和两类边阅读完成并标记通过（学习者决定不再复述，与 2026-07-30 缓存边界题同例）。已确认 Entity–Passage 边是归一化比例，相邻 Passage 通过数字前缀和正则表达式建立固定权重边。
- 2026-08-01：完成 `retrieve() → get_seed_entities()` 默认非向量化分支阅读。能够解释有 Seed Entity 时进入图检索、无 Seed Entity 时回退 Dense Passage；确认 Seed 匹配位于 Question Entity–Corpus Entity 空间，Dense 回退位于 Question–Passage 空间；每个问题实体各执行一次 `argmax`，不同问题实体可以映射到同一个语料 Entity，当前实现没有最低相似度阈值。
- 2026-08-02：完成 `graph_search_with_seed_entities() → calculate_entity_scores()` 默认 BFS-style 分支阅读。能够解释逐层状态、Entity→Sentence→Entity 传播、Sentence Top-k 与全局去重、传播分数乘法、两次阈值剪枝和两个终止条件；确认同一 Entity 的图节点权重按路径累加，而 `new_entities` 只保留最后一次字典赋值。数值检查中明确：若两条路径产生的传播后分数为 0.6 和 0.7，则累计权重增加 1.3。

# 遇到的问题

- LinearRAG README 只说明数据下载位置，没有完整说明 `questions.json/chunks.json` 的代码契约；以后必须区分“README 明示”和“源码确认”。
- LinearRAG official medical 数据的上游医学基准尚未确认，不推测为 MedQA、PubMedQA 或其他数据集。
- 当前没有 gold passage 标注，不能用只有标准答案的数据声称得到 Recall@k、MRR 或 nDCG。
- R2RAG 已从主路线剔除，不再安排 `VanillaAgent` 的 `sid`、跨轮去重和引用映射验证。

# 下一步

下一次只阅读 `calculate_passage_scores()`：追踪 Dense Passage 排名归一化、激活实体出现奖励、tier 衰减和 Passage 节点权重写入。暂不阅读属性关键词增强和 `run_ppr()`；在 LinearRAG 单轮图检索产生真实指标前，不开始 PageIndex。

# 待补知识

- spaCy/SciSpaCy 模型在通用实体与医学实体上的实际识别差异（待后续 toy 实验验证）；
- Dense Passage 先验、激活实体奖励与 tier 衰减；
- Dense Passage Prior、Personalized PageRank 和 Dense fallback；
- Recall@k、MRR、nDCG 与 QA Accuracy 的评测边界。

# 实验结果总表

以下均为 3 条 toy 文档上的接口验证，不是正式医学实验：

| 方法 | Query | Top-1 | Top-1 分数 | 验证内容 |
|---|---|---|---:|---|
| BM25/Lucene | `facial nerve` | `toy_0` | 1.1835999489 | 倒排索引、docid 与 chunk 映射 |
| Dense/FAISS | `facial nerve` | `toy_0` | 0.7347357273 | Embedding、IndexFlatIP 与 metadata |
| RRF | `facial nerve` | `toy_0` | 0.0198019802 | 两路排名按 `1/(k+rank)` 累加 |

正式 Recall@5、Recall@10、MRR、QA Accuracy、延迟和显存峰值均尚未实验。

# 失败案例

| 日期 | 问题 | 根因 | 处理结果 |
|---|---|---|---|
| 2026-07-20 | Pyserini 导入时报 `Unable to find javac` | 环境只有 JRE，缺少 JDK 编译器 | 补齐 JDK 后 toy BM25 建库成功 |
| 2026-07-20 | `task6_toy_rrf.py` 找不到 `src` | 从 `docs/` 直接运行时项目根目录不在 `sys.path` | 根据 `__file__` 加入项目根目录后成功 |
| 2026-07-23 | Wikipedia Git LFS 多次出现 `EOF` | Hugging Face Batch API/并发压力 | 降低批量与并发后完成，`git lfs fsck` 通过 |

已解决失败只保留根因和最终处理；后续新增失败必须附真实日志或可核验输出。
