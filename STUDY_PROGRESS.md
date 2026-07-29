# 当前阶段

- 阶段 1：MedRAG 基础代码、toy BM25/Dense/RRF 和四套语料核验已完成；正式全量索引、生成与评测暂缓。
- 阶段 2：R2RAG 核心动态路由思想已学习；外围 API、服务和完整复现停止推进。
- 阶段 3：LinearRAG 图结构检索与医学迁移（当前阶段）。

# 当前项目

**Medical Routing GraphRAG**：MedRAG 提供医学语料与基础检索基线，LinearRAG 提供关系无关图检索，R2RAG 后续只作为动态路由与停止控制层。

当前只学习和最小复现 LinearRAG。先使用官方 medical 小数据验证，再考虑迁移 MedRAG 的 Textbooks/StatPearls 子集。

## LinearRAG 真实学习边界

- 已亲自阅读：`LinearRAG/readme.md`、`LinearRAG/run.py`、`src/config.py`、`src/embedding_store.py`、`src/utils.py::compute_mdhash_id()`、`src/LinearRAG.py` 的 `__init__()` 与 `load_embedding_store()`。
- 尚未阅读：`src/ner.py`、`src/LinearRAG.py` 的离线建图与在线检索部分、`src/evaluate.py`、`src/utils.py` 的其余部分。
- 助手添加注释、执行语法检查或检查调用链，只算材料准备，不算学习者完成。
- LinearRAG 官方源码版本：`bcc94e66c221f798801255efba09311d6fbcd8d6`。

## 已确认的路线与边界

- 顺序保持为：MedRAG → R2RAG → LinearRAG → MedicalGPT → Search-R1 → MedSearch-R1。
- 阶段 6 使用 `MedSearch-R1`；设计文档位于 `docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`。
- 不依赖 MIMIC-CDM，不模拟真实临床检查，不声称提供临床诊断。
- 暂缓：MedRAG 原生全量索引/生成/评测、R2RAG 外围工程、PageIndex 候选项目。
- 不同时修改路由、图构建、Embedding、重排和生成模型；每次只验证一个主要变量。

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

2026-07-29：阅读 Passage NER 与 Question NER 的数据契约：

```text
SpacyNER.__init__()
→ batch_ner()
→ extract_entities_sentences()
→ question_ner()
```

只查看 `LinearRAG.py` 中 `SpacyNER(...)`、`batch_ner(...)` 和 `question_ner(...)` 的调用位置，不展开阅读 `index()`、`get_seed_entities()` 或图构建逻辑。预计 45～75 分钟。

# 完成标准

- 能说明 `SpacyNER.__init__()` 如何从模型名或路径加载 spaCy/SciSpaCy pipeline；
- 能写出 `batch_ner()` 的输入与两个返回映射；
- 能解释 `extract_entities_sentences()` 如何同时生成 Passage→Entity 和 Sentence→Entity；
- 能区分 Passage NER 与 Question NER 的大小写、去重和返回类型；
- 能指出 `batch_size` 计算和字典顺序映射中的两个待验证边界；
- 回答检查题并引用具体函数或变量；
- 不下载 SciSpaCy 模型、不运行 NER、不使用 GPU、不产生实验指标。

## 今日检查题

1. `SpacyNER.__init__()` 接收的 `spacy_model` 是已经加载的模型对象，还是模型名/路径？
2. `batch_ner()` 为什么接收 `hash_id_to_passage`，而不是只接收 Passage 文本列表？
3. `passage_list` 与 `passage_hash_id` 是如何按位置重新配对的？这依赖什么顺序假设？
4. 当 Passage 数量小于 `max_workers` 时，`batch_size` 会变成什么？为什么需要后续 toy 验证？
5. 当前 `self.spacy_model.pipe(...)` 是否真的启用了多进程并行？请从调用参数判断。
6. `extract_entities_sentences()` 为什么同时维护 `unique_entities` 和 `sentence_to_entities`？
7. `ORDINAL` 与 `CARDINAL` 为什么被过滤？过滤发生在 Passage NER、Question NER，还是两者都有？
8. Passage NER 在哪些层级去重？实体文本是否统一转为小写？
9. `question_ner()` 与 Passage NER 在输入、输出类型、大小写处理和用途上有什么不同？
10. 如果问题没有识别到任何实体，`question_ner()` 返回什么？今天只说明返回值，不展开后续检索分支。

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

## R2RAG（2026-07-22 至 2026-07-24）

- 官方仓库：`https://github.com/rmit-ir/NeurIPS-MMU-RAG`；源码保存在 `R2RAG/`，未保留嵌套 `.git`。
- 已学习 simple/complex 路由、单轮 `VanillaRAG`、多轮 `VanillaAgent`、查询改写、证据审查和停止控制的核心思想。
- 未运行 vLLM、外部搜索或正式实验；不保留 API、服务启动和流式展示为当前任务。

## LinearRAG（2026-07-25 至今）

- 官方仓库：`https://github.com/DEEP-PolyU/LinearRAG`。
- 已下载源码并确认顶层调用链：`load_dataset → index → qa/retrieve → evaluate`。
- 已发现 `run.py` 硬编码 `CUDA_VISIBLE_DEVICES="4"`，正式运行前必须改为单卡可配置方式。
- 已完成 README、`run.py`、配置对象、初始化链和 `EmbeddingStore` 阅读。
- 2026-07-28：完成 10 道初始化与缓存检查题及补答；纠正了“Parquet 会保存映射字典”和“对象类型等于 Python 数据类型”的混淆。

# 遇到的问题

- LinearRAG README 只说明数据下载位置，没有完整说明 `questions.json/chunks.json` 的代码契约；以后必须区分“README 明示”和“源码确认”。
- LinearRAG official medical 数据的上游医学基准尚未确认，不推测为 MedQA、PubMedQA 或其他数据集。
- 当前没有 gold passage 标注，不能用只有标准答案的数据声称得到 Recall@k、MRR 或 nDCG。
- 若未来重新接入 R2RAG，需要用 toy 多轮测试验证 `VanillaAgent` 的 `sid`、跨轮去重和引用映射。

# 下一步

完成今天的复述和检查题后，下一次进入离线索引总流程：`index() → load_existing_data() → merge_ner_results() → save_ner_results() → extract_nodes_and_edges()`。在 LinearRAG 单轮图检索产生真实指标前，不恢复 R2RAG 控制层，也不开始 PageIndex。

# 待补知识

- spaCy/SciSpaCy 的 Passage NER 与 Question NER；
- Sentence 语义桥和 Entity→Sentence→Entity 传播；
- Entity–Passage 与相邻 Passage 边；
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
