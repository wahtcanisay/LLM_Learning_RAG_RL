# 维护记录

- 2026-07-30 及之前：由 GPT 维护本学习进度。
- 2026-07-31：由 DeepSeek 接续维护，并完成图节点/边阅读记录与路线调整。
- 2026-08-01 起：GPT 根据 `STUDY_PROGRESS_deepseek.md` 同步最新进度并继续维护；根目录 `STUDY_PROGRESS.md` 恢复为主进度文件。

# 当前阶段

- 阶段 1：MedRAG 基础代码、toy BM25/Dense/RRF 和四套语料核验已完成；当前回到阶段 1，使用 `MedicalGraphRAG/pubmedqa_hard_v1` 补齐正式 BM25 检索硬指标。
- 阶段 2：LinearRAG 默认主干代码第一轮阅读已完成；图检索端到端实验与医学迁移尚未开始。
- 阶段 3：MedicalGPT LoRA/QLoRA SFT。
- 阶段 4：Search-R1 搜索强化学习。
- 阶段 5：MedSearch-R1 医学证据搜索 Agent。
- R2RAG 已于 2026-07-31 从路线剔除，详见“路线变更记录”。

# 当前项目

**Medical GraphRAG + Agent**：MedRAG 提供医学语料与基础检索基线，LinearRAG 提供关系无关图检索，MedicalGPT/Search-R1 提供领域 SFT 与搜索 RL，MedSearch-R1 组合为医学搜索 Agent。

当前先完成并理解 `MedicalGraphRAG/pubmedqa_hard_v1` 的正式 BM25 基线；随后再继续 Dense baseline 和 LinearRAG 官方 medical 小数据验证。

## 当前执行焦点（2026-08-03）

新项目目录为 `MedicalGraphRAG/`，与 `MedRAG/`、`LinearRAG/` 平级。当前先补齐阶段 1 的可比较基线：使用 PubMedQA PQA-L 的 1,000 个 gold documents、4,000 个确定性 MedRAG PubMed distractors 和 document-level qrels，运行 BM25 并产出 Recall@1/5/10、MRR@10、nDCG@10。BM25 完成前，不启动 LinearRAG 医学迁移实验。

## LinearRAG 真实学习边界

- 已亲自阅读：`LinearRAG/readme.md`、`LinearRAG/run.py`、`src/config.py`、`src/embedding_store.py`、`src/ner.py`、`src/evaluate.py`、`src/utils.py` 的相关辅助函数，以及 `src/LinearRAG.py` 默认非向量化主干：初始化、离线 `index()`、正式图构建、在线 `retrieve()`、Seed Entity、实体传播、Passage 先验、PPR、Top-k 与 `qa()`。
- 尚未验证：官方 medical 数据上的端到端最小运行、真实耗时/显存/指标；向量化实体传播和属性关键词增强属于可选支线，暂不计入默认主干完成条件。
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

本周先完成 `pubmedqa_hard_v1` 的 BM25 正式基线，再回到 LinearRAG 官方 medical 小数据端到端运行。当前已经能够区分：

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

2026-08-03：运行 `pubmedqa_hard_v1` 的正式 BM25 检索基线（已完成）：

```text
7,562 chunks
→ Pyserini/Lucene BM25
→ 每题最多 Top-100 chunks
→ 同 doc_id 取最高 BM25 分数
→ document Top-k
→ Recall@1/5/10、MRR@10、nDCG@10
```

主设置只索引 chunk `content`（`abstract_only`）；`title_abstract` 只作为后续泄漏对照，不混入主结果。

Lucene 只返回至少命中一个 query 词项的文档。真实运行中 995/1,000 题返回 100 hits，另 5 题返回 3、29、58、71、72 hits；这些短排名原样进入评测，没有补齐零分文档或丢弃问题。

# 完成标准

- [x] Pyserini collection 与 Lucene index 成功生成，并记录真实命令和索引时间；
- [x] 1,000 个问题全部产生 chunk 排名与折叠后的 document 排名；
- [x] 评测脚本真实输出 Recall@1/5/10、MRR@10、nDCG@10 和 mean/P50/P95 查询延迟；
- [x] 保存配置、manifest、结果 JSON、失败问题和 Git commit；
- 能解释为什么 chunk 排名需要按 `doc_id` 折叠，以及为什么使用最高 BM25 分数；
- 不生成答案、不调用 GPU，不把检索命中率写成 QA Accuracy。

## 今日检查题

1. Pyserini 的 `id/contents` 与我们的 `chunk_id/doc_id/content` 如何映射？
2. 为什么不能直接拿 chunk 排名与 document-level qrels 比较？
3. Top-100 chunks 折叠后如何得到唯一 document 排名？
4. Recall@k、MRR@10、nDCG@10 分别回答什么问题？

# 已完成

## MedicalGraphRAG（2026-08-03）

- 新建独立项目 `MedicalGraphRAG/`；实现 PubMedQA 加载、MedRAG PubMed 确定性干扰采样、文档边界安全切块、questions/documents/chunks/qrels 组装、原子 I/O、manifest、审计和检索硬指标函数。
- 测试：`cd MedicalGraphRAG && python -m pytest -q`，真实结果为 `25 passed`。
- 20 题人工审计门禁：20/20 通过，0 空 context、0 exact duplicate gold/distractor，涉及 40 个确定性 PubMed 分片和 75 个 gold chunks；证据位于 `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/audit_20.json`。
- 完整数据构建命令：`python -m medical_graphrag.cli build --config configs/pubmedqa_hard_v1.json --pubmedqa-dir data/raw/pubmedqa --medrag-pubmed-dir ../MedRAG/corpus/pubmed/chunk --output-dir data/processed/pubmedqa_hard_v1`。
- 完整数据真实结果：1,000 questions、5,000 documents（1,000 gold + 4,000 distractors）、7,562 chunks、1,000 qrels；500 dev + 500 official test；构建耗时 13.18 秒。
- 数据配置：seed `20260803`，tokenizer `sentence-transformers/all-mpnet-base-v2`，max tokens `512`，overlap `64`，主检索文本 `abstract_only`。
- manifest：`MedicalGraphRAG/data/processed/pubmedqa_hard_v1/manifest.json`，SHA-256 为 `cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa`；四个数据产物哈希均已独立复核。
- BM25 环境：WSL2 `LLM-Ubuntu-22.04`，Docker 容器 `llm-pytorch`，Pyserini `0.22.1`；旧 toy Lucene index `/tmp/medrag_task4_index` 仍可读取。
- 正式 BM25 参数：`abstract_only`、Porter stemmer、移除 stopwords、`k1=0.9`、`b=0.4`、请求最多 100 chunks、按 `doc_id` 取最大 chunk score。
- 正式索引：7,562 chunks，耗时 `2.942967939` 秒，空间 `1,254,677` bytes；容器 Python 3.12.3、Java 21.0.11、Pyserini 0.22.1。
- 正式检索：1,000/1,000 query 完成；995 题返回 100 hits，5 题为短排名，最少 3 hits；短排名分布已写入 `search_run.json`。
- dev（500 题）：Recall@1 `0.920`、Recall@5 `0.972`、Recall@10 `0.978`、MRR@10 `0.943250`、nDCG@10 `0.951905`；mean/P50/P95 延迟 `1.565/1.224/2.724 ms`。
- official test（500 题，主结果）：Recall@1 `0.926`、Recall@5 `0.974`、Recall@10 `0.984`、MRR@10 `0.945825`、nDCG@10 `0.955147`；mean/P50/P95 延迟 `1.479/1.275/2.766 ms`。
- 独立复算：直接从 raw hits、metadata 和 qrels 重新聚合，dev/test 五项检索指标与 `metrics.json` 在 `1e-12` 绝对误差内逐项一致。
- 显存峰值：不适用；本实验是 CPU Lucene BM25，没有调用 GPU。QA Accuracy：未评测，不能由检索指标推断。
- 结果位置：`MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/`；raw rankings、collection、index 位于 ignored `outputs/`、`indexes/`。
- 正式结果代码 commit：`ec7f3cf`；数据 manifest SHA-256：`cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa`；raw rankings SHA-256：`0d32b54d4c72761eab6a58a0feb70f0e9cf6e176c8fd0154ebdf7cc40f033386`。
- 当前完整测试：`39 passed`。

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
- 2026-08-03：完成 `calculate_passage_scores() → run_ppr() → retrieve() Top-k → qa() → Evaluator.evaluate()` 默认主干阅读。能够解释 Passage 先验由归一化 Dense 分数和带 tier 衰减的激活 Entity 奖励组成；确认 Node 权重负责问题相关的 PPR 重启位置，Edge 权重负责沿图传播的路线与比例，`damping` 平衡沿边传播和重启。PPR 对全部图节点打分后只保留 Passage，Top-k Passage 先进入生成 LLM 得到 `pred_answer`，随后 Evaluator 才使用 LLM Judge 和标准化包含指标比较 gold answer；二者均不是检索指标。

# 遇到的问题

- LinearRAG README 只说明数据下载位置，没有完整说明 `questions.json/chunks.json` 的代码契约；以后必须区分“README 明示”和“源码确认”。
- LinearRAG official medical 数据的上游医学基准尚未确认，不推测为 MedQA、PubMedQA 或其他数据集。
- 当前没有 gold passage 标注，不能用只有标准答案的数据声称得到 Recall@k、MRR 或 nDCG。
- R2RAG 已从主路线剔除，不再安排 `VanillaAgent` 的 `sid`、跨轮去重和引用映射验证。
- 根 `.gitignore` 的通用 `data/` 规则一度误伤 `MedicalGraphRAG/src/medical_graphrag/data/` 和 compact manifest；已用白名单与 5 项 Git 跟踪回归测试修复，大型 JSONL/TSV 仍保持忽略。
- tokenizer 对超过 512 tokens 的文本发出长度警告；当前切块器会在编码后按 512 tokens 滑窗，不把完整长序列送入模型。该警告不影响本次数据产物，但后续可单独抑制日志噪声。
- metadata 第 5,862 条 title 含 Unicode U+2029；`str.splitlines()` 会错误拆分合法 JSONL。搜索脚本与 evaluator 已改为文件对象逐行读取，并加入回归测试。
- `pyserini` 顶层模块没有 `__version__`；审计脚本已改用 `importlib.metadata.version("pyserini")`，索引报告成功记录 0.22.1。
- 5 个稀有词 query 的 Lucene 命中少于 100。原因是 BM25 不返回零词项匹配文档；当前保留短排名并记录完整分布，不做无依据补齐。

# 下一步

下一步唯一任务：结合保存的 5 个成功和 5 个失败案例，用自己的话解释一个术语明确问题为什么 BM25 排名高、一个宽泛问题为什么 BM25 失败，并说明 BM25 依赖词项匹配而不是向量相似度。完成这一步后开始 Dense baseline 设计，不直接跳到 LinearRAG 医学迁移。

# 待补知识

- spaCy/SciSpaCy 模型在通用实体与医学实体上的实际识别差异（待后续 toy 实验验证）；
- 官方 medical 数据的真实来源、字段契约和最小样本规模；
- 向量化传播与默认 BFS-style 分支的一致性（可选，端到端主干跑通后再验证）；
- Recall@k、MRR、nDCG 与 QA Accuracy 的评测边界。
- chunk-level ranking 到 document-level ranking 的聚合偏差，以及 Top-100 candidate depth 对 Recall@10 的影响。

# 实验结果总表

以下均为 3 条 toy 文档上的接口验证，不是正式医学实验：

| 方法 | Query | Top-1 | Top-1 分数 | 验证内容 |
|---|---|---|---:|---|
| BM25/Lucene | `facial nerve` | `toy_0` | 1.1835999489 | 倒排索引、docid 与 chunk 映射 |
| Dense/FAISS | `facial nerve` | `toy_0` | 0.7347357273 | Embedding、IndexFlatIP 与 metadata |
| RRF | `facial nerve` | `toy_0` | 0.0198019802 | 两路排名按 `1/(k+rank)` 累加 |

正式 BM25 检索指标和延迟已实验；QA Accuracy 尚未实验，GPU 显存对 CPU Lucene BM25 不适用。

### MedicalGraphRAG 正式数据与 BM25 检索

| 数据版本 | Questions | Documents | Chunks | Qrels | 构建耗时 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| `pubmedqa_hard_v1` | 1,000 | 5,000 | 7,562 | 1,000 | 13.18 秒 | manifest 与哈希核验通过 |

| 方法 | Split | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 abstract-only | dev (500) | 0.920 | 0.972 | 0.978 | 0.943250 | 0.951905 | 1.565 ms |
| BM25 abstract-only | official test (500) | 0.926 | 0.974 | 0.984 | 0.945825 | 0.955147 | 1.479 ms |

该封闭基准只含 5,000 documents，且 PubMedQA 问题与 gold article 主题高度一致；不得把高 Recall 外推为全 PubMed 检索性能。主设置没有索引 title，但术语明确问题仍容易依靠 abstract 词项命中。

# 失败案例

| 日期 | 问题 | 根因 | 处理结果 |
|---|---|---|---|
| 2026-07-20 | Pyserini 导入时报 `Unable to find javac` | 环境只有 JRE，缺少 JDK 编译器 | 补齐 JDK 后 toy BM25 建库成功 |
| 2026-07-20 | `task6_toy_rrf.py` 找不到 `src` | 从 `docs/` 直接运行时项目根目录不在 `sys.path` | 根据 `__file__` 加入项目根目录后成功 |
| 2026-07-23 | Wikipedia Git LFS 多次出现 `EOF` | Hugging Face Batch API/并发压力 | 降低批量与并发后完成，`git lfs fsck` 通过 |
| 2026-08-03 | 根 `data/` 忽略规则隐藏源码包与 compact manifest | 通用 Git pattern 同时匹配源码目录和生成数据目录 | 添加 MedicalGraphRAG 精确白名单及回归测试，大型产物继续忽略 |
| 2026-08-03 | metadata JSONL 在第 5,862 条被解析为两段 | title 含 U+2029，`str.splitlines()` 将其误判为记录边界 | 改为文件对象逐行解析；Unicode 回归测试通过 |
| 2026-08-03 | 索引完成后审计脚本报 `pyserini.__version__` 不存在 | 错误假设顶层模块暴露版本属性 | 改用 distribution metadata；正式报告记录 Pyserini 0.22.1 |
| 2026-08-03 | 5/1,000 query 少于 100 Lucene hits | 稀有词 query 可匹配文档不足 100，BM25 不返回零匹配文档 | 保留真实短排名、记录分布、全部问题进入指标，不伪造补齐 |

## BM25 案例观察（official test）

- 成功：`7482275`（“Necrotizing fasciitis … hyperbaric oxygenation”）gold rank 1。`necrotizing fasciitis` 与 `hyperbaric oxygen` 是高区分度术语组合，gold abstract 的词项匹配明显强于普通干扰文档。
- 成功：`7547656`（epinephrine / uterine blood flow / pregnant ewes）gold rank 1。多个具体药物、生理指标和实验对象词项共同缩小候选范围。
- 失败：`11570976`（“Is it Crohn's disease?”）gold 未进 Top-100。query 过短，只提供疾病名；大量 Crohn's disease 文档共享该词，BM25 无法理解问题意图或语义证据差异。
- 失败：`18359123`（“Is it better to be big?”）gold 未进 Top-100。词项几乎都是通用词，缺少可区分的医学实体，BM25 排名被无关但词频匹配的文档占据。
- official test 共有 8 题 gold 未进 Top-10；已保存其中 5 题的 Top-10 文档、分数、gold rank/title/chunk excerpt。

已解决失败只保留根因和最终处理；后续新增失败必须附真实日志或可核验输出。
