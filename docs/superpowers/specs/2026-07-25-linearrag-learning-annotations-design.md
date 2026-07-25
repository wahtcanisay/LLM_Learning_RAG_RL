# LinearRAG 学习注释设计

## 目标

在不改变 LinearRAG 官方算法、命令行接口和运行输出的前提下，为项目中的 7 个 Python 文件补充面向初学者的中文学习注释，并在 `LinearRAG/run.py` 中按真实执行顺序展示顶层调用链。

注释要帮助学习者回答四类问题：

1. 当前变量保存的是什么数据结构；
2. 当前函数在离线构图或在线检索中的职责是什么；
3. 分数、节点和边为什么这样计算；
4. 当前步骤的输出会被哪个后续步骤消费。

默认知识起点是：学习者已经阅读过 MedRAG，理解 BM25、Dense Retrieval、Embedding、Top-k、RRF、Prompt 上下文、基础 QA 生成和基础评测，但尚未系统学习 GraphRAG。

## 已批准方案

采用“核心链路逐段精注、外围文件按职责精注”的方案：

- `LinearRAG/src/LinearRAG.py` 详细解释离线构图、在线检索、实体传播、Passage 打分与 Personalized PageRank；
- 其余 Python 文件解释模块职责、关键数据契约和重要边界，不逐行讲解普通 Python 语法；
- `LinearRAG/run.py` 使用文件级调用链总览和 `main()` 内编号注释展示运行流向；
- 不新增 `--show_flow` 参数，不在程序启动时打印额外内容。

## 运行流向

`LinearRAG/run.py` 中展示以下顶层流程：

```text
parse_arguments
    ↓
load_embedding_model + load_dataset + setup_logging + LLM_Model
    ↓
LinearRAGConfig
    ↓
LinearRAG.__init__
    ↓
LinearRAG.index
    ├─ Passage Embedding 持久化
    ├─ SpaCy NER
    ├─ Entity / Sentence Embedding 持久化
    ├─ Entity–Sentence 映射
    ├─ Entity–Passage 边
    ├─ 相邻 Passage 边
    └─ GraphML 保存
    ↓
LinearRAG.qa
    ├─ LinearRAG.retrieve
    │   ├─ Question NER
    │   ├─ Seed Entity 匹配
    │   ├─ Entity → Sentence → Entity 传播
    │   ├─ Dense Passage 打分与实体奖励
    │   └─ Personalized PageRank
    └─ LLM 生成答案
    ↓
保存 predictions.json
    ↓
Evaluator.evaluate
```

## 文件范围

### `LinearRAG/run.py`

- 文件级说明入口的五个阶段；
- 解释参数如何进入 `LinearRAGConfig`；
- 解释 `load_dataset()` 为 Passage 添加顺序前缀的用途；
- 在 `main()` 中用编号注释标明参数、资源、构图、检索生成、保存评测五个阶段；
- 标出硬编码 `CUDA_VISIBLE_DEVICES="4"` 是官方环境假设，本轮只解释、不修改。

### `LinearRAG/src/config.py`

- 按基础资源、持久化、检索、传播、PPR 和可选属性回退解释字段；
- 解释 `max_iterations`、`iteration_threshold`、`top_k_sentence`、`passage_ratio`、`passage_node_weight`、`damping` 的作用；
- 不修改任何默认值。

### `LinearRAG/src/embedding_store.py`

- 解释三套映射 `hash_id_to_idx`、`hash_id_to_text`、`text_to_hash_id`；
- 解释内容哈希、增量去重、批量编码和 Parquet 持久化；
- 解释归一化 Embedding 使点积可作为余弦相似度；
- 不替换向量库或存储格式。

### `LinearRAG/src/ner.py`

- 解释 Passage、Sentence 与 Entity 三类文本之间的映射；
- 解释为什么过滤 `ORDINAL` 和 `CARDINAL`；
- 解释问题实体统一转小写，而语料实体保留原始文本；
- 标注批处理规模由 Passage 数量和 `max_workers` 推导，但不在本轮修正潜在边界问题；
- 不更换为医学 NER 模型。

### `LinearRAG/src/LinearRAG.py`

- `__init__()`：解释资源初始化、BFS/向量化分支和无向图；
- `index()`：逐阶段解释离线构图及缓存复用；
- `extract_nodes_and_edges()`：解释关系无关图中句子为何作为语义桥接记录，而不直接加入最终 igraph 节点；
- `add_entity_to_passage_edges()` 与 `add_adjacent_passage_edges()`：解释两类正式图边及权重；
- `retrieve()`：解释有种子实体时走图检索，无种子实体时回退 Dense；
- `get_seed_entities()`：解释 Question Entity 到语料 Entity 的语义对齐；
- `calculate_entity_scores()`：解释 BFS 式 Entity → Sentence → Entity 传播、阈值、Top-k 句子和去重；
- `calculate_entity_scores_vectorized()`：解释稀疏矩阵版如何近似复现 BFS 语义；
- `calculate_passage_scores()`：解释 Dense 分数、实体出现奖励、层级衰减和 Passage 节点权重；
- `run_ppr()`：解释个性化重启向量、阻尼系数与最终 Passage 排序；
- `qa()`：解释检索结果如何组成 Prompt 并生成答案；
- 不重构函数、不修正算法、不更改已有分支。

### `LinearRAG/src/evaluate.py`

- 区分 LLM Judge Accuracy 与标准化字符串包含指标；
- 解释并发评测、逐样本回写和聚合结果；
- 明确该代码当前不计算 Recall、MRR 或 nDCG；
- 不调用真实 API，不生成或填写实验指标。

### `LinearRAG/src/utils.py`

- 解释带 namespace 的 MD5 ID；
- 解释 OpenAI 兼容客户端和确定性生成配置；
- 解释答案标准化、日志落盘和 Min-Max 归一化；
- 不改变 API、超时或日志级别。

## 注释风格

- 使用中文解释“为什么”和“数据如何流动”，保留必要的英文类名、函数名和论文术语；
- 函数使用 docstring 说明职责、输入、输出和流程位置；
- 复杂代码块前使用阶段注释，关键公式旁使用局部注释；
- 不给显而易见的赋值和循环机械加注释；
- 对潜在问题使用“学习注意”标记，不在注释任务中顺便修复。
- 新术语第一次出现时解释四件事：术语定义、对应代码数据结构、为什么需要、它与 MedRAG 基础检索的区别；
- 后续再次出现同一术语时只引用名称，不重复整段定义，避免注释淹没算法主线。

## 相对 MedRAG 新增的术语

以下术语需要随源码在首次出现处解释：

- **GraphRAG**：先把语料组织为图，再利用图结构传播或聚合证据；区别于 MedRAG 直接对独立 Chunk 做稀疏或稠密排序；
- **Relation-free graph（关系无关图）**：不让 LLM 抽取带类型的知识关系，而使用实体共现、句子桥接和 Passage 邻接建立连接；
- **Entity / Sentence / Passage**：实体是概念节点，Sentence 是在线传播时的语义桥，Passage 是最终返回给生成模型的证据单元；
- **NER（Named Entity Recognition）**：从 Passage 或 Question 中识别人名、地点、疾病等实体提及，是图构建和查询入图的入口；
- **Semantic bridging（语义桥接）**：通过与问题语义相似的句子，从当前实体激活同句中的其他实体，从而完成多跳扩展；
- **Seed entity（种子实体）**：从问题实体出发，在语料实体库中找到的初始激活节点；
- **Entity activation / propagation（实体激活与传播）**：沿 Entity → Sentence → Entity 扩展候选实体并传递分数；
- **Tier / hop（层级或跳数）**：实体距离种子实体的传播轮次，用于限制搜索和衰减远距离证据；
- **BFS-style iteration（广度优先式迭代）**：逐轮扩展当前激活实体；代码并非调用标准 BFS API，但控制结构具有分层扩展特征；
- **Sparse adjacency matrix（稀疏邻接矩阵）**：只保存实际存在的 Entity–Sentence 连接，用矩阵乘法并行实现传播；
- **COO sparse tensor**：用“坐标索引 + 非零值”表示稀疏矩阵，是 PyTorch 构造稀疏张量的格式；
- **Vectorized retrieval（向量化检索）**：这里特指把图传播改写成稀疏矩阵运算，不等同于 MedRAG 中“Dense 向量检索”；
- **Personalized PageRank（个性化 PageRank，PPR）**：从与当前问题相关的重启分布出发，在图上扩散权重并排序 Passage；
- **Reset vector / personalization vector（重启向量）**：PPR 每次随机重启时回到各节点的概率分布，由当前问题的 Entity 和 Passage 分数组成；
- **Damping factor（阻尼系数）**：PPR 继续沿边传播的概率，控制局部相关性与全图扩散的平衡；
- **Passage prior / node weight（Passage 先验或节点权重）**：把 Dense Passage 相似度及实体奖励注入 PPR 重启分布；
- **Dense fallback（稠密回退）**：问题没有识别到实体时跳过图传播，直接使用 Dense Passage Retrieval；
- **GraphML**：保存图节点、边和属性的通用文件格式；
- **Content hash / namespace（内容哈希与命名空间）**：用文本内容生成稳定 ID，并用 `passage-`、`entity-`、`sentence-` 防止不同对象发生 ID 混淆；
- **Cache reuse / incremental indexing（缓存复用与增量建索引）**：只对尚未出现的文本执行 NER 或 Embedding，复用已有 Parquet 和 JSON 结果；
- **LLM judge**：调用模型判断答案是否与标准答案一致；它与可程序化的字符串包含指标不同，也不是检索 Recall 或 MRR。

需要特别澄清：论文描述中的句子语义桥接层，在当前代码里主要保存为 Entity–Sentence 映射并参与在线传播；`add_nodes()` 实际只把 Entity 和 Passage 加入最终 igraph。注释不得把 Sentence 错写成最终图中的第三类正式顶点。

## 行为边界

本轮允许的变更只有：

- 注释；
- docstring；
- `STUDY_PROGRESS.md` 中的真实学习记录；
- 实施计划文档。

本轮禁止：

- 修改表达式、控制流、默认参数或导入；
- 下载数据、Embedding、SpaCy 模型或 LLM；
- 调用 OpenAI 兼容 API；
- 运行 GPU 实验；
- 编造检索、生成、延迟或显存指标。

## 验证

1. 对 7 个 Python 文件执行 `python -m py_compile`，确认语法有效；
2. 将工作区文件与 `HEAD` 版本解析为 Python AST，移除 docstring 后比较结构，确认可执行语义未变化；
3. 执行 `git diff --check`，确认没有空白错误；
4. 检查 `git diff`，确保除设计、计划、进度文档外，Python 文件只包含注释和 docstring；
5. 不把语法检查描述成端到端运行或实验复现。

## 今日学习产出

实施完成后，今天只安排一个核心任务：沿 `run.py → LinearRAG.index()` 阅读离线构图链路，能够画出并解释 Passage、Entity、Sentence 三类数据及两类正式图边。

完成标准：

- 能从 `run.py` 复述 `load_dataset → LinearRAG.__init__ → index`；
- 能说明 Sentence 在传播中是语义桥，但没有作为最终 igraph 节点加入；
- 能说明 Entity–Passage 边和相邻 Passage 边的来源与权重；
- 能指出 Embedding、NER JSON、GraphML 三类持久化结果的位置；
- 回答必须引用具体函数或变量，不以“看懂了”作为完成依据。
