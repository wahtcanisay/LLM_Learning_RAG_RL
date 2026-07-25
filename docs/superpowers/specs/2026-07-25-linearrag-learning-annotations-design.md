# LinearRAG 学习注释设计

## 目标

在不改变 LinearRAG 官方算法、命令行接口和运行输出的前提下，为项目中的 7 个 Python 文件补充面向初学者的中文学习注释，并在 `LinearRAG/run.py` 中按真实执行顺序展示顶层调用链。

注释要帮助学习者回答四类问题：

1. 当前变量保存的是什么数据结构；
2. 当前函数在离线构图或在线检索中的职责是什么；
3. 分数、节点和边为什么这样计算；
4. 当前步骤的输出会被哪个后续步骤消费。

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
