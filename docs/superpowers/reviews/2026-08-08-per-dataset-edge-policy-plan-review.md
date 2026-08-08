# Per-Dataset Edge Policy 实施计划审阅与 DS 交接

**审阅日期：** 2026-08-08
**被审阅计划：** `docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md`
**被审阅提交：** `7b4c6cb`
**冻结设计：** `docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md`
**审阅结论：** 不批准按 `7b4c6cb` 直接实施；DS 必须先修订计划中的阻断问题，再据此修改代码。

## 1. 本轮边界

本轮不恢复合并大库实验，不建立跨数据集索引，不构造 merged qrels。每个数据集继续独立建库、独立冻结 manifest、独立评测。

当前只批准处理 PubMedQA Phase 1：

- PubMedQA 使用完整 `documents.jsonl` 作为 document-level retrieval unit；
- 医疗摘要不使用固定相邻边传播；
- Graph-Sim 使用基于同一冻结 document embedding 的 similarity 软边；
- Graph-EP 保留为不添加 Passage–Passage 边的对照；
- BM25-document、Dense-document、Hybrid-document、Graph-EP、Graph-Sim 必须使用相同 document retrieval unit；
- NFCorpus、SciFact 和 MedRAG textbooks/StatPearls 暂不实施，必须在 PubMedQA Phase 1 代码审阅通过后另写计划。

禁止为了提高 test 指标而使用 test qrels 调参。任何真实实验必须保留配置、随机种子、数据 manifest、Git commit、日志及产物哈希。

## 2. 审阅结论摘要

计划的总体方向符合冻结设计，但目前不是可直接执行的 implementation plan，原因包括：

1. 多个计划内测试按原文实现必然失败；
2. document hit schema 与现有 evaluator 契约冲突；
3. graph build、BM25-document、Hybrid-document 和 runner 仍存在占位描述；
4. Dense build report 与 search consumer 所需字段不闭环；
5. Dense、Similarity kNN 和 Graph passage prior 并未真正消费同一个冻结 embedding artifact；
6. Phase 2/3 步骤过早且不可机械执行；
7. MedRAG adapter 的测试、实现和 CLI runner 相互矛盾。

在以下 P0 阻断项全部关闭前，不得启动正式 PubMedQA 实验。

## 3. P0 阻断项

### P0-1：修复完整摘要 embedding 的覆盖证明

原计划的 mock tokenizer 只有 `encode()`，实现却调用 `decode()`，测试会直接失败。`set(flat)` 会丢失 token 位置及重复信息；重叠窗口下用所有窗口长度之和推算截断数量也不能检测遗漏。

必须改为：

- 用原 token 序列的索引区间构造窗口，不用 token 值集合证明覆盖；
- 为每个原始 token 位置维护 coverage count 或布尔 mask；
- 断言所有原始位置至少覆盖一次，首尾都覆盖，且没有越界位置；
- 加入包含重复 token、恰好一个窗口、多窗口、尾窗口不足、超长文本和空文本策略的测试；
- 验证加上 special tokens 后，每个实际送入模型的窗口不超过模型最大长度；
- 不允许通过 decode 后重新 tokenize 来宣称原 token 序列被精确覆盖，除非测试明确证明 tokenizer round-trip 等价；
- `truncated_token_count` 必须由位置覆盖结果计算，不能由重叠窗口长度之和计算。

### P0-2：修复 Similarity 边测试和输入校验

原计划把三元组边 `(source, target, weight)` 当作二元组判断成员，测试必失败。

必须：

- 在测试中显式投影 `pairs = {(a, b) for a, b, _ in edges}`；
- 校验 passage ID 唯一；
- 校验 embedding 是二维数组、行数与 passage 数一致、数值有限；
- 明确并验证输入向量是否必须已经 L2 归一化；
- 覆盖阈值隔断、top-k、并列分数确定性、无自环、无重复无向边、度分布诊断和最终边权范围；本轮不实现 hub penalty；
- 固定默认实验参数，不使用 test qrels 调整阈值或 k。

### P0-3：统一 GraphBuildConfig 构造和运行时校验

原测试没有传 `embedding_model`、`ner_model`，计划实现却把它们声明为无默认值必填字段，导致预期校验前先出现 `TypeError`。

必须：

- 给出唯一、完整、可运行的 dataclass 定义；
- 测试与生产调用使用同一构造契约；
- 为 `retrieval_unit` 和 `passage_edge_mode` 做显式运行时校验，不能只依赖 `Literal`；
- 校验 document 只能使用 `none` 或 `similarity`，chunk 只能使用 `none` 或 `adjacent`；
- 校验 overlap 为非负数且小于实际可用 token window；
- 使用 `dataclasses.asdict()` 等稳定结构生成 config hash；
- document + `none` 必须标记为 Graph-EP，不得错误标记为 similarity profile。

### P0-4：完整展开 graph build 重构

原计划中的“沿用现有实现”“已有哈希字段保持”和省略号不是可执行代码。

DS 必须在修订计划中给出完整函数级改动，至少覆盖：

- retrieval passage 的加载与校验；
- passage IDs、texts、doc IDs、orders 的来源；
- entity extraction、sentence bridge、Entity–Passage 边及节点构建如何消费新 passage；
- Graph-EP、Graph-Sim、Graph-Adjacent 三种互斥模式的分支位置；
- Passage–Passage 边如何加入图；
- passage prior 如何读取冻结 embedding artifact；
- graph report 的完整字段和哈希链；
- 历史 chunk-level `none` 行为如何保持兼容。

计划中的测试不能加载真实 MPNet 或 BC5CDR。单元测试必须使用可注入 mock；真实模型只允许出现在明确标记的 integration/smoke 阶段。

### P0-5：冻结统一 ranking schema

新 document 结果使用 `rank`，现有 `validate_hit_rows()` 无条件读取 `chunk_rank`。必须先解决这一契约冲突。

推荐统一新 schema：

```json
{
  "query_id": "...",
  "split": "test",
  "latency_ms": 1.23,
  "hits": [
    {
      "doc_id": "...",
      "rank": 1,
      "score": 0.9
    }
  ]
}
```

要求：

- document hit 不包含伪造的 `chunk_id`；
- chunk 历史产物继续可读；
- validator 必须按显式 `retrieval_unit` 校验字段；
- 为合法 document、合法 chunk、缺少 rank、重复 rank、混合 schema 和重复 doc ID 添加测试；
- BM25、Dense、Hybrid 和 Graph 的 document raw rankings 必须完全同构。

### P0-6：修复评测哈希链

`graph_build_report_sha256` 必须是 graph build report 文件本身的 SHA-256，并与 `run_context["index_report_sha256"]` 一致，不能使用 `graph_sha256` 替代。

所有测试应先构造合法完整的 build → search → evaluation 哈希链，再验证 retrieval-unit 分支。禁止用一个会提前被审计校验拒绝的 fixture 测试后续逻辑。

### P0-7：完整实现五条公平基线

修订计划不得出现“BM25-document 同理”“runner 对称实现”等占位描述。必须完整覆盖：

1. BM25-document build/search/evaluate；
2. Dense-document build/search/evaluate；
3. Hybrid-document RRF/evaluate；
4. Graph-EP build/search/evaluate；
5. Graph-Sim build/search/evaluate。

每一路都要给出：

- 精确文件路径和函数签名；
- 输入输出 schema；
- report 字段；
- 审计哈希关系；
- 单元测试和 integration test；
- CLI runner 注册；
- 精确运行命令及预期产物。

计划引用的 `configs/pubmedqa_hard_v1_document.json` 必须有明确创建步骤和测试。

### P0-8：三路必须消费同一个冻结 document embedding artifact

不能由 Dense 和 Graph 分别重新计算一份“理论上相同”的向量。必须建立一个唯一 artifact：

```text
documents.jsonl
    ↓
build_document_embeddings
    ├─ document_embeddings.npy
    ├─ document_embedding_metadata.jsonl
    └─ document_embedding_report.json
          ├─ Dense index consumer
          ├─ Similarity edge consumer
          └─ Graph passage-prior consumer
```

要求：

- artifact 记录数据 manifest、documents hash、模型标识、模型文件/版本标识、window 参数、聚合规则、维度、文档数和产物 hash；
- 三个 consumer 只加载，不重新编码 documents；
- 三个 consumer 的报告都记录同一个 embedding report SHA-256 和 embeddings SHA-256；
- 任一数据、模型、参数或产物 hash 不一致时 fail closed；
- 测试必须断言三路报告引用完全相同的 artifact hash。

## 4. P1 必修项

### P1-1：阶段拆分

本轮修订后的可执行计划只保留 Phase 0 和 PubMedQA Phase 1。NFCorpus/SciFact 作为后续独立计划；MedRAG textbooks/StatPearls adapter 和 adjacent 模式再作为另一份独立计划。

原 Phase 0 中“提交已经提交的计划”会导致 nothing to commit，应删除或改为只记录基线 commit 和测试日志。

### P1-2：Adjacent 的连续性语义

长文本模式仅允许同一 `doc_id` 内且 `order` 相差 1 的 chunk 建边，边权固定为 `1.0`。如果 order 存在间隙，不得跨间隙补边，因此实际边数不一定是 `n - 1`。

虽然本轮不实现 MedRAG Phase 3，但后续计划必须用这一语义，防止重新引入 LinearRAG 全局 enumerate 的跨文档错边。

### P1-3：MedRAG adapter 契约

原测试期望混入其他文章的 row 被自动过滤，实现却规定 prefix mismatch 直接失败。后续计划必须二选一并与冻结设计一致：按文件读取时，文件内任何 ID prefix mismatch 都 fail closed，不静默过滤。

同时必须拒绝：

- 空 `content`；
- 无法解析或负数 order；
- 重复 passage ID；
- 同一文档重复 order。

`graph-adjacent` runner 在注册、配置、测试完整之前不得出现在可执行命令中。

### P1-4：工程指标测量

- build time、search latency、evaluation time 分开记录；
- 不得把完整 pipeline wall time 标成 build time；
- 单次 `nvidia-smi` 快照不得标成 peak VRAM；
- 若使用 GPU，必须给出可复现的峰值采样方式及日志；
- CPU-only 路径明确记录 VRAM 不适用；
- index size 使用明确目录及字节统计方式；
- 真实实验命令不得覆盖历史输出目录。

### P1-5：失败案例比较

Phase 1 至少导出 Graph-EP 与 Graph-Sim 的成对比较，而不是各自随便取前三个成功/失败案例。每题应能看到：

- query ID；
- gold doc IDs；
- 两路 gold rank；
- rank delta；
- 新增 similarity 邻居及边权；
- 是否发生改善、退化或无变化；
- 可复核的文本摘要和原因分析字段。

## 5. 修订计划的书写要求

DS 提交的 v2 implementation plan 必须：

- 只覆盖 Phase 0/1；
- 给出精确文件路径、完整代码或完整补丁；
- 不出现 `TODO`、`TBD`、省略号、“同理”“沿用”“对称实现”等占位表达；
- 每个任务遵循可执行的 red → green → focused regression → commit 顺序；
- 每条命令说明运行目录、依赖条件和精确预期结果；
- 单元测试不访问网络、不下载模型、不依赖宿主机已有模型缓存；
- 真实模型 smoke 与 unit tests 分离；
- 保持历史 `GraphConfig`、历史 chunk 结果和现有 runner 的兼容性；
- 每次只改变一个主要变量。

## 6. DS 编码完成后的最低验收门禁

在回交代码审阅前，DS 必须提供以下真实证据；不得编造通过数、指标、耗时或显存：

1. 修订后的 implementation plan 路径和 commit；
2. 代码实现的起止 commit 或 commit 列表；
3. `git status --short`；
4. 完整 pytest 命令、退出码、passed/failed 数和日志路径；
5. document embedding artifact 及 report 路径与 SHA-256；
6. 三个 embedding consumer 引用同一 hash 的证明；
7. 五路 runner 的最小 fixture 命令及输出目录；
8. PubMedQA 正式运行的配置、manifest、原始 rankings、search report、metrics 和日志路径；
9. Graph-EP 与 Graph-Sim 的成对案例文件；
10. 所有新增/修改文件清单；
11. 已知失败、未完成项和与计划的偏差说明。

如果正式实验因模型、CUDA、Pyserini 或磁盘环境没有运行，必须明确写“未运行”及阻塞原因，不能用 unit-test 结果替代正式实验结果。

## 7. 后续代码审阅范围

DS 回交后，由 Codex 进行代码审阅，重点检查：

- 是否逐项关闭本文件 P0/P1；
- 实际代码是否符合冻结设计，而不是只符合修订计划文字；
- 是否真正复用唯一 embedding artifact；
- 是否存在跨文档 adjacent 边或隐式固定边传播；
- ranking schema、report schema 和哈希链是否端到端一致；
- tests 是否能真实捕获错误，而非只覆盖 happy path；
- 五路检索是否在相同 document unit、相同 questions/qrels 和相同 top-k 下公平比较；
- 是否修改或覆盖历史实验产物；
- 报告中的指标、耗时、显存和索引空间是否有真实日志支撑。

代码审阅通过前，不进入 NFCorpus、SciFact 或 MedRAG adjacent 实验。

## 8. 当前状态

- `7b4c6cb`：仅作为第一版计划和审计历史保留；不得直接执行。
- 本文件：DS 的强制修订与实现交接依据。
- 下一责任人：DS，先提交 v2 计划，再按 v2 实现 Phase 1。
- 下一个审阅门禁：DS 回交代码 commit、测试日志和实验产物后，由 Codex 审阅代码。
