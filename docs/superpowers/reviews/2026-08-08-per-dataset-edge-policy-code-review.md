# Per-Dataset Edge Policy 代码审阅结果

**审阅日期：** 2026-08-08
**DS 实现基线：** `ff95021` 及其前序实现提交
**Codex 修复提交：** `7377678`
**冻结设计：** `docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md`

## 结论

代码主方向正确，但 DS 生成的 v1 正式实验不能作为最终结论：完整摘要虽然先按 token ID 切窗，实际编码前又经过 `decode -> tokenize`，窗口边界会漂移。对 PubMedQA 5,000 篇文档的审计中，6,037 个窗口有 205 个发生 token 数不一致。

该问题及其直接影响实验正确性的配套问题已经在 `7377678` 修复。历史 v1 产物保留，不覆盖；后续正式实验必须使用 v2 路径重新生成。

## 已修复问题

1. **完整摘要窗口编码失真**
   - 改为直接把冻结 token ID 窗口送入 SentenceTransformer `forward`，不再 decode 后重新分词。
   - 产物报告新增 `token_window_encoding=token_ids_direct_v1`；旧报告或不匹配报告 fail closed。
   - 真实 `all-mpnet-base-v2` 单窗口对照中，新路径与标准 `SentenceTransformer.encode` 的 cosine 为 `1.0`，最大绝对差为 `0.0`。

2. **Embedding artifact 校验不完整**
   - Dense、Graph、Similarity 统一使用同一加载器。
   - 校验 dataset manifest、documents hash、embedding/metadata hash、模型、overlap、聚合规则、维度、有限值、L2 norm、文档数量及冻结 doc_id 顺序。
   - 不再静默复用损坏、旧格式或部分生成的产物。

3. **Similarity union-kNN 与边权契约**
   - 每个源节点最多贡献 `k` 个非自身候选，避免 self 未进入 FAISS 返回集时多取一个邻居。
   - union-kNN 只决定无向边成员；最终边权从冻结 embedding 重新计算 dot product，再乘固定 scale，符合冻结规格。

4. **Adjacent 连续 order 契约**
   - 只连接同一文档内 `next.order == current.order + 1` 的 passage，权重严格为 `1.0`。
   - order gap 只记录、不跨越；图构建的期望边数已正确扣除 gap。
   - 拒绝布尔、浮点、负数、缺失或重复 order，避免隐式 `int()` 改变数据语义。

5. **实验可审计性**
   - BM25 metadata 改为冻结文档顺序，不再从 set 输出非确定顺序。
   - document embedding、BM25、Dense、Hybrid、Graph、Reranker 和 paired cases 全部迁移到 v2 路径，避免覆盖或误读历史 v1 结果。

## 验证证据

- Windows 针对性回归：`46 passed`。
- WSL Docker 全量测试：`125 passed, 2 failed`。两项失败均为容器中 spaCy `3.8.14` 加载基于 spaCy `3.7.4` 训练的 `en_ner_bc5cdr_md` 时发生配置类型不兼容；失败发生在模型加载阶段，与本次边策略代码无关。
- `git diff --check`：通过。
- 正式 v2 PubMedQA 重跑曾启动，BM25-document 已完成；其 test 指标为 Recall@10 `0.990`、MRR@10 `0.9619`、nDCG@10 `0.9689`。随后完整摘要 embedding 阶段未使用 GPU、6 分钟后仍未产生 embedding 产物，因此已停止；Dense/Hybrid/Graph 本次不报告新指标，也不以单元测试替代正式实验。

## 后续执行要求

DS 下一步只需要在可用 GPU 环境中基于 `7377678` 重跑以下既定实验，不要增加新变量：

1. `hybrid-document`（同时生成 BM25-document、Dense-document 和共享 v2 embedding）；
2. `graph-document --profile ep`；
3. `graph-document --profile similarity`；
4. `graph-pairs`。

正式报告只能引用 v2 目录中的 metrics、rankings、manifest 和 hash。旧 v1 指标仅作为失效历史保留，不得与 v2 混合比较。

## 非阻断项

- spaCy/BC5CDR 容器版本兼容属于实验环境问题，单独修复，不纳入本次代码范围。
- 本轮不加入 hub penalty、学习型边权、UMLS 实体链接或额外调参。
- NFCorpus、SciFact、textbooks 和 StatPearls 继续按独立数据集后续推进，不恢复合并大库实验。
