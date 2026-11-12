# Luna V2 候选复核进度（2026-09-26）

状态：**部分完成；不可用于训练或分配 split**。这里的 Luna 指代理的题干/题答文本审阅，不是临床专家认证。

## 固定输入与隔离

- 来源：`FreedomIntelligence/medical-o1-verifiable-problem`，revision `0b347655bd332537e6814d53c6fe4bc03b555bdc`；本地原始文件 40,644 行，SHA256 `f5998431acb0eb35e9365f81b6e434d67c44392a5667280068c63ec072ba53cd`。
- 首轮候选 3,000 行，`candidate_keep.jsonl` SHA256 `53be1b32aa17f9b2389eeb23445bcf31eea5e4e30cca7421a03e1b12a5a8cd55`。逐行回查原始文件的 `source_id`，题干和 gold 文本均未改写。
- 现有 RL train/dev、Stage-A train/dev、Search100、protocol dev、MIRAGE 共七个边界输入的 ID 与规范化题干精确冲突为 0；各边界路径、行数和 SHA256 见 `distillation_v2_candidate_isolation_container.json`。该结果不代表近重复为 0。

## 近重复复核

- 本地 MPNet、余弦阈值 0.92 的 Top-1 队列有历史配对 232、候选内部配对 72，涉及 336 个候选 ID。Luna 逐对阅读这 304 组题干，判断 281 组同题改写、23 组条件不同或证据不足；逐对理由见 `semantic_pair_review.jsonl`。其中 15 组与冻结 MIRAGE 测试题是同题改写。
- 另用既有 Docker 容器和已安装的 scikit-learn 执行字符 3–5 gram TF-IDF、余弦阈值 0.8 的 Top-1 扫描，历史配对 144、内部配对 40，涉及 213 个候选 ID。与 MPNet 队列相比新增 19 个候选 ID、146 条配对关系。新增关系中 14 条题干同题、10 条不确定、122 条的候选已因其他配对拒绝，未独立裁决；见 `lexical_pair_review.jsonl`。这一轮又识别出 4 条与 MIRAGE 冻结测试同题改写。
- 高相似度不能自动等于重复。例如候选 `2901` 与旧 RL 对照的灵敏度、特异度、患病率及 PPV/NPV 问法不同；`4928` 的 renin 高低相反；`6599` 的妊娠超声与 hCG 数值不同。它们留在隔离集合。两种 Top-1 扫描都不是穷尽式语义去重。

## 候选级账本

`review_ledger.jsonl` 覆盖全部 3,000 条，逐行保留原题、原 gold、source_id、未分配 split、决策、理由、审阅依据及关联近邻。当前计数：

| 决策 | 条数 | 含义 |
|---|---:|---|
| reject | 294 | 227 条与历史/冻结测试同题改写、56 条内部同题改写、11 条确实依赖缺失图/视频/心电图等附件 |
| quarantine | 120 | 近重复条件不一致、缺关键上下文、无选项的排除题或题答疑点，尚待进一步裁决 |
| pending | 2,586 | 只通过旧规则初筛，未完成题答医学质量审阅 |
| accepted | 0 | 本轮未宣称任何样本已通过完整质量门槛 |

其中与冻结 MIRAGE 同题改写的候选级拒绝共 19 条。所有明确题干配对判断只证明数据隔离问题，不证明 gold 医学正确。`manual_review_notes.jsonl` 记录人工文本判断；其余样本的规则筛查在账本中标为 `rule_screen_only`。

## 验证与剩余工作

- Windows 和同一 `llm-pytorch` Docker 容器中的账本定向测试通过。源文件哈希、候选题干/gold 保持不变、两类队列与账本哈希、决策数及精确隔离结果由 `MedSearch/tools/validate_v2_review.py` 校验。Docker 的 `/workspace/code_list` 挂载仍为空，本轮用容器 `/tmp/luna_review_workspace` 临时副本执行；未启动训练或付费 API。
- 需要继续复核 120 条隔离项和 2,586 条待审项的医学题答质量；尚未进行独立临床专家校验。语义 Top-1 阈值之外可能仍有漏网近重复。
- 冷启动 1,000、RL 新增 717、dev 新增 104 的划分尚未进行；目前 `training_ready=false`。即使待审数量在算术上足够，也不能当作合格数量。

账本 manifest：`review_ledger.manifest.json`，ledger SHA256 `c4fe31411bd3d4cb657b9dcf1643624a1385bce74d85c99c4ccc3ff5c583cf85`。
