# 蒸馏问题池合并与 dev 隔离（2026-09-26）

用户决定丢弃两批候选账本中的 374 条 reject、140 条 quarantine；其余 3,486 条沿用原数据集 gold 作为弱标签，不要求逐条医学专家审核，也不宣称临床正确。旧 RL train/dev 原文件保持不变。

## 固定产物

最终目录：`MedSearch/data/distillation/pools_v2_20260926_devclean/`。构建入口：`python -m MedSearch.tools.build_distillation_pools`；输入及输出 SHA256、固定种子 20260919、分配规则见目录内 `manifest.json`。

| 用途 | 条数 | 来源 |
|---|---:|---|
| 冷启动候选 | 1,000 | 新 3,486 条中按固定哈希顺序划出 |
| RL train | 5,756 | 旧 RL train 3,283 条排除与 dev 同题改写的 13 条，再加新题 2,486 条 |
| dev | 396 | 原 `MedSearch/data/rl/dev.parquet` 原样复制，SHA256 不变 |

冷启动与 RL 题目分属不同 split。新题原问题与原 gold 未改；RL Parquet 保留旧 veRL schema。`new_assignments.jsonl` 记录全部 3,486 条新题的去向。原旧 RL train 的 13 条仅从本次合并版本排除，不回写历史实验文件。

## dev 隔离证据

合并前的全训练池与 dev 对比：ID/规范化题干精确冲突 0；字符 3–5 gram TF-IDF（Top-1，阈值 0.8）标出 5 对；本地 MPNet（Top-1，阈值 0.92）标出 15 对。合并两队列后涉及 16 个不同的训练 ID，全部来自旧 RL train。逐对题干判断记录在 `MedSearch/docs/audits/distillation_dev_pair_review_20260926.jsonl`：13 对同题改写从新合并版排除，3 对保留，因孕周/稳定状态、问法正反或数值不同。此判断不涉及 gold 的临床正确性。

重建最终版本后，再在既有 `llm-pytorch` 容器对 RL train + 冷启动候选（合计 6,756）和固定 dev 396 条直接扫描：精确冲突 0；词面队列仅余 1 对、MPNet 队列仅余 3 对，均为上面已记录的条件不同题。输出保存在最终目录的 `exact_isolation.json`、`dev_lexical_neighbors.jsonl` 和 `dev_semantic_neighbors.jsonl`。两种 Top-1 扫描都不能数学上证明不存在任何语义重复。

初版目录 `pools_v2_20260926/` 曾包含这 13 条旧 RL 近重复，manifest 已标为 `superseded_dev_near_duplicates_found_do_not_use`，不得训练使用。最终目录 manifest 标明来源 gold 未经医学专家核验，且长训练前仍需完成 7B 教师身份/tokenizer 核对、checkpoint 保存—重载—动作推理 smoke；不沿用 V9 的 `save_freq=-1`。
