# Medical verifiable V2 第二批候选复核（2026-09-26）

## 结论与边界

从冻结的 `candidate_other.jsonl` 中 `status=reserve` 的 25,904 条，固定补抽 **1,000** 条作为独立第二批。它们不是原先已经存在的另一千条。未对首批 3,000 条或原始题目、gold 做修改，也没有因筛查拒绝而自动补抽。两批合计 4,000 条，仍没有可称为医学质量已验收的训练集，也没有分配 train/cold-start/dev。

| 批次 | 候选 | reject | quarantine | pending | 医学质量已验收 |
|---|---:|---:|---:|---:|---:|
| 原批 | 3,000 | 294 | 120 | 2,586 | 0 |
| 第二批 | 1,000 | 80 | 20 | 900 | 0 |
| 合并 | 4,000 | 374 | 140 | 3,486 | 0 |

`pending` 表示没有被现有规则和抽查证据排除，**不是**医学答案正确或可直接训练的认可。合并账本保留所有 4,000 条供追溯，其中 `screening_eligible=true` 仅对应 3,486 条 pending；所有 reject/quarantine 为 false；所有行的 `training_accepted=false`、`split=unassigned`，清单 `training_ready=false`。没有生成训练输入文件。

## 抽样及来源

- 原始数据：`FreedomIntelligence/medical-o1-verifiable-problem`，冻结 revision `0b347655bd332537e6814d53c6fe4bc03b555bdc`，原始文件 SHA256 `f5998431acb0eb35e9365f81b6e434d67c44392a5667280068c63ec072ba53cd`。
- 从首批 `candidate_other.jsonl` 的 25,904 个 `reserve` 行，根据 ASCII 字符串 `20260919:source_id` 的 SHA256 摘要排序，取最小 1,000；输出按 `source_id` 升序。相同冻结文件和 seed 产生相同候选。脚本核对原始题干和原 gold 后才写出。
- 第二批候选 SHA256 `2d826cf0fdf6084cbe5213fcbbcd89a067d0388e0ce9452ee2acb8f1430f17b3`，Windows 与现有 WSL Docker `llm-pytorch` 重建结果一致。
- 第二批候选、清单和复核资料位于 `MedSearch/data/medical_verifiable_v2_batch2_20260926/`；没有覆盖首批目录 `MedSearch/data/medical_verifiable_v2_20260919/`。

## 隔离与逐对复核

精确检查覆盖历史 RL train/dev、Stage-A train/dev、Search100、protocol dev、冻结 MIRAGE final test、首批 3,000 条以及第二批内部。8 个边界的文件 SHA256 都在 `exact_isolation.json`；题干/gold 与原始行一致；精确冲突、缺失、非法题干均为 0。**精确无冲突不等于语义无重复。**

本地 MPNet（模型权重 SHA256 `a8fd120b1a0032e70ff3d4b8ab8e46a6d01c2cb08ffe7c007a021c1788928146`）对第二批与 42,320 个参照题（历史 39,320 + 首批 3,000）及批内做 Top-1、cosine ≥ 0.92 筛查，得到历史/首批 74 对、批内 8 对。我逐对读了这 **82 对**题干，记为同一核心问题 76 对、不能确认同题 6 对；这仅是题干重复判断，非临床 gold 审核。冻结 MIRAGE 中的 3 个同题近重复为第二批 source ID 32799、32933、40185，已拒绝。

字符 3–5gram TF-IDF Top-1、cosine ≥ 0.8 产生历史/首批 44 对、批内 6 对。与 MPNet 队列去重后有 44 条新增关系：其中 40 条的候选已进入 MPNet 队列，这些额外配对**未分别裁决**，保持候选拒绝或隔离；其余 4 条独有候选的题干配对已人工读为同题并拒绝（source ID 10969、11373、26415、39755）。两种 Top-1 阈值只是可追溯筛查，不能保证找全所有近重复。

对明显质量疑点另作了 8 条题干/gold 记录：7 条隔离、1 条拒绝。示例：source ID 19255 依赖未给出的 MRI 影像描述，23705 声称给出用药清单但题干没有，19058 没有可识别施害人的证据却把母亲写作 gold。它们的具体理由在 `manual_review_notes.jsonl`。这些判断仍不能替代逐题医学专家审阅或独立证据校验。

## 文件与复现

- `candidate_batch2.jsonl` / `manifest.json`：候选、抽样规则、seed、revision 和输入/输出 hash。
- `exact_isolation.json`：精确隔离结果和边界 hash。
- `semantic_neighbors.jsonl`、`lexical_neighbors.jsonl` 及各自 manifest：两种近重复候选队列。
- `semantic_pair_review.jsonl`、`lexical_pair_review.jsonl`、`pair_review.manifest.json`：题干配对裁决及未分别裁决标记。
- `manual_review_notes.jsonl`：按候选 ID 可定位的拒绝/隔离理由。
- `review_ledger.jsonl`、`combined_review_ledger.jsonl`、`review_ledger.manifest.json`：第二批和两批合并视图、状态数和 SHA256。合并账本 SHA256 `fa443193a93dd4db34126249f175895a490c1c29f90f0a8cb6f3277b2c53164e`。
- 抽样、复核重建和验证脚本：`MedSearch/tools/build_distillation_v2_batch2.py`、`adjudicate_distillation_v2_batch2_pairs.py`、`adjudicate_distillation_v2_batch2_quality.py`、`build_distillation_v2_batch2_review.py`、`validate_v2_batch2_review.py`。人工配对裁决保存在脚本和审阅行中；运行脚本只重现这些已有判断，不应解释成自动医学审核。

在项目根目录的 CMD 中直接运行：

```cmd
python -m MedSearch.tools.validate_v2_batch2_review
python -m unittest MedSearch.tests.test_build_distillation_v2_batch2 MedSearch.tests.test_build_distillation_v2_combined -v
docker exec -w /tmp/luna_review_workspace llm-pytorch python -m MedSearch.tools.validate_v2_batch2_review
```

上述验证在 Windows 与 WSL Docker 各通过；4 个定向单元测试在两处均通过。Docker 中还依次重建了第二批候选、人工配对账本、8 条质量记录和合并账本，候选与合并账本 SHA256 均与 Windows 相同。Docker `/tmp/luna_review_workspace` 是临时审计副本；正式产物在项目 `MedSearch/data/medical_verifiable_v2_batch2_20260926/`。

## 待完成

剩余 3,486 条 pending 需要医学题干/gold 质量审阅和更充分的近重复检查，然后才能确定 train/cold-start/dev 分配及训练验收。当前 374 reject、140 quarantine 一律不得进入后续训练或划分。没有调用付费 API，也没有启动训练。
