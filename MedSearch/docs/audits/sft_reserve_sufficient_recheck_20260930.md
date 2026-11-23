# SFT reserve 中 sufficient 轨迹复核

日期：2026-09-30  
范围：`sft_environment_review_20260929_full/review.jsonl` 中 `sft_recommendation=reserve` 且 `support_label=sufficient` 的 51 条。  
依据：逐条对照对应 accepted 轨迹、审查理由、动作标记、gold 关系、题目判定与可见检索 observation。此项只复核轨迹适用性，不构成临床专家核验。

## 结论

- **直接升级为严格 SFT 正例：0/51。** 每条都至少有答案关系、轨迹动作、题目状态或审查记录一致性方面的未决问题。
- **可考虑重新生成一条完整轨迹：8 条。** 这些题当前题目为 `keep` 且答案关系为 `equivalent`，但旧轨迹不能直接复用。
- **其余 43 条继续 reserve。** 不改题目或 gold，不以 verifier 通过代替医学正确性判断。

## 可考虑重生成完整轨迹的 8 条

| 原因 | question_id | 复核要点 |
|---|---|---|
| 非法动作 | `medical-o1-verifiable:2809`, `:4452`, `:5203`, `:6057`, `:8476` | 源记录和审查标记均记录 `invalid_action=true`；旧轨迹不作为正例。 |
| 重复搜索 | `medical-o1-verifiable:5853` | 第二个搜索查询是同一计算任务的改写；重新生成时需确认查询有新增信息。 |
| 推理含未被引用支持的额外说法 | `medical-o1-verifiable:5287` | 最终答案与 gold 等价，但推理额外声称改良 Thayer–Martin 培养基含白喉因子，当前可见片段没有支持该说法。 |
| 审查标记与源轨迹矛盾 | `medical-o1-verifiable:6859` | 标记为无非法/重复动作，但复核理由称存在无效搜索；两次返回的 observation 与文档哈希相同。先按失败轨迹处理，不升级。 |

重新生成仅表示可复用**题目**作为一次新 rollout 输入。必须保留新的完整 Search-R1 轨迹、查询和 observation；不能在旧轨迹上删动作、补片段或改写 reasoning 后冒充真实生成。

## 继续 reserve 的 43 条

- 教师答案与 gold 标为 `different`：24 条，答案关系需裁决：
  `4282, 4515, 4584, 4617, 4764, 4863, 5039, 5292, 5337, 5339, 5447, 5455, 5687, 5780, 5867, 5885, 5918, 6197, 6397, 6684, 6704, 6807, 7737, 9700`。
  其中 `6684` 另有非法动作，`5687` 另有重复搜索；若后续裁决决定保留题目，也要生成新轨迹。
- 教师答案关系标为 `uncertain`：17 条，继续待裁决：
  `2883, 40061, 4033, 4168, 4329, 4430, 4472, 4832, 5006, 5581, 5610, 6090, 6214, 6347, 6417, 6561, 6856`。
  其中 `5006`、`6417` 另有非法动作；`2883`、`4863`、`5292` 还带有环境冲突记录。
- 教师关系等价但题目判为 `uncertain`：`3517, 5561`。先裁决题目，不凭 sufficient 标签升级。

以上 ID 均为 `medical-o1-verifiable:<数字>`。

## 引用核对与局限

- 51 条的 56 个 `support_map` 引用均能按查询映射到轨迹中的可见 `<information>` 内容，且记录的 `document_hash` 出现在相应搜索的返回哈希中。这只核对引用是否真实出现在模型可见文本，不判断医学含义。
- 50 条没有单独保存 BM25 完整返回（`full_return_support=not_saved`），1 条标记为 `not_saved_separately`；因此无法用这批存档确认完整返回中的上下文。
- 50 条数据集级近重复状态仍是 `unverified`，另 1 条为 `not_checked_in_pilot`。轨迹内重复搜索检查不等于数据集去重。

## 复核依据

- 原始 51 条审查记录：[review.jsonl](../../data/distillation/sft_environment_review_20260929_full/review.jsonl)
- 对应教师轨迹：[accepted.jsonl](../../data/distillation/teacher_cold1000_20260929_answered/accepted.jsonl)
- 冻结批次的抽样、证据规则及边界：[数据复审执行书](../superpowers/plans/2026-09-29-sft-environment-data-review.md)
- 本记录由只读复核生成；未改动源轨迹、question 或 gold。
