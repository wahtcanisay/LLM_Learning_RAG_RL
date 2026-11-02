# RL增量抽检操作书：GPT-6.1 Sol中等思考专用

## 0. 当前操作卡

**Terra中等主审 → Sol中等子代理按包抽检 → Terra处理纠正并发布 → 回读后按用户终点收口。**

用户2026-10-28明确改为 [Luna Max 批量清洗、Terra 中等思考复核发布](2026-10-27-rl-cleaning-terra-sol-sampling.md)。清洗过程中不使用 Sol 子代理或例行 Sol 抽检；本文件保留四项内容审查原则和历史记录，旧 Sol 分工、抽样与签字要求不适用。500 条完成后是否抽查由用户决定。

1. 子代理只读本操作卡、§2及主代理指派的ID和原件路径/hash；批次接续、全量清单和当前配额由主代理处理，不让子代理重复全仓库探索。
2. 范围锁定为当前包随机样本、轮末2条失败样本及已给具体疑点的定向项。没有具体实质漏筛，不增加样本、不看全部100题或历史release。
3. 对照原题/完整gold、7B实际窗口及episode、终答/原分，先判断再对照Terra理由；分4–6题读取只是控制长度，不增加调用或审查层。
4. 本包一次交付逐题结果到主代理指定的`sol_review/`返回文件，最少包含question_id/decision/evidence_grade/semantic_grade/issue_code/rationale，问题项附可定位依据。原始C2/C3、gold和分数不改。
5. 一次聚焦核对后仍未定就quarantine；字段/格式错误定点纠正。19210无终答应记empty，题面范围疑点另记；不能混淆两轴。
6. 实质漏筛先报具体ID和同类范围，由主代理安排当前100题内受影响类型的扩查；只在无法界定时扩至本轮拟发布项。不自行启动第三位裁决或重跑7B。
7. 返回实际文件路径及本次ID覆盖，停止等待同轮后续指派。**子代理不写活动账本、不发布、不调用7B、不派其他代理；Terra主代理校验、纠正和发布。**

用户若明确采用人工切换Sol或旧版全候选复核，可按其指定范围使用本书内容标准；不能因此把新模式未抽样项标成Sol已审核。

## 1. 两方职责与数量口径

| 阶段 | 工作与产物 | 能否计入正式训练数据 |
|---|---|---|
| Terra中等，100原题/轮 | 完成C0/C1/C2、7B盲搜索、C3及候选保存；组织抽检，单独纠正、发布及回读 | 候选不可计入；真实发布并回读后可计入 |
| Sol中等，本轮抽样和明确争议 | 对照原题与真实窗口检查，保存retain/quarantine/reject及依据 | 抽检文件本身不是正式训练数据 |
| 共用脚本 | 哈希、ID/JSON/引用、去向、发布、loader/collator及计数 | 脚本成功不能替代内容判断 |

分别报告符合当前配额的拟发布分母、实际随机样本数、失败项抽检数及定向/扩查数；不能用100原题稀释漏筛率，也不能把小样本的一致率当全候选准确率。逐项判断可见gold，不是gold盲审；两模型一致不证明医学金标准可靠。未抽中项只归Terra主审，原有隔离结论复用。

## 2. 一题只检查这四件事

1. **题面和gold是否对得上。** 检查所问对象、条件、否定、单位、集合范围及粒度。原题没给的影像、选项、紫癜、因果前提，不能从检索或答案补入。提问“部位/原因/活动”没有明确细分要求时，不能自动拿更细gold拒绝合理回答；同样不能因为有其他抽象可能就否定正常“最可能”题。
2. **实际窗口是否新增有效帮助。** 沿用冻结E0–E3；E2可为必要答案组成、明确中间诊断/机制或有效鉴别/排除，不要求覆盖全部gold。泛主题、复述题面、候选名单里出现答案词，不自动是E2。另一对象/场景的共性机制若窗口已建立可用；桥接实质未定则隔离。读全实际rank，窗口外信息不算证据。
3. **终答与评分是否有具体矛盾。** 同时看完整final_answer和原verifier_prob。正确答案误罚、未答所问对象却获奖、合理替代答案未覆盖的ID暂缓；原分数保留。普通教师错误、空答、动作违规不单独排除有有效证据的RL问题，不把CoT补成终答。只把评分疑点隔离，不自行改gold/别名/分数。确有具体疑问时最多一次聚焦核对，仍未定就收口。
4. **发布的是否仍是纯问题。** prompt必须是固定协议加原题；gold仅在评分字段，检索、教师回复、审核理由仅在sidecar。复用身份匹配的机械回执；模型/输入/文件变更或发现具体异常才重查对应项，不每题重新载入tokenizer。

结论为`retain`、`quarantine`或`reject`，证据等级另记。保留原E2/E3但因评分疑点quarantine是合法情况；不把所有quarantine叫医学错误。`retain`只表示本轮范围未发现阻塞，不重新认证未审核的历史基线。

逐题通常1–3句，最少记录：`question_id/decision/evidence_grade/issue_code/rationale`。问题项加对应实际`search_index/rank/document_id/quote`或原题/终答的关键原文，并绑定Luna输入/raw/episode的路径和hash；留引用，不复制整段教材。已有有效结论不反复改写。

## 3. 保存和正式发布：复用入口

Terra已调用`record_rl_preview`保存staging。Sol子代理只写指派返回文件；Terra校验抽检ID覆盖和引用，单独将问题项在活动账本置`assignment=quarantine/reject`，保留原C3及纠正历史。被抽中的保留项写真实`sol_review_ref`，未抽中记录not_sampled并沿用Terra主审引用，不伪填Sol全量审核。若有串题/引用错误，只处理受影响ID。

以下发布命令**仅由Terra主代理执行**：复用`publish_joint_rl`与`finalize_rl_preview`，需要累计增量时沿用campaign既有步骤；**不要再调用receive读取旧raw覆盖复核去向**。本包抽检结果入账、实质问题处置后发布该包其余合格项，不等整轮；具体未决ID隔离后不拖住通过项。发布前确认当前基线仍与批次冻结基线一致；若指针改变，比较实际差集并处理重复，不能覆盖别人新结果。

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring"
docker exec -w /workspace/llmlearning llm-pytorch python -m MedSearch.data.cleaning_v2.publish_joint_rl --batch RUN_NAME --publication UNIQUE_INCREMENT --r1-report MedSearch/data/cleaning_v2/rl_20261021_batch04/reward_interface.json
docker exec -w /workspace/llmlearning llm-pytorch python -m MedSearch.data.cleaning_v2.finalize_rl_preview --run RUN_NAME --previous-run BASELINE_RELEASE_FOLDER
```

依次用实际值替换RUN_NAME、未占用增量名、冻结baseline对应目录；R1只复用reward/config身份一致的回执。两个命令分开，确认Exit0后继续。完成全部100题的最终去向后才调用`rl_cleaning close`，把Sol复核摘要与最终指针追加到报告，不让模板覆盖审计产物。每个已通过小包可先发布，不必等本轮全部候选复核结束。

如果Luna沿用旧模式提前正式发布了候选，保留原release为历史，按Sol具体退出ID生成更正release、做实际loader和回读后切换当前指针；不得把原release覆盖成新hash，不重建未变旧题、不重新开展全历史医学审核。

正式累计文件可包含旧基线。manifest/sidecar必须区分历史保留、Terra主审与Sol实际抽检范围，不能写成历史或新增全量均由Sol审核。

## 4. 每轮结束与Luna书的小幅改进

复用现有`sol_review/`保存抽样清单、各小包返回、汇总决策和SUMMARY及已有发布回执，不另加专家签字、评分专审或第三位审核者。待定项当轮隔离；原题/gold/预算不变，不重跑episode争取通过。主代理单独写账本及累计状态。

每轮报告记录：原题100、Terra证据候选数及配额合格拟发布数、随机/失败/定向/扩查各自数量、retain/reject/quarantine、新增正式数与累计、E2/E3、问题类型与ID、可得耗时/搜索数和两模型用量；未知写unknown。比较效率只用口径相同的分母和计时边界，不从账户余额推算费用，不把7B tokens当云端模型tokens。

用户既有的小幅规则纠错授权继续适用；Sol提交具体建议，由Terra主代理在每轮收口时统一维护当前操作卡及旧书相应内容，避免两方同时修改，范围如下：

- 实际发生且有原件的通用漏筛或重复工作：优先改原对应句、替换重复反例；必要时新增简短反例和正确处理。
- 反例须给题ID、触发原文、错误判法和正确动作；有争议的案例标uncertain，不能为了教育Luna就把它写成已证实医学事实。
- 有效E2、正常教师失败和跨rank补全也保留正例，避免越改越像SFT。单次模型疏忽通常补一句检查提示即可，不自动新建门。
- 只修改后续轮执行文档。记录本轮修订摘要/旧新hash，旧轮RULE_SNAPSHOT、raw、episode与gold不改。
- 不自行更换语料/模型、增episode/Search预算、放宽隔离/纯问题底线或改变reward实现；这些若必要需另向用户说明并取得相应范围授权。用户本次同意每轮100不等于授权无限连续扩池。

## 5. 首轮已知问题与接续

2026-10-22首个Sol复核样本是旧模式已发布的46条：37条未发现新增阻塞，9条按用户要求退出当前正式训练集合。逐题依据沿用`rl_20261022_lunamax_batch02/NEW46_CONTENT_AUDIT.jsonl`与`NEW46_HOLD_RECOMMENDATIONS.jsonl`，不在本次再次医学审核。

典型提醒：14366/14400误罚未暂缓；14428把red spots补成紫癜；14523把名单命中判E2；14530跨手术场景桥接未定；14396/14455/14550/14655问答粒度疑点。正例14653应合看rank1和rank2的endoderm关系。失败类别和去向保持事实边界。

本轮撤回后的数量、路径、hash必须读`CURRENT_RL_RELEASE.json`及`rl_20261022_lunamax_batch02_sol_filter9/receipt.json`，不能依靠文档快照。后续冻结新100题时，选题游标继续使用最近有冻结原题的`rl_20261022_lunamax_batch02`（或后继真实清洗批次），基线则取当前正式指针；更正发布目录不是一个新的原题池。

旧版人工切换、全候选复核的启动例子仅在用户明确重新选择旧模式时使用：

> 按Sol增量复核操作书，复核本轮Luna已放行的全部候选；只处理本轮，通过项及时正式发布，问题项单独退出，回读并说明是否有必要小幅更新Luna书，完成后暂停。
