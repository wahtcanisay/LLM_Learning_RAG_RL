# RL 数据清洗执行书：GPT-6 Luna Max 专用（单主代理版）

> **2026-10-28现行分工：**按 [Luna Max 批量清洗＋Terra 中等思考复核发布操作卡](2026-10-27-rl-cleaning-terra-sol-sampling.md) 执行；Luna 可作为子代理，Terra 逐题复核拟发布项，清洗过程中不使用 Sol 子代理。下文旧角色、等待 Sol 及禁止子代理的规定让位于现行操作卡；§2–8 内容标准、成功命令和纠错案例继续适用。

> 适用于用户明确选择“直接用 Luna Max、不使用子代理”的清洗任务。**一个 Luna Max 主代理完成接续、C1/C2/C3和收口；本地7B负责真实搜索，共用脚本负责机械操作。** 不调用子代理，不切换Sol，不另建任务，不创建定时任务。
>
> 本书在单代理模式下替代[通用清洗书](2026-09-29-sft-environment-data-review.md)的角色分工，保留其内容与发布底线。日常直接读本书及当前批实物，不再两本全文重复加载。文档不授权新清洗、扩池或训练，范围和停止点以用户最新指令为准。

## 0A. 保留模式：Luna每轮100题，Sol复核放行增量（2026-10-22，已非默认）

用户已选择两阶段工作方式：**一个Luna Max完成100道原题的C1/C2/C3和候选落盘，暂停；主对话Sol复核Luna放行项，合格即正式发布。** 两个模型顺序工作，不开子代理，不互相自动发任务。“100题”是冻结原始输入数，不是保证放行100条；本次修订本身不启动下一轮。

本节在此模式下优先于下文“Luna通过即正式发布”“不设独立二审”等旧职责表述；下文内容、7B盲输入、预算、隔离和E2/E3标准继续适用。Sol使用[增量复核操作书](2026-10-22-rl-cleaning-sol-increment-review.md)。Luna无需读取Sol操作书全文。

1. 每轮冻结100题；C2沿用40题包、C3沿用20题包及实数尾包，**不是一次把100题全部轨迹塞进上下文**。每题仍1个真实7B episode、最多2次Search。
2. 每个C3完整包返回后，先保存raw并通过现有入口入账：

   ```cmd
   docker exec -w /workspace/llmlearning llm-pytorch python -m MedSearch.data.cleaning_v2.record_rl_preview --run RUN_NAME --return-file raw_01.jsonl
   ```

   将`RUN_NAME`和文件名换成当次实物。此入口已支持引用校验、账本及`staging/rl_content_passed.jsonl`；**不调用会立即正式发布的`rl_cleaning receive`，也不调用正式发布收口的`close`**。这不是新增审批工具，只把已有保存和发布入口按职责拆开。
   若主对话已接手机械命令，Luna只保存本次指派门的raw并交回路径；绑定、7B调用及入账由主对话执行，避免两方同时写账本。判断标准和原始预算不变，不再重复执行同一命令。
3. 本阶段“一条通过立即落盘”指立即保存候选；候选`training_ready=false`，不计入`CURRENT_RL_RELEASE.json`。未决项不能拖住已完成候选的入账。先处理已返回未绑定，再生成下一包。
4. 同一次C3说明终答语义、原分数和是否有评分疑点；已知误奖/误罚、题目范围未定、窗口桥接未定者按现有`uncertain=true`隔离。E等级仍可保留E2/E3，不能因为证据好就忽略R1。
5. 100题全部去向入账后，只交付现成放行清单、原题/gold、episode、实际窗口、C2/C3原件、原分数的路径/hash；这些已在账本和引用链中，不另抄全文。`RUN_REPORT.md`与检查点标`awaiting_sol_review`，报告`100=候选+reject+quarantine+未完成`，并明确候选数不是正式新增数。暂停等待用户切换/指令；不为等Sol而不落盘，也不擅自开始下一轮。
6. 本模式下Sol尚未复核的候选不自动发布。若用户明确选择Luna单独端到端发布，才使用下文原`receive/close`工作方式；不能自行退回旧模式。

**本轮反例提醒（详细原件见[46条复核报告](../../../data/cleaning_v2/rl_20261022_lunamax_batch02/NEW46_AUDIT_REPORT.md)）：**

- 14366/14400：已经看到正确核心答案被低分拒绝，仍以E2/E3发布，违反R1。保留原分数和证据，受影响ID隔离。
- 14428：原题`red spots`不能在理由中变成已确认的`purpura/petechiae`。不能把检索片段里的关键条件补到题面。
- 14523：眶肿瘤名单含gold，没有新增题目所需鉴别关系，不能仅因答案词出现就升E2。
- 14530：另一手术场景的CO2气栓监测不自动建立本题坐位开颅空气栓塞与零ETCO2的桥接；仍未定就隔离。
- 14396/14455/14550/14655：先明确题问活动、部位、病因或具体病变，以及所需粒度；不能只因gold更具体，就把实际合理回答自动判错。这些是具体范围疑点，不声称全部gold错误。
- 正例14653：应读完实际rank；rank2已写endoderm，不可因主引句没写就漏掉有效支持。有效E2与正常7B错误仍可保留，不提高为SFT全答案标准。

## 0. 每次接续只执行这张操作卡

**结构/隔离 → 题目与完整gold一次首审 → 7B原题盲输入真实搜索 → Luna审实际投递窗口 → 合格即发布 → 收口暂停。**

1. 读本节、`STUDY_PROGRESS.md`最新RL摘要、`CURRENT_RL_RELEASE.json`及当前批`continuation_checkpoint.json`；需要运行命令时才读对应`COMMANDS.md`。不复读全部历史或归档。
2. 优先处理“已返回未绑定”“已通过未发布”；然后推进当前批缺失门，最后才取授权的新题。已有有效结论直接复用。
3. 新批先冻结原题/gold、保护集、预算、规则和模型身份。C0/C1之后，同一个Luna一次合并审题目与gold；通过者进入7B。现有C2包通常40题，不把它改成逐题调用。
4. 7B每题默认1个episode、最多2次搜索。输入只能是固定协议＋原题，不给gold、离线query、诊断证据或审核结论。
5. 每个完整episode返回即保存不可变单题原件；C3通常20题一包，尾包按实数。有已完成的小包就审，不等100题全部生成。审阅整包实际窗口，一次交齐逐题结论。
6. **结果返回 → 保存原件 → 校验JSON/有序ID/引用并入账 → 本次全部通过项完成发布与真实loader检查 → 回读数量/hash → 才推进下一任务。** 一条通过也发布；同次多条通过合并增量。未决与重试不拖住通过项。
7. 已通过的E2无需补成E3；证据不够、存在冲突或适用性不明则隔离，不调用第二位审核者，不反复检索争取通过。
8. 全部授权题去向入账、合格项已发布后，写报告和检查点，按用户终点暂停。数量/hash和loader未确认前，不报告“可训练”。

按当前门读取§2–5；§6只在遇到对应边界时查例，§7在首次接线或具体运行问题时读取。已完成的机械检查复用身份匹配的回执，不每包重读全书、复述全部反例或重跑同一检查。

**接线状态：§7的共用接口已在`rl_20261021_lunamax_batch01`实际贯通。后续复用，不重复接线；已知评分误判按§6.4处理，loader/hash通过不能替代答案语义核对。**

## 1. 职责、批量和额度

| 执行者 | 唯一职责 |
|---|---|
| 当前Luna Max主代理 | 根据当前实物选择下一步；C1疑似对判定；C2题目/gold合并首审；C3真实窗口判断；执行既有入口并交付结果 |
| 本地7B进程 | 从原题自然生成Search、调用现有检索服务并形成真实episode；它是本地模型运行，不是Codex子代理 |
| 共用脚本 | 冻结、去重召回、预算、输入构造、保存、JSON/ID/引用/hash校验、账本、增量与累计发布、真实loader/collator、报告 |

- **Luna阶段不设额外二审、专家签字或子代理门。** C2和C3由同一个模型分阶段完成，记录为单代理自审；不能标成两名独立审阅者或gold盲审。§0A的Sol增量复核另行记录，也不伪称gold盲审。7B是否盲输入以真实prompt/token IDs核对。
- 用户授权100题就处理100题，单批最多200题；不擅自扩池。C1按现成疑似对包、C2按现成有序包、C3默认20题处理；输出太长可机械分段读取，每次4–6题，仍交付一个完整包。**7B的`batch-size=1`是生成配置，不是逐题审阅指令。**
- 内容判断只写够复核的1–3句。无疑点不查资料、不写逐题长篇推演；遇具体事实疑点最多一次聚焦外部核对，仍说不清就隔离。不要为了得到pass搜到满意为止。
- 不每批新造prepare/record/finish脚本，不人工计算hash，不重复誊写数字。恢复只读当前阶段需要的文件；不得把历史失败日志、完整教材全文或前批全部轨迹塞进上下文。
- 省去子代理可以减少重复输入、调度和返回成本；**实际总额度、速度和质量仍需实跑衡量。** 记录整批墙钟时间、episode/搜索/重试数及可得tokens，未知记`unknown`。不把账户余额变化倒推为准确费用。

## 2. C0/C1/C2：一次完成前门

### C0结构与C1隔离

- C0：稳定源ID、非空完整原题与gold、来源/revision/hash可追溯。缺图且文字不够、缺必要选项/映射、源文本混入回答或解析者，拒绝或隔离；不猜补、不把回复当题目。文字已充分描述影像的题正常判断。
- C1：先由`freeze`冻结候选及保护集，再用既有`audit_v2_semantic`产生近邻JSONL/manifest（阈值0.92、top-10）；Luna逐对保存`c1_pair_raw.jsonl`的`candidate/match/pair_class/rationale`，最后`rl_cleaning c1`只绑定该文件，不负责生成近邻或自动判对。只比对新增范围与本批、当前SFT/RL及固定dev/test；疑似对判`same_task/related_distinct/uncertain`并给具体理由。
- 同知识点/同答案不自动同题；same-task簇不能跨保护集合，uncertain隔离相关簇。无近邻命中不等于证明全历史零泄漏。保护资产真实缺失只阻断受影响范围，不重查退役实验或旧release。
- 固定dev/test保持原分布，不能按检索命中或学生正确率删题；SFT保持冻结。

### C2题目与完整gold合并首审

同一Luna可见原题及source gold/已有解释，一次检查：**所问对象 → 完整gold是否回答 → 条件、限定与评分覆盖是否成立**。检查否定、数值/单位、年龄/病程、机制方向、复合答案，以及“最佳/首选/唯一”。

- 明确坏题或错gold：`reject`；缺条件、实质争议、多个合理答案而gold/reward覆盖不足：`quarantine`；可靠且可答：`pass`。不把教师成功率当题目可靠性的依据。
- “可以使用”不证明“首选”；普通推荐治疗也不自动要求证明所有病例的唯一最优。正常同义表达可以接受，不逐字比答案；不自行扩大或改写gold。
- 对“哪项实验室发现/疫苗/不良反应”等开放提问，必须区分“gold是正确例子”和“当前题目与reward能覆盖合理回答”。缺必要选项、范围或数量限定，且能指出具体未覆盖的合理回答时，C2隔离；不能只写“经典答案/传统考试用法”就通过，也不因抽象地存在替代方案而一概隔离。
- 开放列举题看范围/数量/选项及合理答案覆盖，不能因gold中某个成员正确就认定完整。不能忽略gold的错误附加机制、单位或限定。
- `one example/举一种`只限定回答数量，不限定答案身份。若能指出具体未覆盖的合格成员，仍按上条隔离；不能写“gold给一种就足够”后直接通过。例如15617可举AIDS定义疾病，实际窗口列Kaposi/NHL/浸润宫颈癌，单一Kaposi gold的覆盖仍未解决。
- 一次正常首审结束即入账。只有具体反证、串题理由等无效返回，才对受影响ID纠正一次；原件保留，仍不确定则隔离。不要把改结论、换引句、补医学前提称为格式修复。

## 3. 7B真实盲搜索：不走离线准入捷径

1. policy输入必须由现有代码构造：一个user消息，固定`INSTRUCTION`＋`\nQuestion: `＋原题。不得手写相似协议替换。gold只在评分字段中，离线证据、检索建议、C2/C3和教师历史答案均不进prompt。
2. 7B自行生成query并调用BM25服务；原题导致其自然猜到诊断/答案词不算外部泄漏。Luna不得看gold后指定query，不回放离线结果冒充自然搜索。
3. 每题默认**1个episode、最多2次Search**。无搜索即答对也不能通过这次检索准入。无效方向、重复搜索、无有效证据者保留原迹并隔离，不追加episode刷通过；换批次/模型不重置旧预算。增加预算须用户专项授权。
4. 保存prompt IDs、模型/revision/tokenizer、动作及token、query、原始检索返回、doc ID/rank、真正投递的观察文本/token IDs及投递标志、终答与原始评分。共享追加日志不是不可变episode引用；单题完成后固定副本并记录hash。
5. 使用真实`encode_observation`已截断并闭合标签的窗口：当前top-k=3、每次观察**整段总计500个实际7B tokenizer tokens**，并非每篇500。窗口外、未投递内容和原教材全文不能用于C3；Luna不自行重tokenize/截断，不用别的tokenizer重建后冒充实际窗口。
6. 学生tokenizer与7B不同时分别记录；实际准入审的是本次7B看到的窗口。不得静默加长窗口追命中，不能把离线学生窗口替代7B窗口。
7. 环境失败与语义失败分开记。确实未完成的环境失败最多原参数重试一次，保留失败回执及原预算；已有完整episode不得借故重跑。原Q0/Q1/Q2离线路径不用于新批准入。

## 4. C3：只依据实际窗口判断有无新增帮助

C3包由脚本带上原题、已通过的gold、实际窗口及不可变episode引用。Luna审全包各题已提供的实际rank，**不要只复述gold、旧引句或verifier结果**。本门可见gold，不能伪称独立盲审。

| 等级 | 判定 | 去向 |
|---|---|---|
| E3 | 原题＋实际片段及正常推理已支持最终答案关键关系，不缺关键前提 | 可发布 |
| E2 | 实际新增必要答案组成、中间诊断/机制或有效鉴别/排除依据，尚缺完整终答关系 | 可发布，明确缺口 |
| E1 | 仅主题/词面相关、泛背景、复述题干，或无适用桥接的另一对象/机制 | 隔离 |
| E0 | 无支持或存在未解决的实质冲突 | 隔离，写明失败关系 |

- E2/E3理由写成“**窗口支持的关系 → 对本题新增哪一步 → 尚缺什么**”，引句逐字对应实际search轮次/doc ID/rank。正常同义和推理可用，不要求答案词出现；不能靠Luna已知gold补出缺失主体、机制或适用前提。
- C2有效通过且无新反证时，一个必要组成或明确中间步骤就足以E2，不能要求覆盖全部gold。跨病例共享机制若窗口确有支持可E2；只有相似症状、同成分/终点或同主题不够。
- 缺完整证据不是gold错误；发现具体题面/gold反证才纠正该ID。引句存在/hash正确只证明绑定，不证明推理有效。未解决冲突、`uncertain`或原题＋标准解析镜像`exact_qa_leak`均不入正式集合；普通教材与答案一致不自动泄漏。
- **检索证据、终答语义、verifier分数分别记录。** 7B答错、未答或停得不好，但实际窗口已达E2/E3，问题仍可用于RL；失败轨迹不是SFT示范。准确命中原批准文档ID不是要求，其他文档同样可以提供有效证据。
- 同一C3若已判`semantic=correct`且原verifier拒绝，必须标`uncertain=true`并隔离，不能只写“答案正确、E3”留下放行状态。反例15465：`radial nerve`对gold `Radial`仅0.310694被拒；词形差异不改变其具体误罚事实。不新增评分调用，也不改原分。
- 终答缺否定、对象或关键限定，不能用推理过程替它补齐。已有verifier同义漏判保留原分数，受影响类型训练前定点处理；不为每题再开一个评分审查层。
- **在同一次C3里顺带看已有episode的`final_answer/verifier_prob`。** 用既有理由字段简记“终答语义：正确/错误/不完整/空/不确定；原评分；是否误奖/误罚”。不把`prob>0.5`当正确答案证明，不用CoT补终答，不新增生成或独立审核调用。真实答错且正确判负，与正确答复被误罚是不同情况。

## 5. 保存与发布：模型不手工重造机械流程

1. 返回先落UTF-8临时文件，脚本确认JSON可解析、完整有序ID覆盖且无重复、与输入及来源绑定后，固定为raw原件。格式修复保留原件；无歧义的格式问题不重开内容审核。返回漏题先补缺题，不重审已有有效结论。
2. C1沿用`candidate/match/pair_class/rationale`；C2沿用`question_id/decision/rule_code/rationale`，`decision`为`pass/reject/quarantine`。沿用现有审阅输入/schema，不因角色合并改数据字段。
3. C3沿用`question_id/evidence_grade/conflict/uncertain/citations/key_relations/rationale/rule_code`；每条citation含`rank/document_id/quote`。脚本保存search轮次映射及原文身份；不可混合不同窗口或把rank当全局唯一ID。题目、gold、理由不串题。
4. **R0**：原题/gold/split/来源及审计键不变；导出的policy prompt仍只有原题与协议。离线证据、C3理由、7B assistant回复只放sidecar，不能混入问题。
5. **R1**：复用身份匹配的现有reward接口固定样例回执，覆盖正确/错误/空/非法格式及适用的否定、附加结论、多答案、映射、单位。接口未变不逐批重跑；具体误判只修受影响类型，不把verifier当gold认证。已发现误奖/误罚、gold拼写歧义或答案覆盖缺口的受影响ID，保留E等级和原分数，按既有quarantine/staging去向暂缓训练就绪发布；不能沿用通用回执把它标成已解决。其他通过项立即发布。修正答案映射/评分须留变更依据并只回读受影响固定案例，不临场手改分数或自行扩写gold。
6. **R2**：当前通过未发布项组成不可变增量；发布器验证单题`teacher_7b_ref`、hash、实际prompt IDs、模型/预算/搜索以及引句在实际投递窗口。只对新增项执行真实`RLHFDataset`逐条读取和固定小批collator，确认无丢题/截坏输入、prompt无gold泄漏；不加载生成模型或优化器。
7. **R3**：回读实际数量、唯一ID、文件/manifest hash和loader回执后更新累计release与`CURRENT_RL_RELEASE.json`。账本/绑定/发布/累计合并单一写入，**前一命令真正Exit0才启动依赖命令**；尚在运行的poll返回不算完成。
8. 内容/隔离待定留staging；内容和隔离已过但loader环境不可用，也导出实物并标`data_ready=true/runtime_verified=false/training_ready=false`，不冒充可训练指针。已验证通过项继续发布，不能只交空manifest或blocked统计。

发布只检查新增范围，不复审旧release。每次接续先清待发布积压；连续两次新增0时，只盘点一次待发布队列，有就发布，无就推进缺失门，不另开审计。明确漏筛才定点撤回，旧原件和原因保留。

## 6. 必须保留的纠错与正反例

历史ID仅用于理解边界，**不授权重审、恢复旧查询预算或重新纳入撤回数据**。正常执行在相应首审内应用，不为下表另加审查门。

### 6.1 题目/gold与串题

| 例子 | 正确处理 |
|---|---|
| 12587问链球菌多糖，返回理由却写失神发作/EEG | ID/JSON正确也不能pass；对该ID当前门纠正一次，仍无效则隔离，不能只换理由保留pass。 |
| 13019折叠辅助蛋白、13045 Raynaud病因：开放题只列部分有效成员 | 检查范围/数量/选项与合理答案覆盖；缺口未解决则C2隔离，成员有证据不能修复评分覆盖。 |
| 15277：原题Hb `9.5 mg/dL`却按常见贫血数值解读；iron终答和窗口均正确 | C2仍须检查原题数值/单位；不能静默把mg/dL改成g/dL后发布。保留正确终答和有效E2，隔离原题单位错误，不回审无关旧数据。 |
| 16381：同组ABG写pH7.50、pCO2=20、HCO3=24 | 数值关系不自洽；不能只见低pCO2便按常见中毒题通过，也不能默改成相容数值。按已有C2数值检查隔离，不增加独立审核或重新生成。 |
| 16325：题问用药预防CAH发生，实际只有“预防virilization”的文献题名 | 预防疾病发生与预防其表现不是同一目标，不能以“经典考题”补齐。题名不当成已核实的教材机制；本题隔离，不推广成所有产前用药题错误。 |
| 15392：原题只写early positive HIV ELISA，C2自行加“第四代”后判anti-HIV Ab错误 | 未给检测代次/感染窗口限定，不得补前提；实际抗体ELISA片段支持该回答时，按具体范围/评分疑点隔离，不占真实错答配额。 |
| 15424：开放问中性粒细胞趋化介质，gold C5a；实际窗口同时列IL-8、C5a、LTB4 | 原题无补体等鉴别条件，不能以C5a确为有效成员便通过单一gold覆盖。保留C5a关系及原分，按已有C2范围规则隔离；无须新增审核层。 |
| 13064 T315I问drug of choice，理由仅说ponatinib可用 | 若缺分期、治疗背景或比较条件，首选限定未成立，隔离；不夸称gold必错。 |
| 13222 gold含V/Q mismatch with dead space ventilation | 附加机制方向冲突不能忽略；拒绝当前gold，不能只留正确主词后当E2。 |
| 13202增殖期内膜问首选激素 | 缺周期时点/无排卵条件，不得按gold补出病理性孕酮不足，隔离。 |
| 13969半径减少1/3而gold说阻力81倍；14006将破伤风痉挛说成瘫痪 | 前者剩2/3，按该关系为81/16；后者概念不等价。拒绝具体错误，不凭主题近似通过。 |
| 13714 IgA class-switch与分泌/成熟混淆 | 题目起始条件和单一IL-5评分有歧义则隔离；不能推广成所有IL-5题都错。 |
| 13960把药物抗菌机制与其遇酒不良反应机制混为同一所问对象 | 先看本题究竟问什么，不制造假冲突；窗口支持有效药物识别组成时可E2。 |

### 6.2 证据：既不放水，也不强求完整答案

| 实际窗口例子 | 正确分级/理由 |
|---|---|
| 12655测压→贲门失弛缓，缺地域感染病因；12688 CRL未达确诊阈值，缺复查间隔 | 已有中间诊断或排除立即确诊的关系，可E2。 |
| 12587只说蔗糖与龋病；13019只说折叠涉及二硫键 | 未支持细菌产物关系或折叠辅助蛋白主体，E1，不能凭gold跳到答案。 |
| 13458题干已有随机/盲法，片段只介绍另一随机试验 | 没有新增定义、关系或机制，E1；关键词不是新增证据。 |
| 12571已穿刺确诊感染，引用痛风的诊断抽液 | 不支持该题治疗性引流，E1。 |
| 13065常规复方OCP问题引用紧急LNG延迟排卵 | 同成分/终点不能无桥接迁移主要机制，E1。 |
| 13085缩瞳、心动过缓、出汗/分泌→抗胆碱酯酶机制 | 虽缺有机磷病因，明确共享中间机制可E2，不因来源病例不同全盘排除。 |
| 13278心衰→zone 3缺血性坏死，缺CCl4；13536 DKA→鼻脑毛霉菌病→眼眶侵犯，缺Rhizopus属级频率 | 支持必要组成或中间诊断，可E2；缺另一个组成/属级关系不自动推翻已过C2。 |
| 13625 Meissner为黏膜下神经丛；13797同源染色体配对与synapsis | 明确同义关系可正常推理，不要求答案逐字出现，不机械降级。 |
| 13581只支持t(15;17)未给精确带位；13587支持危象使用吗啡但无儿童/静脉等限定；13926支持HCV抗体后查RNA但未给PCR方法 | 已支持关键组成而不全覆盖，保留E2；不凭常识补齐后升E3。 |
| 13753支持肉毒中毒诊断而缺ACh释放；13722支持静坐不能而缺propranolol；13623只支持知情同意理解/能力框架 | 有具体中间关系可E2；最后一例不能据此宣称失忆必然无能力。 |
| 13792题干已给溶血、问实验室指标，片段只补HUS诊断 | 若未推进所问指标或新增有效机制，为E1，不以诊断相关代替所问关系。 |
| 13953同一窗口出现20mm与25mm阈值 | 不能用“25也大于20”化解标准冲突；未解决不得E2/E3发布。 |
| 14117引用生殖器疣HPV6/11和非黏膜角化描述，支持额头/头皮3/10型 | 部位/亚型适用桥接不足，不能凭“都是疣”通过。 |
| 13954近同义重复query仍未查到药物；14041检到部分鉴别而终答只给诊断 | 有效诊断/鉴别关系最多按实际给E2，不声称已完成治疗/指标证据；终答另记。 |
| 14101/14129命中不同文档；14058 spontaneous clearance与Transient infection被verifier判不等价 | 前者有效证据不必同ID；后者分开记检索、语义和原始分数，已知同义漏判训练前定点处理。 |
| 13962终答只有Dorsiflexion of the foot，缺loss | 不用CoT补否定，终答语义不确定；检索有效性独立记录。 |

### 6.3 执行错误

- **口头数量不当事实：**曾有6题口述E2/E3为4/2、实际文件为3/3；只采用经JSON/ID检查的文件和账本统计。
- **读取截断不重审：**长包两次读不全应机械分成4–6题读取，保持原ID/schema；不经CMD参数传大JSON，不另造医学审查。
- **receive未结束就close：**曾被ID守恒断言拦截；必须等实际Exit0再合并/收口，不能把轮询返回当完成。
- **不借格式修复改内容：**大小写/纯格式可按原文无歧义修复并留原件；串题、换证据、改变E等级或结论须记为受影响ID纠正。
- **不反复预检或重启：**非致命GPU日志不等于失败；npipe权限、服务端口、CUDA/显存分别处理，最多一次对应层重试。不抢停其他GPU工作、不重启正常Docker，不把NVML异常当所有数据无法导出的理由。
- **已通过马上出实物：**不等100题全审完、不等失败重试、不重审旧release，不以空manifest和“还需审查”收尾。

### 6.4 新增32题审计：误评分与覆盖反例（2026-10-21）

以下`prob`由本机现有`/verify`接口复现，阈值为`>0.5`；原始回执和逐题依据见[审计报告](../../../outputs/rl_lunamax_new32_audit_20261021/AUDIT_REPORT.md)。这些是已知受影响案例，不要求每批对全部题重跑verifier。

| ID与实际返回 | 必须避免的误判与处理 |
|---|---|
| 14156：gold=`Monocytic`，终答`Acute Monocytic Leukemia (AML-M5)`，prob=0.2583 | 完整亚型名称被误罚，不能当7B医学答错；记录别名/全称评分缺陷，修复前暂缓该ID训练就绪。 |
| 14211：gold误写`FTS-ABS`，正确终答`FTA-ABS`，prob=0.0106 | C2理由写对FTA却未处理导出gold的拼写歧义，不能静默按正确拼写理解后发布。窗口还列TP-PA等合理确认试验；应处理原题/gold覆盖，而非奖励错误拼写。 |
| 14268：题目问当前无症状胆结石管理，终答`Observation`，prob=0.0165 | 正确当前管理被较长的条件式gold误罚；同义策略与真实错误分开记，不能只看字符串长度/原分数。 |
| 14283：终答`Td`，gold=`Single dose of tetanus toxoid`，prob=0.3294 | 本题情境下的疫苗简称/规范联合制剂表达被误罚；核对具体情境及必要剂次，不把所有Td字符串无条件判对。 |
| 14170：问酶缺陷，终答`Homocystinuria`，prob=0.6077 | 只答疾病名，未答CBS酶缺陷，却获奖励；主题相关不等于所问对象正确，不能用推理补出酶名。 |
| 14165：要求的gold含ANA、抗Scl-70、抗着丝点，终答缺抗Scl-70仍获0.5832 | 先明确原题是在问完整集合还是有效成员；不能既以完整集合通过C2，又按部分集合宣称完整答对。范围不清则隔离。 |
| 14160：开放问旅行前疫苗，甲肝疫苗对MMR gold仅0.0053；14270：开放问达托霉素显著不良反应，嗜酸性肺炎对横纹肌溶解gold仅0.0023 | 两个gold本身可正确，但题目没排除具体合理替代回答。C2应发现缺选项/范围或reward覆盖；检到gold某个成员不能修补这个问题。 |
| 12955：只问噻嗪类常见副作用；rank1明确列低钾，rank2列gold高尿酸；终答低钾仅0.0017 | 不能写“虽有证据支持，但不是gold所以wrong”。这是具体合理替代答案被误罚，保留E3与原分、隔离，不占教师错答配额。 |
| 13487：题面限定与2,3-BPG的亲和力，终答仅`higher affinity` | 应承接题面对象判方向错误；不能从窗口/CoT补入未在终答出现的oxygen，将其重解释为正确或不完整。有效E2、判负合理可保留。 |
| 13242：search1/rank2称alprazolam为SSRI，search2/rank2称其benzodiazepine | 同次窗口在本题治疗类别上直接冲突；不能只看支持gold的rank就标conflict=false。保留原片段、评分并隔离，不临场修订教材。 |
| 14149/14241/14281：开放问某实验室发现/相似肾处理物质/下肢进一步发现 | 指向一种病或净分泌，只能说明某答案合理，不能自动确定唯一具体发现/物质；写出缺失的选项或评分范围，不能泛称“还有别的答案”。 |
| 14243：以“传统考试用法”证明高剂量DEX为唯一最合适下一步 | 本题缺具体定位检查背景/选项，且实际窗口提示抑制反应重叠；“可用于鉴别”不足以单独证明“最合适下一步”。不直接宣称该检查无用或gold必错。 |
| 14190：窗口支持单甘油酯＋脂肪酸，但未建立gold的2位特异性 | 保留有效E2并记缺口，不能补常识升E3；无须为此重跑7B。 |

正常的7B错误终答不等于坏RL问题；**已确认的gold/reward错误要先解决**。每批报告分别统计：冻结题数、实际7B题数、人工语义正确/错误/不完整/空/不确定、verifier接受数及已知误奖/误罚。正确率以实际完成并接受该项核对的题数为分母，不能用100题冻结数或32题证据通过数混算。没有语义核对的记unknown，不把verifier拒绝全部叫“7B答错”。

## 7. 运行入口与一次性接线状态

### 7.1 已验证能力与复用入口（2026-10-21核对）

- 共用入口：`MedSearch/data/cleaning_v2/rl_cleaning.py`，现有动作包括`freeze/c1/combined/teacher-input/teacher-bind/receive/close/query-parts/compare`。`--run`只传批次目录名；`freeze --rule-path`冻结实际规则路径/hash。新模式已禁止旧离线准入。
- 7B入口：`MedSearch.tools.evaluate_agent_rollout`已有11题真实运行成功，保存实际检索及投递窗口；不需要再证明7B是否能启动。
- **已贯通回执：**`rl_20261021_lunamax_batch01`的59题完成真实7B搜索，3个包发布11＋12＋9条并close；本轮审计重新核对32条prompt IDs、40段观察token及3个增量的真实loader。复用`teacher-input`准备输入、`teacher-bind --shard N`保存/绑定完成包。**§0A协作模式仅用该节独立模块`record_rl_preview`保存候选（不是`rl_cleaning`子命令，`--return-file`仅传文件名），随后等Sol发布回执；`receive/close`仅适用于另获授权的Luna端到端模式。**不手填引用绕过保护，也不再重复实现这条链路。
- 当前实际路径为`teacher/input_01.jsonl`、`teacher/rollout_01/`、`teacher/episodes/episode_01_001.json`、`review/input_01.jsonl`，后续包依次02/03。工程链路通过不代表内容或评分零缺陷，已知问题按§6.4局部处理。规则修订仅供后续新批冻结，不改写旧批快照/hash。

### 7.2 成功运行配置与CMD用法

工作目录为`D:\code_list\some tricks\LLMLeanring`，容器`llm-pytorch`、根目录别名`/workspace/llmlearning`。从根目录用`python -m`，不用PowerShell或子目录直接跑模块。

| 参数 | 现有成功值 |
|---|---|
| 7B模型/revision | `MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct` / `74d46a43224daa612b5f27174e1c58b35587ee92` |
| 检索语料 | `MedSearch/data/retrieval/medrag_textbooks.jsonl`，沿用BM25 Textbooks |
| 生成 | batch-size 1，temperature 0，top-p 1，seed 42；action 500、response 4096、context 4608 |
| 搜索 | 最多2次、top-k 3、每次整段观察500 tokens；8001 retrieve、8002 verify |
| 显存相关 | 1024批tokens、固定KV 1GiB、utilization 0.55；这是旧次空闲19092MiB时的成功值，先核对当次可用显存 |

以下为**已有7B命令的参数模板**。CMD中先设置`RUN`为新批目录名、`PACKET`为两位小包编号（如01）、`N`为该包实际C2通过数（通常20题，尾包按实数）；沿用`teacher-input`生成`teacher/input_%PACKET%.jsonl`，确认无旧输出和重复episode后才运行。前包通过项发布回读后再生成下一包。动作500是现有蒸馏模式的代码配置，记录实际值。

**完整复制命令，只替换批次/包/数量/日志路径，不凭记忆重组或省略环境开关。** `legacy_recheck01`曾两次漏掉`VLLM_WSL2_ENABLE_PIN_MEMORY=1`，导致WSL下vLLM在生成前报`UVA is not available`；补两个MEDSEARCH内存参数不能修复，完整命令恢复后20/20成功。这是漏参数，不应直接报成GPU/驱动异常。完整实际命令先记入本轮`COMMANDS.md`；共用的绑定/入账入口也继续使用成功的Docker环境。

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring"
docker exec -e MEDSEARCH_MAX_NUM_BATCHED_TOKENS=1024 -e MEDSEARCH_KV_CACHE_MEMORY_BYTES=1073741824 -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -w /workspace/llmlearning llm-pytorch python -m MedSearch.tools.evaluate_agent_rollout --model MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct --data MedSearch/data/cleaning_v2/%RUN%/teacher/input_%PACKET%.jsonl --distillation-dir MedSearch/data/cleaning_v2/%RUN%/teacher/rollout_%PACKET% --model-revision 74d46a43224daa612b5f27174e1c58b35587ee92 --retrieval-corpus MedSearch/data/retrieval/medrag_textbooks.jsonl --limit %N% --batch-size 1 --search-url http://127.0.0.1:8001/retrieve --verifier-url http://127.0.0.1:8002/verify --max-turns 2 --response-length 4096 --max-model-len 4608 --observation-length 500 --top-k 3 --gpu-memory-utilization 0.55 --temperature 0 --top-p 1 --seed 42 > MedSearch\data\cleaning_v2\%RUN%\teacher\generation_%PACKET%.log 2>&1
```

成功原命令/实测回执在[7B COMMANDS](../../../outputs/rl_batch06_7b_audit_20261021/COMMANDS.md)。新批可用不同小包路径分次生成，已完成episode不能重跑；不能等待全部生成完才保存已完成原件。

共用发布已有命令形式如下，`RETURN_FILE`是本批`review`下已完成raw文件名；必须先有真实episode绑定和对应审阅包，不是凭这个命令自动补齐缺门。`R1_REPORT`指向与当前reward身份匹配的已有效回执。

```cmd
docker exec -w /workspace/llmlearning llm-pytorch python -m MedSearch.data.cleaning_v2.rl_cleaning receive --run %RUN% --return-file %RETURN_FILE% --r1-report %R1_REPORT%
docker exec -w /workspace/llmlearning llm-pytorch python -m MedSearch.data.cleaning_v2.rl_cleaning close --run %RUN%
```

两条须分开执行并确认退出码；第二条只在整批去向完整且前者真正结束后运行。环境异常先保存结果，定位具体层，最多一次相应重试；不重复加载7B做无关预检，不关闭用户其他程序。现有服务正常就复用。

## 8. 长度、数据规模和停止条件

- RL交付的是唯一问题，不是SFT示范轨迹。无需教师终答全对，不追求所有E3；E2已足够有训练价值，但E1不能因想凑数而放入。500题用于首轮训练诊断、1,000题作为首版，之后是否扩至2,000–3,000依据开发集、覆盖与预算，数量本身不授权训练或扩池。
- 记录模型generated tokens（排除题目、检索观察、环境纠错、padding），检索tokens及完整序列另记；缺失`unknown`。可复用同配置、正常结束未截断参考集的P20/P80分short/medium/long，边界含等号归medium，记录参考身份和算法；不为分级新增生成或按比例凑数。
- 长度不代表难度，长短均不单独拒绝；无终答、截断、环境失败不混入完整长度参考分布。达到上限不自动证明截断。后续学生难度需用已授权学生真实尝试统计，不凭7B长度替代。
- 训练中的长度惩罚、loss、重采样、OPD及学生评测另行授权；此处不改训练策略。训练前核对实际`TRAIN_FILE`与manifest及已知评分问题，旧成功命令不证明新版对齐。
- 本批结束要求：`冻结输入 = 已发布 + reject + quarantine + 未完成/阻塞`，ID互斥且覆盖；待发布单列不得藏入完成数。报告新增/正式累计、E2/E3、未通过具体原因、环境阻塞和实际路径/hash。
- 检查点只保留：当前阶段/包、已返回未绑定ID、已通过待发布ID、去向计数、剩余原预算、下一条实际命令/文件、真实阻塞、用户停止点。保留账本中的原始证据，不把大段推演写成恢复材料。
- 用户终点到达或实际额度不足时，先保存返回并发布可发布项，再写可恢复检查点暂停；不要创建自动唤醒任务。没有已完成返回时，不靠重复状态更新或另开审计消耗额度。

## 9. 本书落盘时的接续快照与启动提示

2026-10-21快照，实际执行仍先查指针：**正式RL 234（E3=120/E2=114），SFT1000冻结。** RL234为旧规则基线，不声称经过新7B必经门。batch06原新增51全部撤回，不默认重洗；旧285/283文件及11题7B诊断保留追溯，不作为当前入口。

- 正式指针：`MedSearch/data/cleaning_v2/CURRENT_RL_RELEASE.json`。
- 当前RL：`MedSearch/data/cleaning_v2/audit_20261021_rounds20_question_only/release/rl/train.parquet`，SHA256 `9938d4a9fb757020cd27b1e97b48cedbf834b8d7639718a658a0cdeebc826b3e`。
- 最新撤回/恢复回执：`MedSearch/data/cleaning_v2/rl_20261021_batch06_withdraw51/receipt.json`及同目录checkpoint、报告；带`WITHDRAWN.md`的旧批不能恢复收件/发布。
- 选下一批时复用已有冻结记录和来源游标，跳过已冻结/撤回ID。旧批可供脚本读取游标，不能将它作为待恢复批次，也不为“重新从原库挑题”重复纳入被撤回51题。
- 本次只新增本书与交接说明，未启动新清洗、模型或训练，没有新增RL。

下面提示词供用户填写授权数量后发起下一轮；**此处不是启动命令**：

```text
使用GPT-6 Luna Max单主代理，按 MedSearch/docs/superpowers/plans/2026-10-21-rl-cleaning-lunamax-only.md §0A 清洗下一批100道原题，候选落盘后暂停交Sol复核。禁止子代理、自动任务和旧池重审。
先读当前release和最新检查点，处理已返回未绑定；一次合并题目/gold首审，7B原题盲输入真实搜索，审实际投递窗口。沿用每题1 episode/2 searches预算，不走离线检索准入、不重置次数。
复用teacher-input/teacher-bind，每个C3包保存raw后直接调用record_rl_preview落账与staging；不要调用会正式发布的receive/close。已知误评分、缺题面条件、范围或桥接未定的ID隔离，其他E2/E3立即保存为training_ready=false的候选。100题去向完整后交付候选清单与原件引用，标awaiting_sol_review并暂停；正式新增为0是本阶段职责结果，不能因此重审或另开任务。
```
