2026-10-27 batch31：100题完成，5发布/74隔离/21拒绝；35真实7B。Sol抽拟发布3/5、失败2，未发现可恢复误杀。累计p108 393=368正确+25错答，距500还差107正确，loader/hash回读通过；batch32已冻结并继续。详见 MedSearch/data/cleaning_v2/rl_20261027_terra_batch31/RUN_REPORT.md。

2026-10-27 batch30：100题完成，7发布/74隔离/19拒绝；49真实7B。Sol正向抽3/7、失败抽2，19886证据等级修正E2且维持隔离。累计p106 388=363正确+25错答，距500还差112正确，loader/hash回读通过；batch31已冻结并继续。详见 MedSearch/data/cleaning_v2/rl_20261027_terra_batch30/RUN_REPORT.md。

2026-10-27 batch29：100题完成，6发布/70隔离/24拒绝，45真实7B；Sol抽拟发布3/6及失败2条，未发现误杀。累计p103 381=356正确+25错答，距500尚差119正确；真实loader/hash回读通过，batch30已冻结并继续。详见 MedSearch/data/cleaning_v2/rl_20261027_terra_batch29/RUN_REPORT.md。

2026-10-27 batch28：100题清洗完成，6发布/76隔离/18拒绝；44真实7B、62搜索、8313生成tokens、2未搜索。Terra主审，Sol抽拟发布3/6及失败2条，未发现误杀。累计p100 375=350正确+25错答，training_ready=false，距500还差125正确；真实loader与数量/hash回读通过。报告 MedSearch/data/cleaning_v2/rl_20261027_terra_batch28/RUN_REPORT.md；本轮结束暂停。

## 2026-10-23：7B评分子进程验收通过，继续P6桥接

- 主对话按既定工程书补辅助MedSearch/distillation/teacher_process.py，复用已有load_teacher/score_failed_batch；每批一次独立进程，空失败跳过，退出0才读CPU分数，非零退出传出异常，任务文件不带gold/references。未改写用户核心函数或loss。
- WSL真实本地7B BF16验收通过：2条合成失败轨迹、518模型token，含4096上下文＋500目标token；峰值allocated16.126GiB，加载＋评分49.831秒，退出0且PID消失，整卡占用前后均1.637GiB。正确/padding项与observation保持零OPD。runtime只记录资源调用，没有加载真实学生/optimizer或启动Ray/FSDP/vLLM；完整P6尚未验收。
- 联合CPU真实小模型子进程和调度/评分/回填测试28 passed，2条tqdm弃用警告；日志tmp/teacher_process_regression_20261023.log。GPU日志tmp/teacher_process_7b_20261023.log；机器报告MedSearch/outputs/distillation/teacher_process_smoke_20261023/acceptance.json，详见MedSearch/docs/audits/teacher_process_acceptance_20261023.md。
- 当前唯一学生编码任务：在V1 trainer_base.py新增_compute_failure_teacher_scores(self,batch,score_fn)，从TransferQueue读当前完整批次，调用run_failure_teacher_phase，只将teacher_logprobs/opd_mask写回原keys/partition并返回原batch。该方法单独验收后再接_step_once启用分支，随后接actor合并loss。未启动训练或调用付费API，未修改RL/dev数据或发布指针。

## 2026-10-22：dev题答与教材证据清洗完成（仅dev）

- 当前仍为阶段5 MedSearch-R1；本轮唯一任务是清洗原dev396，不审查/修改RL数据或其发布指针。
- 已完成全部396题分流：137保留、238隔离、21排除（13同题边界冲突＋8题答/数据冲突）。不以7B/学生/verifier表现筛选，没有7B/学生推理、训练或定时任务。
- 新dev：`MedSearch/data/dev_cleaning/dev_evidence_20261022/release/dev.parquet`；SHA256 `47f11019c82dc63e354e643ce2c1c3229ca50571a1dcb90d4c8489c631052c91`。原dev、题目/gold/prompt不改。
- 178引用/176段已逐字回绑；真实veRL RLHFDataset读取137/137、四条collator及512-token门通过，最长281 tokens。语义审阅为当前主代理模型判断，未经医学专家复核。
- 逐题理由、证据、同义表达/答案覆盖清单、原题盲检索和500-token窗口、隔离依据及hash见同目录RUN_REPORT.md及release/manifest.json。qrels为已知正例非穷尽；gold-aware查找不算Agent召回。
- 后续读取规则：`MedSearch/docs/superpowers/plans/2026-10-22-dev-evidence-cleaning.md`。普通策略输入不得预塞gold证据/评分清单；固定名单后才能做同预算对照，最终超越教师结论必须由独立冻结测试验证。
- 本轮已收口停止；不自动开始评测、训练、扩池或处理隔离题。RL数据审查属于其他工作，本轮只使用开工时训练实物作只读dev查重边界。

## 2026-10-21：新增32条RL审计与误评分反例修订完成（未启动下一批）

- 上轮`rl_20261021_lunamax_batch01`已完成100题去向：新增32（原E3=17/E2=15），reject10、quarantine58；当前正式指针仍266，release SHA256为`02ce30699a0278b756b22532889f40eead92b7821379722e25dea1e07ae9082b`。本次审计未改正式release、gold或原评分。
- 按用户授权全量审计新增32：32个原题/7B盲prompt绑定、40段实际观察token、52条引句通过；11/12/9条增量均重新通过真实RL loader及固定collator，旧234条保持原样。14190证据仅覆盖单甘油酯＋脂肪酸，建议E3降E2，仍有RL价值。
- 7条已复现评分/覆盖缺陷：14156、14160、14170、14211、14268、14270、14283；另5条范围/完整答案口径须明确：14149、14165、14241、14243、14281。审计建议受影响题修复/澄清前暂缓训练；并未自动撤回，也没有声称现有verifier已修好。其余20条未发现阻断问题。
- 依用户要求，Luna专用清洗书§6.4补入误奖/误罚与答案覆盖案例，原C2/C3中顺带核对既有终答与原评分，已知异常局部隔离；通用书同步引用。纠正了“接口尚未接通”和旧输入路径模板，复用已成功的`teacher-input/teacher-bind/receive/close`。不新增二审、子代理或7B重跑门；旧批规则快照/hash不改。
- 关于教师每100题只错3–4条，用户正在讨论目标，尚未确认新准入比例。现有verifier接受率不能当真实正确率；先区分语义正确、错误、不完整、空、不确定和误奖/误罚。没有授权新批清洗、改评分器、训练或定时任务。
- 审计报告：`MedSearch/outputs/rl_lunamax_new32_audit_20261021/AUDIT_REPORT.md`；逐题32条结论：`audit_decisions.jsonl`；实际loader/token/hash回执：`mechanical_audit.json`；评分定点复现：`additional_verifier_probes.json`。下一步等待用户确定新批范围及教师成功优先策略。

## 2026-10-21：Luna Max单主代理专用清洗书已落盘

- 按用户要求另写专用入口：`MedSearch/docs/superpowers/plans/2026-10-21-rl-cleaning-lunamax-only.md`。单个Luna负责接续、C1疑似对、题目/gold一次合并首审及7B实际窗口C3；不使用子代理，不切换Sol；本地7B与共用脚本负责真实搜索及机械入账/发布。
- 保留有效E2/冲突与跨对象边界、原题盲输入、预算、旧纠错正反例和“保存→校验入账→即时发布→真实loader→数量/hash回读”。取消角色往返、例行二审、重复历史加载；批量审阅而非逐题调度，实际节省额度尚未实测。
- 已写明当前工程差异：共用入口规则路径仍固定为通用书，尚无rule-path参数；7B单题原件/窗口包/teacher_7b_ref统一绑定与收口尚未按新批贯通。下次用户授权后在首个实际小包最小接通，成功发布回读再处理剩余题，不把已有保护断言说成全流程已完成。
- 本次仅新增专用书并更新本摘要；未启动新清洗、7B或训练，无新增RL。CURRENT仍为234条旧规则基线（E3=120/E2=114），SFT1000冻结，撤回51题不自动重洗；下次启动范围/终点以用户命令为准。

## 2026-10-21：用户指定整批撤回51条，恢复7B检索准入

- 已撤回batch06原新增51（本次移出活跃49、此前已隔离2），当前正式RL恢复234（E3=120/E2=114），本批正式保留0，SFT1000/hash不变。没有删除原件，没有把51条默认放回重洗队列。
- CURRENT恢复MedSearch/data/cleaning_v2/audit_20261021_rounds20_question_only/release/rl/train.parquet，SHA256 9938d4a9fb757020cd27b1e97b48cedbf834b8d7639718a658a0cdeebc826b3e。234条逐行/ID与批前基线及原283去掉本批49完全一致；manifest/hash、既有真实loader234回执与SFT hash已核对。
- 清洗书主动流程已替换为C1/C2→逐题真实7B盲episode（默认1次、最多2次搜索）→Luna Max审实际投递窗口→即时发布；旧Luna离线查询准入移除。RL仍接受有效E2而不要求教师终答全对，问题/答案/评分分开记，冲突阈值和跨对象不当迁移明确禁止。
- 共用freeze新增7B必经模式；新批旧离线查询入口禁用，发布核对真实episode/prompt IDs/实际窗口引句，旧批WITHDRAWN禁止恢复收件/发布/累计合并。7B runner在11题已有实际成功证据；新模式统一收件/teacher_7b_ref绑定尚需下次首个小包接通，未伪称全流程已跑通。
- 撤回清单、数量/hash及恢复点在MedSearch/data/cleaning_v2/rl_20261021_batch06_withdraw51/。原285/283文件及7个增量加WITHDRAWN标记保留追溯。当前234为旧规则基线，不宣称已按新7B门复验；没有启动新清洗、7B或训练，等待用户命令。

## 2026-10-21：batch06新增数据20%真实7B抽检完成，已暂停（历史）

- 从新增51条均匀随机抽ceil(20%)=11条（seed2026102107，原E3=4/E2=7）。真实Search-R1 Qwen2.5-7B，temperature0，max_turns2、top3/整段500tokens，无gold/离线证据预塞；11条全部真实搜索共14次，8条合法终答。
- 原批准引句命中3/11；逐题实际窗口复核3条完整支持、5条部分有效关系、1条跑偏、1条阈值冲突、1条跨对象适用性不明。冻结verifier仅2/11判对，语义明确正确3/11，13962缺loss表述待定；14058同义答案被verifier错判，未在本次偷偷改reward。
- 发现原内容审批13953将20/25mm阈值冲突当一致、14117将生殖器疣片段迁移到额头/头皮的证据不足，均定点隔离。原285条版本保留；当前正式283（E3=137/E2=146），本批净保留49（E3=17/E2=32），SFT1000不变。真实纠正loader283及collator4、ID/hash回读通过。
- 当前release在MedSearch/data/cleaning_v2/rl_20261021_batch06_7b_audit/release/rl/train.parquet。评测全轨迹/窗口/原始分数/逐题对照与REPORT.md在MedSearch/outputs/rl_batch06_7b_audit_20261021/。未改清洗内容门或启动训练，未扩新候选。
- 本次未满足“7B均可检索并利用证据”的验收；不据一次11题结果宣称全池无用。建议先修复审批边界及verifier具体漏判，再讨论下一批；等待用户命令。

## 2026-10-21：batch06新100题已收口，已暂停

- 本批新增RL51（E3=18/E2=33），正式累计285（E3=138/E2=147）；100=51发布+6拒绝+43隔离，未完成/待发布/待续/环境阻塞0。SFT1000及hash不变。
- 按精简清洗书执行：C1 Luna Max26对→4题重复排除；主代理96题题面/gold合并首审66通过/28隔离/2拒绝；Luna Max C3常规首审，Luna轻度独立盲改写。Q0/Q1=66/36，102次实际检索0传输重试；无第三次查询、7B、训练或自动任务。
- 通过即发布7次，真实新增RLHFDataset逐条及collator、累计数量/唯一ID/hash回读通过。正式文件MedSearch/data/cleaning_v2/rl_20261021_batch06/release/rl/train.parquet，SHA256 dc74510b5d691869e39bb9f9e8e6adb598999ede5a7aaee7049db55ae1b8e153。
- 冻结到收口1324.1秒（22.07分钟），138.66条新增/小时，仅为本批不同题目的实际吞吐；tokens/额度成本unknown，不推断总体准确率或额度节约。
- 13960一处问题对象混淆作一次定点纠正后按E2发布，原返回保留；1次提前close被ID集合保护断言拦截，待最后receive真正Exit0后close成功，详见COMMANDS。没有重启环境或重审旧池。
- 本轮完整报告、逐题排除、样例和恢复位置均在rl_20261021_batch06的RUN_REPORT.md、excluded.jsonl、PREVIEW_10.md及continuation_checkpoint.json。等待用户命令后再启动新批。

## 2026-10-21：两轮各10题新抽查与全量输入纯净性检查完成

- 本次样本避开上一轮已查20题，seed2026102102混排。两批各E3/E2=5/5，证据质量接近；旧批13714有IL-5/TGF-beta语义/单一gold评分歧义，已定点撤回。13587、13581、13926由E3改E2，仍保留。当前正式RL234（E3=120/E2=114），SFT1000不变。
- 原235条逐条原始题字段/单用户prompt精确比对通过，题干回复/推理标记扫描0；实际loader的raw_prompt只含固定协议+原题。golden_answers/reward_model为评分标签，不预塞给policy；无教师assistant回复、离线证据或审批理由。修正后234条真实loader与collator4、数量/ID/hash回读通过。
- 当前文件MedSearch/data/cleaning_v2/audit_20261021_rounds20_question_only/release/rl/train.parquet，SHA256 9938d4a9fb757020cd27b1e97b48cedbf834b8d7639718a658a0cdeebc826b3e；旧235/236版本保留。报告audit_20261021_rounds20_question_only/REPORT.md及逐题results_with_provenance.jsonl。样本小，不声明总体医学准确率或两批质量相等。
- 本次完成并暂停，无新候选、生成或训练；后续接续读CURRENT_RL_RELEASE与本次continuation_checkpoint，不以旧235条记录当当前状态。

## 2026-10-21：第二轮100题与质量比较已完成，已暂停

- 最终净新增RL32（E3=18/E2=14），正式235；100=32发布+16拒绝+52隔离。未完成/待发布/环境阻塞0，SFT1000及hash不变。
- 初次清洗1396.8秒（23.28分钟）；按最终净新增折算82.47条/小时，上轮64.16。前置接线126.2秒、用户指定比较及纠正约12.97分钟单列。不同题批不是控制实验，tokens/额度成本未知。
- 两轮各随机10题混排C3：原始盲复核上轮7/10、本轮9/10；一次定点处理4项分歧，旧3题符合原E2中间关系定义保留，新13792缺新增实验室指标关系撤回。最终样例上轮10保留、本轮9保留1撤回；不推断整体正确率或独立gold真实性。
- 正式路径MedSearch/data/cleaning_v2/rl_20261021_batch05/release/rl_comparison_corrected/train.parquet，SHA256 de2dc7ab57292d472f80815dc4f6c771779e917eb32e56cc1d6ae1cc8f0f61cc；实际RLHFDataset逐条235与collator4通过，数量/ID/hash回读通过。原236条文件保留，CURRENT已更新。
- 报告batch05/RUN_REPORT.md、COMPARISON.md；原始盲评及裁决comparison/，逐题排除excluded.jsonl。下一步等待用户命令，不启动新池、7B或训练。

## 2026-10-21：按精简书执行第二轮100题（历史启动记录）

- 用户授权下一100题及与上一轮效率/质量对比。新批rl_20261021_batch05，源ID13763–13942，基线RL203/SFT1000；冻结规则为合并首审、默认Q0/Q1两次；本批完成后暂停。
- 已新增共用rl_cleaning入口，接通合并收件、冻结预算读取及结果绑定→发布→回读顺序调用；本批实际运行验证各门，未更改旧batch04的三次预算与已发布数据。语义召回top10/.92已实际返回12个疑似配对，Luna Max只判C1，主代理随后直接合并C2。
- 本轮比较另记一次性接线耗时与冻结到收口时间；通过率/E3占比不当医学正确率。当前以新批账本/checkpoint和CURRENT实物为准。

## 2026-10-21：用户批准RL清洗流程精简，已更新项目书

- 已直接替换清洗书对应条款：500唯一问题试训练、1,000首版、之后按效果/预算扩2,000–3,000；取消先满3,000/5,000的前置目标。
- 新批题面/gold由Sol轻度一次合并首审，取消独立题面Q1盲审；Luna Max保留C1/C3，简单盲查询交独立Luna轻度；新批默认Q0+Q1两次，第三次仅专项授权，旧批原预算不改。
- E2/E3门槛、gold可靠/隔离/盲查询、实际窗口、引用、真实loader及数量/hash底线保留；通过项即时发布，独立只读任务可继续，账本/发布单一写入，不确定项不默认增加审核层。
- 本次只改项目书与进度，未启动清洗/训练。真实共用脚本尚有合并收件、读取冻结两次预算及统一发布编排差异，已在清洗书§6.1逐项标明；下一次新批执行前一次性接通，不冒充已实现，不复制新批脚本或重审旧池。
- 正式RL203（E3=105/E2=98）、SFT1000及其指针/文件hash不变；batch04保持paused_batch_complete，旧6题不恢复。接续入口：MedSearch/docs/superpowers/plans/2026-09-29-sft-environment-data-review.md §8。

## 2026-10-21：RL batch04新100题收口，已暂停

- 新100题收口：新增RL39（E3=22/E2=17），正式累计203；reject14、quarantine47、环境阻塞0，未完成/待发布/待补检索0，已暂停。
- 按更新清洗书执行C1/Q1→主代理C2→有限盲Q0/Q1/Q2→Luna C3→即时发布；没有整包例行二审。
- 正式路径：`MedSearch/data/cleaning_v2/rl_20261021_batch04/release/rl/train.parquet`；SHA256 `51d88256293323011ec4d8c3f943895e4173f848f53931eef1f6596122310f79`。
- 数量/hash/唯一ID/真实新增loader回读通过；SFT1000及hash不变。旧6题已按用户意愿放弃，未重置预算。
- 过程、逐题理由、品鉴和恢复位置：`MedSearch/data/cleaning_v2/rl_20261021_batch04/RUN_REPORT.md`、`execution_metrics.json`、`excluded.jsonl`及`continuation_checkpoint.json`。没有启动下一批、7B、训练或自动任务。

## 2026-10-21：P6教师阶段调度函数复验通过

- 用户已将score_fn返回值赋给scores。WSL Docker复验test_failure_teacher_phase.py与test_distillation_hook_jobs.py：25 passed，退出0；日志tmp/failure_teacher_phase_recheck_20261021.log。
- 全对跳过教师、混合/全错组正确回填、评分异常恢复学生并传出异常均通过；原批次身份、顺序、response_mask和advantages保留。测试使用真实veRL TensorDict及CPU模拟评分，不代表真实7B子进程或V1训练接线已验收。
- 下一步仍需补辅助教师评分子进程，并在V1循环接入调度、持久化新增字段到TransferQueue，之后接actor合并loss；本次没有改写用户核心函数或启动训练。

## 2026-10-21：P6教师阶段调度函数首次验收待修正

- 用户完成run_failure_teacher_phase，主对话在现有WSL Docker用真实veRL TensorDict及CPU模拟评分验收：2 passed、2 failed。全对跳过教师、评分异常恢复学生通过；混合组和全错组的OPD分数回填失败。
- 唯一定位：score_fn返回值赋给score，最终attach_teacher_scores读取的scores仍是初始空列表，导致失败项teacher_logprobs/opd_mask也全零。用户核心实现未被主对话改写；将该赋值变量改为scores后复验。
- 新增验收MedSearch/tests/test_failure_teacher_phase.py；日志tmp/failure_teacher_phase_acceptance_20261021.log。未加载7B、未进行GPU训练、未调用API；V1调用及actor loss接线尚未实施。

## 2026-10-21：P5运行闭环、checkpoint保存/原生重载/动作推理通过

- 用户授权后，在现有WSL Docker完成真实Ray/FSDP/vLLM单步纯GRPO：一题双rollout，save_freq=1，训练退出0。模型、optimizer、RNG/scheduler、dataloader状态均落盘；最终原生resume及一题搜索→答案推理退出0。
- 两条训练reward均为-1，GRPO advantage/grad_norm为0，LoRA参数差为0；optimizer step为1。本次证明运行与保存恢复链路，未证明非零学习更新。旧dev一题功能检查acc1，不是完整dev评测。
- 为通过原生重载，veRL checkpoint模型torch.load增加CPU暂存，补丁MedSearch/patches/verl_fsdp_checkpoint_cpu_load.patch。保留CUDA直接重载失败、热改运行脚本导致wrapper127和W&B退出清理异常日志；最终使用固定脚本副本重跑通过。
- 产物MedSearch/outputs/distillation/grpo_p5_20261021/checkpoints/global_step_1；最终机器报告同目录P5_ACCEPTANCE.json。命令、资源与证据见MedSearch/docs/audits/grpo_p5_acceptance_20261021.md。
- 当前进入P6准备：7B冻结评分而非训练7B；失败提取、评分、回填及loss辅助函数已有，V1循环教师阶段和actor实际loss仍未接入。下一项核心任务是失败批次串行教师评分接线；继续由用户写核心代码、主对话指导验收，不直接长训。

## 2026-10-20：reward已通过；同卡脑电训练期间停止P5 GPU验收

- 用户说明同卡有脑电模型训练。主对话已仅终止本次grpo_p5_retry03_20261020的driver PID43670，保留日志，未向其他进程发信号；未停止全局Ray集群或脑电训练。对应exec退出134由主动SIGTERM触发，原生进程检查未发现GRPO driver。
- 本次用户reward修改17项测试通过的结论保留。P5已实际加载3B基座、定稿SFT adapter并建立FSDP，但尚无rollout、更新、checkpoint保存/重载/动作推理验收；不称P5完成，不开启长训。
- vLLM比例式KV预算在共享GPU环境下初始化失败；小并发/eager尝试仍失败。辅助脚本现有512MiB显式KV配置在用户提示后尚未验收，GPU资源合适时先复验，不冒充修复成功。详情MedSearch/docs/audits/reward_acceptance_20261020.md；没有自动重试或定时任务。

## 2026-10-20：用户reward修改验收通过，进入P5

- 已核对用户compute_verifier_outcome_reward修改：读取extra_info.final_answer，无答案直接返回原失败结果，HTTP只提交最终答案；主对话未改写核心实现。
- WSL验收17 passed，1条既有Ray弃用警告。日志tmp/reward_acceptance_20261020.log；报告MedSearch/docs/audits/reward_acceptance_20261020.md。临时本地HTTP服务实测无答案不请求、有答案请求体正确、轨迹全文的检索/其他answer标签不影响结果，未调用付费API。
- 当前唯一任务进入P5：一题双rollout、单步纯GRPO、save_freq=1，之后保存/重载/一次动作推理。首次真实启动发现ppo_mini_batch_size按题数计的配置错误，辅助脚本已改为1，实际mini-batch仍为两条轨迹。重跑日志tmp/grpo_p5_retry01_20261020.log。P5尚未验收，不进入较长训练。

## 2026-10-14：最终答案输出与原生reward传递通过

- 用户授权主对话完成MedSearchAgentLoop.run的final_answer初始化、两个answer分支赋值和extra_fields返回。正常与预算结束作答均返回内容；检索answer标签不能形成最终答案。
- 复用veRL原生NaiveRewardManager传递tool_extra_fields→extra_info，辅助GRPO launcher改为reward.reward_manager.name=naive；GRPO、不滤全错组、无reference与单步save_freq=1保持原配置。没有修改veRL源码。
- WSL命令：`C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\loop_reward_fields_green_20261014.log python -m pytest MedSearch/tests/test_agent_loop_smoke.py MedSearch/tests/test_teacher_trace_export.py MedSearch/tests/test_reward_wiring.py -q`。16 passed，1条现有Ray弃用警告；launcher bash -n通过。实际veRL manager回调与原token/mask/logprob验收；未调用verifier API、未运行GPU训练。
- 当前唯一学生任务：MedSearch/rl/reward.py::compute_verifier_outcome_reward从extra_info读取final_answer，替换全文答案正则；无答案直接返回原失败reward，有答案只把该内容发给verifier。验收后进入P5真实RL单步与checkpoint保存、重载、动作推理，再做P6失败OPD接线。

## 2026-10-14：动作解析本土化通过，继续RL答案传递

- 用户授权主对话修改解析，并要求MedSearch独立发布无需Search-R1运行依赖。`rl/protocol.py`本土化Search-R1首个完整动作提取逻辑；非空内容要求保留，移除唯一动作/额外标签数/尾随文字限制。离线capture重复解码检查已删除。
- WSL Docker验收：`C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\searchr1_protocol_green_20261014.log python -m pytest MedSearch/tests/test_teacher_trace_export.py MedSearch/tests/test_agent_loop_smoke.py -q`，11 passed，1条现有Ray弃用警告。新增两项在修改前实际失败，修改后通过；真实tokenizer同token句号边界、原IDs/mask、训练loop搜索与logprob前缀通过。
- 当前唯一学生核心任务：`MedSearchAgentLoop.run`初始化final_answer=None，在现有两个answer分支赋值cur_action.content，并通过extra_fields返回。该函数尚未改写；reward桥接及P5实际RL保存、重载、动作推理仍待验收。

## 2026-10-14：两版SFT定稿，回到RL代码构建

- 用户选定CE epoch3/checkpoint375与CE+Top32 epoch8/checkpoint1000；完整checkpoint保留并逐文件hash核验。索引MedSearch/outputs/distillation/FINALIZED_SFT_CHECKPOINTS.json。
- 清理其他本轮SFT权重及非选中版本完整rollout，共32个目标，释放787.231MiB。对照评分、配置、日志、报告以及输入数据和基础/教师/verifier权重保留。
- P5脚本已接入veRL原生model.lora_adapter_path。两版Hydra配置解析通过；读取当前RL正式correction04发布164题，hash验收。没有运行RL GPU训练，P5尚未完成。
- 代码核对发现loop尚无final_answer、reward仍扫描全文。1342进一步定位为离线capture额外检查误拒绝同token尾随句号，训练loop字符截断后能接受，不据此增加严格门槛。下一项唯一学生编码任务：在MedSearchAgentLoop.run记录已有解析器接受的final_answer并返回到extra_fields，避免reward从含检索结果的全文抽取答案。之后桥接reward消费，再做P5单步。
- 既有失败评分/回填/OPD/合并loss模块已完成，但上游运行接线未完成；V1真实hook位置是_balance_batch之后、_compute_old_log_prob之前。
- 任务详解MedSearch/docs/audits/rl_next_task_20261014.md。核心实现由用户编写，助手负责辅助配置、审查与测试。

## 2026-10-14：8epoch冷启动回答质量抽查完成

- 用户询问继续增加SFT轮次的收益与生成质量；本次仅审查已有产物，没有追加训练。
- 按固定seed20261014分层抽查20条唯一ID，诊断结果见MedSearch/docs/audits/sft_epoch8_quality_20261014/REPORT.md及sampled_cases.jsonl。
- Top32第7→8轮训练总loss窗口均值0.44211→0.43909（约-0.68%），调度已到尾部；dev非单调，不能断言继续无收益或已经完全收敛。建议结束本轮冷启动并优先推进既定RL接线验收。
- epoch8最终答案被环境接受392/396；107题两次搜索中48题重复查询且观察hash相同。发现答案正确但病因编造、被局部词面噪声带走、拼接无关检索结果及预算结束仍搜索等案例。自动答案得分不代表推理真实性。
- 新诊断“环境接受答案且verifier通过”：Top32 epoch8=99/396，CE epoch3=96/396；与原100/97口径都净差3题，初始化候选不变。原始评分、checkpoint和数据发布均保留。
- 下一步仍是用户负责核心RL接线，助手讲解、审查与测试；当前没有启动RL或新增推理评分器。

## 2026-10-13：纯CE与Top32八组对照完成

- 两组1/3/5/8epoch各396题rollout先保存后评分，全部完成，数据/训练调度/解码/评分实际核对一致；失败题不剔除。
- top32 e1=85/396, top32 e3=96/396, top32 e5=92/396, top32 e8=100/396, ce e1=85/396, ce e3=97/396, ce e5=91/396, ce e8=91/396。
- 当前选top32 epoch8作为RL初始化候选；这是固定dev和本地verifier上的选择，非独立test或真实下游RL结论。
- 完整checkpoint保存、重载动作、adapter生效和W&B服务器finished均已核验；未启动RL，veRL加载尚未执行。
- 报告MedSearch/outputs/distillation/sft_loss_comparison_20261013/REPORT.md，候选见selected_checkpoint.json。W&B：https://wandb.ai/lixle-northwestern-polytechnic-university/medsearch_r1/runs/hcr2rry6。
- 下一步在用户授权的RL入口验收该初始化；保留两种loss的全部结果供下游受控比较。其他清洗发布指针未修改。

## 2026-10-13：追加纯CE的1/3/5/8epoch对照（运行中）

- 用户授权仿照CE+Top32完成纯CE训练和dev比较。Top32四组已完成，verifier判对85/96/92/100（各396题），其8epoch是该组当前候选；总体选择等待CE完成。
- CE数据由同1000条缓存只删除target_positions/teacher_topk_ids/teacher_topk_logprobs三列而得，剩余所有列逐值一致，未重rollout/重清洗；SHA256 647cb87dbdbb96cfb8bbcb582c579428ef2ec59e7be29920255dac3ee69739c3。
- 沿用用户已写DistillationTrainer纯CE分支。配置除数据/输出/名称外完全一致；两组都是1epoch单轮cosine后原生resume扩至累计8epoch/1000步，同一seed20261013。
- 纯CE真实3B最长轨迹保存/重载logits与动作一致、合法动作、续训参数更新非零已验收。报告MedSearch/outputs/distillation/sft_ce_checkpoint_smoke_20261013/report.json。相关6项回归通过。
- 单卡pipeline运行中，各epoch先完整保存dev396题rollout再同条件本地verifier评分；检查点完整状态逐轮保存。无付费API、未启动RL。
- 命令：`C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\sft_ce_pipeline.log python -u -m MedSearch.tools.run_sft_comparison_pipeline --method ce`。
- 产物根MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013，完成后汇总8候选于sft_loss_comparison_20261013；当前不宣称CE优劣。

## 2026-10-13：SFT四组dev对比完成，推荐epoch 8为RL初始化候选

- 1/3/5/8epoch各396题的搜索rollout完整保存后本地评分，ID/顺序/hash回读一致；数据与评测参数固定，检索异常0，不过滤失败题。
- verifier判对：epoch1=85/396, epoch3=96/396, epoch5=92/396, epoch8=100/396。按最高dev判对数/完全同分优先较早轮次，选epoch8。这是当前dev选择，非独立测试或医学专家结论。
- 累计1000步、8epoch完成；后续七个完整checkpoint均保留。前1epoch旧cosine，后续扩展1000步映射；已记录调度变化与真实更新验收。
- 候选路径/sha见MedSearch/outputs/distillation/sft_epoch_comparison_20261013/selected_checkpoint.json；重载验证adapter确实生效并生成合法search动作；veRL初始化和RL尚未执行。
- W&B dev：https://wandb.ai/lixle-northwestern-polytechnic-university/medsearch_r1/runs/bs5y1ka2；训练：https://wandb.ai/lixle-northwestern-polytechnic-university/medsearch_r1/runs/1n9661zq。报告：MedSearch/outputs/distillation/sft_epoch_comparison_20261013/REPORT.md。
- 数据准备状态/冻结RL发布未改；下一步可在RL入口接入该adapter并验收实际权重。原生续训目录创建异常已修复并留存，未重复epoch1采样或评分。

## 2026-10-13：执行1/3/5/8 epoch的dev对比（运行中）

- 用户授权先保存已有1epoch的dev rollout，再评分；继续累计3/5/8epoch并选择下游RL初始化候选。
- epoch1固定dev396题已全部保存，rollout耗时777.932s；合法最终作答385/396、verifier判对85/396（21.46%）、词面判对46/396，平均搜索1.1086。检索异常0，原始失败轨迹保留。
- 产物位于MedSearch/outputs/distillation/sft_epoch_comparison_20261013/epoch1；raw/score分文件，hash和顺序已保存；同一dev题干与SFT源轨迹标准化精确交集0。本次没有新做完整语义审计。
- 续训真实3B验收已通过：checkpoint125恢复至126，loaded_lr0→resumed_lr9.6556e-5，adapter_delta_sq0.00120355。新的cosine horizon1000；前125步不重训，恢复其余训练状态。后续每epoch完整checkpoint均保留。
- 原生Trainer先创建输出目录导致辅助脚本mkdir冲突，尚未更新参数；已修复并保留resume_smoke.failed_attempt1.log和失败启动目录。epoch1轨迹及评分未重跑。
- 正式续训、epoch3/5/8生成与评分按单卡串行执行，状态见pipeline_status.json。启动命令：`C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp/sft_comparison_pipeline_resume.log python -u -m MedSearch.tools.run_sft_comparison_pipeline --resume`。
- 当前实验未结束，不能选择最佳epoch；不启动RL。设计记录：MedSearch/docs/audits/sft_epoch_comparison_20261013_plan.md。等待真实结果完成后更新。

## 2026-10-13：Top-32 冷启动 SFT 一轮完成，W&B 在线同步已核验

- 当前阶段：MedSearch 新蒸馏冷启动；今日唯一任务为用户授权运行1000条SFT一轮并上传W&B。1 epoch、125步完成，dev评测继续暂停。
- BF16 LoRA r8/alpha16/q_proj+v_proj，lr1e-4，batch1×accum8，max_length4096，seed20261013；CE+0.1*Top32，缓存hash与Git/源码hash保留在run_report.json。
- 全程平均总loss 0.642486；首/末日志窗口总loss 0.9505→0.5589（不同训练批次，仅表示趋势）。Trainer耗时208.832s，CUDA peak allocated 10.348GiB。
- checkpoint-125保存完整训练状态，最终adapter与checkpoint逐张量相同；重载后生成合法search动作。W&B服务端finished，摘要125步及曲线已回读：https://wandb.ai/lixle-northwestern-polytechnic-university/medsearch_r1/runs/6ied183a。
- 输出：MedSearch/outputs/distillation/sft_top32_epoch1_20261013/REPORT.md；命令：`C:\Users\dell\anaconda3\python.exe tmp\run_sft_epoch.py`。回归43项通过；首次Windows日志转发GBK错误已修正，容器训练未中断、未重复启动，W&B原始日志已保存。
- 下一步：该SFT adapter可作为后续RL初始化候选；dev/教师数据库匹配核查按用户决定仍待开展。未宣称医学QA准确率提升，未改动RL清洗发布或启动RL。

## 2026-10-13：按补反例版本RL batch03已收口暂停

> 2026-10-21用户新指令：放弃此前6题后续检索，已按closed_user_declined_retrieval关闭并保留原预算；不再作为默认待办。新批rl_20261021_batch04已冻结100道未处理题（13573–13760），基线正式RL164，执行新版按门分工，完成本100题后暂停。旧段落中的“下次先6题”由此次明确指令覆盖。

> 2026-10-21接续前只读核对通过：正式RL164（E3=83/E2=81），撤回16、待补检索6题/剩余10次预算、SFT1000及hash不变。清洗书已补C2/C3边界与正反例、新版规则路径和代理保存失败收件方式，沿用现有按门分工及审核上限。未恢复清洗或训练；下一次授权先接续6题原预算。核对记录见rl_20261014_targeted_review/RUN_REPORT.md。

> 2026-10-14后续定向复查已完成：68题（4既知＋64风险项）保留52、撤回16，正式RL更新为164（E3=83/E2=81）。112题沿用旧有效结论，不称全池重认证。撤回为reject1、隔离9、待补检索6（原剩余预算合计10次），本次复查未决/待发布/环境阻塞0；下一次授权先处理6题明确预算待办。SFT1000与hash不变。
>
> 正式文件MedSearch/data/cleaning_v2/rl_20261014_targeted_review/release/correction04/train.parquet，SHA256 00af8fd0bf91837e1589cd38d7d37f3474cf9e36ffdc50a92e51cc5dbe7b0b2f。4次修正导出均真实loader、collator及数量/hash回读通过。报告、逐题decisions、withdrawals、deferred_queries、COMMANDS及暂停检查点在同目录；旧原件保留，未补检索、生成、训练或定时任务。下方为10月13日历史收口记录。

- 新200题已全部完成：新增RL98（E3=42/E2=56）、拒绝20、隔离82；未完成/待发布/环境阻塞0。正式累计180（原修正版82＋98），SFT1000与hash保持冻结。
- Q0/Q1/Q2实际135/78/43次，共256次返回，运输重试0。7个即时增量26/14/26/16/6/8/2均通过真实loader、collator、ID/数量/hash回读。13065沿用剩余预算完成后仍E1隔离，不重置、不再检索。
- 正式文件：MedSearch/data/cleaning_v2/rl_20261013_batch03/release/rl/train.parquet；SHA256 5b921ac01859522f7d2f86c4d9a4316a748fbba5ff67731e2c0d2d5299362391。
- 本批RUN_REPORT、COMMANDS、excluded.jsonl、PREVIEW_10.md、continuation_checkpoint及execution_metrics已落盘。冻结到收口实测4268.7秒，token/额度unknown。清洗书仅同步§8进度，内容规则未追加。
- 已暂停等待用户恢复；未扩展下一批，未启动7B生成、训练或定时任务。下方为恢复时的过程记录。

- 用户授权继续清洗。正式基线RL82、SFT1000不变，未发现已通过待发布积压。
- 新批rl_20261013_batch03冻结200题（13200–13572），top10/.92语义召回50对待审，题面Q1分4包各50题交GPT-6 Luna Max。结果返回依次入账、通过项及时发布。
- rl_20261013_resume13065接续旧1题，复制原C3_queries保留已用Q0一次/剩余两次，未重置；独立盲代理只见题面和真实反馈生成Q1。
- 本次不启动教师生成/训练/定时任务，新批完成后暂停。细节以本批RUN_REPORT、COMMANDS、账本和CURRENT实物为准。

## 2026-10-13：清洗书加入样例反例警示，保持暂停

- 用户授权补充警示：清洗书§3加入13019/13045开放答案覆盖及13064“可用不等于首选”反例；§4.1加入13019/13065证据误判反例与13085有效E2正例，说明13045的C2/C3结论应分开。
- §9同步Luna短指令，警示在现有首审执行，沿用原字段和批量流程，不新增审核层、逐题verifier调用或例行重审。
- 本次仅修改文档，正式RL82题、SFT1000题及清洗暂停状态不变。

## 2026-10-13：RL展示10题复核完成，定点撤回4题，暂停

- 当前正式RL82题（E3=43、E2=39），SFT1000不变。只复核PREVIEW_10中的10题，不估计全池错误率，不扩展其余76题。
- 5条E3保留；5条E2中1条保留、2条证据判宽、另2条C2评分覆盖存疑。撤回13019/13045/13064/13065；前三条隔离，13065保留未用Q1/Q2两次预算，待授权。
- 数量守恒200=82发布+49 reject+68 quarantine+1 deferred_query；待发布/环境阻塞0。此次新增0，明确撤回4，未新增检索、教师生成或训练。
- 实际新文件：MedSearch/data/cleaning_v2/rl_20261013_batch02/release/rl_quality_review_10/train.parquet，SHA256 66efe76f15499899fecee193a7e182824c9aac6b4e935dbced1b2d1ebe8012bf。真实RLHFDataset逐题prompt/ID/gold及collator4题通过，数量/hash回读成功；CURRENT已指向82题修正版。旧86题文件及不可变增量保留。
- 复核报告：MedSearch/data/cleaning_v2/rl_20261013_batch02/QUALITY_REVIEW.md；原绑定上下文及逐题结论已保存。清洗书只同步接续位置；建议小幅补C2开放题覆盖、E2跨方案负例/共享机制正例，等待用户决定，不另加审核层。
- 本轮已暂停；后续读取CURRENT及撤回清单，不从历史86题重新导入撤回项。

## 2026-10-13：清洗书精简为RL唯一执行入口，等待下轮启动

- 用户同意依据品鉴问题修改规则，并移出已完成SFT的多余流程。原清洗书647行已完整归档到MedSearch/docs/archive/2026-10-13-cleaning-guide-before-rl-simplification.md；原路径正文精简为143行，保留旧章节映射。
- 当前规则：C0–C3、R0–R3、Sol轻度调度＋GPT-6 Luna Max批量审阅、脚本即时发布；补入E2新增解题关系正反例、C2完整限定/串题无效返回、Q0不足后的有限Q1/Q2及deferred_query状态。保留数据隔离、真实loader、长度unknown与全错组规则。
- 文档已明确工程待办：prompt_tokens错误取BatchEncoding字段数；现有检索入口仅实现Q0。此次未修脚本、未恢复数据清洗/生成/训练。
- RL已导出数量仍140，但最新品鉴定位的7题尚未纠正，不能称内容全部重新确认；下一轮按清洗书§8先处理这些局部问题和剩余检索预算，不重审全池。详细反证见rl_20261013_batch01/QUALITY_REVIEW.md。
- 已回读章节与行数，清洗书单文件git diff --check无空白错误；进度文件整体检查另报既有末尾空行，未据此改动历史正文。等待用户启动下一轮。

## 2026-10-13：既有RL检查与新200题清洗完成，已暂停

- 当前为RL数据准备。既有87题核对通过且逐行保留；42题绑定82个旧episode，结束/截断元数据不足，长度记unknown，不阻止准入。报告：MedSearch/data/cleaning_v2/rl_20261013_previous_check/REPORT.md。
- 新原始200题全部分流：53正式RL（E3=24、E2=29）、34排除、113隔离。增量13→26→14及时发布，正式累计140题；SFT仍1000条，指针与hash未变。
- 排他漏斗：C1近重复20、Q1不通过47、C2不通过13、C3窗口不足67、C3通过53，合计200。120题实际盲检索top3合并后前500学生tokens，引用逐字绑定。E0/E1隔离保留剩余检索预算；未新增7B生成或训练。
- 新增53题通过真实loader/collator检查；累计140唯一ID及数量/hash回读通过。正式文件：MedSearch/data/cleaning_v2/rl_20261013_batch01/release/rl/train.parquet；SHA256 05e79d3f128e98a20520a9e47f47d8cd725797a99ebfe85e2fe82a8697bc5fd4。
- 本批RUN_REPORT.md、readback.json、PREVIEW.md、排除/隔离账本和continuation_checkpoint.json已保存。待发布0、未决0、环境阻塞0。所有审查任务结束，无定时任务；等待用户下一轮命令。

## 2026-10-13：RL长度分级规则已补充

- 根据用户确认，在清洗书10.16.C.1补充长度统计和short/medium/long分级：同配置完整未截断轨迹的固定参考集P20/P80，只作描述与覆盖观察，不增加准入门或长度配额。模型生成token与检索/环境token分开；截断、无终答、环境异常及unknown单列。
- 复用已有日志，由脚本批量统计；缺字段不阻塞发布，不逐题补跑7B。长轨迹不直接等于难题，后续结合3B有效尝试的成功率与失败类型；全错组和既定failure OPD处理沿用原方案。
- 本轮只修改文档，未实现统计脚本、未执行新清洗/生成/训练，未改变既有release。长度惩罚与采样调整留作独立训练实验。已回读新增段落并确认清洗书其余内容未变。

## 2026-10-11：RL首批112题已完成，暂停等待品鉴

- 用户批准§10.16后，冻结new14/new15/new16已有112个候选；112题全部审查入账，87正式RL（E3=35、E2=52），25因当前窗口仅主题相关、关键关系不足（E1）隔离，未决和通过未发布均0。
- 发布增量60→18→9，均通过真实RLHFDataset逐行ID/gold/raw_prompt与collator检查，长度过滤0。累计回读87个唯一ID，和25隔离ID合计覆盖112；数量/hash一致。
- 正式文件：MedSearch/data/cleaning_v2/rl_20261011_preview01/release/rl/train.parquet；SHA256 a0363130f798491f72c74b73994d96ada60c1316242c19db2e52737c08589d34。CURRENT_RL_RELEASE已指向本轮87条，不将历史5756候选计入新版正式数量。
- 10条分层品鉴样例见本目录PREVIEW.md；全量可读accepted.jsonl、quarantine.jsonl、RUN_REPORT.md、readback.json已保存。
- 题目/gold和原隔离结论复用；仅补top3整段前500学生token的C3窗口判定。当前分配近重复新增0；检索新增0，7B episode新增0，每题原预算未重置。verifier固定接口8/8通过，不冒称逐题答案认证。
- SFT仍正式1000，当前指针和数据hash确认未变。没有启动训练或定时任务。当前状态paused_user_preview_batch_completed，无发布阻塞；等待用户品鉴与下一轮命令。

## 2026-10-11：RL 清洗方案完成，待用户启动执行

- 当前阶段：SFT 数据准备已收口，后续为 RL 数据准备；本次仅查阅公开论文/技术博客并更新规则，没有检索、教师生成、清洗发布或训练。本次新增 RL 为0，不代表历史候选已完成新版准入。
- SFT 当前指针 `sft_20261001_joint_run_v1_inc171_TAKEOVER_BATCH_12251`，正式1000条，保持冻结。历史 RL 池 manifest 为5756候选、dev396；候选状态为 `weak_gold_split_ready_not_medically_verified`，不冒称新版 `retrieval_ready/training_ready`。
- 已更新清洗书 §10.16 和 RL 清洗说明入口：首版3000合格唯一题目、扩充约5000；约200题接线/诊断子集属于训练池，不额外计数。保留冻结 dev/test，目标未满也即时发布。规模为本项目工程建议，非医学最优规模实验证明。
- 执行门精简为 C0–C3、R0–R3：E2/E3均可；不要求正确教师轨迹，不开 S1/S2/S3。旧有效结论复用，实际RL top-3/整段500-token窗口发生实质变化时只补受影响证据门。奖励固定样例验证复用，不每题生成7B答案。分工和失败/全错组保留遵从现有蒸馏方案。
- 来源：Search-R1作者规模实验附录E（https://arxiv.org/html/2505.15117v1）；s3 §4.1（https://arxiv.org/html/2505.14146v1）；Search-R1/SGLang技术教程及HF 3B GRPO复现博客，链接已写清洗书。3,000→约5,000需结合后续真实评测、覆盖与预算决定，未授权固定训练步数/epoch/时长。
- 下一动作：收到用户开始RL清洗指令后，按“已通过待发布→现有RL候选缺失门→历史池缺失门→必要新候选”接续；不重审正式SFT、不展开退役历史审计、不创建定时任务。

## 2026-10-13：暂停 dev、清理退役脚本与临时权重

- 用户最新决定：当前只推进SFT训练，dev评测暂停；后续先确认教师与现有数据库在dev题目上的匹配情况。1000条冻结冷启动数据不拆分，原396题dev及test保持原归属；更新训练建议的eval_strategy=no，正式入口还需允许eval_data_path=None。
- 清理依据：工程书明确旧Stage A/B退役，当前GRPO替代旧DAPO；当前MedSearch/tests和sft/scripts没有对旧Stage A/B验收模块的运行依赖。移出30个已退役/完成的脚本，包括旧docs/tests验收套件、旧DAPO启动入口、已完成的不保存smoke及一次性探针；先压缩归档并逐文件校验SHA256，再删除原路径。
- 删除3个失败checkpoint探针目录的33个文件，包含临时adapter/optimizer/scheduler及重复tokenizer。保留 `sft_checkpoint_smoke_20261013_final` 的通过验收checkpoint、report.json及当前3B学生/7B教师/verifier/公开比较基线。MedSearch历史DAPO输出经盘点无权重文件，未伪称释放其显存或模型空间。
- 实际净释放90.39MiB。精确路径/大小/hash记录 `MedSearch/docs/audits/cleanup_20261013.json`；旧脚本恢复源 `MedSearch/docs/archive/retired_scripts_20261013.zip`，通过zip完整性和逐文件hash检查。历史清理执行程序也移入docs/archive。
- 清理后运行 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：43 passed, 14 dependency deprecation warnings in 13.29s。当前源码/缓存/验收checkpoint完好，未编写新的训练脚本或启动正式训练。
- 下一唯一核心任务：现有sft/train.py在eval_data_path=None时跳过dev loader，复用已经验收的DistillationTrainer；其后辅助配置及WSL launcher由助手准备。

## 2026-10-13：SFT 调研、旧 dev 决策与 checkpoint gate

- 调研 DGPO ACL论文/官方仓库、HF TRL PEFT指南、Smol SFT课程及PEFT文档，建议与来源记录在 `MedSearch/docs/audits/sft_training_recommendation_20261013.md`。参数是本项目建议，不包装为论文针对3B/1000条的最优配置；正式训练尚未开始。
- 用户决定1000条冷启动全部训练，验证沿用原dev，不划900/100。工程书对应固定dev为 `MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet`，实际回读396条，字段id/question/golden_answers/data_source/ability/prompt/reward_model/extra_info，没有teacher缓存，应走已有搜索QA评测，不直接送入SFT轨迹Dataset。
- 只读统计 `tmp/summarize_sft_training_inputs.py`：1000条不同question_id，长度878–2071，p50=1004/p90=1121/p99=1808，359条超过1024，无超过4096；有效目标136864 tokens。正式长度建议4096完整保留。
- checkpoint脚本 `MedSearch/tools/smoke_sft_checkpoint.py` 复用现有加载/LoRA/Trainer，CMD命令 `C:\Users\dell\anaconda3\python.exe tmp\run_sft_checkpoint_smoke.py`，最终退出0，报告 `MedSearch/outputs/distillation/sft_checkpoint_smoke_20261013_final/report.json`。最长2071-token轨迹真实更新，step1保存adapter/optimizer/scheduler/RNG/trainer state，重载logits最大差0，正常初始问题生成合法search动作且前后token一致；原生resume到step2继续更新，adapter delta平方和0.0127587。
- 独立小验证 BF16、r8/alpha16/q_proj+v_proj、lr1e-4、batch1，峰值CUDA allocated10.3618GiB，44.983s。动作合法性通过，不代表检索命中或医学答案正确。
- 前几次辅助探针失败已保留：保存/重载adapter权重均完全相同，但Trainer autocast与直接重载前向精度不同，统一BF16 autocast后logits一致。停止条件错误检查输入上下文标签、历史强制回答前缀产生后缀格式问题，最终改为只检查新token并使用正常初始问题入口。未修改用户核心loss/Trainer，也未把失败运行报告为通过。
- 正式训练候选：1e-4、batch1/accum8（125步/epoch）、r8/alpha16/qv、CE+0.1*Top-32/T1、先1epoch，warmup7步/cosine、epoch checkpoint、原396题dev搜索QA评测；尚未创建正式训练产物。

## 2026-10-13：真实 3B CE+Top-32 不保存短验证

- 用户授权核对后运行一轮不保存验证。新增辅助入口 `MedSearch/tools/smoke_sft_top32.py` 和 CMD launcher `tmp/run_sft_top32_smoke.py`，复用现有模型加载、LoRA、Dataset/collator、build_trainer 和 DistillationTrainer；未修改用户核心训练代码。现有仓库没有专用正式 SFT YAML，本轮参数完整存于报告，不视为正式训练预算。
- 命令：`C:\Users\dell\anaconda3\python.exe tmp\run_sft_top32_smoke.py`；现有 WSL Docker llm-pytorch，退出码 0。基座 models/Qwen2.5-3B-Instruct，原始学生，无旧 adapter；teacher cache SHA256 与冻结 manifest 一致。Git HEAD a2861fd44828450db884c4d65c79e5a892d7bb88（本地未提交实现另在当前文件中）。
- 使用冻结训练缓存中前 8 条长度不超过 1024 的完整轨迹，实际长度 988–1021，956 个有效目标 token，无截断、无 dev/test 样本参与。BF16 LoRA r=8/alpha=16/dropout=0，q_proj/v_proj，1,843,200 可训练参数；batch=1/accumulation=1，lr=2e-4，seed=20261013，gradient checkpointing，1 epoch/8 optimizer steps，save_strategy=no。
- 同一训练样本按有效目标 token 数加权复算：CE 1.0208699→0.7310077，Top-32 0.9074779→0.6223525，总 loss 1.1116177→0.7932430；8 条各自 CE/Top-32/总 loss 均下降；adapter delta L2=1.1550754，8 步梯度范数均有限且非零。
- Trainer 更新阶段 1.9553s；整轮含模型加载、前后复算和两次生成 52.6821s；CUDA peak allocated 8.1096GiB（非整机显存占用）。训练前后固定首题 greedy 64-token 输出变化，均以 <think> 开头；本轮输出被 64-token 上限截断，未完成最终答案或工具调用验证，不能宣称任务准确率或搜索能力提升。
- 产物目录 `MedSearch/outputs/distillation/sft_top32_smoke_20261013` 仅 report.json，没有模型权重或 checkpoint。报告含配置、样本身份、hash、逐样本 loss、训练 history、输出及时间显存指标。
- 当前结论：真实 3B 能跑通 loss 接线并在这 8 条训练轨迹上学动，不代表完整 4096 上下文、泛化评测或保存恢复已验收。下一步正式长训练前仍需 checkpoint 保存、重载和完整动作推理验收。

## 2026-10-13：build_trainer 接线验收

- 用户授权助手完成入口替换。现有 DistillationTrainer 导入已存在，将 `MedSearch/sft/train.py::build_trainer` 的实例化从 Trainer 改为 DistillationTrainer，保留其他参数。
- 将已有 Trainer 验收测试改为通过真实 build_trainer 构造，确认类、dataset 和 collator 接线正确；CE 和 Top-32 两条路径均完成极小 Qwen2 模型的一步梯度累积更新。
- 命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：43 passed, 14 warnings in 14.08s，warnings 为依赖弃用提示。未启动正式训练。
- 下一任务：核对 SFT 数据/训练配置，随后验收真实模型 checkpoint 保存、重载和动作推理；不得绕过保存验收启动长训练。

## 2026-10-13：DistillationTrainer loss 组合验收

- 用户的 `MedSearch/sft/distillation_trainer.py` 已验收，未修改核心实现：浅拷贝 inputs，移除教师元数据后单次模型前向；无缓存返回 CE，有缓存返回 CE+0.1*Top-32；支持 return_outputs；model_accepts_loss_kwargs=False。
- 新增 `MedSearch/tests/test_sft_distillation_trainer.py`，验证输入不变、组合数值、返回格式、教师无梯度、学生有限且非零梯度。复用真实 Transformers Trainer，在随机初始化极小 Qwen2 模型上分别执行 CE 和带缓存训练，各两个 microbatch 累积为一个 optimizer step，权重确实更新。
- 现有 WSL Docker 命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：43 passed, 14 warnings in 14.52s。warnings 为依赖的 torch.jit.script_method 弃用提示。未加载真实 3B 权重，未进行正式训练或 checkpoint 验收。
- 下一唯一核心任务：sft/train.py 的 build_trainer 改用 DistillationTrainer；正式长训练前继续保留 checkpoint 保存、重载、一次动作推理验收门槛。

## 2026-10-12：SFT forward_topk_loss 验收

- 用户的 `MedSearch/sft/losses.py::forward_topk_loss` 通过验收，未修改核心实现：按 target_positions-1 选 logits，截取共同词表 151665 项，float32 全词表归一化后 gather 教师 Top-32，教师 logprob detach，每位置 clamp_min(0) 后按 batch 总目标数平均。
- 新增 `MedSearch/tests/test_sft_topk_loss.py`：CPU/CUDA 数值对照、不同轨迹长度的 token 平均、未映射输出行排除、非目标位置无梯度、学生梯度方向、教师无梯度、逐位置截零。首次 CUDA 测试参考值设备不一致，修正测试脚本后通过，用户 loss 未改。
- 运行 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：41 passed in 7.63s，CUDA 项实际执行。仅合成 logits 验收，未启动模型训练。
- 下一唯一任务：实现自定义 Transformers Trainer.compute_loss，同一次前向组合 CE + 0.1*Top-32；之后再接 build_trainer，正式训练前仍须 checkpoint 保存、重载及动作推理验收。

## 2026-10-12：SFT Dataset / collator Top-32 接线验收

- 用户实现的 `SFTDataset.__getitem__` 和 `build_data_collator` 已验收：普通字段复用 Transformers 右侧 padding，稀疏教师字段保持变长 Tensor 列表；位置和 ID 为 int64，logprob 为 float32。未修改用户核心实现。
- WSL Docker 回归命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：38 passed in 10.72s。新增检查覆盖输入不被修改、环境和 padding labels=-100、变长教师字段、目标位置与 labels 对齐及纯 CE 兼容。
- `C:\Users\dell\anaconda3\python.exe tmp\check_sft_parquet_loader.py` 用本地 tokenizer 和真实 1000 条缓存检查 500 个 batch，全部通过，136864 个评分位置保持对应。当前入口已设置 padding_side=right。
- 下一唯一核心任务：自定义 Transformers Trainer 的 compute_loss，读取缓存计算 CE + Top-32 loss；目前普通 Trainer 尚未消费教师字段，未进行训练。

## 2026-10-12：SFT Parquet 读取验收

- 按用户授权，补充 `MedSearch/sft/data.py::load_samples` 的 Parquet 分支，使用 `pyarrow.parquet.read_table(path).to_pylist()`；保留 JSONL 读取与行序。Dataset 和 collator 的 Top-32 字段接线尚未修改。
- 现有 WSL Docker `llm-pytorch` 回归验收：`python -m pytest MedSearch/tests/test_distillation_hook_jobs.py MedSearch/tests/test_distillation_teacher.py MedSearch/tests/test_failure_opd_loss.py MedSearch/tests/test_grpo_failure_opd_loss.py MedSearch/tests/test_teacher_topk_cache.py MedSearch/tests/test_sft_search_distill.py -q`，37 passed in 10.72s。
- 实际缓存通过新接口读取：1000 条、136864 个评分位置；身份、行序、prompt/response IDs 和 mask 与冻结 JSONL 一致，稀疏教师数组每位置 32 项。核查入口：`tmp/check_sft_parquet_loader.py`。
- 下一唯一任务：在 `SFTDataset.__getitem__` 和 collator 中保留并传递稀疏 Top-32 字段；自定义 Trainer 尚未接入。

## 2026-10-11：离线 Top-32 Parquet 缓存工具

- 全量任务已收口，进程退出码0：1000 rows、136864 scored tokens，196.48s（含加载/评分/保存重读），CUDA peak allocated15.018GiB。最终 train.parquet SHA256 `83dd481cbb2f248b407490367d787a42f0dbde021b70ddacd647c717ed45c4e5`；train.manifest.json status=complete/scope=full，输入hash与冻结1000条SFT release一致。下一单项任务是 sft/data.py 的 Parquet读取及稀疏Top-k字段随batch保留；自定义Trainer尚未接入。

- 用户明确同意一次处理后持久化，不逐条切换师生。新增 `MedSearch/tools/cache_teacher_scores.py`，一次加载教师、逐动作评分、逐轨迹流式写 Parquet（zstd）、完整后重读验收并发布；目标位置是 input_ids 中目标 token 的下标，训练时读取 position-1 的 logits。只保存 token/mask/身份/目标位置及 Top-32，不复制 gold；保留教师身份、输入/输出 hash 与配置 hash manifest。未修改用户核心训练代码。
- 保存重读单测及既有回归：`33 passed in 10.94s`。真实 Search-R1 7B BF16 两条轨迹缓存完成：2 rows、315 scored tokens、49.69s（含模型加载）、CUDA peak allocated14.66GiB；产物 `MedSearch/data/distillation/top32_sft1000_20261011/smoke2.parquet` 与同名 manifest。输入hash与当前1000条release指针一致。
- 已启动全量命令 `C:\Users\dell\anaconda3\python.exe tmp\run_teacher_topk_cache.py --full`；最终目标 `MedSearch/data/distillation/top32_sft1000_20261011/train.parquet`，本行不代表全量已完成。正式训练仍待缓存全量收口、CE+Top-k Trainer 与 checkpoint 保存/重载/动作推理验收。

## 2026-10-11：SFT B 档 Top-32 教师评分函数验收

- 用户增加 `score_tokens(mode="topk")`，返回 topk_ids/topk_logprobs [B,T,32]。现有容器命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py` 返回 `32 passed in 10.75s`。
- 新增固定 logits 数值测试验证非平凡 prefix 的因果位置、双样本双目标位置、token ID 及 float32 logprob、排除未映射输出行、完整共同词表归一化而非 Top-32 再归一化、无梯度；既有 sampled/RL 回归通过。未进行本轮真实 7B Top-32 GPU 前向或离线缓存任务，未修改用户核心实现。
- 当前训练分工：SFT 使用现有 Transformers/PEFT 装配，自定义 DistillationTrainer 接入 CE+0.1*Top-k loss 数学；参考/复用 veRL 数学函数，不开启 veRL RL 教师管理器。RL 继续用 veRL sampled 失败 OPD。当前 SFT release 指针为1000条，Top-k离线缓存和自定义Trainer尚未实现；RL actor接线暂缓。

## 2026-10-09：GRPO + OPD 合并函数首轮验收

- 用户明确按最小 demo 范围不增加重复形状 guard：已有评分/回填保证逐轨迹长度，移除针对任意 [B,1] 输入的两项防护测试。重新运行同一容器命令 `31 passed in 11.04s`，合并函数在约定输入契约下验收通过，无核心代码变更。真实 Trainer 接线仍需验证字段传递与对齐；此前两项失败是被用户撤回的防护要求，不算当前未解决阻塞。

- 用户完成 `grpo_with_failure_opd_loss`；新增真实 veRL ppo_loss/变长 TensorDict 测试。正常路径验证通过：系数为零且无教师字段时与原生 loss/梯度一致；系数非零时合并数值与手算梯度增量正确，因果 response 对齐正确；全对组与原生相同；保留原日志和数据字段。本轮未修改用户核心代码。
- 最终容器命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：`2 failed, 31 passed in 6.84s`。缺少此前约定的五张量形状检查：teacher_logprobs 或 opd_mask 为 [B,1] 时自动广播到 [B,T]，不报错，造成逐 token 分数或门控错误。需在四字段 padding/设备转换后、调用 failure_opd_loss 前对比 student_log_probs.shape，不一致抛 ValueError。完整验收暂未通过，真实 Trainer 接线仍未执行。

## 2026-10-08：failure_opd_loss 首轮验收未通过

- 后续用户将 policy loss 关键字修正并保存后，原容器命令复验 `28 passed in 10.04s`，failure_opd_loss 函数级验收通过。验证教师/旧概率无梯度、全对与无动作批次零损失可反传、混合组仅失败动作有梯度且成功项计入分母、padding 排除、全错组梯度方向、固定全局分母下 microbatch 损失与梯度一致、原生 clipping 生效。未运行真实 Trainer 更新，合并 GRPO 与 actor/teacher 生命周期接线仍待完成。

- 增加 `MedSearch/tests/test_failure_opd_loss.py`，覆盖无 OPD 零梯度、混合组分母/屏蔽、全错组梯度方向、microbatch 全局分母以及 native clipping。测试配置已补齐真实 ActorConfig 必填项，global_batch_info 按原生字典方式设置。
- 现有容器命令 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py` 最终结果 `4 failed, 24 passed in 6.78s`。既有 22 项回归与本轮 2 项无 OPD 零损失/零梯度通过。
- 有 OPD 的路径被调用接口阻断：`compute_policy_loss_vanilla` 接收 `log_prob`、`advantages`，用户代码传 `logprob`、`advantage`；运行报 unexpected keyword argument 'logprob'。需改两处关键字后继续数值与梯度验收；不能将被阻断的 4 项称为通过。本轮只写测试与记录，未修改用户核心实现。

## 2026-10-07：下一单项编码任务——失败 OPD loss

- 上一项 attach_teacher_scores 已通过函数级验收；当前编写 `MedSearch/distillation/losses.py` 的二维张量 `failure_opd_loss` 核心，复用 veRL `kl_penalty` k1 与 `compute_policy_loss_vanilla`。
- 失败门控仅作用于 detach 后的 OPD advantages；原 response_mask 保留，OPD 使用 seq-mean-token-mean，成功项为零但保留平均分母。无 OPD 位置返回带计算图的零，教师与旧策略不参与梯度；系数留给后续 GRPO wrapper。本轮为指导与接口约定，无新 loss 实现或训练结果。

## 2026-10-05：attach_teacher_scores 函数验收通过

- 在现有 `llm-pytorch` 容器复验命令：`C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`；最终结果 `22 passed in 4.93s`，包括既有 13 项回归和本轮 9 项用例。未修改用户核心实现。
- 真实 KVBatchMeta/TensorDict 下，等长与变长序列、乱序评分回填、空评分、成功/padding 项置零通过；保留原始字段及 batch metadata，返回原 data。索引越界、身份不匹配和两个评分张量形状错误均报 ValueError，且不部分写入新字段。
- 测试初轮 `2 failed, 20 passed`：失败由测试使用普通高级索引造成（PyTorch jagged tensor 不支持 `aten.index.Tensor`）；改用 veRL 原生 `index_select_tensor_dict` 后通过，新增字段与原 responses 在重排选行后保持一致。本轮未运行真实 actor microbatch/TransferQueue 链路，不能视为完整接线验收。

## 2026-10-04：score_failed_batch 首轮验收

- 改名后复验：形参与循环引用均为 `failure_batch`；同一容器命令返回 `13 passed in 5.27s`，该函数验收通过。下一核心任务为 `attach_teacher_scores(batch_meta, data, scores)`，在原完整 TensorDict 中按轨迹身份回填评分字段；真实 Trainer 接线尚待实现。

- 在现有 `llm-pytorch` 容器执行 `C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`：`1 failed, 12 passed in 5.27s`。
- 核心行为通过：空失败列表不评分；变长轨迹逐条调用既有评分函数；保留原始索引与 trajectory_id，observation 分数/mask 为零，输出 CPU float32；不修改输入，第二条评分异常向上传递并停止后续评分。
- 未通过项：参数拼写为 `failure_bacth`，与约定及 docstring 的 `failure_batch` 不一致；按 `failure_batch=` 关键字调用抛 TypeError。用户需将形参及循环变量引用一并改名后复验。本轮仅补测试与记录，未替用户修改核心代码。

## 2026-10-03：build_failure_batch 函数验收

- 按用户要求，本轮暂不验收有效轨迹 acc 缺失的处理；未修改用户核心实现。
- 在现有 `llm-pytorch` Linux 容器中，使用真实 `transfer_queue.KVBatchMeta` 与 veRL `list_of_dict_to_tensordict` 构造输入；覆盖全错组、混合组、全对组、合成 padding、变长 token 序列和嵌套非张量字段。
- 失败项身份、原始索引、token/mask 均正确，成功项与 padding 不进入教师任务，输出不携带 gold；原完整 batch、keys、tags 保持不变。
- 验收命令：`C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`；结果 `9 passed in 6.54s`（本轮新增 2 个参数化用例，已有评分回归 7 个）。测试位于 `MedSearch/tests/test_distillation_hook_jobs.py`。
- 此结论为函数级输入兼容与筛选验收；尚未执行真实 V1 Trainer hook、教师批次进程及完整训练更新。

## 2026-10-02：SFT 增量清洗续接（当前状态）

- SFT 当前 release 已于2026-10-07回读核对为 **869 条**，距1,000差131；new06本次实际新增16（853→869）。文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc125_TAKEOVER_BATCH_8322/release/sft/accepted.jsonl`，SHA256 `ad26620d6feb39cb88855c9a8f3dd643da32b83fd89ce18f61aeef00078a268a`；manifest SHA256 `ef0b9d5e829fd8bb8499d9aa11c21b3cf9f963f7ec4523b70da91cbc36f5b0f1`。**用户最新授权自主迭代至正式1000条，解除旧五批cap及逐批暂停。new06已实际生成54 episodes，待17题S2、6题独立S3；协作工具agent thread limit reached、旧已完成子代异常pending_init使隔离审查无法派发，输入/预算/hash全部保存，未恢复旧暂停且未标目标完成。现无通过未发布、没有活动教师；环境解锁后先消费本批冻结审核包，再自主继续new07。**
- 来源 401–420 的 C1 语义筛查 20/20 通过、无近邻标记；20 条 Q0 检索已保存。当前 C3 累计分级：E3=1、E2=2、E1=5、E0=2、待分级=10。`medical-o1-verifiable:32797` 为 E1，`32816`、`32823` 为 E2；`32826` 的 Q0 证据为 E3，但原题写“from the following”而源记录无选项，按缺选项规则 reject。每项判定、引用及账本 hash 均已写入并回读。
- E1/E2 尚未达到 SFT 所需 E3；32826 虽达 E3 但题目缺选项而 reject。当前没有新 SFT 可发布项。下一步按 §10.5 有限预算继续处理剩余已保存 Q0 结果，每条结果完成保存/回读闭环后才开始下一条；E1/E2 再按题干导向 Q1 目标寻找 E3。
- 目前未启动 401–420 教师 rollout；没有 C3 E3 题目进入 SFT 生成队列。batch03 输出位置：`MedSearch/data/cleaning_v2/sft_20261001_rollout_prep/batch03_401_600/`。检索服务已成功返回本批结果；Docker 健康不是当前阻塞。
- 最新逐项账本 SHA256：`7fc6d8ea398bd975b73bec97daec5d04b7004d2a0746bfcbead835722b7a3c88`；attempt ledger SHA256：`a3b83a2c7e11d2a3e742c43ae98dcdaa6b43612faf47df0b3d951eb6a3452ac5`。

## 2026-10-01：SFT 增量收口与实际落盘

> 本节是当前最新回读状态；下方按日期保留的旧进度记录仅作历史，不覆盖本节。

- SFT 最新统一收口包 v2 已落盘：636 条 release、224 条 staging、8 条已知当前保护集同题冲突排除。复用 868 条内容门通过的完整轨迹；648 条基础账本中的 9 reject、3 quarantine 均未收入。现有逐条语义对账与当前 RL train/dev、冻结 MIRAGE final test 精确筛查均未发现这 186 条旧 staging 的保护集冲突，因此按用户收口指示转入 release；其余 224 条隔离/切分仍待定。本轮未重开医学审核或扩大历史排查。数据包和 hash 见 `MedSearch/data/cleaning_v2/sft_20261001_all_available_closeout_v2/manifest.json`；真实 SFT loader/collator 仍未验证，`runtime_verified=false`。
- Cold1000 增量 batch01 的 35 条完整轨迹通过 Q2/S2/S3，且 S2 全部为 E3，已落盘到 `MedSearch/data/distillation/teacher_cold1000_sft_incremental_review_20261001/batch01/staging/content_pass_pending_isolation.jsonl`，SHA256 `e17c37894c8d529408036861f8fa659bc201551b148398a9242789f017038967`。
- Batch02 的 187/187 个 S2 槽位现已完成盲审：pass 57、uncertain 34、exclude 96；与既有 Q2/S3 通过记录相交且 S2=E3 的 28 条完整原轨迹已落盘到 `MedSearch/data/distillation/teacher_cold1000_sft_incremental_review_20261001/batch02/staging/content_pass_pending_isolation.jsonl`，SHA256 `c3041e33f02d3276641967cbd6657e3188877e0b5aaef8e375974b39126688f6`。此前 7 条中间包保存在 `batch02/staging_partial_s2_0001_0045/`，不重复计入当前库存。
- Batch10 现存轨迹中另有 10 条通过已有题目、证据、动作质量及答案等价门，且与当前 858 条无规范化精确重复，已原轨迹落盘到 `MedSearch/data/distillation/teacher_batch10_incremental_sft_export_20261001/staging/content_pass_pending_isolation.jsonl`，SHA256 `87fbdb4acaa55302ffe9789161e94c089d6b603e5dee4df995d31dbffbd3724f`。10 条语义近重复仍待闭合，故列 staging；未用 verifier 状态作为准入依据。
- Batch8 的 54 条完整门禁通过轨迹和 Batch9 的 100 条逐行门禁通过轨迹均已在现有库存内逐条匹配，新增数均为 0；Batch8 另 3 条新候选保留原有 action-review hold。Batch10 的 928 条剩余 queue 只是题目、没有完整 Search-R1 轨迹，不计入库存。
- 当前来源中有 868 条内容通过的完整轨迹：636 release、224 staging、8 条已知当前保护集同题冲突排除。当前 RL train/dev 与冻结 MIRAGE final test 精确核验为 0 冲突；928 条待生成题目也已对同一保护范围做精确筛查，0 冲突。首批 200 题已冻结但尚未 rollout，不计轨迹。release 距 1,000 条目标差 364；若 staging 全部通过隔离并晋级，仍差 140 条。真实 SFT loader/collator 未验证，`runtime_verified=false`，环境状态单独记录且没有阻止已有轨迹导出。详情见 `MedSearch/data/cleaning_v2/sft_20261001_all_available_closeout_v2/manifest.json`。
- 旧 E2E 文件：release [accepted.jsonl](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/release/sft/accepted.jsonl>)；staging [content_pass_pending_isolation.jsonl](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/staging/content_pass_pending_isolation.jsonl>)。新 batch02 staging [content_pass_pending_isolation.jsonl](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/teacher_cold1000_sft_incremental_review_20261001/batch02/staging/content_pass_pending_isolation.jsonl>)。
- 完整库存回读核验通过：868 条数量、question/trajectory ID 与规范化题目唯一；旧 E2E 636 条来源行 hash、训练字段和 token/mask 长度通过，新增 batch02 28 条与 Batch10 10 条均与来源原始行及 token/mask 结构一致。真实 `SFTDataset`/collator 运行验证未通过，故 `runtime_verified=false`、`training_ready=false`；这是单独的环境阻塞，不影响已有轨迹导出。
- 清洗书符合性：新增数据复用已有内容审查结论，三门通过且隔离待定的轨迹进入 staging；没有将 verifier 单独当准入、没有把待隔离项放入 release，也没有把 loader 环境失败当作阻断导出。当前剩余工作是为合格 staging 处理语义近重复/最终 split、从已有候选队列产生足够的新增完整轨迹，以及实际 loader/collator 验证。

## 2026-10-01：SFT §10.12 收口导出

- 基础 648 条已有轨迹内容门通过；排除 reject 9、quarantine 3 后，636 条完整轨迹分为 release 450、staging 186。另并入既有清洗导出 232 条完整轨迹；统一包合计 450 release、410 staging、8 条已知当前保护集冲突单独排除。实际包：[report](MedSearch/data/cleaning_v2/sft_20261001_all_available_closeout/report.md)、[manifest](MedSearch/data/cleaning_v2/sft_20261001_all_available_closeout/manifest.json)。另有 928 条尚无 rollout 的新题，不计作完整轨迹。
- Release：`MedSearch/data/cleaning_v2/sft_20261001_e2e/release/sft/accepted.jsonl`，SHA256 `8feeb4bd6955c881fcdd9cc98c7d0a29c50641bd433f82896b7b184d5d5a6125`；staging SHA256 `49cda5083cd57ec40d466788b4322f3120c999fc71fc4a60837beedb65b2c84d`。
- 已逐条回读文件数量、ID、来源 attempt 行 hash 与 prompt/response token-mask 结构；真实 `SFTDataset`/collator 未运行，`runtime_verified=false`、`training_ready=false`。运行环境问题单独记录，未阻挡导出。
- 完整报告：[SFT 收口报告](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/report.md>)；[release 数据](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/release/sft/accepted.jsonl>)；[staging 数据](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/staging/content_pass_pending_isolation.jsonl>)。
- 本轮收口完成；后续只需在 Torch/Transformers 可用环境验证此导出包 loader，不重做医学审核。


## 2026-10-01：新版 §10 SFT 清洗 run 收口检查点

- 当前仍在阶段 3 的 MedicalGPT SFT 数据准备；本次只完成 SFT §10 端到端账本/证据回连，不代表开始训练。
- 冻结 run `MedSearch/data/cleaning_v2/sft_20261001_e2e/`：1,576 个有序 ID = 648 条既有轨迹候选 + 928 条待生成题目队列。队列固定为 200/200/200/200/128 五批；本轮没有新检索、教师生成、训练或付费 API 调用。
- 648 条候选的历史证据定位补齐：首轮冻结漏绑 17 条，已从原始 sanitized model-review 行回连精确可见引用；再核对发现 4 条起始 `Doc N` rank 错记并修正，5 条引用跨多个 Doc 段并逐段列出对应 hash。当前 `evidence.jsonl` 保存 648 条完整模型可见 observation、其 hash、quote offset、query、起始/覆盖 Doc rank 与 document hash、attempt 和 review 行 hash。648/648 的确定性 offset/hash/rank/pointer 校验通过；这不是新增语义审查。
- 截断边界已记录：646/648 个模型可见 observation hash 与检索前完整 observation hash 不同，且可见长度为 699–700/700 tokens；其余 2 条长度 632、646 tokens，hash 与检索输出一致。证据引用按模型实际看到的文本核验。
- 最新冻结门禁：C1 `reject=9 / quarantine=3 / blocked=1564`；C3 `pass=648 / pending=928`；S0–S3 既有候选各 648 pass；S4 648 blocked、真实 `SFTDataset`/collator 未验证。终态 `reject=9 / quarantine=3 / blocked=1564 / ready=0`；正式 SFT-ready 仍为 **0/1,000**，不把旧批次数、verifier accepted 或轨迹候选误报为准入。
- 928 条新队列已补做可用范围内的精确 C1 子门：按 NFKC/casefold/空白归一化与 648 条现有候选及队列内部比较，交叉重复 0、队内重复 0。该结果已写回 928 条 ledger C1；完整历史暴露/split 仍缺 Stage-A/B、UltraChat 映射，故 C1 总状态继续 blocked。
- 清洗书遵循情况：本 run 已按 §10 冻结输入和有序 ID、复用历史门禁、补齐证据锚点、留存 ledger/evidence/pairs/release manifest/hash，并按硬停条件不启动受阻工作；**整个端到端发布尚未完成**。完整历史暴露/split 映射仍缺，928 条增量 C1 和教师轨迹未完成，真实 loader 验证与最终发布抽检未完成。
- 环境预检：`nvidia-smi -L` 为 NVML Unknown Error，Torch import 与本机 socket 初始化报 WinError 10106，`tasklist /FO CSV` 报模块无法找到；8001/8002 listener 状态未知。按 §10.2 停止重试，本轮没有启动新服务。
- 落盘报告：[SFT §10 run 报告](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/report.md>)；[run manifest](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/run_manifest.json>)；[ledger](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/ledger.jsonl>)；[evidence](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/evidence.jsonl>)；[空 release manifest](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/cleaning_v2/sft_20261001_e2e/release/manifest.json>)。
- 今日清洗任务已完成的部分：把 648 候选与 928 队列纳入同一冻结账本，修正终态优先级/证据定位，完成 928 条可用范围内精确 C1 子门，并通过确定性完整性核验。下一项只处理全历史暴露与 split 证明缺口；缺失证明的样本继续 blocked，环境通过前不做 rollout。

## 2026-10-01：追加有界 SFT / RL 清洗路线（仅文档）

- 针对用户反馈“持续审计而未追加数据”，追加 §10.14 强制调度：结果返回先保存，已通过项优先增量发布，再启动下一任务；一条可发即发，未决分片不阻塞，发布检查只覆盖新增项，写入须回读且可恢复。本次仅修改清洗书及说明，未读取/测评本轮样本，未替 Luna 导出或认证636/637等数量。

- 用户授权 SFT/RL 联合清洗，已追加执行书 §10.13 并修正“RL 必须等待 SFT 完成”“候选转用必须下一版”的旧限制：共用前置门、按同题簇互斥分流，优质完整示范优先 SFT，满足可靠题目/gold 与有效检索证据门的其他题进入 RL，剩余一次模型裁决后封账；两路分别交付。本次仅更新清洗规则，未执行审计、生成、分配数据或训练。

- 收口规则修订（§10.12）：已读取 `data/cleaning_v2/sft_20261001_e2e/` 报告及 manifest，确认当前648条中9 reject、3 quarantine、636因历史边界待定而blocked；loader另受Torch导入失败阻塞。新增内容/隔离/运行三类状态，要求导出实际 staging/release 轨迹，旧训练历史按当前模型和评测的关联判断，禁止用GPU故障阻止静态交付。本次只修订执行规则，未重新认证636条，也未改变既有run账本/hash或声称已导出训练包。

- 用户进一步明确：没有医学人工审核人员，也不需要人工审核。已补充清洗执行书 §10.11，所有语义审阅、有限裁决和发布抽检由 Luna Max 完成；不确定项在预算内裁决后隔离，不等待用户/医生批准。输入隔离、证据门槛与真实 loader 验证仍保留。本次仅修改文档。

- 已按用户要求在 `MedSearch/docs/superpowers/plans/2026-09-29-sft-environment-data-review.md` §10 追加端到端生产契约：共同结构/隔离/题目/证据门，SFT 完整高质量轨迹与答案等价门，RL 有效相关证据与 reward 接口门，最终真实 loader 验证和版本化导出。
- 明确一次首审、疑难一次裁决、hash 复用、有限检索/重生成、一次发布验收；未决项隔离，不以反复审查阻止可用部分收口。不足目标与不可训练分开报告。
- 今日本次任务是制定并追加执行路线，已完成文档更新；未运行数据清洗、教师生成、训练或变更现有准入数量。下一步按 §10.9 先映射现有候选账本，仅补缺失门。

## 最新 SFT 清洗核验（2026-10-01）

### 持续清洗补记（2026-10-01）

- 正式 SFT-ready 仍为 **0/1,000**；未生成最终训练导出，`training_ready=false`。题面 keep、verifier accepted、rollout 有答案均不等于准入。
- **产量复核纠错：**10个生产rollout队列共 **5,088次原始尝试**（cold1000 1,000；Batch3 775；Batch4 517；Batch4 replay 53；Batch5 491；Batch6 305；Batch7 316；Batch8 338；Batch9 606；Batch10 687）。各manifest的机器 `accepted` 合计1,380；该标签只供诊断，5,088含replay且未做全池语义去重，不是唯一合格SFT数。之前提到的“100”只指Batch9逐行质量门通过的子集，不代表全部清洗产量。
- Batch9 Search-R1 7B rollout 已完成 **606/606**，attempts 606 行且 question_id 唯一；accepted 171、rejected 435，完整输入顺序、分区及逐行一致性核对通过。537 条有答案、69 条无答案只是诊断数。
- Batch9 606条 sanitized gold-blind 轨迹审阅包分成6×101行，review input SHA256 `4937522a385ac4290e35c796c1aac846471442dc16ea62ebfc0722adfa767b6d`。reviewer未见gold/reference、verifier、accepted/rejected、question_id、来源或文档hash。37条动作/observation绑定异常和5个same-task重复均hold。盲审完成606/606：题目keep580、uncertain25、exclude1；原advance194、hold411、exclude_current_trace1。主代理精确文档引用绑定审计将10条存在跨文档边界/不可定位引文的advance暂hold，当前有效advance184、hold421、exclude1；37条flag全部hold。答案等价复核现已完成184/184：equivalent100、different44、uncertain40；其中100条通过逐行SFT候选门，仍待全池去重、历史暴露与split隔离。advance不代表SFT准入。
- Batch10 Q1 题面 2,000/2,000 完成；adjudication 为 keep 1,783、uncertain 190、exclude 27。曾被排除后改回 keep 的 48 条已完成 Q2–Q4 审查：keep 15、uncertain 33、exclude 0；主代理抽查 6 条，均只确认题面/答案映射可暂留，不代表医学认证。
- Batch10 其余 **1,735** 条 Q1 keep 的gold-visible Q2–Q4审查全部完成，七片ID/顺序/schema/原文引文校验通过：keep1,600、uncertain129、exclude6。加上另审的48条中keep15/uncertain33，Batch10 Q2–Q4阶段合计候选keep1,615；这仍不代表轨迹或准入。
- 从首批 Q2–Q4 keep 冻结的687条 Batch10 rollout 已完成 **687/687**，reconciliation核对通过：accepted183、rejected504；609条有非空答案、78条无答案；37条有非法动作标记。SHA256 `8ee3a4b83e107d72c7878a93da4200515060c8d18863e031b6ad728c7afe574a`；gold-blind sanitized轨迹审查与答案等价复核均已完成：第一轮advance181/hold477/exclude29，答案等价 equivalent101/different57/uncertain23，根校验errors=0。928条剩余keep队列SHA256 `3aabe8b0493ae62ba57e4a00806adc9adbf2befc469f5e7072691a7b6cb5c435`，输入精确去重预检通过，但尚未启动。
- Batch8 sanitized 盲审 338/338 完成；答案等价与 response grounding 交集 57。Query/action 绑定二审 57/57：pass 54、hold 3、reject 0；仍需总池去重、历史暴露和 split 核验。
- 旧 reserve 8/8 仍为 hold：5 条包含 invalid action segment，2 条存在重复/近重复搜索；其余未解除的问题仍保留 hold。
- 清洗计划符合性为**部分符合、整体未完成**。已按计划纠正 Batch10 Q1 对开放题的过度排除（48 改回 keep、4 改 uncertain、27 确认 exclude）；Batch8 曾泄漏 full_observation 的旧盲包降为预审，338 条改用 sanitized 包重审。全池证据、答案、动作、去重、历史隔离及最终 split 尚未完成。
- `DISTILLATION_PLAN.md` §P3 的旧冷启动标准仅要求合法非空 `<answer>`，不要求 verifier 正确或必须搜索；当前目标还要求 source-gold 等价、强可见证据、干净动作/推理和历史隔离。因此旧 886 条 answered 轨迹只是待审候选，不能按旧规则直接计入 1,000 条。
- 393条旧候选、Batch8逐行通过54条、Batch9逐行通过100条与Batch10逐行通过101条形成648条跨来源exact-unique候选并集。全历史语义裁决并合并旧RL专项复核后，9条候选与受保护RL集合确认同题、1条因新旧判断冲突hold，3434/15993构成需择优的候选内部同题对；语义筛查单项理论上限为637条。另有187条候选命中raw来源同题参考，实际模型暴露仍未知；Stage-A/B、Search100、UltraChat 821行映射缺失。正式SFT-ready仍为0/1,000。
- 当前报告：[SFT 清洗进度与计划符合性](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/sft_cleaning_progress_20261001_update.md>)；Batch9 审阅输入：[manifest](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/teacher_batch9_fresh_trace_20261001_sanitized_review/manifest.json>)。

### 今日唯一清洗任务

- 核心任务：已完成648条候选的全历史语义裁决、旧RL专项结果合并和候选级影响对账；检查928条待生成队列所需环境。
- 完成标准：候选级账本覆盖648/648；明确protected overlap、冲突hold、内部同题与source-reference待确认数量；不因embedding自动排除。928条仅在GPU、搜索服务及后台任务归属可确认后生成。
- 完成状态：9条受保护集同题block、1条新旧裁决冲突hold、3434/15993一个候选内部同题对（择一）；187条raw来源同题保持provenance pending。检查得到NVML Unknown Error、8001/8002无监听、tasklist进程枚举报错、WSL E_ACCESSDENIED，928条未启动。

### 较早快照（2026-10-01，保留历史记录）

- Batch9 rollout manifest 到 `processed=496/606`，attempt 文件 499 行、question_id 全唯一且 JSON 无解析错误；结束后再核实完整输入覆盖。
- Batch10 Q1 已完成 2,000/2,000：keep 1,735、uncertain 186、exclude 79。独立校验八份账本 ID/顺序/schema/题面引用均无问题；79 条 exclude 正在独立二次审计，这只确认题面完整性。
- Batch8 的 57 条双复核交集已生成 query/action 绑定盲审队列，输入只含问题、回答、查询和模型可见检索文本；独立审查进行中。
- 正式 SFT-ready 仍为 **0/1,000**，计划符合性仍是部分符合；全历史暴露与 split 隔离未完成。

- 正式 SFT-ready：**0 / 1,000**；没有最终训练导出或 `training_ready=true` manifest。
- Batch9 question-only 盲审完成 2,000 题：keep 1,120 / uncertain 847 / exclude 33。只表示候选筛选，不表示 SFT 准入。
- Batch9 语义近邻 895 对已完成盲审和 hash/schema/ID 核验：same_task 864 / related_distinct 31 / unclear 0；仍需将配对映射到候选和历史集后裁决。
- Batch9 7B 教师轨迹队列 606 条运行中；2026-10-01 最近检查 manifest processed 352/606、attempts 352 行，状态 incomplete。verifier 仅作诊断，不作为准入。
- Batch9 语义近邻复核 895 对已完成并对账：same_task 864 / related_distinct 31 / unclear 0。双向映射 1,120 个 question-keep 后为 601 无已筛近邻、495 可能与历史同题、9 内部同题、15 仅相关不同题。606 条生成队列中有 5 条作为内部 same_task 的匹配侧漏入，保持生成不中断，但输出必须 hold。
- Batch8 原 338 条盲审输入中 337 条含完整检索返回，旧账本降为预审参考。重建后的 sanitized 队列已完成 338/338 条复核并通过校验：clear/sufficient/legal_relevant 交集 119 条，4 条缺教师终答 hold；115 条已同时送入独立答案-gold 等价与 response grounding 复核，均不等于正式 SFT 准入。
- Batch8 的 115 条答案-gold 复核已按原 UUID、输入顺序复原并通过独立 ID/schema/hash 校验：equivalent 82 / different 20 / uncertain 13；response grounding 为 grounded 81 / partial 29 / conflicting 3 / uncertain 2。两项同时通过 57 条，仍需 query/action 绑定、语义去重和历史/split 隔离。
- Batch10 从剩余 10,146 条 exact-known-history reserve 中固定抽样 2,000 条题面候选，切成 8×250 gold-blind 题面审查包；独立审查已开始。该批次尚无搜索轨迹，且完整 history exposure pending。
- 本地原始 `medical_o1_verifiable_problem.json` 有 40,644 行题目与 gold，没有已生成搜索轨迹；`medical_o1_sft.json` 有 19,704 行文本推理数据，也不能直接计为 Search-R1 搜索轨迹。可继续扩充题池，但需要教师生成和逐关复核。
- Reserve 中 8 条 evidence-sufficient 候选复核后仍全部 hold；5 条有 invalid action segment，2 条有重复或近重复搜索。
- 清洗计划符合性是**部分符合、整体未完成**：核心质量门槛和盲审流程在执行；答案-gold 等价、全历史隔离、最终去重、train/dev/test manifest 与 1,000 条正式准入仍未完成。Batch8 的盲审输入隔离问题在导出前发现，已通过 sanitized 队列纠正并重审。
- [完整进度及计划符合性报告](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/sft_cleaning_progress_20261001_continuation.md>)；[独立只读对账](</D:/code_list/some tricks/LLMLeanring/tmp/sft_plan_status_independent_20261001.md>)。

# 当前阶段

## 2026-10-01：失败轨迹评分函数验收通过

- 用户完成 `score_failed_trajectory`，评分分段及按 `response_start` 回填的源码逻辑符合当前任务；验收发现成功轨迹与空动作分支返回 tuple，正常评分分支返回 dict，违反统一返回协议。由用户将两处提前返回统一为包含 `teacher_logprobs` 和 `opd_mask` 的字典后复验。
- Agent 扩充 `MedSearch/tests/test_distillation_hook_jobs.py`，覆盖分段评分与回填、成功跳过教师、全 observation 跳过教师、失败动作缺教师报错。测试尚未执行到用例：本机 PyTorch 导入触发 `WinError 10106`，现有 WSL Docker 调用返回 `Wsl/Service/RPC_S_SYSTEM_HANDLE_TYPE_MISMATCH`，提升权限重试仍未解决。没有真实 7B 评分或训练，不报告通过。
- 用户已将两处提前返回统一为 dict，Agent 按授权补齐缺少教师的 `ValueError` 消息。复验时函数与测试 `py_compile` 通过；完整 pytest 再次在 PyTorch 导入阶段被 `WinError 10106` 阻止。已尝试提升权限调用现有 WSL Docker，以及备用 Node 进程启动通道：WSL 仍报同一 RPC 错误，Node runtime 初始化也失败。返回协议问题经源码复核已修正，但动态评分回填及真实教师验收仍待运行环境恢复，未声明测试通过。
- 后续定位并解决执行通道问题：工具启动命令缺少 `SystemRoot`/`WINDIR`，为当前进程补回 `C:\Windows` 后 WSL RPC 和 Python 原 Winsock 错误消失。默认 WSL 发行版无 Docker socket，改用 Windows Docker Desktop CLI 进入同一个 WSL-backed `llm-pytorch` 容器，不重启服务或容器。容器内运行 `python -m pytest MedSearch/tests/test_distillation_hook_jobs.py MedSearch/tests/test_distillation_teacher.py -q`：`7 passed in 7.88s`，退出码 0。
- 新增辅助脚本 `MedSearch/tools/smoke_failed_trajectory.py`，对本地 Search-R1 7B BF16 教师执行两动作夹一段 observation 的真实短评分：response 44 token，其中模型 token 28、observation 16；mask、CPU float32 回填、有限分数、成功项零分且不需要教师均通过。总耗时（含加载）51.53 秒，PyTorch CUDA allocated 峰值 14.29 GiB。报告 `MedSearch/docs/audits/score_failed_trajectory_smoke_20261001.json`；未启动训练，未验收 V1 整批 hook 或 checkpoint。
- 本次宿主机可复现命令：`C:\Users\dell\anaconda3\python.exe tmp\run_medsearch_score_checks.py`；真实教师 smoke 加 `--teacher`。该辅助启动器以参数数组调用现有 Docker CLI，并仅为子进程恢复 Windows 环境变量。
- 下一项用户核心任务：在 `rl/distillation_hook.py` 实现 `build_failure_batch(batch_meta, data)`，按真实 V1 `KVBatchMeta`/TensorDict 字段选择 acc=0 的有效失败轨迹，跳过合成 padding，保留原批次索引与 uid/trajectory key，不改变原 GRPO batch。工程书已把旧 RayPPOTrainer 示意接入点更正为 V1 `_balance_batch` 后、`_compute_old_log_prob` 前；本步只做失败项提取，教师生命周期与 loss 接线待后续验收。

## 2026-10-01：Batch10 SFT 清洗检查点（最新）

- Batch10 首批687条 gold-blind sanitized 轨迹复核已完成，七片通过独立根校验：advance到答案-gold复核181、hold 477、排除当前轨迹29；题目 keep/uncertain/exclude=667/18/2；730条引文绑定正确；41条待处理旗标均未advance。引用索引86处偏差及instructions SHA元数据笔误均已修复，根校验 errors=0。
- Batch10 的181/181条独立答案等价复核完成：equivalent101、different57、uncertain23，根校验errors=0。当前本批新增101条行级质量候选，尚非正式SFT。
- 本轮判断：181条是“可进入答案等价复核”的高精度中间候选，尚未证明与source gold等价；不是可训练SFT。此前抽查的一条模型答案与可见证据同为`tinea corporis`，source reference是`Tinea cruris`，将由等价审阅判定，不能从可见证据直接推成正确。
- 全池盘点：10个队列共5,088次原始rollout（含53次replay），verifier accepted 1,380只属诊断；候选并集648条。全历史语义裁决并合并旧RL专项复核后，9条同题候选block、1条有冲突hold，另有3434/15993一个内部同题对需择优。仅此语义/保护集检查后的理论上限637条；187条raw来源same-task仍待溯源。Stage-A/B、Search100、UltraChat 821行暴露边界未闭环，正式SFT-ready仍为0/1,000，无最终训练导出。
- 清洗计划：本批Batch10盲审遵守gold-blind、模型可见证据及分关审查；全历史语义复核按计划仅做人审线索、不因相似度自动删除。整体计划仍部分符合、未完成，因raw来源暴露核验、缺失历史资产、最终去重及train/dev/test freeze未闭环。旧360条partial/none题仍是RL重检意向，不计SFT；reserve 8条仍hold。
- 928条后续rollout输入已准备，但当前nvidia-smi返回NVML Unknown Error、8001/8002无监听、tasklist进程枚举报错且WSL返回E_ACCESSDENIED，无法安全确认运行环境与现存任务；暂不重复启动。648条候选语义/保护集对账已完成，下一步先处理source-reference与缺失历史边界；服务恢复后再继续928条。
- 详情与可复验账本：[Batch10 SFT质量检查点](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/batch10_sft_quality_checkpoint_20261001.md>)；[SFT清洗进度报告](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/sft_cleaning_progress_20261001_update.md>)；[候选级语义/受保护集对账manifest](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/sft_candidate_history_exact_20261001/full_history_semantic_review_20261001/candidate_level_semantic_reconciliation_20261001.manifest.json>)；[扩展历史exact join](</D:/code_list/some tricks/LLMLeanring/MedSearch/data/distillation/sft_candidate_history_exact_20261001/combined_report.json>)。

## 2026-10-01：SFT 清洗进度复核（继续执行）

- **正式 SFT 准入：0/1,000；清洗未完成。** 现有候选审查、题目 keep、priority、独立配对标签和 verifier 结果均不等于最终训练行；尚未生成最终训练导出或通过全历史隔离的 manifest。
- 已按 `MedSearch/docs/superpowers/plans/2026-09-29-sft-environment-data-review.md` 的质量门槛推进。原计划是先做 40 条 pilot；用户随后明确授权扩大到所有 SFT 并补足 1,000 条，故扩展范围但不降低题目、可见证据、答案等价、动作质量、泄漏与隔离要求。蒸馏总计划把“有合法非空最终答案”作为一般行为模仿的最低候选条件；按用户后来明确要求“题目正确、检索有强证据、生成高质量”的更高目标，本轮仍把这类轨迹仅视为候选，不视为 SFT 准入。题目盲审、gold/轨迹审查分开；诊断/补查不拼回旧轨迹；保留原始数据；uncertain/reserve 不当正例；仍无最终训练导出。
- 旧 cold-start 886 条已有轨迹的全量审查：question keep 581 / uncertain 293 / exclude 12；visible evidence sufficient 179；teacher answer equivalent 228 / different 344 / uncertain 314；此前建议 priority 128 / reserve 746 / exclude 12。这个分类仍是候选建议，不是最终准入。
- 跨批库存只读核对：393 条独立二审候选中有 91 条属于旧 cold1000 池，故 886 不能另行累加；移除已知 2 个 same-task 及 23 条被旧 priority 新对账降为 hold/reject 的记录后，旧候选保守约 368 条。Batch6 有 86 条独立二审 confirm 且与该池不重叠；Batch7 有 73 条首审 accept 尚待独立二审；Batch5 二审 confirm 的 85 条已包含在 393 中；Batch4 replay 53/53 是原题重跑。即使假设这些都通过并计入 Batch8 338 条，当前已知来源上限约 865，至少还缺 135 条唯一轨迹；这只是规划上限，非训练可用计数。
- 补充 gold-blind question-only 审查已完成且只读聚合校验通过：886/886，keep 488 / uncertain 394 / exclude 4；ID、原序、题面全文、引句与分片哈希检查通过。Q2（答案映射）及 Q5（暴露/重复）未评估，故不能据此认证 SFT。旧 128 个 priority 与本轮题面决定交叉为 101 keep / 27 uncertain / 0 exclude；27 个继续 hold。
- 旧 128 priority 独立轨迹二审与跨账本对账已封账：独立审查 confirm 108 / hold 18 / reject 2，strict audits 合并 confirm 91 / hold 19 / reject 18，Q-only 子集 keep 101 / uncertain 27；保守分流 provisional keep 68 / hold 39 / reject 21。86 条 independent-confirm + Q-only-keep 中仍有 7 条 strict hold、11 条 strict reject；18 条按较保守结果保留。`source_reference_overlap_flag` 在获准输入中 128/128 缺失；原 cold1000 attempts/accepted 没有字段或映射定义，Batch3 的专属定义不适用于旧池，因此保持 unknown，不解释为无重叠，细节见目录中的 `source_reference_overlap_reconstruction_audit_20261001.md`。对账报告位于 `MedSearch/data/distillation/sft_environment_review_20260929_full/priority_cross_review_reconciliation_20261001/`，`training_ready=false`。
- Batch8 全历史语义关系：候选级 provisional disposition 已生成且主代理复核 3 个主要输出 SHA 一致。390 对完整覆盖（382 新盲审 + 8 条精确复用），pair 关系 same-task 376 / related-distinct 13 / unclear 1；其中历史关系导致 373 条 provisional exclude、11 条 related-distinct 标记、1 条 unclear hold，558 条仅表示在已扫描边界无近邻。另有 3 个内部 same-task 连通簇，代表选择待定、未自动删除。旧 unmatched 58 条不迁移，cosine 阈值不自动决定重复；历史暴露边界仍不完整，所有这些状态都不是 SFT 准入结论。
- 全历史隔离仍为 `history_exposure_pending`：旧 Stage-A/B 资产、manifest 及 UltraChat 被选中的 821 条映射缺失。故不能声称历史暴露或最终 train/dev/test 隔离通过。
- 已在当前仓库检索 Stage A train/dev、Search100、Stage B protocol-dev 的精确文件名及 Git 全分支路径历史；均无可恢复副本或提交记录。缺失历史边界仍需从原训练产物/存档恢复，暂不以原始 UltraChat 全量替代 821 条选择映射。
- Batch8 的 338 条 question-only 合格候选已转成独立 fresh teacher rollout 输入，并于 2026-10-01 完成 Search-R1 7B rollout：`processed=338/338`，verifier 自动标记 accepted 93 / rejected 245。输入 SHA256 `c822b2ae87882871b7db48c30e90fcf0ed8f349fa4db88981d50b06b291764cf`；attempts SHA256 `fc229c2201d9856554ba622e2b4689add6a80aa2cb7a5a6dbd651769cc7143b6`，accepted SHA256 `7fecc8ecd91f38b549216fd1ba9639cf1dc56cacaca6c2dd7ded0ab3b5422172`，rejected SHA256 `d266fabde03023047858c6e837d4ee824aa631789807eef8db322acbb0cce493`。沿用历史成功配置：`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、显存比例 0.75；BM25/verifier 健康检查均 200。93 只是自动 verifier 结果，仍须独立 gold-blind 审阅，不计 SFT 准入。
- Batch9 已从冻结 reserve 另行生成 2,000 条 provisional 题目候选；seed `20261006`，完整 raw source 与 batch3–8 按 source ID/规范化题干精确去重后可选 12,146 条。source/reference 映射回查通过，抽样没有读取 gold/reference；候选 SHA256 `7ef60c52a3fa2b04ebc6fc4d5d4b52eb6cf6f992581bf3164914c27f82c12f15`，builder SHA256 `fc402bc4d3316803bd094465f9653d74f805fb5b2a1fe3650ee2969c03a5cfe8`。另建 gold-blind 8×250 审阅包（仅 `question_id/question`，manifest SHA256 `fed455084673f51aa7036c3561bef903a2e444eecdf76641162eb50036c06cc3`）；题面盲审正在分片推进，语义近邻脚本已准备但未运行，教师生成尚未开始。完整历史暴露仍 pending，`training_ready=false`。
- Batch9 question-only shard01–07 已独立审完并由主代理逐条复验 1,750/2,000：keep 984、uncertain 697、exclude 69；ID/顺序、完整题干引句、Q2/Q5/SFT `not_assessed` 与输入/账本哈希均通过。七片 ID 和 NFKC/casefold/去标点规范化题干均唯一（重复 0）。shard05 的 exclude 为 44/250，已另交不看旧判断的盲审质控；shard06 的唯一 exclude 是明确依赖缺失的肾上腺标注图。shard08 正在审阅。
- Batch9 MPNet 语义筛查已在 `llm-pytorch` 容器内用本地 RTX 5090 启动，输入/历史/模型哈希固定；容器检查前 GPU 仅用 1,337 MiB 且没有可见计算进程。CPU 尝试仅完成约 5,000 条后中断，未产生输出；主运行强制 8 个 CPU 线程用于数据装载，MPNet 在 GPU 编码。当前进度输出为 15,008/269,592 个历史问题，候选和历史 embedding 全部完成后才会写人工复核队列及 manifest，`training_ready=false`。
- Batch8 fresh trajectory gold-blind 审阅包已校验：338/338 question 与 attempt 映射、338 个唯一 ID；436/436 搜索文档哈希可解析、observation 哈希匹配、可见 information 是精确前缀；allowlist 不含 source ID、reference/gold、verifier/reward/自动结构决定。队列三片 113/113/112，manifest SHA256 `cc795596b0fef93c0dc694758f3717721b50e855d378d6db9380eb42799843d8`。326 条 search action/query 对齐；另 12 条内部校验为 query/action 绑定不一致，保留在盲审队列并默认 hold，不能据此预先拒绝或准入。训练准入仍 `training_ready=false`。
- Batch8 shard01 首次审阅尝试误读 `full_observation`，该审阅者已判污染并停止，不写 ledger；未读 gold/reference/hidden ID。替换审阅者最终账本经主代理独立核验：113/113 覆盖，队列/输出 SHA 正确，可见证据引句错误 0，schema/禁止字段错误 0；support 为 sufficient 52 / partial 38 / none 19 / conflicting 4，动作 legal 45 / needs_hold 56 / invalid_or_redundant 12。结论仍是盲审标签，不是 SFT 准入。shard02 的第一位替补审阅者误读完整 validation_report，看到了内部自动 hold ID；已停止并作废、未写账本。新 fresh reviewer 只拿 shard02 投影输入，禁止读取 manifest/validation_report；shard03 待分派。
- 历史 `reserve ∩ sufficient` 的 51 条已独立轨迹复核：0/51 可直接升级，8 条只可作为重生成题目，另 43 条继续 reserve。那 8 条当前仍是 hold（0 accept / 0 reject），需对新生成的完整轨迹重新审核；`sufficient` 仅表示旧可见片段覆盖答案所需证据，不能抵消旧轨迹动作、答案关系或审查一致性问题。此前所述 360 条题目可用但证据 partial/none，不进入 SFT；按用户决定保留为 RL 候选意向，须核对对应 ID 与正式 RL manifest 后才能称为已转入 RL。
- **下一步：**独立复审 Batch8 的 338 条完整轨迹（不暴露 gold/verifier），并审核 Batch9 的题面与语义近重复；只有通过题面 keep、无未决重复后才启动下一轮 fresh rollout。所有新增轨迹仍须通过可见证据、答案等价、动作质量、泄漏/去重及历史隔离门槛。先补齐历史暴露边界，再冻结 split 与最终导出。正式准入维持 `0/1,000`，直至逐行复验通过。

## 2026-09-30 17:30 UTC：cold1000 题面盲审封账，旧 priority 交叉复核

- 旧 cold-start 886 条的补充 gold-blind question-only 审查 4/4 分片已完成。独立脚本式回读逐行检查输入—输出 ID/题面/顺序，核验每条精确引句和 Q2/Q5 `not_assessed`：886/886 通过；汇总 keep 488、uncertain 394、exclude 4。原输入和分片 ledger 未改；aggregate report/manifest 正在单独封账。
- 与旧全量医学/证据审查产生的 128 个 priority ID 交叉：101 条 Q-only keep、27 条 uncertain、0 条 exclude、0 条缺失。27 条先按 hold 处理，不自动覆盖原审查或认定题目无效。独立 reviewer 正复审 priority 轨迹，已读 24/128；目前其中 `medical-o1-verifiable:3024` 因可能存在 exact-QA leak 暂记 hold，等待整体复审封账。
- Batch8 full-known-history 382 对已完成关系审查及独立结构核验。candidate-level 暂定去重映射正在生成：必须一对一合并 382 条新裁定与 8 条严格匹配的旧裁定；58 条旧 unmatched pair 不得迁移，cosine 不作为自动去重结论。完成前不得更改候选准入。
- 正式 SFT 仍为 0/1,000；缺失的 Stage-A/B 资产与 UltraChat 821 条选择映射继续使全历史隔离状态为 `history_exposure_pending`。上述 886 题面审核及语义关系账本都不是训练导出。

## 2026-09-30 17:16 UTC：SFT 清洗进度与计划符合性核对

- **结论：SFT 清洗尚未完成，正式 `training_ready` 导出仍为 0/1,000。** 题目候选数、人工 `priority`、轨迹盲审 accept 和可疑似去重数不能直接相加成训练样本；尚无最终训练文件或最终隔离 manifest。
- 既有 40 条 pilot 已按计划完成：抽样 40 个唯一 ID，审查结论 priority 4、reserve 34、exclude 2；原始数据未覆盖，有限补查独立落盘、没有拼回旧轨迹，主代理复核了 4 条 priority。pilot 报告明确指出未完成全量训练隔离，priority 清单不是训练导出。
- 旧 cold-start `accepted.jsonl` 的 886 条已有轨迹已有全量环境/医学审查：question keep 581、uncertain 293、exclude 12；可见 evidence sufficient 179；教师答案 equivalent 228、different 344、uncertain 314；最终建议 priority 128、reserve 746、exclude 12。引用/schema 问题为 0；所有 exclude 已主审确认，抽检 5 条 priority。与冻结 dev/final test 的 exact 冲突为 0，128 条 priority 在 5,852 条 dev/test 上以 cosine 0.92 做语义筛查匹配 0。此结果只验证相应旧池及其 dev/test 边界，不覆盖缺失的旧 SFT Stage-A/B 暴露；没有创建训练导出。
- 为进一步守住“先看题面、再看 gold/轨迹”的盲审隔离，旧 886 条另做 question-only 补充盲审：shard01/02/03 已封账 665/886，keep 375、uncertain 287、exclude 3；shard04 正在独立审阅。此轮 Q-only 结果不覆盖全量环境审查账本，也不能单独判训练准入。旧 886 全量报告的 128 个 priority 现已派独立二审，结果未回收。
- Batch8 全历史近邻复核的 382 个未复用配对已完成盲审：373 `same_task`、8 `related_distinct`、1 `unclear`；独立聚合核验 382/382 pair ID、顺序、题面、764 条引句均通过。shard01 reviewer report 的输入 SHA 少写最后一个 `f`（原报 63 字符）；实测输入 SHA 与封存 manifest 一致，ledger SHA 正确，已单独写 erratum 保留原始审计轨迹。该关系审查不自动排除或准入样本；前序 MPNet 队列共有 390 个 near-neighbor（另精确复用旧关系 8 个），源扫描为 943 个候选对 277,727 条可用历史记录。
- **历史暴露仍待解决：**四个旧 Stage-A/B 数据资产及 manifest 缺失；UltraChat 原始 shard 存在，但曾选入的 821 条 ID/映射不可恢复；旧记录数量口径不一致。因此 `history_exposure_pending` 必须保留，不能宣称全历史隔离通过。
- 计划符合性：40 条试点、输入哈希、题目/轨迹分审、模型可见 `<information>` 证据核查、答案等价性与动作质量分开判断、逐条引用、verifier 不作为唯一准入条件、原数据不覆盖，均有相应产物；用户明确授权从 pilot 扩展到全量和补足 1,000 条，范围扩大但质量门槛未放宽。尚未完成最终跨池去重、缺失历史暴露边界处置、1,000 条轨迹准入和训练导出。
- `reserve` 表示有潜力但当前至少一个 SFT 闸门未过，不等同坏题。此前 51 条 reserve 的可见 evidence 已足够支持答案，其中 8 条同时具备可用题面、证据充分和答案等价；这 8 条已做独立轨迹复核，8/8 仍为 hold（accept/reject 均 0），不能升为 SFT 正例。
- **下一步唯一核心任务：**把已完成的全历史配对决策安全应用到 Batch8 候选池，并独立复核旧 886 条中 128 个 priority 轨迹；并行中的 cold-start Q-only shard03/04 完成后，逐条通过 evidence、answer-equivalence、动作与全池暴露门槛，按实际合格数补生成 7B 轨迹。任何阶段不把 `training_ready` 置 true，直到恰有 1,000 条通过全部检查且隔离账本可回读。

## 2026-09-30 17:04 UTC：全历史配对盲审包已封账并开始人工复核

- 复核包已由独立代理生成且核验：390 个 MPNet near-neighbor 中复用旧账本 3 个 history + 5 个 internal 精确配对，382 个 history pair 待人工审阅；旧账本另有 58 个历史配对未与本次 top-neighbor 精确匹配，因此不迁移其关系标签。382 个待审 pair 保持原序、ID 唯一，分片 128/127/127；reviewer 输入只含两道原题、ID 与 pair_type，隐藏 cosine、完整 locator、gold/答案/轨迹/来源质量。manifest 当前 `prepared_not_reviewed`、`review_started=false`、`training_ready=false`。Shard hashes：01 `3cdd34b5e2af7a19a980fd82d04af6975d16fd4259a8fe61080d5a391729b18f`；02 `af5a1fef30507cbe75ae1ad55223d9d01836aca211b274a7681956476ea18819`；03 `9f98c643721a18f57fc5d875beaf93568de81aca4c529ba3e5fd3d7b83f19064`；lookup `da4a9890565f34069fde8645def1840ba5fa48d8a6ffa82cc07df5a480954f6b`；manifest `5523af9cfc3c0b06bb8c5a181af09e15d7e2ce5610fdd83c8cde2eab88d328aa`。
- Reviewer01/02 已接手 shard01/02，第三片待 reviewer03 完成当前旧 886 题面 shard02 后再审。没有自动排除 candidate。
- Git/工作树路径搜索均未找到四份缺失 Stage A/B 文件、manifest 或 UltraChat 821 条 ID；原始三个 UltraChat shard 不能被误认为实际使用的 821 条。

## 2026-09-30 16:57 UTC：旧 886 条题面盲审 shard01 封账

- 对旧 Search-R1 cold-start 的 886 条 answered 轨迹，已建立只含 `question_id/question` 的四分片 gold-blind 审查包（222/222/221/221）。shard01 的 222/222 账本完成并独立核验：keep 117、uncertain 104、exclude 1；ID/源序/题面全文/短引句全部匹配，明确未审 gold 映射 Q2、历史重复 Q5，也未看任何轨迹或 verifier。输入 SHA `edd434a6371c0a5d898638e9005ff3a3b2ebb6fc96476b1148c8367740b3fcf3`，ledger SHA `ec4f4118714b8d7e9d931bc9630f49be32f2e5dea1e4076593bd5767ed2e78d1`，review manifest SHA `31828e55bceda284eb8b3f0d3b035aeaa431e9d3421b76322ccd479c665e355e`。该 shard 不是轨迹或 SFT 准入结论；其余 664 条未审。

## 2026-09-30 16:52 UTC：全历史语义审计完成，进入配对复核

- GPU 全历史审计已完成并回读校验：943 个 Batch8 语义预筛候选 × 277,727 条完整可用历史记录，MPNet cosine 阈值 0.92、device=`cuda`、283.33 秒。输出 390 行：385 个候选到历史近邻（383 指向 batch1 history、2 指向 RL train），5 个批内近邻。输出 SHA256 `304a627a40580c77024756a7c0fc5e03f23ab0ee4794562aad395d162ecc4d36`，manifest 声明值一致；manifest SHA256 `935f9fea907d7ca4d72ba34f497ee082556fbf470ab9065ee1caedd1a555ff56`。这是人工复核队列，不是 390 条重复或自动排除。
- 从 629 道 question keep 中，251 道落入该次历史近邻队列，378 道没有达到 0.92 的历史近邻阈值；前者仍需逐对判 same-task/related-distinct，后者也不等同全历史唯一。保留旧人工pair账本只有在 candidate ID 与两题原文精确匹配时才重用：已实测可复用 3 个历史 pair、5 个批内 pair；剩余 382 个历史 pair 待盲审，review pack 正在准备，training_ready=false。
- 重新审阅旧 886 条 answer-presence 轨迹的 Q-only 包已开始构建；分片为 222/222/221/221，shard01 目前读到 50/222，尚无封账决定。旧 886 仍全部只是待清洗候选。
- Git 历史只读审计未能恢复缺失的 Stage A/B 文件或 UltraChat 821 条 ID；refs/unreachable 对象扫描找不到对应 data/manifest blob。仅找到 Stage-A 清理脚本和 Search100 汇总，无法重建逐条训练暴露，因此 `history_exposure_pending` 继续保留。详细记录见 `MedSearch/data/medical_verifiable_v2_batch8_20260930/git_history_exposure_boundary_audit.md`，SHA256 `e6eaf624ff5060e32a1d46482084ca8a0bcea4f293e0980f2a2fbc01e0432167`。
- 新增清洗文件未覆盖 canonical 数据。计划继续先完成人工近邻审阅与旧 886 题面/轨迹分审，再按通过数确定是否需要生成新教师轨迹；正式冻结仍 **0/1,000**，没有最终 split。

## 2026-09-30 16:43 UTC：全历史语义审计切换至 GPU

- 原 CPU 进程无检查点，最后报告 25,024/277,727；在核实 RTX 5090 可用后停止该进程，未产出或覆盖任何审核文件。随后以相同候选文件、相同完整历史输入、阈值 0.92，在新目录启动 GPU 复核：`semantic_screen_gpu/full_known_history_20261001/semantic_neighbors.jsonl`，batch size 128。GPU 会话 80580 首个 30 秒输出 25,088/277,727，运行正常；最终近邻清单及 manifest 尚未生成。该步骤只产复核队列，不自动剔除。

## 2026-09-30 16:36 UTC：SFT 清洗进度与历史边界复核

- 本轮新增的高置信 SFT 正式冻结仍为 **0/1,000**，无最终 train/dev/test 划分。旧版 886 条“有合法最终答案”轨迹只是答案存在性筛选结果，不能替代本轮医学证据、轨迹质量和历史隔离门槛；候选数、question keep 和 blind accept 均不计作最终 SFT 数。
- Batch7 轨迹盲审三分片已封账 **268/268**：accept 73、hold 103、reject 92。shard01/02/03 分别为 22/39/29、18/38/33、33/26/30。所有分片的 ID/顺序和题干绑定复核通过；已报告的 query/observation hash 绑定与可见 Doc 内引文定位均通过。review 中的 accept 仍须全局去重、暴露核查、必要的二审与最终轨迹准入；shard03 输入未提供 reference-overlap 字段，该项保持未知。
- Batch8 HuatuoGPT-o1 question-only blind review **1,000/1,000** 已封账：keep 629、uncertain 311、exclude 60。三个分片各自覆盖 323、343、334，ID/原序/题面及引文短语校验通过；审查未读 gold、教师答案、来源解释、轨迹或语义标签，因此 keep 不表示医学正确或 SFT 准入。语义候选池仍有 3 组内部 same-task 需要择优；`medical-o1-verifiable:7087` 与 `:36007` 的近重复需复核。
- 只读交集计数：question keep 629 与旧语义预筛后候选 943 的 ID 交集为 **589**；这是潜在教师轨迹输入上限之一，不是最终可训练数，仍待当前全历史 MPNet screen、批内代表选择、真实 7B 轨迹审查和全局隔离。
- Batch8 全历史 MPNet screen 进程仍在既有会话 53812 运行；最后一次明确输出为 **20,000/277,727**，无最终近邻产物，不能据此宣称语义暴露审计通过。只读 `docker stats` 实测该容器约 576.86% CPU、6.041 GiB/62.53 GiB 内存，说明编码进程仍消耗 CPU；完成 ETA 未确认。按原设计只生成近邻队列，不自动删除。
- 只读来源对账报告 `MedSearch/data/medical_verifiable_v2_batch8_20260930/source_inventory_reconciliation.md` SHA256 `ffb19c0389c87436967cc1def974cd0080fef7f4a98779712724de5e514bea10`：21/21 已声明历史输入路径与 SHA 匹配；`semantic_history.jsonl` 268,592 行、hash 与 manifest 相符。四个 Stage A/B 历史数据文件仍缺失，旧 isolation 快照的 `missing: []` 与当前状态矛盾；PROJECT_BOOK 的 Stage A 数量 16,077/328 与 Batch8 缺失边界记录 29,433/640 不一致。另有 UltraChat 原始三片共 207,865 行，但记录只说其中选出 821 条，821 条映射、Stage A manifest 与 finalizer 均无法恢复；不把整个原始语料错误当作历史训练集，也不能排除那 821 条可能被训练/用于 smoke。
- 计划遵从：执行了题目与轨迹分离、gold-blind 题审、模型实际可见 observation 引文绑定、问答相等性/证据/动作分别判定、manifest/hash 留存及不让 verifier 单独决定准入；没有覆盖 canonical 输入或把补查伪装成原检索命中。原 2026-09-29 书面计划最初只授权 40 条 pilot；用户随后明确授权扩展到所有 SFT 和补足 1,000 条，故按用户新范围继续，未降低证据门槛。尚未完成全历史隔离和最终交付，当前不能称清洗完成。
- 下一步：继续查 Git 历史是否能恢复 Stage A/B/821 条来源映射；待全历史近邻队列产出后逐对人工复核。先按本轮门槛重审已有的 886 条“有合法最终答案”旧轨迹（题面盲审与轨迹证据审查分开）；其通过数与 Batch6/7 候选、Batch8 题面 keep∩语义预筛 589 条合并去重后，按真实缺口补生成 Search-R1 7B 轨迹。任何旧/新增来源均需逐条核对可见证据、动作、答案等价性及暴露；达到 1,000 条后再冻结 split。

## 2026-09-30 16:20 UTC：全历史语义审计已在容器启动

- 通过已授权的 `llm-pytorch` Docker 执行环境绕过 Windows Python 导入阻塞。容器挂载路径和 MPNet 权重、Batch8 候选及历史 JSONL 均已只读确认存在；复用现有 `audit_v2_semantic.py`，由仓库根目录以模块方式运行，CPU 限 6 线程，输出到 `semantic_screen_cpu/full_known_history_20261001/`。进程会话 53812 仍运行，尚无近邻结果；不自动剔除任何题。
- Batch7 reviewer01 shard01/02 均已独立封账：179/268，accept 40、hold 77、reject 62；ID/顺序、题干 hash、可见 quote 及检索 locator 核验通过，`training_ready=false`。shard03 的 89 条已读完，账本/机器校验仍在收尾。
- Batch8 question shard02 的 343/343 已读完，正在写 ledger 与校验；因此三份题面均已读完 1,000/1,000，但目前只有 shard01/03 的 657 条正式封账，keep 430、uncertain 168、exclude 59。
- Windows `find /v /c ""` 初次误报 252,687 行；容器 `wc -l` 独立复核 `semantic_history.jsonl` 为 268,592 行，与 exact-dedup manifest 完全一致，无 15,905 行差异。reviewer01 被改派核对来源覆盖构造及 Stage A/B 边界，而不再追查已不存在的计数差异。

## 2026-09-30 16:07 UTC：历史隔离审计环境阻塞

- 准备将 Batch8 的 943 条 provisional 题与大历史清单做全量 MPNet 近邻复核。Windows `find /v /c ""` 初次误报 252,687 行；容器 `wc -l` 实测 `semantic_history.jsonl` 为 268,592 行（65,295,734 bytes），与 manifest 精确去重行数相符，包含原始医学/通用 SFT、MedQA 与 MedMCQA 来源。
- 复用 `MedSearch/tools/audit_v2_semantic.py` 的首轮执行在导入 `torch` 前失败：Windows Python 3.11 与 Anaconda 的 `asyncio`/`_overlapped` 均稳定报 WinError 10106；WSL 枚举返回 `E_ACCESSDENIED`，Docker API named pipe 返回 permission denied；Bundled Python 3.14 未安装 torch。审计未开始、未生成近邻结论，未改动系统配置。
- Batch7 shard02 已由 blind reviewer 读完 89/89，逐条 ledger 正在按 qid 封账；shard03 审核中。Batch8 question shard02 343 条仍审理中。其余轨迹、语义人审与 split 状态未变，正式 SFT 仍为 0/1,000。
- 后续先核清 full-history JSONL 与 manifest 的记录差异、恢复可用隔离审计运行时，再扩充候选语义近邻检查；不因环境错误把缺失边界视为通过。

## 2026-09-30 15:57 UTC：SFT 清洗进度复核

- 正式冻结 SFT 仍为 0/1,000；没有最终 train/dev/test 划分。此前全局语义二审池 391 条与 Batch6 独立二审 confirm 86 条分别报告，尚未合并成全局唯一数，均非 training-ready。
- Batch7 轨迹盲审 shard01 已封账 90/90：accept 22、hold 39、reject 29；引文及检索定位均绑定到相应可见轨迹，且 reviewer 未看结构审计/verifier。shard02、03 正在独立审核，余 178/268。
- Batch8 题面盲审已完成 shard01、03 共 657/1,000：keep 430、uncertain 168、exclude 59；shard02 的 343 条仍在审。该批尚未生成 7B 教师轨迹。
- Batch8 语义派生清单 943 行只排除了 57 条历史 same-task，批内 3 对还需择优；清单标为 provisional、training_ready=false。Stage A train/dev、Search100、Stage B protocol dev 等历史边界未解决，不能声称完整暴露隔离。
- 执行计划：继续封账 Batch7/8 盲审；对通过题面和语义门槛的候选再生成/审阅 7B 轨迹；并行盘点可用历史源和缺失 Stage A/B 边界；最终统一全池去重与 train/dev/test 冻结。40 条试审计划已按用户指令扩展为全量清洗，但没有放宽证据、动作、答案等价或隔离门槛。

## 2026-09-30 15:38 UTC：动态核验补充

- 正式冻结 SFT 仍为 0/1,000。原 393 条暂定二审候选的 3 个历史语义近邻已完成人审：2 条判 same_task，需在派生清单中排除；1 条判 related_distinct，暂保留。因此暂定有效候选最多按 391 条继续核验，尚未生成最终过滤 manifest，也未通过完整暴露审计与 split 冻结。
- Batch6 首轮 accept 的 99 条独立二审最终为 confirm 86、hold 12、reject 1；99/99 条响应 `<information>`、104/104 个检索轮次已按相同 tokenizer 与 700-token 截断边界重建核验，全部可见 Observation 与响应块逐字一致，检索 hash 重建 0 失败。1 条（ID 25683）因原始表格抽取拆断关键句从原始汇总 87/11/1 改为 hold，最终账本、报告、Manifest 均经覆盖及 SHA 校验；86 条 confirm 仍须全局去重/暴露核验，不等于正式训练准入。
- 393 条暂定确认池已按三对语义人审生成非覆盖派生清单：same_task 历史重复 2 条排除，related_distinct 1 条保留，产物 391 行、training_ready=false、无 split。脚本：`MedSearch/tools/apply_provisional_semantic_decisions.py`；清单 SHA `0853528eff00d5d0bee61a30a193960f5cf221a4270b397e22f49a07cc4bbc94`；来源计数 batch3 111、batch4 100、batch4 replay 4、batch5 85、old SFT 91。
- Batch7 trajectory blind shard01 已逐条核读 90/90；正在修正草拟账本与原始输入的行映射后再封存，故尚无正式decision计数；另外 178/268 条盲审队列待审。审查只使用 blind inputs，未读取结构审计/verifier判断。
- Batch8 的 1,000 条仅是 HuatuoGPT-o1 的 gold-blind 题面候选；全量 CPU 语义编码及输出完整性复核通过，对比可用历史题 25,131 行/11,601 个规范化唯一题，精确历史重合 0。66 对语义近邻已盲审：历史 61 对中 same_task 57、related_distinct 4；批内 5 对中 same_task 3、related_distinct 2；无 unclear。已生成 943 行历史语义过滤候选与独立排除账本，输出 SHA `617aa536368b43fd9d3db594a43b048418b745fa277b8695a9d44f250e7406b9`；批内 3 组需在轨迹/题面质量审完后择优留一，故本轮语义后理论最多 940 个不同题。题面审核输入 323/343/334 正在独立复核，尚无 Search-R1 7B 轨迹、training_ready或完整历史隔离结论。

## 2026-09-30 15:20 UTC：SFT 清洗在途状态

- 最终冻结的 SFT 训练集仍为 0/1,000；尚未完成全池去重、历史训练暴露核验和最终 train/dev/test 隔离。人工确认数和教师 rollout 候选均不计作可训练样本。
- 目前有 393 条跨批独立二审确认候选（batch3 112、batch4 101、旧 SFT 91、batch5 85、batch4 replay 4）；已回溯到原始教师轨迹，question_id、原题和 NFKC/大小写/字母数字归一化题干精确去重均为 393 unique、0 个重复组。跨历史语义去重、训练暴露核验和最终 split 仍未完成，不能称为最终可训练的 393 条。batch5 的 105 条引用已验证到模型可见片段；1 条拒绝，第 88 条无支持引文，第 67 条题干与证据冲突。batch4 replay 二审为 accept 4、hold 3、reject 0。
- batch6 的 305 条 7B rollout 有 273 条合法最终答案、32 条无答案；结构筛选后 256 条进入人工复核，17 条结构排除（标签原因可重叠：无可见 Doc hash 5、非法动作 13、无搜索 4）。三份盲审账本均已完成 256/256 覆盖：accept 99、hold 79、reject 78；205 条引用逐字绑定到可见片段。三份均保持 `training_ready=false`。99 条首轮 accept 的盲审包已生成，JSONL 与 Manifest SHA 经 `certutil` 核对一致；独立二审正在进行，首轮 accept 尚未计入二审确认数。
- batch7 题面盲审完成 1,000 条：keep 543、uncertain 433、exclude 24；231 对语义关系全审完：exact/near_duplicate 194、same_task 32、related_distinct 2、unclear 3。派生 316 条 7B 教师轨迹输入，223 条与历史重复簇关联、1 条批内重复排除，3 条语义不清暂缓。7B rollout 已完成 316/316，verifier 标签 100 accepted、216 rejected；它们不是人工证据审查或 SFT 准入结论。结构审计已覆盖 316/316，268 条进入盲审队列、48 条结构排除；408/408 条搜索 observation、1,545/1,545 个可见 Doc-title/hash 对及 2,040/2,040 个返回文档 hash 均绑定通过。268/268 blind review 副本已核对 ID/顺序和必需内容，隐藏结构判定字段，不含 verifier 分数；队列 90/89/89，医学证据与轨迹人工复核待开始。模型、语料、输入、配置及输出均留有 SHA256。评估入口初始提示只含题目，references 仅传给后台 verifier。
- batch8 已从 HuatuoGPT-o1 准备 1,000 条 provisional 候选，seed=20261005，并生成 323/343/334 三份 gold-blind question-only 审查输入；候选 SHA 已独立核验。仅排除了目前可用的 exact-known-history，状态为 `history_exposure_pending`：Stage A train/dev、Search100、Stage B protocol dev 四个历史输入共 30,185 行缺失，因此题目审核可继续，但不能宣称完整隔离或进入 7B/训练。该批 semantic review 与题面审核仍待完成。
- 393 条 provisional confirmed pool 的历史语义 CPU 核验已完成：MPNet Top-1 cosine≥0.92 的历史近邻 3 条、精确/规范化历史重叠 0 条；3 条仅进入人工关系复核队列，不自动排除。11,601 个规范化可用历史问题完成编码；CUDA 已禁用。Stage A/B 缺失集单独列为暴露边界。
- 旧 reserve 中 8 条独立复核全部为 hold、accept/reject 各 0，training_ready=false；9 条引文、10 条查询/observation 绑定及 50 个语料文档哈希均通过验证。暂缓原因为无效动作、重复搜索、答案范围过宽/证据不足或引用落在 `</information>` 外。
- 当前顺序：batch6 99 条首轮 accept 的独立二审收尾；完成 393 条池中 3 个历史语义近邻的人审；batch7 268 条轨迹盲审输入已就绪；batch8 1,000 条题面盲审；全池历史暴露盘点并行推进，最后完成 train/dev/test 隔离。最终达到 1,000 条前不启动正式 SFT。
- 清洗遵循计划书的题目有效性预筛、语义近邻人工判定、7B 教师轨迹、可见检索证据与动作结构筛选、独立复核、verifier 不作为唯一筛选器和末端数据隔离门槛。最初 40 条试点已按用户要求扩至全部 SFT；历史 Stage A/B 源文件缺失造成的暴露审计边界仍须保留。

## 2026-09-30：SFT 清洗状态续报

- SFT 清洗尚未完成，正式冻结的最终训练样本为 0/1,000。此前独立二次确认的 304 条（batch3 112、batch4 101、旧 SFT 91）仍需全池语义去重、训练暴露审计和 split 隔离，不能直接作为最终数。
- batch5 的 402 条轨迹人工审核现已按真实输入 ID 完成三批复核：accept 106、hold 163、reject 133。第 03 批因此前索引映射不一致曾撤回临时结论，后按原始输入 134/134 重审并逐引文校验。106 条 accept 的决策盲独立复核输入已由 `MedSearch/tools/prepare_batch5_independent_accept_audit.py` 准备在 `trajectory_review/independent_accept_audit_final/`，尚未复核；通过后仍需跨池去重。
- batch4 hold 中挑出的 53 条 replay_candidate 已用固定 Search-R1 7B revision 重放并完成首轮人工复核：53/53，50 条有合法最终答案、3 条无答案；48 条结构进入队列，2 条因非法动作排除。48 条复核结果为 accept 7、hold 34、reject 7；42 条引用逐字可回查，另发现 7 条不能把 `</information>` 外的 Doc 文本算作证据。verifier 记为 accepted 30、rejected 23，但只作日志信息；7 条新 accept 仍需独立复核和跨池去重，不计入最终 SFT。账本在 `MedSearch/data/medical_verifiable_v2_batch4_20261001/replay_trajectory_review/`，整理脚本为 `MedSearch/tools/prepare_batch4_replay_trajectory_review.py`。
- batch6 的 1,000 题 gold-blind 题面审核与完整性校验已完成（keep 506、uncertain 420、exclude 74）。仅 keep 相关的 202 对近邻已全部人工复核：195 对直接命中历史同任务，2 对为池内同任务，4 对 unclear，1 对 related-distinct。对同题内部重复按关系簇处理：其中一个内部候选还连到历史重复，因此整个相关簇都排除；另一纯内部对保留较小 source_id。生成 305 条 teacher rollout 输入、196 条历史重复排除、1 条内部重复排除、4 条 unclear 暂缓；只进入教师轨迹审核，不代表 SFT 合格。整理包在 `MedSearch/data/medical_verifiable_v2_batch6_20260930/teacher_input_review/`，`training_ready=false`。
- batch7 仍处于候选准备阶段，1,000 题尚待题面盲审、相关语义近邻裁定和 teacher 轨迹审核。
- 清洗计划执行情况：当前候选冻结与哈希、gold-blind 题面审查、轨迹可见证据逐字回查、query/Doc 绑定、verifier 不作准入、near-duplicate 人工复核等门槛均按计划推进；全池历史暴露审计和 train/dev/test 隔离未完成，因此整体清洗不能判为完成，也不能启动正式 SFT。
- 当前推进顺序：收尾 batch5 第 03 批并复核其 accept；完成 batch6 keep 近邻裁定；审核 batch4 replay 新轨迹；随后对所有候选统一做历史暴露、语义去重和 split 冻结，再判断 1,000 条目标缺口并扩充。

## 2026-09-30：SFT 高置信检索轨迹清洗继续进行

- 当前按用户要求优先构建 1,000 条高置信 Search-R1 SFT 轨迹；尚未启动训练。清洗目标是题目可答、gold 等价、当时可见检索证据充分、推理受证据支持、动作轨迹干净，并保留原始 token/mask 和数据来源。
- batch3 的 775 条 7B teacher attempts 中，674 条有合法答案；638 条通过结构门槛进入人工轨迹复核，另 137 条未进入人工复核（101 条无合法答案、36 条结构排除）。638 条人工结论为 accept 127、hold 201、reject 310。
- 根级 teacher-input 汇总与独立回读已通过：batch4 1,000 题分为 teacher_input 517、hold 258、exclude 225；7 个内部重复成员随 teacher input 保留，最终 SFT 导出须每个重复组最多保留 1 条。teacher input 的 SHA256 为 `f10add3438c43ea1afa7ca7ec835cbd02c579035ab940dbf1eb8f1449c9b178a`，输出清单位于 `MedSearch/data/medical_verifiable_v2_batch4_20261001/`。batch4 rollout配置与结果见下文。
- batch3 的 127 条初审 accept 已完成独立二次复核：confirm 112、hold 9、reject 6；未复核 0。112 条单独派生至 `trajectory_review/independent_confirmed.jsonl`，对应引用和哈希清单为 `independent_confirmed_citations.jsonl`；独立审计与派生 manifest 同目录。初审 `strict_accepted.jsonl` 原件保持不变。跨批语义去重和最终 split 隔离仍待完成，不能把 112 条直接称为最终训练集。
- 启动 batch4 前已核验：Search-R1 7B 容器运行中，RTX 5090 显存使用 3,122 MiB、GPU 利用率 0%；检索与 verifier 的 OpenAPI 路由均返回 200；容器工作区可用磁盘约 881 GB。batch4 使用与 batch3 相同的 7B revision、检索配置、响应长度与轮数，仅换入审核后的 517 道题和独立输出目录。
- batch4 的 Search-R1 7B rollout 使用冻结配置：517题、revision `74d46a43224daa612b5f27174e1c58b35587ee92`、top-k 5、最多2次搜索、observation 700、response 4096、batch 8；rollout 已完成，数量与审核情况见下条。
- batch4 rollout 已完成：517/517，164 verifier-accepted、353 rejected；answered 派生集为456条有合法答案、61条无答案。结构门槛通过430条进入人工证据/推理复核，26条唯一轨迹结构排除；invalid_action 22、no_search 2、无可见检索文档2、精确重复search2是可重叠问题标签；430条拆为144/143/143三批。verifier accepted 不等于人工确认或 SFT 通过。
- 旧 SFT 128 条 priority 轨迹严格复核闭环：第一半 64 条为 confirm 48、hold 12、reject 4；第二半 final 64 条为 confirm 43、hold 7、reject 14。两半合计91条 confirm、19条 hold、18条 reject。第二半 final SHA256 `43a8c5e8992357b200257142a69e3c20e02b63e8ded3024ac68d72097c4818bb`，经 agent01 独立核对；旧错版保留并标记作废。旧语义筛查 manifest 已对 5,852 条 dev/final-test 题检查且高相似匹配为0；训练集隔离仍需独立核验，因此91条未计为最终 SFT。
- batch5 1,000 候选的 gold-blind 题目复核三批全部完成：keep 579、uncertain 383、exclude 38。分批计数为 324条（178/130/16）、334条（220/100/14）、342条（181/153/8）；三份输出哈希均已复核。排除项主要是题干依赖缺失列表/检查结果或明确同题重复；keep 仅代表可答性预筛，不代表轨迹可进入 SFT。
- batch5 1,000 条候选的题面与候选级语义隔离已完成：题目 keep 579、uncertain 383、exclude 38；464 对语义审查为 same_task 450、related_only 14、uncertain 0。聚合为 491 条可进入教师轨迹审核、152 条历史同题排除、320 条题目暂缓、37 条题目排除；297 条 raw MedQA/MedMCQA 来源重叠仅标记，14 条 related-only 未排重。`candidate_semantic_eligibility_batch5.jsonl` SHA256 `400be676fade51af3ebede8010652df9ca6874a586103dc33d0cd8f5ec73d3aa`，manifest 显示 `sft_ready=false`、`rollout_started=false`；这些仍只是教师轨迹审核候选。历史 SFT 同题动作冲突已统一归一化并记录。
- batch4 的 430 条检索轨迹人工复核已全部完成：batch1 144/144（accept 44、hold 55、reject 45）；batch2 143/143（41/44/58）；batch3 143/143（35/60/48）。总计 accept 120、hold 159、reject 151；batch3 的 143 行 canonical ledger 为 `trajectory_review/trajectory_review_manual_03.jsonl`，SHA256 `d7d3c1c0f5affc16bd647a185149bbe230413aa4365cfe0d1ba5e11a5ce2fb14`，105 条可见证据引文逐字匹配正确查询和 Doc，48 个 source-reference overlap 标记原样保留，所有 accept 通过严格准入门槛。每个 accept 仍需跨批去重和冻结 split 隔离，不能直接计入最终 SFT；batch2 中 14738 因答案虽算对但无检索支持已由 accept 降为 hold。
- batch4 的 120 条初审 accept 已完成独立二次复核：40 条分组结果分别为 30/10/0、34/6/0、37/3/0（confirm/hold/reject），合计 confirm 101、hold 19、reject 0。独立审计文件分别为 `trajectory_review/independent_accept_audit_001_040.jsonl`、`_041_080.jsonl`、`_081_120.jsonl`，没有修改 canonical 初审账本；暂缓理由包括题目缺少范围或选项、证据未支持首选/关键关系、证据仅适用于特定情境、引文归属不准。101 仍须跨池去重及冻结 split 隔离，不能直接计入最终 SFT。
- batch4 全部 159 条 hold 的重跑适配性分流已完成：53 replay_candidate、70 question_or_gold_issue、30 answer_issue、6 other_or_uncertain。三个独立输出 `replay_suitability_batch_01/02/03.jsonl` 的 SHA256 依次为 `ce3dde5d5552ccf5935031869678403eb2c3e6e32d958d4abcb049cf346f7f77`、`f88bb3512f54b072fb6d2c38faa8a2cf45986104d244d05493e6968487e7f945`、`dda8f1a430cf58af872f7cc38beb6b53563abec11a339028d834ac8614e7aefc`；canonical ledger 均未修改。53 条已整理为 `MedSearch/data/medical_verifiable_v2_batch4_20261001/replay_retry_input.jsonl`，仅含明确 replay_candidate，未含其他分流类别。另行逐字复核 39841、40192 后，旧 hold 均成立：39841 的 Doc 5 引文实际在 `</information>` 标签后，不能按严格可见证据门槛收录；原“检索片段没有该菌”理由不准确，已记录正确边界原因。40192 题目无选项，rocker-bottom feet 与 small, upturned nose 都可接受，答案不唯一。独立核对文件 `hold_adjudication_39841_40192.jsonl` SHA256 `010eb67b21afee1e991c017a8b9f21cfa5feef7758f5ccad6dec6d8d4432d7f4`。
- batch5 候选池按同一 Huatuo revision 从剩余 reserve 独立构建：16,146 条精确隔离 reserve 中抽取 1,000 条，源文件 `candidate_batch5.jsonl` SHA256 `019704053a2e273ea192cfebd10891550b0f81f8a93655f0351cfb64ffe35126`。491 条审核队列已派生为独立 rollout 输入，SHA256 `3e2b707a3ae406417630a536a0326930f8d795bc67fc95ea987380cf308e3e08`。先前尝试均在生成前失败：脚本入口修正为 `python -m`；WSL2 使用 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；sm120 与容器 nvcc 12.8 不兼容后，改用已验证可用的 vLLM 非 FlashInfer sampler（`VLLM_USE_FLASHINFER_SAMPLER=0`）；随后从日志发现教师导出还需显式 `--limit`。已使用同一模型 revision、输入、检索配置和采样参数带 `--limit 491` 完成运行；任务状态与日志记在 `teacher_rollout_runtime.json` 和 `MedSearch/outputs/distillation/teacher_batch5_20260930.log`。rollout 完成 491/491：verifier accepted 138、rejected 353（只记录，不作 SFT 过滤）；有合法 final 431、无答案 60。结构预筛后 402 条进入人工证据/答案/推理审核，29 条结构排除（no_search 5、无可见文档 hash 5、invalid_action 20、重复查询 4，原因可重叠）；token/mask 不匹配和响应超 4096 均为 0。结构 manifest 在 `MedSearch/data/medical_verifiable_v2_batch5_20260930/trajectory_review/manifest.json`；人工审核仍待完成，尚无 batch5 条目计入 SFT。未启动训练。历史 Stage A/B 缺失限制仍保留。

- batch4 的 53 条明确 replay_candidate 已整理到 `MedSearch/data/medical_verifiable_v2_batch4_20261001/replay_retry_input.jsonl`；manifest 记录 `training_ready=false`、`model_started=false`，且题目/gold/来源/前序轨迹逐项对齐。等 batch5 7B rollout 完成后再决定重跑；其结果需重新过证据和轨迹人工门槛。

- batch6 的 Huatuo 候选池已冻结到 `MedSearch/data/medical_verifiable_v2_batch6_20260930/`：1,000 题、seed 20261003，精确隔离 reserve 15,146 条，semantic history 266,592 条；`candidate_batch6.jsonl` SHA256 `8f382f583b3017a22b5a64f3686ef8f8b41d47cf37de8fdef41436a62e8992c7`，与 batch5 及历史题面无精确重叠，`training_ready=false`。gold-blind 问题审核已完成，分片 322/351/327，keep 506、uncertain 420、exclude 74；1,000 条 assignment/输出 ID、题面哈希与必需字段独立完整性审计通过（不代表医学答案审核）。MPNet semantic history 266,592 对候选 1,000 题、cosine>=0.92 筛查生成 395 个近邻记录（history 386、internal 9），无自动排除，输出与 manifest 哈希一致。仅对 question-keep 相关的 202 对建当前 SFT 人工复核队列（history 200、internal 2；分批 55/77/70）；其余 193 对涉及 uncertain/exclude 题，留在全池审计记录，不进入当前训练候选审核。`training_ready=false`。
- batch7 候选仅预处理已完成：同一 Huatuo revision，从 14,146 条精确隔离 reserve 中按 seed 20261004 抽取 1,000 题；语义历史含 batch5/6，计 267,592 条。`candidate_batch7.jsonl` SHA256 `5d7772649f337f6ff4a496620524823ca755e31bfbd8283143969bea5f4ecf7c`，与 prior exact history 隔离；gold-blind 问题审核分片 335/343/322 已准备。MPNet cosine>=0.92 筛查产生 394 个近邻记录（history 387、internal 7），独立校验输出数/类别/阈值/哈希一致，不自动排除；题面盲审、keep 子集近邻人工裁定和 teacher rollout 尚未完成，`training_ready=false`。

## 2026-09-29：MedSearch 冷启动 A 接线完成，SFT 训练前做医学质量试审

- 按 `MedSearch/DISTILLATION_PLAN.md` 和工程书，MedAgent harness 暂缓。冻结 `cold_start.jsonl` 1,000 题与 RL train 5,756 题、dev 396 题分开；本次仅对冷启动池按文件顺序每题做一次本地 Search-R1 7B rollout，没有把 gold 放入生成 prompt。
- Agent 在现有 `MedSearch/tools/evaluate_agent_rollout.py` 增加教师导出模式，保存原始 `prompt_ids/response_ids/response_mask`、动作段、检索来源哈希、模型/tokenizer/语料/代码身份、失败原因；4 项容器回归测试通过。Luna 在既有 WSL Docker 中先做 2 题 smoke，再运行全量 1,000 题。脚本及测试分别在 `MedSearch/tools/evaluate_agent_rollout.py`、`MedSearch/tests/test_teacher_trace_export.py`。
- 原始产物在 `MedSearch/data/distillation/teacher_cold1000_20260929/`：attempts 1,000 条，按当时严格规则 accepted 200 条、rejected 800 条，manifest 为 complete。Agent 独立复跑 `MedSearch/tools/validate_teacher_cold1000.py`，确认 1,000 个唯一题号、顺序、token/mask/segments、gold 隔离及三文件哈希均通过。完整命令、运行日志、弱标签与首题缺图限制见 `MedSearch/outputs/distillation/teacher_cold1000_20260929_REPORT.md`。7B 七个权重分片重算哈希与固定 revision 旧审计 7/7 一致。
- 本次没有训练，也没有付费 API。未连续采样 GPU 峰值，不报告峰值；约 71 分 42 秒是模型加载、rollout 和 verifier 的总时间。accepted 仅表示当前协议、检索与 verifier 通过，不能解释成医学专家审核。
- 当时安排的唯一编码任务是由用户给冷启动 A 档接入 Dataset/批处理输入，复用现有 `encode_distillation_trace` 与 SFT collator，让模型只学习 `response_mask=1` 的原始教师 token；现已完成并通过数据层验收。Top-32 缓存及 B 档损失在 A 的小步输入/梯度验收后接入。任何较长训练前，必须验证 checkpoint 保存、重载和一次动作推理；训练配置不得沿用 `save_strategy=no` 或 V9 的 `save_freq=-1`。
- 已给该任务写好 `MedSearch/tests/test_sft_search_distill.py`；容器内运行 `python -m pytest MedSearch/tests/test_sft_search_distill.py -q` 如预期 2 failed，原因均是现有 `SFTDataset` 尚不支持 `search_distill` stage。用户完成最小接线后复跑验收。
- 2026-09-29 用户明确改定冷启动轨迹筛选：只删除无合法最终答案的轨迹，不以 verifier 是否判对、是否曾有无效动作或是否检索作为过滤条件。Agent 只从已冻结的 1,000 条 attempts 派生新清单，未重跑 7B、未改原始导出。`MedSearch/data/distillation/teacher_cold1000_20260929_answered/accepted.jsonl` 有 886 条，`excluded_no_answer.jsonl` 114 条；原严格通过的 200 条全部保留，旧规则拒绝但有合法答案的 686 条加入。新 manifest 记录父产物哈希和新文件哈希。容器内新筛选脚本测试 2 passed；下一步 Dataset 应读取这个新 `accepted.jsonl`，而非旧的 200 条文件。
- 用户已完成 `sft/data.py` 的 `search_distill` 接线；Stage A/B 及相关文件由用户明确授意废弃并移除，不再按旧阶段做回归要求。Agent 在 WSL Docker 运行 `test_sft_search_distill.py` 与 `test_distillation_data.py`：3 passed；逐条编码新清单 886 条，最长 2,274 token、每条目标 token 41–673，均未触发 4,096 上限；本地 3B tokenizer 拼两条真实样本得 `[2,1820]`，padding 的 label 均为 -100，目标 token 总数与原始 mask 相等。`sft/data.py`、`sft/train.py`、`distillation/data.py` py_compile 通过。尚未加载学生权重训练或验收 checkpoint；后续唯一门槛是一次可保存 checkpoint、重载并完成动作推理的短训练 smoke，禁止沿用默认 `save_strategy=no`。
- 2026-09-29 按 `MedSearch/docs/superpowers/plans/2026-09-29-sft-environment-data-review.md` 完成 40 条 SFT 医学质量与证据试审：从 886 条有答案轨迹抽 random 20，从 1,000 条原始尝试补 boundary 20。结论为 priority 4、reserve 34、exclude 2；`:1652` 缺图及 Structure C 映射，`:15070` 缺痰液 Gram stain 结果。主代理复核全部排除项、冲突项及 4 条 priority 的模型可见证据；40 条原始数据引用、query 与 document hash 均通过定位校验。
- 有限补查仅作诊断：Search-R1 7B forced-search 34 条，每题检索 1–2 次；题目导向 BM25 28 条、全部成功，2/28 找到直接支持 gold 的文本。补查未拼回旧轨迹，未生成训练 JSONL；原 `accepted.jsonl`、attempts 和冻结题集哈希复算一致。报告、审阅账本、候选名单与诊断原文在 `MedSearch/data/distillation/sft_environment_review_20260929_pilot/`。没有启动训练或付费 API。vLLM 日志记录 SM 12.x/CUDA 12.9 能力探测警告，随后使用 FlashAttention 2 并完成输出。
- 2026-09-29 用户决定：SFT 数据继续在独立窗口清洗，本对话暂时跳过 SFT 训练与 checkpoint smoke；它们仍是任何较长训练前的必经门槛。清洗结果未验收前，不固定正式训练清单。
- 本对话已核对真实入口：`main_ppo.py` 在默认 `trainer.use_v1=true` 下进入 `ppo/v1/trainer_base.py` 的同步训练器。V1 AgentLoop 先算 reward 再尝试教师评分；`ReplayBuffer.sample()` 在 `algorithm.filter_groups.enable=true` 时会清除组内 `acc` 全相同的组，包括全错组。采样后由 V1 trainer 计算 old logprob、GRPO advantage，再更新 actor，`save_freq>0` 时保存 checkpoint。下一步唯一编码任务是为新蒸馏实验配置关闭组过滤并用最小全错组测试验证它会进入训练 batch；不改旧 V9 脚本，也不启动长训练。
- 2026-09-30：Agent 已新增 `MedSearch/scripts/run_medsearch_grpo_distill.sh` 作为独立单步接线入口：关闭动态组过滤、冻结 reference KL 与 veRL 原生蒸馏，V1 sync、双轨迹、`save_freq=1`，要求显式提供冷启动 checkpoint；Git Bash `bash -n` 通过。现有 WSL Docker 调用返回 `Wsl/Service/CreateInstance` 错误，尚未运行 ReplayBuffer 全错组 CPU 验收或训练，不把配置检查当作完整链路通过。下一项学生核心编码任务为从一条失败轨迹构造逐动作教师评分任务，后续再把评分结果接入整批 OPD。
- 2026-09-30：学生完成 `rl/distillation_hook.py` 的 `build_teacher_score_jobs`。Agent 以双动作夹两段 observation 和全 observation 两例验收：前缀只含当前动作之前的真实上下文，目标只含模型 token，`response_start` 可用于原轨迹回填；本机禁用无关 pytest 自动插件后 `2 passed in 0.02s`，`py_compile` 通过。尚未接真实教师评分、批次失败筛选或 V1 trainer。

## 2026-09-30：全量 SFT 数据质量与证据复审完成

- 按 `MedSearch/docs/superpowers/plans/2026-09-29-sft-environment-data-review.md` 完成固定 `accepted.jsonl` 的全量 886 条审查。最终分为 priority 128、reserve 746、exclude 12；仅 priority 是本轮高精度候选，尚未生成训练导出，也没有启动训练。
- 主审复核原标记为 conflicting 的 55 条：39 条改为 partial、6 条改为 uncertain、3 条改为 sufficient，7 条保留 conflicting；9 条回看完整原题和原始 observation。另逐条复核 12 条 exclude 和 5 条 priority 抽检。所有 priority 引文均定位到模型实际可见的 query/Doc/hash/observation。
- 固定 dev 396 条、MIRAGE final test 5,456 条与 accepted SFT 做 ID/规范化题干精确隔离核对，冲突 0。128 条 priority 候选以本地 all-mpnet-base-v2、384-token 截断、cosine 阈值 0.92 对冻结 dev/test 和候选集内部筛查，0 个近重复匹配，人工队列为空；该筛查不证明阈值以下绝无重复。
- 引用与 schema 问题均为 0；原 accepted、attempts、cold_start、dev、final test 哈希保持不变。复审报告、manifest、ID 清单和逐条裁定位于 `MedSearch/data/distillation/sft_environment_review_20260929_full/`。运行复核命令：`python -X utf8 -m MedSearch.tools.finalize_sft_environment_review`。
- 边界：priority 是当前可以考虑进入小规模 SFT 的候选集，不等于医学专家核验或模型效果验证；reserve 保留供后续证据补查/人工裁定，exclude 只排除出本次训练候选，不删除源记录。
- 下一步唯一任务：与用户检查 128 条 priority 清单及其答案轨迹，再按 MedicalGPT SFT 阶段要求设计小规模训练与 Base/SFT 评测；在清单和 checkpoint smoke 验收前不启动正式训练。

## 2026-09-28：7B 现有参数与本地 StatPearls 扩库对照完成

- 当前仍在 Search-R1/MedSearch-R1 搜索 Agent 评测阶段。今日唯一核心任务：固定同一 dev64 与 7B，检验检索观察加长、搜索轮数增加、响应/模型总预算增加，以及本地 StatPearls 扩库。五组新增 rollout 均正常退出且各保存 64 条，配置、逐题结果、日志与限制见 `MedSearch/outputs/searchr1_7b_param_corpus_ablation_20260928/REPORT.md`。
- 原 Textbooks/700-token/2 轮基线词面 13/64、verifier 20/64、合法答案 56/64。只加观察到 1200 为 12/64、20/64、59/64；只加到 3 轮为 13/64、18/64、59/64，且一题耗尽 4096 响应预算；3 轮改用 6144 响应/6656 总长后为 13/64、13/64、61/64且无耗尽；1200 观察+3 轮大预算为 10/64、19/64、62/64。各配置均无稳定 QA 提升，长观察使平均响应 token 从 1129.7 增至 1535.5。
- 本地 Textbooks 125,847 段与 StatPearls 371,651 段流式合成 497,498 段，BM25 索引全部入库且无错误；在相同 700-token/2 轮下扩库结果词面 12/64、verifier 15/64、合法答案 59/64，低于原库。固定 63 条首轮查询的完整 Top-5 答案字串命中 Textbooks 16/63、扩库 14/63，均只是无 qrels 的词面代理。保留原 Textbooks 配置为当前答题基线，不把扩库默认接入。
- 磁盘：模型直接读现有 Docker 宿主机挂载，未复制进容器。合并语料约 665 MB、索引约 49 MB；本次 8004 服务已停止、其约 645 MB 可再生缓存已清除。D 盘空闲从 109,503,950,848 到 108,782,333,952 字节，净减少约 688 MiB，与保留产物量级一致。
- 下一步唯一学习检查（预计 30–60 分钟）：阅读报告中的 `5574` 基线与 1200-token/扩库轨迹，用自己的话指出最先出现的错误证据或推理转向。完成标准：引用各轨迹的检索片段、最终 `<answer>` 和标准答案，解释“检索信息增加但答题变差”的具体位置；之后再决定是否值得处理查询或证据呈现问题。

## 2026-09-28：7B 现有参数与语料覆盖诊断

- 当前仍在 Search-R1/MedSearch-R1 的搜索 Agent 评测阶段。今日唯一核心任务：判断先调整观察长度、搜索轮数或语料，是否比加入新机制更有依据。已完成 393 条已保存首轮 BM25 结果的离线观察窗口重放；不改检索器、不下载、不建索引。结果见 `MedSearch/outputs/searchr1_7b_retrieval_sampling_pilot_20260928/REPORT.md`。
- 700/1,000/1,200/1,600-token 可见标准答案字串命中分别为 71/87/90/90（总 393）；完整 Top-5 可放入分别为 8/135/342/393。此为词面代理，非 Recall 或 QA。7B dev64 修复协议后无预算耗尽，所以先单独试观察长度 1,200；只调大模型总上下文而仍截 700 不会增加证据。
- 诊断时的 7B BM25 只用 Textbooks 125,847 段；本地 StatPearls 原始切块约 371,651 段当时尚未索引。MedRAG 本地 corpus 不含 PubMed/Wikipedia；Search-R1 另有 wiki 索引。用户决定暂不增加 reranker 等机制；后续已完成 StatPearls 单变量扩库对照，结果见上节。
- 当时安排的下一步任务为同一 dev64 上把 `observation_length=700` 改为 `1200`；现已完成 64 条 rollout、逐题比较和 D 盘验收，结果见上节。

## 2026-09-28：7B/3B 检索观察修复、MedCPT Hybrid 与采样复核

- 当前仍在 Search-R1/MedSearch-R1 的搜索 Agent 评测阶段。今日唯一核心任务：检查 7B 的 BM25 返回段落，并在同一小样本上验证 Dense 融合和采样温度。完成标准：索引与服务可用、逐题检索审计及 rollout 文件保存、同题配对指标、失败轨迹与磁盘占用记录。已完成；详见 `MedSearch/outputs/searchr1_7b_retrieval_sampling_pilot_20260928/REPORT.md`。
- 发现旧评测在截断 700-token 检索观察时会丢失 `</information>`。修复并通过 Docker 中的回归测试；旧 64 题子集的 7B/3B 预算耗尽分别从 11/19 降到 0/0。旧 dev396 结果保留，不能与新协议结果混算。
- 用宿主机原有 125,847 段教材语料建立 MedCPT Article Encoder 的 768 维 FAISS 索引（386,602,029 字节）；Query Encoder 与 Article Encoder 直接从 D 盘项目路径加载，未复制进 Docker 可写层。BM25+MedCPT 使用 RRF-60、各 Top-20、最终 Top-5。
- 对旧 7B 的 393 个首轮查询重放检索：BM25/Hybrid 原始 Top-5 答案字串命中 90/102，700-token 可见文本命中 71/85；没有证据 qrels，此为字面代理指标，非 Recall@5。
- 固定 dev 64 题、seed 42、Top-5、2 工具轮、4096 回答预算、700 观察预算：7B BM25 greedy 词面/verifier 13/20；7B Hybrid greedy 9/18；7B BM25 temperature 0.3 为 11/15，0.7 为 10/17；3B BM25 greedy 5/10。Hybrid 虽增加字面命中，但本样本未改善最终答题。7B 相对 3B 在 verifier 上逐题独赢 12 对 2，协议合法作答则 7B 56/64、3B 64/64。两个检查点训练方法不同，不能把全部差异归因于参数量。
- 同一 63 条首轮查询的服务检索延迟：BM25 平均 2.3 ms、Hybrid 38.9 ms；不含生成与 verifier。抽查记录了 Hybrid 成功、失败、查询错误及判分器误判案例。新增权重 safetensors 合计约 835 MiB、FAISS 索引约 369 MiB；D 盘空闲约 102 GiB，未出现大量复制。
- 用户追问 BM25 命中与是否需要 reranker：以 `39029` 为命中良好例，BM25 Top-3 可见 Colchicine 证据且 7B 答对；以 `887` 为失败例，完整原始查询下相关尿路瘘段落在 BM25 Top-4/5 且被 700-token 观察截掉，Hybrid RRF 提到 Top-1 后字面证据可见；但修复协议下 7B 又把查询偏成 urethral fistula，两种检索候选均跑偏，说明 reranker 只能处理召回后排序，不能修错误查询。结论：先修查询漂移、700-token 内容预算和 passage 标题/正文格式，再做 BM25 Top-20 + reranker 的小规模单变量消融；当前证据不足以宣称 reranker 会提升最终 QA。报告已补上案例和方案。
- 下一步唯一学习检查：从报告的 `33340`、`5574`、`26934` 三条轨迹中选一条，用自己的话说明“检索段落、模型答案、标准答案、判分器”四者分别发生了什么，再决定是否针对查询改写或证据压缩做单变量实验。

## 2026-09-28：Search-R1 7B/3B + MedRAG BM25 dev396 对照完成

- 当前项目仍为 MedSearch-R1 阶段的搜索 Agent 评测；本次唯一任务是完整保存同一 `dev.parquet` 396 题的 7B 与 3B rollout，并抽查轨迹。两组各 396 条、题号和题干逐一相同，运行进程正常退出。原始结果、日志、配置、磁盘验收与逐题点评见 `MedSearch/outputs/searchr1_7b_3b_bm25_dev396_20260928/REPORT.md`。
- 7B：词面 59/396（14.90%），CPU verifier 107/396（27.02%），合法 `answer` 动作 302/396（76.26%），至少一次无效动作 79/396（19.95%），平均搜索 1.119 次、平均响应预算 1,732 tokens（含工具观察）。3B：词面 21/396（5.30%），verifier 55/396（13.89%），合法 `answer` 动作 262/396（66.16%），至少一次无效动作 124/396（31.31%），平均搜索 0.972 次、平均响应预算 2,013 tokens（含工具观察）。词面配对 7B 独赢 47、3B 独赢 9；verifier 配对 75 对 23。
- 需区别“合法答案动作”和“闭合答案标签”：7B 有 329 条含 `</answer>`，3B 有 273 条；缺失闭合标签分别为 67、123 条，均用满 4096 token 轨迹预算且从未出现 `<answer>` 开标签。绝大多数停在未闭合的 `<information>` 工具观察中，详见报告。不能把 302/262 误称为闭合答案数。
- 判断：7B 相对现有 3B 检查点有明显进步，但绝对 QA 表现和动作合法性仍不足以称为可靠医学搜索 Agent。7B 是 PPO、3B 是 GRPO 检查点，搜索次数也不同，不能将增益单独归因于参数规模。人工抽查发现 7B 正确利用 BM25 的案例，也发现 7B 搜错方向（`18099`）、被相关但错误的材料带偏（`31491`）和 verifier 将症状 `trismus` 误判为正确诊断（`25300`）；词面规则还漏判了 `TSH and T4`（`9951`）。
- 磁盘：模型、索引和 verifier 均从 `D:\code_list` 主机绑定挂载读取，未复制入容器；输出目录全部 12 文件共 7,245,777 字节。开跑前 D 盘空闲约 110,800,379,904 字节，完成后 110,802,640,896 字节，未发生 Docker 数据盘异常扩张。GPU 峰值未连续监控，不能编造峰值数值。
- 下一步唯一学习任务（约 30–60 分钟）：阅读同题轨迹 `25300` 和 `9951`，用自己的话解释为何一个是 verifier 假阳性、另一个是词面假阴性。完成标准：在 `REPORT.md` 对照原始 `<answer>`、标准答案与两种判分规则，指出各自错在哪里，并提出一条可执行的人工复核准则；之后再决定是否改评测器，不直接扩大训练。

## 2026-09-28：7B dev 评测中断后的磁盘恢复与容器重建

- 原计划：固定 Search-R1 7B + 现有 MedRAG BM25 在 `dev.parquet` 396 题生成并保存完整 rollout，再与 3B 比较。7B 的 8 题 smoke 已运行；全量 396 题生成后，CPU verifier 写结果时因 Docker 数据盘占满而中断，只得到约 102 行不完整输出，不能报告全量指标，也不能据此判断 7B 相比 3B 是否进步。
- 事故原因：旧容器挂载 `/mnt/d/code_list` 在 Docker Desktop 中为空，我错误地把已有的 7B、verifier、3B 权重和 BM25 索引复制进容器可写层，而非修正主机绑定挂载。D 盘上的 Docker VHDX 随之扩张。后续还发现容器内另有与主机逐分片 SHA256 相同的 7B（约 33 GB）和 3B（约 6.3 GB）临时副本。
- 恢复：删除本次评测临时目录、上述已校验重复模型及缓存；以旧容器软件环境保存本地镜像，创建挂载主机 `D:\code_list` 的新 `llm-pytorch`，验证 RTX 5090、项目文件可读、两个原缓存卷及 329 个 Python 包版本一致后，删除旧容器。容器配置和包清单见 `MedSearch/docs/run_logs/container_rebuild_20260928/`。
- Windows 空间验收：删除容器内文件和旧容器本身未让 D 盘空间立即回升。管理员 CMD 对指定 `docker_data.vhdx` 执行 `attach vdisk readonly`、`compact vdisk`、`detach vdisk` 后，D 盘空闲从 17,676,083,200 字节到 110,834,720,768 字节，实测回收 93,158,637,568 字节。此前约 6.27 GB 到 17.68 GB 的增加来自用户清理其他目录，不能归因于 Docker。
- 后续约束：模型和索引直接从主机只读/绑定挂载读取；挂载不可用即停止排查，不复制大文件进容器。评测前记录 D 盘余量与预计输出体积；结果写入主机挂载目录。下一步仍是同一 `dev` 上完整 7B rollout 与可比 3B 基线，当前尚未完成。

## 2026-09-27：P2 真实 3B/7B/3B 短生命周期 smoke

- 通过升级权限调用以当前 Windows 用户 `dell` 访问既有 WSL Docker；`llm-pytorch` 一直运行。此前普通沙箱调用返回 `E_ACCESSDENIED` 是本会话权限路径问题，不代表用户 WSL 已停。无需用户手动启动服务。
- 复用本地 3B 权重和固定 Search-R1 7B 权重，在 RTX 5090 上运行 `MedSearch/tools/smoke_phase_switch.py`；学生前后两次下一 token ID 同为 `3862`，教师 16 token sampled 评分全有限。第二次运行的 3B 首次前向/CPU 卸载/7B 加载评分/教师释放/3B 恢复前向分别为 2.35/3.48/5.87/0.30/1.19 秒；阶段结束 GPU 已用 7.797/1.780/16.157/1.780/7.797 GiB。首次 7B 阶段 13.30 秒，说明缓存影响明显。结果及限制见 `MedSearch/docs/audits/phase_switch_smoke_20260927.md` 和 JSON。
- 此为辅助脚本模拟 veRL `param_offload=true` 自动卸载后的真实模型资源探针；hook 接收轻量 runtime，未运行 Ray/FSDP/vLLM、verifier 或优化器。P2 完整真实小批链路仍待验收，长训练和 P3 尚未开始。

## 2026-09-27：P2 资源切换 hook 静态审查与访问问题排查

- 学生已编写 `MedSearch/rl/distillation_hook.py` 的 `prepare_teacher_phase(runtime)` 与 `restore_student_phase(runtime)`。对照当前 veRL `ActorRolloutRefWorker.to()`、FSDP `to()` 及 `train_mode/eval_mode` 自动卸载逻辑审查：固定配置 `param_offload=true` 时，actor 前向上下文退出会把模型参数放回 CPU，hook 的 `model=False` 不重复搬移；未启用自动卸载时，hook 的 `model=True` 才显式搬移。`python -m py_compile rl\distillation_hook.py` 通过。审查未发现需要学生立刻修改的确定性缺陷。
- 曾写出一个要求 `param_offload=true` 时仍强制 `model=True` 的单元测试；继续核对 FSDP 自动卸载实现后确认该断言错误，已撤销测试，不把其失败算作学生代码缺陷。
- 本次 Codex 受限账号无法运行真实 3B→7B→3B smoke：`wsl --status` 返回 `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`；`sc query LxssManager` 显示该按需启动服务当前 `STOPPED`，尝试 `sc start LxssManager` 返回访问被拒绝（exit 5）。补查发现 `WslService` 与 `vmcompute` 均为 `RUNNING`，所以不能据此认定用户平常的 WSL 实例已停止；当前只确认本会话无权访问 WSL，Windows Docker 端也无法连接引擎。P2 资源切换、显存、主机内存、耗时与恢复后前向均未验收；暂不进入 P3 或较长训练。取得既有 WSL/Docker 访问后继续同一项 smoke。

## 2026-09-27：固定 Search-R1 7B 教师下载与真实短评分

- 已从公开 Hugging Face 仓库下载候选教师 revision `74d46a43224daa612b5f27174e1c58b35587ee92`，七个权重分片及配置/tokenizer 共 16 文件保存到 `MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct/`。项目副本七个分片 SHA256 与固定 revision 下载元数据逐个一致；无付费 API。
- 既有 WSL Docker `llm-pytorch` 中用当前 `load_teacher` 和 `score_tokens` 运行真实 7B BF16 短评分：`[1,16]` logprob 全有限，峰值 CUDA allocated 14.286 GiB，加载加评分 5.31 秒。报告 `MedSearch/docs/audits/searchr1_7b_teacher_smoke_20260927.md` 与同名 JSON。首次辅助 smoke 脚本错误将字符串当 token IDs，修正编码后通过；核心教师函数未改。
- P2 单卡资源阶段切换仍未验收：尚无学生 rollout→verifier→卸载/教师评分→恢复学生前向的完整小批流程。未启动训练；较长训练前仍须 checkpoint 保存、重载和一次动作推理，不使用 V9 的 `save_freq=-1`。

## 2026-09-27：本地教师加载函数验收

- 学生已实现 `MedSearch/distillation/teacher.py::load_teacher(local_path, device)`：仅从本地加载 BF16 模型，转指定设备，冻结参数并设为 eval。Agent 在既有 WSL Docker 中以临时小 Qwen2 checkpoint 运行 `python -m pytest MedSearch/tests/test_distillation_teacher_load.py MedSearch/tests/test_distillation_teacher.py -q`，`2 passed`；小模型一次前向、sampled 评分回归均通过。按用户要求，仅补该函数的 Args/Returns 简短注释，补后复跑同一命令仍为 `2 passed`。2 条依赖 `tqdm` 的弃用 warning 不影响结果。
- 验收范围只到小模型本地加载与前向。候选 Search-R1 7B 权重仍未下载，真实模型身份、单卡显存/资源切换和 checkpoint 保存、重载及动作推理门槛尚待验证；未启动训练或付费 API。

## 2026-09-27：教师 sampled 评分验收与下一项任务

- 学生实现 `MedSearch/distillation/teacher.py::score_tokens` 的 `[B,P]`/`[B,T]` sampled 路径；真实共同词表 ID 0–151664 上归一化，逐 token 因果对齐。既有 WSL Docker 运行 `python -m pytest MedSearch/tests/test_distillation_teacher.py -q`，`1 passed`。用户决定暂不验收非 sampled 模式报错，Top-k 留到 B 档接线；已撤下该模式测试。此通过仅为假模型数值验收，没有加载 7B 权重或实测显存。
- 下一项唯一核心编码任务由学生实现 `MedSearch/distillation/teacher.py::load_teacher(local_path, device)`：本地 checkpoint、BF16、冻结、eval、指定设备、一次前向。Agent 已新增 `MedSearch/tests/test_distillation_teacher_load.py`，用临时小 Qwen2 checkpoint 验收；当前因函数尚不存在，在导入阶段按预期失败。真实 7B 下载与单卡阶段切换仍在 P2 待办；没有启动训练或付费 API。

## 2026-09-27：教师 sampled-token 评分改为批量 Tensor 契约

- 按用户确认，`MedSearch/distillation/teacher.py::score_tokens` 输入直接为同设备的 `torch.long` 张量：`prefix_ids[B,P]`、`target_ids[B,T]`；本步批内无 padding，`attention_mask` 全 1；输出为逐位置 `logprob[B,T]` 张量。已同步函数类型/说明、工程书及 `MedSearch/tests/test_distillation_teacher.py` 的 B=2 样例；未替学生实现前向或 logprob 计算。
- `python -m py_compile MedSearch\distillation\teacher.py MedSearch\tests\test_distillation_teacher.py` 通过。Windows 本机 pytest 在导入 PyTorch→asyncio 时遇 WinError 10106，未运行到测试主体；后续仍在既有 WSL Docker 环境验收。本步函数尚未完成。

## 2026-09-26：A 档数据入口与 P1 smoke 通过，进入 P2 首个函数

- 按用户要求撤销 `response_ids`/`response_mask` 不等长的异常测试，仅验正常轨迹。学生已实现 `MedSearch/distillation/data.py::encode_distillation_trace(record)`；Windows 本地测试 `1 passed`，`py_compile` 退出码 0。该结果只验原始 token 拼接、生成位置标签、observation 掩码及 gold 不进入训练样本。
- P1 复用现有 `MedSearch/rl/medsearch_agent_loop.py`，只补两次搜索后最终回答的 smoke；既有 WSL Docker `llm-pytorch` 临时副本运行 `python -m pytest MedSearch/tests/test_agent_loop_smoke.py -q`，`5 passed`。首次测试期望把预算耗尽提示放早了一轮，修正测试后通过；核心 loop 未改。Windows 的 `asyncio` 导入受本机 WinError 10106 阻断，因此按既定 WSL Docker 环境验收。
- 下一项唯一核心编码任务由学生实现 `MedSearch/distillation/teacher.py::score_tokens(model, prefix_ids, target_ids, mode="sampled")` 的 sampled 路径。Agent 已写 `MedSearch/tests/test_distillation_teacher.py`，当前因模块未存在而在导入处失败；要求因果对齐、共同真实词表 ID 0–151,664 上归一化、输出逐 token 教师 logprob。此处只用假模型做数值验收，7B 权重尚未下载。较长训练仍需先验 checkpoint 保存、重载与一次动作推理。

## 2026-09-26：候选教师 tokenizer 已核对

- 暂用 Search-R1 官方集合中的 `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo` revision `74d46a43224daa612b5f27174e1c58b35587ee92`。本地仅有 tokenizer 与 config，没有 7B 权重。审计报告 `MedSearch/docs/audits/searchr1_7b_vs_qwen25_3b_tokenizer.md` 和同名 JSON 显示师生 151,665 个真实 token 的 ID 映射完全一致；输出层额外行不同，B 档 Top-k / 在线 OPD 需在共同真实 token 集合上屏蔽、归一化，实际 Top-k 待权重加载后实测。batch attention mask 依据真实长度。最终教师如变更需重查。
- 当前唯一核心编码任务仍是学生实现 `MedSearch/distillation/data.py::encode_distillation_trace(record)`，由现有 `MedSearch/tests/test_distillation_data.py` 验收；尚无教师轨迹、冷启动训练或 checkpoint 重载结果。较长训练前必须验证保存、重载和一次动作推理，不能用 V9 的 `save_freq=-1`。

## 2026-09-26：MedSearch 蒸馏 P0 输入隔离核查

- 数据池决策落实：用户要求排除两批候选中的 reject 374、quarantine 140，直接用 pending 3,486 条的来源 gold 作未经医学专家核验的弱标签；旧 RL 数据可与新增 RL 题合并，固定 dev 不得重合。新增 `MedSearch/tools/build_distillation_pools.py`，另存最终池 `MedSearch/data/distillation/pools_v2_20260926_devclean/`：冷启动 1,000、新增 RL 2,486、旧 RL 保留 3,270、RL train 合计 5,756、dev 原样 396。直接 dev 近重复扫描发现旧 RL train 中 13 条同题改写，只从本次合并版排除；另 3 对条件不同留存，逐对理由在 `MedSearch/docs/audits/distillation_dev_pair_review_20260926.jsonl`。最终 ID/规范化题干精确冲突 0，词面 Top-1 队列余 1 对、MPNet Top-1 队列余 3 对，均已判为不同题；阈值扫描非穷尽。审计报告 `MedSearch/docs/audits/distillation_pools_v2_20260926.md`。初版 `pools_v2_20260926/` 因含旧 RL 与 dev 近重复已标作 superseded，不可用于训练。未修改旧 RL train/dev、未启动训练。

- Luna 阶段结果已回收并核验：`MedSearch/docs/audits/luna_v2_review_20260926.md`、`data/medical_verifiable_v2_20260919/review_ledger.jsonl` 记录 3,000 条候选中 reject 294、quarantine 120、pending 2,586、accepted 0。304 组 MPNet 高相似配对已逐对审题干，另做词面队列审阅；发现 19 条与冻结 MIRAGE 同题改写。`python -m MedSearch.tools.validate_v2_review` 本地通过，原题/gold 3,000/3,000 未改、精确历史冲突 0。医学 gold 复核未完成，未分 split、`training_ready=false`；Luna 任务已结束，不能将阶段报告称为最终清洗结果。

- 2026-09-26 当前唯一核心编码任务：由学生实现 `MedSearch/distillation/data.py::encode_distillation_trace(record)`，直接从已接受教师轨迹的原始 `prompt_ids`、`response_ids`、`response_mask` 产生 `input_ids`、`attention_mask`、`labels`；模型动作原位保留标签，初始 prompt 和环境 observation 为 -100，不重新 tokenize、不手动 shift、不引入 gold。该函数是冷启动 A 档 CE 的最小数据入口，B 档后续复用同一目标位置，RL 将加载其训练出的学生 adapter。Agent 已新增 `MedSearch/tests/test_distillation_data.py` 两项验收测试并先运行，当前按预期在导入阶段失败（模块尚不存在）；学生提交实现后复测。Luna/P0 模型身份与数据验收仍独立进行，未启动训练。

- 2026-09-26 更正：撤回此前要求实现 `rl/episode.py` / `generation_budget()` 的编码任务。用户决定直接使用 Search-R1/veRL 的长度配置及现有动作截断；本地现有 loop 已有 500 token 动作上限，YAML 现显式写入 `max_action_length: 500`。工程书 P1 改为复用原 loop 的最小 0/1/2 次搜索与第二次搜索后作答 smoke；只有发现具体缺陷才修原 loop，不新建共享状态层。离线 `evaluate_agent_rollout.py` 的 500 上限若在 P3 复用为教师生成时需要，再直接配置 vLLM `SamplingParams`。本轮无新增训练或付费 API。

- 2026-09-26 续：用户要求把新 v2 候选复核交给 Luna，主对话可并行阅读 P1；Luna 已接手 3,000 候选及语义队列。当时工程书的 P1 曾写 `EpisodeState`/`generation_budget()` 等共享规则加两处 loop 接线；上一轮把仅实现预算函数并接离线入口称为完整“P1 任务”不准确。用户随后决定撤回该抽象，工程书已按上方更正修改。P0 未完成项保留，不因此宣称 P0 全部验收。


- **同日补充（v2 重建）**：用户确认旧 Luna 产物位置不明、7B 教师尚未下载。重查后仅找到旧 v1 规则筛查产物；其 `summary.json` 明记 Luna 子任务未执行。新建 `MedSearch/tools/build_distillation_v2_candidates.py`，从固定原始题库和全部历史隔离集合重新生成 `MedSearch/data/medical_verifiable_v2_20260919/` 候选池：3,000 条待复核、5,177 条排除、6,563 条质量隔离、25,904 条备用。候选保留原题与原答案、来源索引；manifest 明标 `training_ready=false`、未分 split、词面近重复/语义近重复/医学审阅未完成，不作为训练输入。
- v2 候选对旧 RL train/dev、Stage-A train/dev、Search100、protocol dev、MIRAGE final test 的 ID/规范化题干精确冲突为 0；报告为 `MedSearch/docs/audits/distillation_v2_candidate_isolation.json`，容器复核报告为同目录的 `_container.json`。候选文件在 Windows 与既有 `llm-pytorch` 容器的 SHA256 完全一致：keep `53be1b32aa17f9b2389eeb23445bcf31eea5e4e30cca7421a03e1b12a5a8cd55`，other `1c8e2705f029141b1ca1ef4e6b1281817938c71bd40f486609ae58ca140ddcfe`；两边定向测试均 7 passed。
- **语义近重复待复核队列**：在同一容器用本地 `MedicalGraphRAG/models/all-mpnet-base-v2`（权重 SHA256 `a8fd120b1a0032e70ff3d4b8ab8e46a6d01c2cb08ffe7c007a021c1788928146`）对 3,000 条候选与 39,320 条历史题做 Top-1 余弦扫描，阈值 0.92；232 条候选命中历史，72 对候选内部命中，合计涉及 336 个不同候选 ID，另有 2,664 个候选 ID 未触发该阈值。报告 `MedSearch/data/medical_verifiable_v2_20260919/semantic_neighbors.jsonl` 与同名 manifest，真实耗时 234.24 秒，设备 CUDA。高相似项只进入人工/医学复核，不自动删除；低于阈值也不保证没有近重复。原始 `sentence-transformers` 版本与当前 Hugging Face Hub 不兼容，脚本改用同一本地 MPNet 权重、384 token 截断和配置指定的 mean pooling；两题探针输出 `(2, 768)`，余弦 0.742793。未更改容器依赖、未调用付费 API。
- 再核对历史记录：8 月曾出现 `data/rl/medical_verifiable_v2/quality_audit/` 与网页审阅文件，但当前工作区该目录及全工作区同名目录均不存在；这些历史记录不能代替本轮 Luna 产物或当前可读的审阅证据。新建的日期后缀目录仅为本轮重建候选，不能称为 Luna 审阅版。最终在既有 Docker 容器对本轮 3 个脚本运行 8 个定向测试，`8 passed`；语义队列文件 SHA256 `8e7eda554c9cbc50ba54c881ebc5d330ddae387631b37513a7d4b3ef7a9b2e39` 与 manifest 相符。
- 环境澄清：本会话 `wsl.exe` 返回 Wsl/Service RPC 错误，先前称“没有可用发行版”不准确。经授权的 Docker 客户端可访问此前的 `llm-pytorch` 容器；原 `/workspace/code_list` bind 当前为空，本次将最小数据/脚本复制到容器临时目录完成复核。正式训练前仍需恢复工作区挂载，不通过这份临时副本训练。
- 当前聚焦 MedSearch 新蒸馏方法，MedAgent harness 暂缓；本轮未编写或运行核心训练代码。
- 已实现 `MedSearch/tools/check_distillation_inputs.py` 的 `check_split_isolation()` 及 CLI，读取真实文件并保存各输入 SHA256、行数、可定位的 ID/规范化题干冲突、缺失和未审阅资产；语义近重复单列为 `not_checked`。
- 用户确认 7B 教师尚未下载。重复搜索 `MedSearch/data`、`MedSearch/outputs` 与临时目录仍未发现 Luna v2；旧 `medical_verifiable_expansion_v1/summary.json` 明记当时 Luna 子任务因 sandbox 缺少 bwrap 未执行，不能把旧规则筛查称作 Luna 医学复核。
- 对旧 v1 另做来源一致性复核：原始题库 40,644 条、v1 保留 1,988 条；逐条核对来源索引、ID、原题、原标准答案、嵌入数据答案及当前正式 RL 文件，发现不一致 0。该检查只证明数据未被改写，不证明官方答案医学正确或题目无歧义；旧 `audit_text_only.py` 中的 `answer_correct=True`/`answer_unambiguous=True` 是规则脚本写入，不能当人工审阅证据。
- 运行命令（当前 Windows 静态核查环境）：`python -m MedSearch.tools.check_distillation_inputs`。脚本因 Luna v2 缺失按设计返回退出码 1，报告位于 `MedSearch/docs/audits/distillation_split_isolation.json`。
- 实际回读：现有 RL train 3,283、dev 396；Stage-A train 29,433、dev 640；Search100 100、protocol dev 12、MIRAGE medqa/medmcqa final test 5,456。已检查集合之间 ID/规范化题干精确冲突 0，空规范化题干 0；这不证明语义近重复已排除。
- 原 Luna v2 清洗产物未找到；同日虽重建同目录候选池，仍未形成可训练的最终划分。Search-R1 7B 教师 checkpoint/revision 仍未固定，不能验收完整 P0。
- 模型身份静态核查：学生基座 `models/Qwen2.5-3B-Instruct/` 的 config 为 `Qwen2ForCausalLM`、vocab_size 151936，Hugging Face 下载 metadata revision 为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`，tokenizer.json SHA256 为 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`；medical-o1 verifier 位于 `MedSearch/models/external_baselines/medical_o1_verifier_3B/`，其 config 为 `LlamaForSequenceClassification`，metadata revision 为 `c285f67f186cebb61d70950af1576dc2d52e4cba`。本地另有 Search-R1 3B baseline，但它不是本计划的 7B 教师；尚未进行教师/学生 token→ID 兼容性实测。
- 验收命令：`set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1&& python -m pytest --noconftest --confcutdir=MedSearch\tests MedSearch\tests\test_distillation_inputs.py -q`，Windows 结果 5 passed；`python -m py_compile` 对脚本和测试均退出码 0。随后经 Docker 客户端进入既有 `llm-pytorch` 容器（Python 3.12.3），在临时复制的同一项目文件上复跑：5 passed；真实审计计数相同，因 v2 缺失按设计退出码 1。容器报告保存在 `MedSearch/docs/audits/distillation_split_isolation_container.json`。本会话的 WSL CLI 报 Wsl/Service RPC 错误，Docker 客户端可连既有容器；其旧 `/workspace/code_list` bind 当前为空，因此本轮临时复制必需文件，没有更改训练容器依赖。
- 较长训练前必须验收 checkpoint 保存、重载和一次动作推理；新实验不得沿用 V9 的 `trainer.save_freq=-1`。本轮未启动训练或付费 API。

## 2026-09-20 当前任务补充：MedSearch 蒸馏工程规划

- 当前仍为阶段 5；在已有 SFT、AgentLoop 与 RL 工程上规划 Search-R1 7B → 3B 冷启动及 GRPO + 失败 OPD。
- 已完成文档：`MedSearch/DISTILLATION_ENGINEERING_PLAN.md`，包括源码复用审计、必要修改、目标目录、逐文件函数职责、数据契约及 P0–P7 验收顺序；同步 `MedSearch/DISTILLATION_PLAN.md`。
- 本轮仅更新文档，没有新增训练代码、GPU smoke 或能力提升结果。单卡教师评分切换与 SFT adapter 导入仍需实测。
- 下一项唯一编码任务：指导学生实现 `check_split_isolation()`，先验收 Luna 实际产物与历史隔离边界；输出真实数量、可定位的重复项与未覆盖项，再进入轨迹接线。

**阶段 5：MedSearch-R1 医学证据搜索 Agent——SFT 已冻结，进入 RL 架构与基线收敛。**

阶段 1、阶段 2 已完成并冻结，阶段 3 已完成最小 MedicalGPT SFT smoke，阶段 4 已完成
Search-R1 3B 单步 GRPO 保底架构。MedSearch-R1 的 SFT 数据与隔离边界已经冻结，Stage-A/
Stage-B tokenization、统一数据入口与 Qwen2.5-3B + LoRA 模型装配均已通过验证。2026-08-31
按学生要求暂缓 Stage-B replay；学生 `build_trainer()` 已通过 CPU 验收，完整回归 22 passed。
学生 `run_training()` 已验收：返回完整真实 metrics 与顶层 global_step；tiny CPU 单步、
有限 loss/梯度、LoRA 更新和基座不变均通过。最新完整回归 27 passed；真实3B也已完成
Stage-A 8条单步验证，loss=1.173087、整卡峰值18,139 MiB。保存接口与测试骨架已搭建，
学生 `save_adapter()` 已通过 tiny CPU 保存/重载验收；最新完整回归 31 passed，无 xfail。
真实 3B 单步保存/重载也已通过：504 个 adapter 张量与 434 个基座张量一致，固定输入最后
位置的全词表 logits 最大差 0。学生 `build_quantization_config()` 已验收，最新 CPU 回归
37 passed，无 xfail。随后已在 Docker 安装 bitsandbytes 0.50.2 并通过微型 NF4 GPU 前向；
学生量化加载分支现已通过：CPU 回归 40 passed、16 条第三方弃用警告，无 xfail；
真实 Qwen2.5-3B 的 252 个 NF4 双重量化层及单次前向通过。尚未验证 QLoRA 训练。
学生 prepare_qlora_model 已通过CPU验收；最新回归42 passed、16条警告，无xfail。
真实4-bit准备与LoRA单样本反向传播已通过：515tokens、loss1.9581148624420166，
504个LoRA张量梯度有限；无optimizer更新，不能称完整QLoRA训练通过。
2026-09-01 正式 Stage-A QLoRA 已完成 2010/2010 步并保存最终 adapter；冻结的 328 条
dev 上，Stage-A 相对 Base 的总体 NLL 从 1.3803 降至 1.0014、PPL 从 3.9762 降至
2.7221。固定 32 条生成的格式与截断检查已通过，但医学正确性仍待人工审阅，不能仅据
loss/PPL 宣称医学能力提升。2026-09-01 首轮正式 Stage-B 已使用全部 100 条 Search100
完成 1 epoch（50 micro-batch、13 optimizer step），最终 adapter 已保存；12 条 protocol
dev 也已完成跨 Search100、Stage-A 与 MIRAGE 最终测试集的冻结去重。

2026-09-04 正式冻结两阶段 SFT：`MedSearch/outputs/stage_b/formal/adapter/` 作为后续 RL
初始化 Checkpoint-B，不再继续增加 SFT epoch、replay 或调整训练配方。该冻结表示训练管线、
adapter 产物、Stage-A 保留能力与 Stage-B 首轮搜索协议已达到进入 RL smoke 的准入线；不表示
已经证明闭环检索质量、开放式医学答案正确性、幻觉改善或最终项目效果。

```text
Medical Verifiable Problems
  → Checkpoint-B 多轮 <search> / <information> / <answer> rollout
  → MedText BM25 检索环境
  → Search-R1 normalized exact-match acc（0/1）
  → DAPO-style group-relative policy update
  → actor policy update
```

当前只收敛 RL 架构与可比较 baseline；不重训 SFT，不同时更换 retriever、corpus 和 reward。

2026-09-13：V8 在医学 dev 的当前宽松 rule_acc 为 6.06%，未形成可保留的训练初始化；已按用户决定永久删除 V8 的 38GB 训练产物、V8 dev 输出及专用脚本。V9 重新从冻结的 Stage-B adapter（Qwen2.5-3B-Instruct + LoRA）开始，不再依赖 V8 权重。

2026-09-13：已按用户决定永久删除不再参与 V9 的 Stage-A 与 Stage-B QLoRA 正式/检查点/smoke 产物（Stage-A 约 712MB，Stage-B 约 319MB）。V9 改为纯 Qwen2.5-3B-Instruct + 新 LoRA 初始化；Stage-A/Stage-B 的历史训练配置仍保留为复现记录，但不再具备旧 adapter 重跑条件。

2026-09-13：已按用户确认永久删除旧 V6 RL 训练目录 `dapo_v6_tokenfix_dynamic_full_s42`（约 26GB）；它没有当前配置依赖。保留官方 Search-R1 3B、medical-o1 verifier 3B 和基础 Qwen2.5-3B，分别服务外部 baseline/V10、verifier reward 与 V9 基础初始化。


## 2026-09-13：V9 verifier smoke 与 4-step pilot

- V9 使用纯 Qwen2.5-3B-Instruct + 新 LoRA r=8/alpha=16，不加载 V8、Stage-A 或 Stage-B adapter；BM25 服务端口已统一为 8001，verifier 在 8002。
- 先完成 1-step smoke：真实 GRPO 更新至 global_step=1、rollout 落盘、退出码 0；为修复此前 Ray 主机内存 OOM，将 SimpleStorage 数据单元从 8 降为 1，并补齐新版 veRL 所需的 ref log-prob micro-batch 配置。
- 随后完成 4-step V9 pilot（seed=42）：42分08秒、global_step=4、退出码 0，四份 rollout 文件 1.jsonl--4.jsonl 均存在。没有评测集验证，因此 Final validation metrics=None，不能据此声明能力提升。
- 40 分钟以上运行监控期间，BM25 与 verifier 的健康检查共 32 次均成功；未见 OOM、rollout failure 或有效 swap 使用。实际 nvidia-smi 峰值为 32,123 MiB/32,607 MiB，余量只有约 484 MiB；虽然本次通过，后续扩大实验前必须保守降低显存压力。
- pilot 四步 verifier score 均值依次为 -0.20、-0.10、-0.275、-0.25；这是极小训练 pilot 的即时训练奖励，不是独立 QA 指标或结论。

## 2026-09-16：MedAgent DSH 选型边界 smoke 开始

- MedAgent 主框架候选改为 DeepSeek Harness；Python 保留医学状态机、业务校验、预算和最终提交权，DSH 只承担可替换的隔离角色执行。
- 固定官方 dsh-v0.1.5-rc.2、commit fb2c4b9e698e30edb738bca4cf0618587db7d203、MIT；源码位于被忽略的 MedAgent/.deps/deepseek-harness/。
- 新增无工具 patch、Planner 薄适配、标准库假 DeepSeek 服务、CMD 入口和决策记录。SDK 的请求 timeout 不覆盖 Session.run() 等待 idle，已增加 Python 外层 deadline，超时关闭 runtime 并拒绝迟到结果。
- scripts\run_dsh_smoke.cmd unit 实测 4 tests passed；Python 编译通过。假服务本机 SSE 响应与请求抓包通过，固定 usage 为 17/9/26。
- 官方 PyPI 只有占位版本、GitHub Release 没有 wheel；本地同版本 Node runtime 已构建并用于开发 smoke。单文件 EXE 封装因官方脚本生产安装时缺失 `lefthook` 失败，正式分发仍待解决。
- 本地真实 DSH SDK→Node runtime→假 DeepSeek SSE 集成已通过：Planner 返回 SEARCH JSON、`completed`、`turn/end` 和 usage=17/9/26；假服务抓包一条 DSH 请求。最终配置导出证实两个 persistent shell 均禁用、session-log 上传关闭。会话隔离、错误重试和真实 API 调用仍为 TODO。

# 今日唯一任务

**验证 DSH Planner 与 Verifier 两个独立会话的实际请求上下文。**

完成标准：假服务抓包中两个会话各有独立 messages，Verifier 不包含 Planner 私有输入；两个角色的事件与结果可对应。真实 DeepSeek 调用本任务不执行。

# 当前项目

**MedSearch-R1：基于领域微调与强化学习的医学证据搜索 Agent。**

当前设计入口：

- 项目书：`MedSearch/PROJECT_BOOK.md`
- SFT 训练管线参考：`MedicalGPT/`
- 后续多轮协议参考：`Search-R1/search_r1/llm_agent/generation.py`
- 已有新版接口适配：`Search-R1/modern_verl/search_r1_modern/search_agent_loop.py`

# 本周目标

1. 为 MedSearch 建立干净、可复现的原生新版 veRL 依赖基线。
2. 先跑通新版 veRL 官方 V1 单轮最小链路，再编写 MedSearch 搜索循环。
3. 按协议、检索客户端、AgentLoop、reward、单步 GRPO 的顺序逐项实现和验证。

# 今日唯一任务

**2026-09-08：由学生在 `MedSearch/scripts/run_medsearch_dapo_smoke.sh` 中接入 veRL 主命令和三个 reward 字段。**

完成标准：测试必须通过新版 veRL 的 `get_custom_reward_fn()` 按配置路径/函数名加载
`compute_outcome_reward()`，并通过 `load_reward_manager()` 实例化注册名为 `dapo` 的 RewardManager；
单条正确/错误 DataProto 分别得到 `reward_score=+1/-1` 与 `acc=1/0`。本轮不启动 CUDA、Ray、vLLM
或 BM25。计划见 `MedSearch/docs/superpowers/plans/2026-09-08-dapo-reward-wiring.md`。

## 2026-09-08：DAPO V1 outcome reward 冻结

- 用户决定以 Search-R1 normalized exact match 计算 `acc`：命中任一 gold alias 为 `1`，否则为 `0`；
  再以 `outcome_reward=2*acc-1` 得到正确 `+1`、错误 `-1`。训练 `score=outcome_reward`，DAPO
  filtering metric 保持为 `acc`；不另设独立 `is_correct()`。
- 只接受完整闭合的 `<answer>...</answer>`；未闭合、空答案、缺失答案或被长度上限截断均不给分，
  不猜测或自动补标签。多个完整 answer 块取最后一个。
- 训练 prompt 沿用 Search-R1 闭合协议说明与 `<answer> Beijing </answer>` 示例；现有严格动作 parser
  对未闭合、空内容、双 action 和 action 后额外文本继续判 invalid。
- 借用 `Search-R1/modern_verl/search_r1_modern/reward.py` 的 response-only 解析与 normalization；
  MedSearch 自定义函数返回 `{"score": outcome_reward, "acc": acc}`，不重新设计语义 verifier。
- DAPO group filtering 只读取 `acc`；Overlong Reward Shaping 暂时关闭，因为原生实现按 attention
  mask 计长、包含 BM25 observation，当前也没有真实截断率证明必须启用。先记录长度与截断率。
- V1 不增加 observation 扫描或转义层；只有真实 rollout 出现协议标签污染时再处理。
- 2026-09-08：学生完成 `MedSearch/rl/reward.py`：`compute_acc_score()` 只计算 `acc=0|1`，
  veRL 回调入口 `compute_outcome_reward()` 返回 `{"score": 2*acc-1, "acc": acc}`。教师实际运行
  闭合命中、闭合未命中、未闭合、空 answer、多个完整 answer 五个分支，输出依次为
  `(+1,1)、(-1,0)、(-1,0)、(-1,0)、(+1,1)`；reward 函数通过 CPU 验收，训练配置尚未接线。
- 2026-09-08：教师新增 `MedSearch/tests/test_reward_wiring.py`，通过新版 veRL
  `get_custom_reward_fn()` 与 `load_reward_manager()` 验证真实配置加载和 DAPO RewardManager 协议；
  现有 Docker 环境实测 `2 passed, 1 warning in 9.53s`，未初始化 CUDA。原先误建的独立计划 MD
  已删除。现有 `run_verl_v1_lora_smoke.sh` 保留为官方 V1/GSM8K 环境基线；另建
  `run_medsearch_dapo_smoke.sh` 作为 MedSearch DAPO 学生工作文件，不新增 trainer YAML。
- 2026-09-08：学生在 `run_medsearch_dapo_smoke.sh` 中用 Bash 数组接入
  `reward.custom_reward_function.path`、`name=compute_outcome_reward` 与
  `reward.reward_manager.name=dapo`；容器内 `bash -n` 退出码为 0。当前只完成 reward 参数组，
  尚未加入 data/model/rollout/trainer 必填配置，也未启动训练。
- 2026-09-08：新版 MedSearch DAPO 单步 smoke 已以退出码 0 完成：`train_batch_size=1`、
  `rollout.n=5`、Checkpoint-B LoRA、BM25、AgentLoop、±1 outcome reward、GRPO advantage 与
  actor update 全链路执行。5 条 reward 均为 -1，因此本组 advantage、loss 与 grad norm 均为 0；
  这是无区分度组的真实记录，不代表正式训练无法学习。step 用时 33.59s，训练器记录 actor 峰值
  allocated/reserved 为 23.11/23.18 GiB；外部 1 秒采样观察到整卡瞬时峰值 31,978 MiB。
- 本次原始轨迹位于 `MedSearch/outputs/dapo_smoke/rollouts/1.jsonl`。该单题的 5 条模型输出均未形成
  最终闭合 answer；轨迹中出现的 `<answer>answer</answer>` 来自环境纠错提示，不计作模型闭合。
  该结果只作为训练前基线，闭合率是否改善应在 RL 训练后评估，不据此新增 parser 补救。
- 按用户批准，RL prompt 恢复 Search-R1 的闭合脚手架：明确 answer 标签并加入
  `<answer> Beijing </answer>` 示例；现有非法 action 反馈与再次生成逻辑保持不变。同步更新
  train/dev parquet 的 prompt 字段，样本、split、问题、gold 与 reward 均未改变；相关测试 6 passed。

当前进度：官方新版 veRL 已克隆到 `MedSearch/.deps/verl`，固定 commit
`23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`，工作树 clean，且根 `.gitignore` 已排除该目录。
现有容器为 Python 3.12.3、PyTorch 2.13.0、vLLM 0.27.1。考虑磁盘约束，首轮决定不运行完整
`uv sync`：通过 `VERL_USE_UV=0` 和显式 `PYTHONPATH` 复用
现有 PyTorch 2.13.0 与 vLLM 0.27.1，只按官方锁定 commit 安装 TransferQueue 0.1.9.dev0。
零安装新版 veRL/AgentLoop 导入以及安装后的 V1 `AgentLoopManagerTQ` 导入均已通过；PyTorch/vLLM
版本保持不变。该组合是兼容性试验而非官方锁定环境。Hydra V1 单卡配置展开已通过，尚需完成
官方 single-turn 初始化。

进入代码前的固定学习进度：

- [x] 能解释 trainer、actor、rollout、reward、reference 五个角色。
  - 2026-09-05：通过状态同步检查：rollout 负责生成；actor 更新后、下一批采样前必须同步 rollout；
    未同步轨迹不代表当前 actor；reference 保持冻结。
- [x] 能画出或口述“问题 → 同组多条 rollout → reward → advantage → actor 更新”的 GRPO 数据流。
  - 2026-09-05：理解同组全同 reward 可作为采样表现观察，但因零方差不能提供有效 advantage；
    BM25 observation 参与后续 token 的 attention，`response_mask=0`，不参与 actor loss。
- [x] 能用官方配置启动单轮最小任务，并解释 Hydra 覆盖后的关键配置。
  - 2026-09-05：已通过固定 sample batch 语义检查：`train_batch_size=2`、`rollout.n=4` 产生
    8 条轨迹，`ppo_micro_batch_size_per_gpu=1` 表示单卡一次处理 1 条；官方示例启用 dynamic batch 后
    实际主要由 `ppo_max_token_len_per_gpu` 控制。
  - 2026-09-05：Hydra `--cfg job --resolve` 已通过，确认 GRPO、V1、FSDP、vLLM、单卡、
    `train_batch_size=1`、`ppo_mini_batch_size=1`、`rollout.n=2`、micro batch 1、KL 关闭及正式
    train/dev 路径均生效；
    V1 入口会在运行时启用 TransferQueue 的 `SimpleStorage`。尚未加载模型或启动 single-turn。
  - 2026-09-05：官方 single-turn `val_only` 进行三次递进验证。第一次由配置校验发现
    `train_batch_size=1 < ppo_mini_batch_size=2`，据源码约束将 mini batch 修正为 1；第二次发现遗漏
    rollout log-prob micro batch，补为 1；第三次已通过配置校验并成功启动 Ray、读取 1 条 train/val、
    启用 TransferQueue SimpleStorage、创建 actor/rollout worker，随后在构建 Qwen2.5 actor 时因环境
    未安装 FlashAttention2 失败。失败后 Ray 已退出，GPU 回到约 2.76 GB。当前停在架构选择：安装
    匹配 PyTorch 2.13/SM120 的 FlashAttention，或使用 veRL 官方支持的 SDPA；尚未作第四次尝试。
  - 2026-09-05：按“保留 PyTorch 2.13、不替换现有 vLLM”的决定，以 `--no-deps` 安装 veRL
    官方 `flash-attn==2.8.3` 预编译 wheel。安装本身成功，但首次导入
    `flash_attn_2_cuda` 即报 `undefined symbol: c10::impl::cow::materialize_cow_storage`，证明该
    PyTorch 2.11 ABI wheel 不能由 PyTorch 2.13 直接加载；因此没有继续做 SM120 前向/反向，也没有
    重跑 trainer。已卸载该 wheel，PyTorch 2.13 与 vLLM 0.27.1 保持不变。下一步只评估针对当前
    PyTorch/CUDA 源码编译 FlashAttention，或显式使用 SDPA，不能把相近版本视为二进制兼容。
  - 2026-09-05：决定当前环境不再使用 FlashAttention，actor 改为
    `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`，并关闭依赖
    `flash_attn.bert_padding` 的 `actor_rollout_ref.model.use_remove_padding`。实测 Qwen2.5-3B 权重、
    FSDP actor、reward manager 均成功初始化，证明 SDPA 已越过原 FlashAttention 阻塞。vLLM 0.27.1
    默认 V2 model runner 在 WSL 中因缺少 UVA 失败；设置 `VLLM_USE_V2_MODEL_RUNNER=0` 后，vLLM
    服务、FlashInfer autotune 与 CUDA Graph 捕获均完成，但首次 actor→rollout 权重同步在 CUDA IPC
    重建处报 `CUDA error: invalid resource handle`。上游 CUDA platform 无条件声明支持 IPC，而代码已
    提供 shared-memory fallback；下一步应先做显式 WSL opt-in 的 SHM 兼容验证，不继续猜调训练参数。
    失败进程已停止；完整 uv 环境仅在现有兼容路径无法成立时另建。
  - 2026-09-05：为 CUDA platform 增加显式 `VERL_FORCE_SHM_WEIGHT_TRANSFER=1` 开关，默认仍使用
    CUDA IPC；测试先得到预期失败，再修改实现，完整 platform 测试 `14 passed`。V1 smoke 日志确认
    “Falling back to shared memory”，此前的 `invalid resource handle` 不再出现；权重同步、AgentLoop
    rollout 和 reward 均已执行，训练进入 actor optimizer step。随后因当前 smoke 对 3B 模型执行全参数
    Adam，在首次创建 optimizer state 时 `CUDA driver error: out of memory`。这不是 batch/rollout
    语义问题；下一次必须恢复项目既定的 Stage-B LoRA adapter 训练，而不是继续测试全参数 actor。
    失败进程已退出，GPU 显存回落至 354 MiB。
  - 2026-09-05：将命令固化为 `MedSearch/scripts/run_verl_v1_lora_smoke.sh`，加载正式
    `outputs/stage_b/formal/adapter`（rank 8），保持 `train_batch_size=1`、`rollout.n=2`，完成
    1 个真实 GRPO optimizer step并以退出码 0 结束。日志确认 Stage-B adapter、vLLM LoRA rank 8、
    SHM 权重同步、AgentLoop rollout、reward、actor backward/update 和更新后同步均执行；step 用时
    23.08s，actor peak allocated/reserved 为 23.11/23.18 GiB，结束后 GPU 为 637 MiB。两条 rollout
    reward 均为 0，因此 advantage、policy loss、grad norm 都为 0：只证明闭环可执行，不证明有效
    学习。相关配置/platform 测试 `15 passed`；后续脚本固定 `data.dataloader_num_workers=0`，消除本次
    成功退出时的 DataLoader teardown 噪声。
- [x] 能解释 `AgentLoopBase.run()` 的输入输出与 `response_mask` 的 1/0 含义。
  - 2026-09-05：通过结构检查：`prompt_ids` 是初始提示 token；`response_ids` 同时包含模型与工具
    token；模型 token 的 mask 为 1，工具 token 为 0。工具 token 不参与 loss，但参与后续 attention。
- [x] 完成以上四项后，由用户开始编写 `MedSearch/rl/protocol.py`。

候选 V2 已完成自动隔离、token 长度和语义复审；本日继续人工核查 V1 剩余队列与 V2 新补位，
不启动 RL，不把尚未完成的答案人工审核写成已经通过。

## 2026-09-04：V2 2,304 条质量审核收拢与二次审计

- 2,192 条剩余 batch 已全部回收；覆盖、ID、顺序、keep/reject 互斥和 reject manifest 均通过检查。
- 合并此前 112 条后，桌面原始裁决为 keep 1,700 / reject 604。二次抽审只进行高置信修正：恢复
  `medical-o1-verifiable:22697`，清理 10 条重复、冲突答案或时效/上下文不稳定样本。
- 当前临时质量池为 1,691 条：train 1,494、dev 197；清洗 613 条，不补位，也暂不覆盖 V2 原
  parquet。证据和输出归档于 `MedSearch/data/rl/medical_verifiable_v2/quality_audit/`。
- 医学质量与搜索需求严格拆层：467 条 moderate/high、205 条 low、1,019 条 unknown。low 保留为
  no-search/control；unknown 暂不准入 search-positive。467 条严格集只用于 search-only baseline。
- 当前仍等待 256-dev 独立复核；返回后按 ID 合并冲突裁决，再冻结最终质量数据和重算训练 steps。
- 质量版本最初在 `medical_verifiable_quality_v1` 候选目录生成，最终已发布为
  `MedSearch/data/rl/train.parquet` 1,494 条与 `dev.parquet` 197 条；保持原 veRL 八列 schema，
  train/dev ID 无交集且 manifest 哈希复核一致。候选目录现已移除，后续不得继续引用旧路径。
  dev 仅作开发评测，冻结 MIRAGE MedQA/MedMCQA 继续承担最终测试。
- 用户决定首轮 GRPO 使用全部 1,494 条质量合格 train。467 条高检索需求子集以及 no-search 对
  BM25 收益诊断后移为消融，不作为首轮训练准入门槛；检索标签继续保留以支持后续分层分析。

## 2026-09-07：token-F1 医学语义风险审计

- 明确 `medical verifier` 指独立的医学语义判分模型；token-F1 只是词面重合规则，二者不能混用。
- 新增 `MedSearch/tools/audit_token_f1_risk.py`，只用可解释规则标记方向、极性、数值与否定敏感答案；
  它是人工或 LLM 复核队列生成器，不宣称判断医学正确性。
- 对正式 train 1,494 条扫描得到 316 个风险记录、覆盖 280 个唯一问题；其中 97 个问题可通过替换一个
  方向/极性词构造医学意义相反但 token-F1 仍不低于 0.5 的反事实答案。因此 token-F1 暂不冻结为
  唯一在线正确性 reward，只作为 rollout 诊断指标。
- 审计结果位于 `MedSearch/docs/audits/token_f1_semantic_risk.csv`；聚焦测试 `4 passed in 0.03s`。

# 完成标准

- 未拒绝的 V1 样本保持原 split，只有被拒绝位置由固定 reserve 补齐；
- manifest 固化 V1 锚点、累计排除索引、裁决来源和最终 parquet 哈希；
- 最终保持 2,048 train + 256 dev，且 MPNet `cosine >= 0.92` 命中数为 0；
- 聚焦测试、Python 编译和 diff 检查通过；
- 剩余人工质量审核完成前只称 candidate V2，不准入正式 RL 训练。

# 已完成

## 2026-09-04：V1 64 条网页审核回收与 dev 审核导出

- `final_web_review_64_audited.jsonl` 完整覆盖输入 64 条，ID 唯一、无缺失或额外项；结果为
  keep 35、reject 29，四个审核字段均无空值，reject 全有 notes，keep 与负向字段无矛盾。
- 29 条 reject 中，14 条早已从当前 V2 删除，15 条是当前 V2 的新增待排除；另有 1 条网页判
  keep 但此前因语义泄漏被删除，因此仍维持隔离。完整报告 SHA256 为
  `e6dbdd127001669ed431a9e2b04ee816e7d62b6e9440bde9f6755a7eeb59943f`。
- 已归档完整报告并导出 `v1_web_review_rejects.jsonl`。在 2,192 条其余候选网页报告返回前不重写
  最终 parquet；届时与补位 24 条 reject 合并，按 ID 去重、只删除不补位。
- 导出当前完整 dev 256 条到 `medical_verifiable_v2/dev_web_review_256.jsonl`，SHA256 为
  `740a8be5b0c4d63dc842d65bcf444284fbe9519a6e1b9ac8c5cab6f4d7fa099d`；审核标准固化在
  `WEB_REVIEW_STANDARD.md`。dev 253 条与 2,192 条 master 重合，另 3 条已在补位报告中审核，
  最终合并按 ID 去重。

## 2026-09-04：63 条补位队列网页审核回收

- 校验 `refill_review_audited.jsonl`：63/63 个唯一 ID 与当前 `refill_review.jsonl` 完全对应，
  无缺失、无额外 ID；结果为 keep 39、reject 24。四个审核字段无空值，reject 均有 notes，
  keep 没有负向字段，reject 没有全正向字段。
- 完整报告 SHA256 为
  `baff963f915cf955da5a6ea7f03bdc969f08a78ef6edc4f20259dea21f19e6ae`；已另存仅含 24 条 reject
  的 `refill_web_review_rejects.jsonl`，SHA256 为
  `2b78e2b9a6e6fb41062cca4b90d9119566e7da123d0d4c02a8c52b59d96d7595`。
- 用户决定不再为这 24 条补位。当前 parquet 暂不重写，待 V1 完整 64 条最后审核返回后合并
  findings 一次性过滤；若没有其他新增 reject，规模将由 2,304 降至 2,280，实际最终数量以
  最后一轮裁决为准。
- 最后一批网页审核文件为 `medical_verifiable_v2/final_web_review_64.jsonl`，完整保留 V1 固定
  64 条，SHA256 为 `3298755f3bf16931fbe877eaff4e2b629f0396cf1ded80362b1e9a14af7a7d95`。
- 为后续独立审核导出当前 V2 其余 2,192 条：63 条补位报告与当前仍存在的 49 条 V1 队列共
  覆盖 112 条；V1 队列另 15 条已不在 V2。`remaining_web_review_batches/` 包含 master 及
  22 个批次，前 21 批各 100 条、最后一批 92 条，顺序固定为当前 train 后 dev。

## 2026-09-04：candidate V2 稳定补样与语义复审

- 首版排除实现错误地在删除 57 条后重新 shuffle 整个候选池，导致 V2 全量漂移；该版本判定
  无效并覆盖。新增稳定补样回归测试后，构建器改为以 V1 train/dev 原始行号为锚，只从固定
  reserve 顺序补空位，未拒绝样本不换 split。
- 第一轮稳定补样保留 V1 的 2,247 条，train/dev 分别只替换 54/3 条；MPNet 复审发现 3 条
  近乎逐字改写（Stage-A 1、MIRAGE 最终测试 2），已固化到
  `stable_refill_semantic_findings.jsonl` 并加入累计排除。
- 首轮补位与 V1 剩余队列的定向人工核查又确认 14 条问题：开放性气胸与无症状胆石处置错误、
  反应性关节炎病原错误、PAR 缺少暴露率、胃癌手术回答回避、IBD/IBS 混写、循环题干、过时
  clozapine 监测目标，以及 6 条 V1 的机制错误、过时疗法、内部矛盾或条件不足。权威依据与
  逐条理由保存在两份新增 findings 文件中；拿不准的样本没有自动删除。
- 当前累计排除 74 条，其中 11 条属于 reserve 中被二次拒绝的样本；candidate V2 仍为
  train 2,048 + dev 256。相对 V1，train/dev 分别替换 60/3 条，共 63 条当前补位记录，跨 split
  移动为 0。最终 MPNet cosine `0.92` 复审命中数为 0。
- 当前 train/dev SHA256 分别为
  `4a9c2dc4117ba5c6429c368ca1a346a27f29b83f7a64603d23c6a6011b6ff4e0` 与
  `c40791681df592dc97e756c8ed6da430e4dbef5b59d270aaa54931e6fe9abe14`；累计排除索引
  `applied_exclusions.jsonl` 共 74 条，SHA256 为
  `a1837c3902419752278ff9b2863bf7866953ea0882156e4409d55a8f68121fac`。
- 当前只通过自动门槛，尚未完成 V1 队列余下 50 条及当前 63 条补位样本的人工正确性审核，因此
  仍称 candidate V2，不启动正式 RL。

## 2026-09-04：candidate V1 第二轮自动审计与高风险人工核查

- 使用本地 `models/Qwen2.5-3B-Instruct` tokenizer 检查全部 2,304 条：train/dev prompt 最大
  `262/265` tokens，超过 `4096` 的样本均为 `0`；答案最大 `86/53` tokens。当前长度不会触发
  prompt 截断，但 rollout 后的完整轨迹长度仍须在真实环境另测。
- token-set Jaccard 阈值 `0.80` 对四个隔离池找到 7 条近重复：2 条命中 MIRAGE 最终测试、
  5 条命中 Stage-A。逐条核对均为同题轻微改写，全部记录为未来 V2 的 reject；candidate V1
  暂不原地修改，避免在 MPNet 审计前反复重建。
- 64 条人工队列先核查 8 条明确高风险记录，均确认应拒绝：4 条存在可由权威资料或定义核验的
  错误，4 条缺失图片、表格、X 光描述或血气数值。其余 56 条仍为 pending；这 8 条是定向核查，
  不能外推为数据集错误率。
- 后续在 `MedicalGraphRAG/models/all-mpnet-base-v2` 找到此前项目使用的固定模型；MedRAG 本身
  默认是 MedCPT/Contriever，不为方便而更换去重口径。复用该 MPNet 在 CUDA 上完成全部 2,304
  条语义审计，模型权重 SHA256 为
  `a8fd120b1a0032e70ff3d4b8ab8e46a6d01c2cb08ffe7c007a021c1788928146`。
- cosine `0.92` 共命中 49 条明显同题改写：Stage-A 30、Search100 1、protocol dev 0、MIRAGE
  最终测试 18；此前 7 条 Jaccard 命中包含于其中。candidate V1 不原地修改，V2 按 split 排除并
  从 reserve pool 补位后，所有补位题重新经过相同门槛。
- 新增 token 长度和 Jaccard 契约测试，并修复近重复报告缺少候选侧原文的问题；测试经历预期
  RED 后达到 `9 passed in 0.42s`。报告位于 `MedSearch/data/rl/medical_verifiable/`。

## 2026-09-04：Medical Verifiable Problems 来源固化与首轮数据处理

- 官方来源固定为 Hugging Face `FreedomIntelligence/medical-o1-verifiable-problem`，数据文件
  revision `0b347655bd332537e6814d53c6fe4bc03b555bdc`，Apache-2.0；本地原文件 SHA256
  `f5998431acb0eb35e9365f81b6e434d67c44392a5667280068c63ec072ba53cd` 与官方文件一致。
- README 已记录 40,644 条官方 train、两个原始字段及其不包含 evidence、difficulty、原始考试
  来源的限制；派生 parquet 的 `extra_info` 新增 source dataset/revision/split，保留原始行号。
- 使用原 seed `20260902` 重建相同 2,048/256 候选选择；新增 `audit.json`：train/dev ID 不相交、
  空问题 0、空答案 0、跨 split 规范化精确重复 0。新 parquet 哈希变化来自 provenance schema
  增加，最新哈希以 manifest 为准。
- 新增固定 64 条 `manual_review.jsonl`：先纳入 train 中全部答案长度 `<=2` 或 `>200` 的样本，
  再由相同 seed 随机补足；长度只用于风险抽样，不作为自动删除规则。
- TDD 先观察到缺少来源常量/manifest revision 的预期失败，再实现最小来源与审计逻辑；聚焦测试
  `7 passed in 0.39s`。尚未完成 Qwen token 长度、词汇/MPNet 语义近重、人工答案质量和
  no-search/BM25 搜索收益诊断，因此 candidate V1 仍只准用于最小链路 smoke。

## 2026-09-04：modern veRL 架构准入审查

- 总体分层合理：Search-R1 XML 协议被收敛到独立 `AgentLoopBase`、protocol 与 reward 文件；
  `<information>` 通过 `response_mask=0` 排除 policy loss，检索仍是 policy-native action，未改成
  generic function calling。WSL2 shared-memory 补丁为显式 opt-in 环境兼容项，不改 RL 算法。
- 当前不能作为 MedSearch-R1 正式引擎：`generated.token_ids` 被 decode、截断后重新 encode，违反
  token-in/token-out，训练 token 可能不是 vLLM 实际采样 token；pinned veRL 的标准单轮实现直接把
  `output.token_ids` 传给 `ct_merge_assistant_token()`。
- observation 先整体编码再右截断，可能截掉 `</information>`；空 query 会被当作合法 search 发给
  检索器；同一生成中同时出现 search/answer 时被静默接受第一个 action，与冻结的 Stage-B
  malformed 判定不一致。
- 当前启动脚本加载 Qwen2.5-3B Base 并新建 rank-16 LoRA，未设置 `lora_adapter_path`；冻结的
  Checkpoint-B 是 rank-8 LoRA。因此已完成的单步结果只是 Search-R1 modern fallback，不是
  Checkpoint-B 初始化的医学 RL smoke。
- 当前 `reward.py` 是开放域 normalized EM；Medical Verifiable Parquet 的 veRL schema 与其
  `ground_truth.target` 接口兼容，但正式 medical verifier 尚未接入。检索异常也未转成可审计的
  environment failure，一次 HTTP/schema 错误可能中断 rollout。
- 从项目根运行测试时，本地旧 `Search-R1/verl` 会遮蔽 `.deps/verl`，导致缺少
  `verl.experimental`；从 pinned `.deps/verl` checkout 运行同一组聚焦测试得到
  `10 passed, 1 warning in 7.82s`。现有测试证明当前既定行为，不覆盖上述正式准入缺口。

## 2026-09-04：两阶段 SFT 正式冻结并准入 RL 架构阶段

- 冻结 Stage-A 医学领域 QLoRA 与 Stage-B Search100 协议冷启动配方；唯一 RL 初始化产物为
  `MedSearch/outputs/stage_b/formal/adapter/`，不再通过追加 epoch、replay 或混合数据追逐指标。
- Stage-A 冻结 dev 上 Base → Stage-A 的 NLL/PPL 为 `1.380321/3.976179` →
  `1.001387/2.722054`；Stage-B 为 `1.038353/2.824561`，相对 Stage-A 有约 `3.7%` NLL
  回升，但未出现灾难性遗忘。
- Stage-B 在 12 条独立 protocol dev 上合法 search 与非空 query 均为 `12/12`、畸形 action
  `0/12`，足以作为 RL 搜索协议冷启动；该结果仅为首轮 action，不代表 query 质量、BM25
  闭环、开放式医学答案正确性或自适应停止已经通过。
- 架构讨论候选方向为 policy-native `<search>` action、`<information>` environment observation、
  token-in/token-out 轨迹和 environment token `response_mask=0`；不采用 generic function calling。
- 网页讨论提出“先旧 Search-R1 baseline、再迁新版 AgentLoop”，但仓库已存在
  `Search-R1/modern_verl/`，并于 2026-08-20 完成过 3B 单步 GRPO 全链路。因此下一步是比较
  两条已有路径并冻结实验职责，不从零重复迁移；该单步两条 reward 均为 0，仍不算有效学习结果。

## 2026-09-04：旧调度退出，原生新版 veRL 作为唯一 RL 路线

- 使用新质量 parquet 对旧 `trainer.use_v1=False` 路径进行两次真实单步尝试。两次均成功读取
  train 1,494 / dev 197，并通过 GRPO 配置检查；失败发生在异步 vLLM engine 的 KV cache 初始化，
  未进入 rollout 或 optimizer step。
- 第一轮沿用旧 smoke 参数（vLLM memory 0.25、上下文 1280）；第二轮仅将 memory 调至 0.35、
  上下文调至 1024、response 调至 256，根因不变：`No available memory for the cache blocks`。
  这不是医学 parquet 字段、BM25 服务或 reward 的错误。
- 用户决定不再投入旧调度兼容性修复。保留失败日志
  `MedSearch/docs/run_logs/modern_verl_fullpool_1step*.log` 作为环境证据；本次临时启动的 BM25
  服务已停止。
- 后续唯一架构路线：锁定新版 veRL 的原生训练调度，薄迁移 `<search>/<information>/<answer>` 协议、
  token-in/token-out 与 medical verifier；不重用旧 trainer/rollout/WSL 兼容补丁。

## 2026-09-02：Medical Verifiable Problems 首版 RL 数据冻结

- 下载官方 `FreedomIntelligence/medical-o1-verifiable-problem`，共 40,644 条开放式医学可验证题，
  Apache-2.0；原始文件 SHA-256 为
  `f5998431acb0eb35e9365f81b6e434d67c44392a5667280068c63ec072ba53cd`。
- 冻结 `MedSearch/data/rl/medical_verifiable/train.parquet` 2,048 条与 `dev.parquet` 256 条，
  seed `20260902`。排除池覆盖 Stage-A 30,073 个 user turn、Search100 100 条、protocol dev
  12 条和 MIRAGE 最终测试 5,456 条；规范化精确排除 1,156 条，源数据内部重复排除 342 条。
- 输出使用当前 modern veRL 所需的 `prompt`、`reward_model`、`golden_answers`、`data_source`、
  `ability` 与 `extra_info`；开放答案目标来自官方 `Ground-True Answer`。
- 教师契约与产物验收：`4 passed in 1.09s`。manifest 与输出哈希位于
  `MedSearch/data/rl/medical_verifiable/manifest.json`。
- reward 决策：一步 smoke 暂用 normalized exact match 验证接口；正式 pilot 使用 HuatuoGPT-o1
  风格 medical verifier outcome reward，不加入关键词、证据命中或搜索次数 shaping。
- 单卡预算：3B 依次运行 1/20/200 updates，7B 仅在 3B 有效后运行 1/20/100，必要时延长到
  200；不遍历全部 40K，也不按 8×H100 的 Search-R1 batch 512 配方照搬。

## 2026-09-02：Stage-B SFT 首轮协议与保留能力准入结果

- 教师新增 `MedSearch/sft/evaluate_stage_b.py`，只生成 protocol dev 的第一轮，遇到完整
  `</search>` 或 `</answer>` 即停；明确不执行 BM25、information 注入、Evidence Hit@k 或闭环
  答案评测。对应教师测试位于 `MedSearch/docs/tests/test_stage_b_evaluator.py`。
- 同一冻结 Stage-A dev（328 条、89,335 supervised tokens）且同一口径下，Base / Stage-A /
  Stage-B overall NLL/PPL 为 `1.380321/3.976179`、`1.001387/2.722054`、
  `1.038353/2.824561`。Stage-B 相对 Stage-A 的 NLL 回升约 `3.7%`，四个 mixture group 均小幅
  回升；尚未出现灾难性遗忘，但不能据此称“完全不遗忘”。
- 固定 32 条 Stage-A 生成中，Stage-B 没有 `<search>` 泄漏；但 24 条普通对话有 4 条出现非预期
  `<think>`（`16.7%`），高于零，需在后续阈值实验中处理。
- 独立 protocol dev（12 条、greedy、seed `20260823`、max_new_tokens `256`）的合法 search /
  非空 query / 直接 answer / 畸形 action：Base `1/12,1/12,3/12,8/12`；Stage-A
  `0/12,0/12,0/12,12/12`；Stage-B `12/12,12/12,0/12,0/12`。这证明 100 条 Search100 的
  Stage-B SFT 已学会首轮协议冷启动，不证明 query 质量、闭环能力或最终医学正确性。
- 完整回归：`72 passed, 18 warnings`。新增评测不修改模型权重、训练样本或冻结 protocol dev。
- 经学生决定，12 条的随机重复生成不作为下一步前置条件：首轮 `12/12` 已足以判断 SFT 是否
  完成协议冷启动；随机稳定性检查移至搜索环境接入后，用于回归而非阻塞 RL smoke。

## 2026-09-02：正式 Stage-B 与 12 条 protocol dev 独立验收

- 正式 Stage-B 使用 Search100 全部 100 条：batch `2`、梯度累积 `4`，因此是 50 个
  micro-batch、13 个 optimizer step；训练 1 epoch，train loss `1.5409584`、runtime
  `912.2916s`，W&B run `krv3cmlr`。
- checkpoint-13 的 13 个 step 均有有限 loss/grad norm；epoch 末 328 条 Stage-A dev 的
  Trainer eval loss 为 `0.9218875`。最终 adapter 的 504 个 safetensors 张量全部有限。
- protocol dev 共 12 条，one/two search 为 `8/4`，MedQA/MedMCQA 为 `6/6`；所有支持文档
  ID 均存在于本地 MedText。输出 SHA-256 为
  `76a670ec05a84e3a6363d0da9c11685e333e50542fd41ae06df58763a6970a80`。
- 去重覆盖 Search100 100 条、Stage-A 30,073 个 user turn、MIRAGE MedQA/MedMCQA 最终测试
  5,456 条；MPNet 语义阈值 `0.92`，实际最高相似度 `0.781795`。未来医学 RL 尚未构建，
  这 12 个 ID 已显式预留为排除项。
- 独立验收和完整回归通过：`67 passed, 18 warnings`。当前仍没有 Stage-B 行为指标，训练完成
  不能替代协议效果评测。

## 2026-09-01：Stage-B 跨阶段 eval 接线与 1-step QLoRA 冒烟通过

- 修正 `build_dataloader` 的 dataset stage 分派：Search100 train 使用 `stage_b`，冻结的
  Stage-A dev 使用配置中的 `eval_stage: stage_a`；定向测试 `5 passed`，完整回归
  `67 passed, 18 warnings`。
- 从 `MedSearch/outputs/stage_a/adapter` 加载 Stage-A LoRA，使用 Qwen2.5-3B NF4 QLoRA、
  Search100、batch `2`、梯度累积 `4`、max length `4096` 完成 `global_step=1`。
- 真实输出：train loss `2.0851536`，train runtime `35.5325s`；随后完整评估 328 条
  Stage-A dev，控制台 eval loss `0.8756`、eval runtime `30.75s`。最终 adapter 与
  `MedSearch/outputs/stage_b/train_metrics.json` 已生成。
- 边界：本次没有运行 Stage-B 行为评测，不能证明 search 协议提升；该 eval loss 的聚合口径
  也尚未与独立 Stage-A token-weighted NLL 对齐，不能直接拿 `0.8756` 与 `1.0014` 比较。

## 2026-09-01：训练中 dev loss 教师接口与 RED 契约

- 固定 `stage_a_dev.jsonl` 328 条作为训练过程 validation；MedQA test 与 MedMCQA dev 继续作为
  项目最终测试资产，不参与 loss、checkpoint 选择或调参。
- 为 `build_dataloader(..., split="train|eval")`、`build_trainer(..., eval_dataloader=...)`
  增加最小接口和学生契约；Stage-A YAML 新增 `eval_data_path`，尚未开启正式 eval strategy。
- 教师测试 `docs/tests/test_sft_eval_wiring.py` 在 `--runxfail` 下准确得到 4 failed：eval 仍读取
  train、未知 split 未拒绝、TrainingArguments 仍强制 no、Trainer.eval_dataset 仍为 None。
- 默认 CPU 回归 `62 passed, 4 xfailed, 18 warnings`；既有训练、量化、保存和最终评测路径未回归。

## 2026-09-01：Stage-A 正式训练完成并完成 Base/SFT 自动评测

- 正式 Stage-A QLoRA 跑满 `2010/2010` optimizer steps、1 epoch；16,077 条训练样本，
  运行 `3638.2171s`，train loss `1.0024849`，最终 adapter 与 tokenizer 位于
  `MedSearch/outputs/stage_a/adapter/`，`checkpoint-2010` 同时保留 optimizer、scheduler、
  trainer state 与 RNG state。W&B run：`gufa7zwo`。
- 评测代码 commit `7d59c8f`；冻结 dev 328 条、assistant supervised tokens 89,335。
  Base 总体 NLL/PPL 为 `1.380321/3.976179`，Stage-A 为 `1.001387/2.722054`；四个组的
  NLL 均下降：single `-0.268887`、multi `-0.541840`、reasoning `-0.470224`、general
  `-0.187681`。
- 固定 seed `20260823`，每组 8 条，共 32 条 greedy 生成。Base 与 Stage-A 的 sample ID
  完全一致；两者 EOS 结束率均为 100%，max-token 截断率均为 0。
- Base 在 8 条 reasoning 抽样上均未输出预期标签；Stage-A 的 reasoning 8/8 均严格输出
  `<think>...</think><answer>...</answer>`，其余 24/24 普通样本均未误输出 `<think>`，
  格式行为成功率为 100%。该结果只证明格式学习，不证明医学正确性提升。
- 产物：`MedSearch/outputs/eval/base/` 与 `MedSearch/outputs/eval/stage-a-final/`，每个目录
  均含 `metrics.json`、`generations.jsonl` 和 `review.md`。CPU 回归 `62 passed, 18 warnings`。

## 2026-09-01：Stage-A dev 评测与医疗对话抽样脚本完成

- 新增 `MedSearch/sft/evaluate_stage_a.py`：同一 Stage-A YAML 下分别加载 4-bit Base 或
  inference-only adapter，在冻结的 328 条 dev 上计算 assistant-token 加权 NLL/PPL，并按
  single、multi、reasoning、general 分组报告；未接触 MedQA/MedMCQA 最终测试资产。
- 每组固定 seed 分层抽样，导出 `metrics.json`、`generations.jsonl` 与待人工填写的
  `review.md`；人工栏覆盖医学正确性、相关性/完整性、幻觉、安全风险与标签行为。
- 标签指标区分完整 think、畸形标签、reasoning 组缺失、普通组非预期 think、多次生成形态一致、
  多次生成稳定符合预期，以及命中 max-new-tokens 的疑似截断，避免把一致犯错或长度截断算作成功。
- TDD 专项测试经历三轮预期 RED 后达到 9 passed；Docker `llm-pytorch` 禁用 CUDA 的完整回归
  `62 passed, 18 warnings in 13.89s`，警告均来自既有 Torch/TQDM/PEFT 第三方路径；compileall、
  CLI `--help` 与 `git diff --check` 退出码均为 0。
- 使用命令与 Base/SFT 公平对比约束已写入 `MedSearch/README.md`；实施计划位于
  `docs/superpowers/plans/2026-09-01-stage-a-evaluation.md`。
- 正式训练仍在运行，未抢占 GPU、未产生 Base/SFT dev loss、生成回答或医学能力结论；脚本验收
  不等于模型效果验收。

## 2026-09-01：Stage-A正式QLoRA启动并接入W&B（运行中）

- 学生将量化开关职责放在main：量化配置函数只接收dtype，模型加载显式接收已解析dtype；
  教师旧测试已按新接口迁移，未要求学生退回旧设计。
- 完整CPU回归：53 passed、18 warnings；真实QLoRA单步保存/释放/4-bit重载通过，
  adapter张量一致，固定输入logits最大差0.0；tiny checkpoint恢复optimizer/scheduler/global_step并继续到第2步。
- 正式配置：16,077条、max_length1024、batch2、累积4、有效batch8、1 epoch、预计2010步，
  QLoRA NF4双重量化/BF16、LoRA r8/alpha16/dropout0.05、lr2e-4、seed20260831。
- Docker后台PID7819；W&B project `MedSearch-R1`，run ID `gufa7zwo`。
  首次记录step10：loss1.174、grad_norm0.4244、lr0.0001991；观察时至少运行到step17。
- 运行清单：`MedSearch/outputs/stage_a/run_manifest.json`；训练日志：
  `MedSearch/outputs/stage_a/train.log`。训练尚未结束，尚无最终adapter/dev指标或能力结论。

## 2026-09-01：正式SFT入口教师骨架与RED契约

- 实施计划：`docs/superpowers/plans/2026-09-01-medsearch-sft-entry-consolidation.md`。
- Stage-A YAML改为正式QLoRA配方：16,077条、1 epoch、max_steps=-1、batch2、累积4、
  每250步保存、最多2个checkpoint；`resume_from_checkpoint` 初始为null。
- Stage-B仅补统一schema，adapter路径指向`MedSearch/outputs/stage_a/adapter`；未启动、未决定replay。
- 教师测试锁定：CLI只允许--config、YAML控制QLoRA、正式训练后保存adapter、恢复路径原样转发。
- 聚焦回归：19 passed、9 failed、14条第三方警告；9个失败全部属于学生尚未实现的旧main或恢复分支，
  无fixture、导入、配置保存策略故障。尚未运行GPU或启动完整训练。

## 2026-08-31：学生main真实QLoRA Trainer单步通过

- 命令 `wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --qlora`，退出码0；
  教师仅扩展docs探针并调用学生main，未改学生函数/main/YAML或数据，未保存adapter。
- Qwen2.5-3B NF4/double quant/BF16，r8/alpha16/dropout0.05，seed20260831；
  固定8条Stage-A、batch2、梯度累积4、max_steps1、max_length1024，最长实际842tokens。
- global_step1、loss1.1438109874725342、grad_norm0.8728765249252319、Trainer runtime7.8915秒。
  252/504个LoRA张量更新；434个基座参数无梯度且数值保留，108个bias仅库内部dtype转换。
- PyTorch训练峰值allocated7,081,850,880 bytes，reserved12,297,699,328 bytes；
  整卡200ms采样250点峰值15,285MiB（含背景），非4096tokens可行性或性能对比结论。
- CPU回归48 passed、16 warnings（16.11s）；报告MedSearch/docs/run_logs/3b-smoke-SQf36DFi/SUMMARY.md；
  产物MedSearch/outputs/smoke/stage_a-_y_9hw2d/，含配置、实际QLoRA参数、指标、全量参数检查。
- GPU进程已退出；未保存更新后权重，不构成Checkpoint A。QLoRA保存重载/dev评估/断点恢复待验收。
- 可开始代码整合：学生负责正式main编排，教师处理docs中的检查逻辑；本轮未删除或迁移代码。

## 2026-08-31：学生修复LoRA返回缩进，主入口CPU验收通过

- attach_lora的return已由学生退到if/else之后，教师未修改学生代码。
- `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：48 passed、16 warnings in 19.09s，无xfail。
  新增入口检查覆盖QLoRA开/关与梯度检查点开/关；先前None返回导致的连锁失败已消失。
- CPU使用tiny加载替身，不等于真实QLoRA Trainer单步；下一项运行学生入口的GPU单步更新验收。

## 2026-08-31：QLoRA主入口检查发现LoRA返回缩进回归（待学生修复）

- 学生main已正确映射dtype并调用prepare_qlora_model；未代改学生实现。
- 新增docs/tests/test_sft_cli.py主入口开/关QLoRA、开/关检查点的4项CPU测试，昂贵加载以tiny替代，
  其余准备/挂载/data路径真实执行，不代表GPU训练验收。
- 完整回归35 passed、13 failed、16 warnings（17.31s）：attach_lora的return peft_model
  缩进到else内，新建adapter分支隐式返回None，引发入口/Trainer/保存等连锁失败。
- 当前先由学生将返回语句退一级至if/else之后，再重跑；暂停GPU Trainer单步验收。

## 2026-08-31：QLoRA命令行参数骨架

- 学生表示不会引入参数，教师仅添加argparse的--qlora/store_true，默认False。
  args.qlora已可读取，量化模型分支仍未接入，留学生填写；不代写训练编排。
- 参数测试先出现2项缺少参数的失败；添加后完整CPU回归
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：44 passed、16 warnings in 11.01s。
- 未启动GPU训练或更改其他函数；当前唯一任务仍是学生编排QLoRA调用路径。

## 2026-08-31：真实4-bit准备与LoRA梯度联检通过

- 命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward`，退出码0。
  复用教师加载检查新增可选backward模式，不改学生函数/main/YAML或数据。
- Qwen2.5-3B、NF4/double quant/BF16、LoRA r8/alpha16/dropout0.05、seed20260831；
  Stage-A第0条medqa-dialog_4790，batch1×515，281个监督token，loss1.9581148624420166。
- 252个量化层保留；504个LoRA张量梯度有限，252个B非零、A首次为零。
  434个冻结基座参数无梯度且数值保留；108个bias仅发生库内部FP32→BF16转换。
- 首次教师全量字节哈希断言失败；核对bitsandbytes源码并记录变化参数后，只允许指定bias
  精确数值不变的dtype转换，其他326个基座参数仍字节一致。失败日志保留，未改学生实现。
- forward+backward0.9319077680120245秒；PyTorch峰值allocated4,657,840,640 bytes，
  reserved4,752,146,432 bytes；无optimizer状态，不是整卡或完整训练显存，不外推4096tokens。
- CPU命令 `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：42 passed、16 warnings in 12.09s。
- 报告：MedSearch/docs/run_logs/qlora-backward-VKJlYdcw/SUMMARY.md。
  未运行optimizer更新/Trainer训练/保存；下一项调用编排归学生，教师继续负责验收。

## 2026-08-31：学生低比特准备接口 CPU 验收通过

- prepare_qlora_model 复用PEFT一次，梯度检查点True/False均原样传递，kwargs固定use_reentrant=False，
  返回原模型；tiny CPU检查基座冻结、非量化参数转FP32、数值/配置/设备保留均通过。
- `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh MedSearch/docs/tests --runxfail`：
  42 passed、16 warnings in 14.78s。移除两项占位标记后默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：42 passed、16 warnings in 11.11s，无xfail。
- 未修改学生函数或main，未启动GPU、训练或保存。测试仍在MedSearch/docs/tests/；
  CPU tiny量化标记只触发PEFT准备分支，不能证明实际Params4bit保留或QLoRA梯度正确。
- 下一项复用现有加载/准备/attach_lora做真实4-bit准备与LoRA梯度联检，不代写正式main。

## 2026-08-31：低比特训练准备教学接口（待学生填写）

- 已核对MedicalGPT/training/supervised_finetuning.py:920-923与当前PEFT0.20.0源码。
  MedicalGPT在创建新LoRA前准备量化基座；本项目额外保持既有non-reentrant检查点设置一致。
- 新增prepare_qlora_model(model, use_gradient_checkpointing)，只留契约和NotImplementedError；
  导入PEFT原生函数，不改加载/挂载/Trainer实现或main/YAML，不安装依赖、不运行GPU。
- 教师测试先出现2项缺少接口的真实失败；骨架后命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：
  `40 passed, 2 xfailed, 16 warnings in 11.35s`。预期失败仅限占位NotImplementedError。
- 测试位置MedSearch/docs/tests/test_sft_kbit_preparation.py，覆盖开/关、原生调用、返回对象、
  基座冻结与精度/数值边界。tiny CPU量化标记是测试替身，不冒充4-bit GPU准备或反向传播。

## 2026-08-31：学生量化加载分支验收通过

- CUDA 提前检查、单卡 device_map、量化对象原样传递和非量化兼容均通过；未修改学生代码。
- 默认命令 `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`：
  `40 passed, 16 warnings in 15.08s`，已撤去两项占位 xfail，警告为已知第三方弃用提示。
- 真实加载命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh`，退出码 0。
  Qwen2.5-3B-Instruct、seed20260831、NF4/double quant/BF16，252 个 Linear4bit 层；
  全部参数在 cuda:0，35-token 单次前向 logits 有限。加载耗时 27.8416 秒。
- PyTorch 加载及单次前向峰值 allocated 2,149,335,040 bytes，reserved 2,176,843,776 bytes；
  不是整卡占用或训练显存，不能据此外推 4096-token QLoRA 训练。GPU 验证进程已退出。
- 报告：`MedSearch/docs/run_logs/qlora-load-jxVC6GU3/SUMMARY.md`，含版本、源码哈希和原始日志。
  首次教师报告引用缺失的 hf_device_map 属性而失败，保留在 qlora-load-Gd9jRIwx；
  改为报告实际参数设备后重跑通过，学生无需为教师报告增加属性。
- 不挂 LoRA、不 backward、不训练、不保存模型。下一项为低比特训练准备，正式 main 仍归学生。

## 2026-08-31：量化加载骨架与 Docker 环境微型检查

- Docker `llm-pytorch` 内新增 bitsandbytes0.50.2，使用 --no-deps 保留原 torch/transformers。
  PEP668保护在明确工具批准后仅对容器安装覆盖，主机Python未修改；版本固定在requirements-qlora.txt。
- 真实RTX5090/CC12.0/CUDA13.0微型NF4+double-quant层前向通过，非Qwen加载或训练。
  命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_environment.sh`。
  日志：`MedSearch/docs/run_logs/qlora-env-2p8oGYHS/check.log`。
- load_base_model新增仅限关键字quantization_config=None；保留非量化实现，量化分支仅留
  NotImplementedError，CUDA检查/加载调用由学生填写。未改main/YAML或已验收配置函数。
- 新测试初始1 passed、2 failed（缺少接口参数）；骨架后完整回归
  `38 passed, 2 xfailed, 16 warnings in 13.06s`。两项占位待验收，第三方弃用警告未屏蔽。
- 参数传递CPU测试不等于GPU实际量化；学生实现后再验收真实加载。
- 安装命令、wheel哈希、依赖版本及容器持久性边界详见`MedSearch/docs/QLORA_ENVIRONMENT.md`。

## 2026-08-31：QLoRA 量化配置验收通过

- 学生实现 `build_quantization_config()`：关闭返回 None，开启返回原生 BitsAndBytesConfig，
  4-bit/NF4/double quant 正确，计算 dtype 按调用方传入，不同时启用 8-bit。
- `--runxfail` 完整回归 37 passed in 14.38s；教师撤去待实现标记后，默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh` 得到 `37 passed in 15.64s`，无警告、无 xfail。
- 配置检查覆盖关闭分支及 BF16/FP16/FP32 三种 compute_dtype；未修改学生核心实现或 main。
- 本轮没有安装 bitsandbytes、加载量化模型或运行 GPU 训练。结论仅为配置对象正确，
  不能称为 QLoRA 训练跑通。下一项为量化加载，依赖先由教师核验，关键适配仍交学生。

## 2026-08-31：QLoRA 量化配置教学骨架（待学生填写）

- 新增 `build_quantization_config(use_qlora: bool, compute_dtype: torch.dtype)` 的签名和契约，
  函数体保留 NotImplementedError，返回类型为 BitsAndBytesConfig 或 None。
- 复用 MedicalGPT 的 NF4/double quant 配置思路；未修改 main、load_base_model、YAML 或既有
  学生实现。QLoRA 属于既有学习路线，不是医学任务额外的自定义量化算法。
- 教师测试 `MedSearch/docs/tests/test_sft_quantization.py` 先出现 4 个接口缺失红灯；
  骨架后完整回归 `33 passed, 4 xfailed in 10.87s`。预期失败限 NotImplementedError，
  学生完成后由教师撤去标记、真实检查配置对象。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- Docker 中 bitsandbytes 尚未安装；本地 Transformers 5.10.4 配置构造没有要求该包，
  不阻碍纯配置任务。真实量化加载前须核验依赖与硬件兼容。本轮未安装依赖或启动 GPU。

## 2026-08-31：真实 3B BF16 LoRA 保存/重载通过

- 最终命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --save-roundtrip`，退出码 0；
  最新 CPU 回归 `33 passed in 10.22s`。
- 固定 8 条 Stage-A、seed20260831、global_step1、loss1.173087239265442、grad_norm0.8145956993，
  Trainer runtime7.3378 秒。252 个 LoRA 张量更新，434 个冻结基座张量哈希不变。
- 保存后释放训练模型，再重载相同底模与 adapter；504 个 adapter 张量、434 个基座哈希、
  tokenizer 均一致。同一 BF16 autocast/SDPA/eval 下最后位置全词表 logits 最大差0。
- adapter 权重59,933,632 bytes；整卡400个采样点峰值18,062 MiB（含背景）。GPU 进程已退出。
- 成功报告：`MedSearch/docs/run_logs/3b-smoke-SrvIu01k/SUMMARY.md`；产物：
  `MedSearch/outputs/smoke/stage_a-kv1h2i46/`，含 adapter、配置、指标及前后 logits 文件。
- 此前两次教师探针失败均保留：Q9PJhbEt 为 chat template 默认返回格式误用；8byqSe5Q 为
  保存前后 autocast 不一致。仅修正 docs 内脚本并补测试，未改学生训练实现或 main。
- 本轮不是 QLoRA、正式 Checkpoint A 或医学效果提升。后续正式 main 由学生编写；
  下一项唯一任务改为 QLoRA 4-bit 配置，不提前代写完整流程。

## 2026-08-31：save_adapter CPU 验收通过

- 学生增加 `save_embedding_layers=False`；原生模型/tokenizer 保存与隐式 None 返回符合契约。
  教师未修改学生实现，也未要求删除无害的重复 mkdir 或显式重写默认 safe_serialization=True。
- 修正前专项测试 3 passed、1 failed：tiny 与引用配置词表大小不同，auto 分支额外保存 embedding；
  修正后全量 `--runxfail` 回归 31 passed in 13.87s。撤去待实现标记后，默认命令
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh` 得到 `31 passed in 11.76s`，无警告、无 xfail。
- 验证 safetensors 仅含 LoRA、adapter 权重逐项一致、tokenizer 词表/特殊 token/chat template 保留、
  同底模副本重载后的固定输入 logits 一致，以及 attach_lora 可训练加载分支。
- tiny adapter 人工设为非零，仅用于序列化验收；不宣称真实 3B 保存已验证或医学效果提升。
  未启动 GPU 训练、未生成正式 Checkpoint A；下一唯一任务为真实 3B 单步保存/重载验证。

## 2026-08-31：adapter 保存教学骨架（学生实现待验收）

- `sft/train.py` 新增 `save_adapter(model, tokenizer, output_dir) -> None`，函数体保留
  NotImplementedError；教师接好仅限 `--train-smoke` 的 `--save-adapter` 参数与独立 adapter 子目录。
- 参考 MedicalGPT 原生保存方法；契约要求只保存 LoRA/config/tokenizer，不合并底模，
  不保存 embedding，不改已验收的 run_training 返回值，不做 optimizer/step 恢复。
- 新增教师测试 `MedSearch/docs/tests/test_sft_save.py`：非法模式提前拒绝，以及 adapter
  权重/tokenizer/同底模固定输入 logits 的 round-trip 验收。tiny adapter 人工设为非零仅用于
  序列化检查，不冒充训练结果；重载底模使用原底模副本，避免随机初始化差异。
- 初始红灯为 4 failed（接口与参数未创建）；保存重载测试保留严格 NotImplementedError xfail，
  等学生实现后再撤去，不能将预期失败算作保存功能通过。本轮未运行真实 3B 或产生正式 adapter。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 最新 CPU 回归：`30 passed, 1 xfailed in 13.41s`，无警告；仅学生保存实现待验收。

## 2026-08-31：真实 Qwen2.5-3B Stage-A 单步 smoke 通过

- CPU回归27 passed in 14.26s；GPU运行退出码0，复用学生run_training和唯一main入口。
- BF16 LoRA rank8/alpha16/dropout0.05、学习率0.0002、batch2×累积4、max_steps1、seed20260831。
  固定8条样本，实际长度143～842，配置上限1024。
- 真实global_step=1、loss=1.173087239265442、grad_norm=0.8148247599601746、
  Trainer训练耗时6.3281s；252/504个LoRA张量变化，434个冻结基座张量全量哈希变化为0。
- 整卡显存306点采样峰值18,139 MiB/32,607 MiB；PyTorch peak allocated 9.134 GiB、
  reserved 14.113 GiB。整卡包含背景占用；真实训练进程已结束，显存释放。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh`。
  报告：`MedSearch/docs/run_logs/3b-smoke-eqK4l3I8/SUMMARY.md`，同目录保存原始日志和显存CSV。
  有效配置、代码/数据哈希、metrics与参数验证保存在`MedSearch/outputs/smoke/stage_a-_rs5vew5/`。
- 没有改动学生核心实现或冻结数据；只增加docs内教师观察脚本。无OOM；初始化较慢后自然完成，
  未禁用监控或修改训练逻辑。tokenizer特殊token提示为Trainer配置对齐，不是训练失败。
- 不宣称医学效果提升，不外推Stage-B4096或全量训练；没有保存adapter，本次不是Checkpoint A。

## 2026-08-31：run_training 最终验收通过

- 学生通过字典展开保留完整 TrainOutput.metrics，并加入真实 global_step；新字典没有修改
  原 metrics。教师未修改学生实现，不再增加该接口的要求。
- 教师修正新增测试的错误假设：当前 Trainer 回调可能从返回 metrics 移除 total_flos，不能
  将 log_history 作为返回字段的真值。此前关于“必须补回 total_flos”的判断撤回；现在直接
  观察真实 TrainOutput，逐字段验证返回值一致，真实训练没有被 mock 替代。
- 撤去 xfail 后完整回归 `27 passed in 12.94s`，命令：
  `wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 验证真实 tiny 训练调用恰好一次、global_step=1、有限 loss/非零有限梯度、LoRA 参数更新、
  基座逐项不变及完整返回指标。未运行真实 3B，未生成训练 checkpoint。
- 沟通改进：今后预留接口应明确返回结构示例，区分“消费端读取哪些字段”和“返回时保留哪些
  字段”，不再让学生因教师说明或测试假设不清反复返工。

## 2026-08-31：run_training 扁平结构复验（指标完整性待修正）

- 学生已修复返回层级；tiny CPU 实际通过 global_step=1、有限 loss/非零有限梯度、LoRA
  参数变化、基座参数逐项不变检查，未生成 checkpoint。
- 教师补上此前遗漏的“保留全部原有 metrics”断言；实际缺失 total_flos、
  train_samples_per_second、train_steps_per_second。未修改学生实现。
- 相同限定目录命令复验结果 `26 passed, 1 failed in 13.48s`。当前只要求完整保留指标，
  不扩展训练功能；未启动真实 3B。

## 2026-08-31：run_training 首次验收未通过

- 保存后的实现正确调用一次 trainer.train() 并读取真实 global_step，但返回
  `{metrics: {...}, global_step: ...}`，不符合下游直接读取顶层 train_loss 的接口契约。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh MedSearch/docs/tests --runxfail`。
  结果 `26 passed, 1 failed in 13.54s`，失败为 KeyError('train_loss')。
- tiny CPU 真实执行至 global_step=1，日志显示 loss 约4.136、grad_norm 约0.5355；这是合成
  token 小模型的管线检查，不是医学模型指标。参数更新/基座不变断言位于失败之后，尚未执行。
- 未修改学生代码；当前等待学生将返回结构改为扁平字典，未启动真实 3B 或正式训练。

## 2026-08-31：单步训练教学骨架（run_training 待学生实现）

- 教师新增 `prepare_train_smoke()` 和互斥 `--train-smoke`，只允许 Stage-A、max_steps=1、
  单进程/单设备；每次新建 smoke 输出目录，原配置和正式 Checkpoint A 目录不受影响。
- 模型/LoRA 初始化前 set_seed，固定抽取一次完整梯度累积的样本数；入口记录有效配置、
  样本 ID、Git HEAD、SFT 源码/数据哈希；训练成功后记录真实 metrics 和 Trainer log history。
- 学生 `run_training()` 仅预留 Trainer 训练调用与真实结果转换契约，未替学生完成实现。
- 新增红灯检查因接口缺失产生 `5 failed`；教师保护实现后，完整 CPU 回归
  `26 passed, 1 xfailed in 12.92s`。唯一预期失败为学生训练接口，未执行 backward。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。新增测试在
  `MedSearch/docs/tests/test_sft_step.py`，待学生完成后检查有限 loss/梯度、LoRA 更新与基座不变。
- 本轮未跑真实 3B、完整 CLI 训练路径或 GPU 显存测量，未产生训练 checkpoint；不能将骨架
  回归通过写成单步训练通过。

## 2026-08-31：学生 Trainer 接口验收

- 学生正确实现 `build_trainer()`，原样复用模型、tokenizer、Dataset/collator，返回 HF Trainer；
  不手写 DataLoader/optimizer，不调用 train，不改 labels 或解冻基座。
- 首次限定目录回归为 `19 passed, 1 failed`：教师断言错误地要求关闭 checkpointing 时保留
  原 cache。核对本地 Transformers 5.10.4 `trainer.py:608`，Trainer 会写入 args.use_cache，
  其默认值 False；修正测试与契约说明，未改学生执行逻辑。
- 撤去 xfail，补验 Stage-B 和完整 batch 三张量对比；CPU fixture 禁用无意义的 pin_memory。
  最终干净结果：`22 passed in 12.31s`，无跳过项或预期失败项。
- 命令：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`；位置：`MedSearch/docs/tests/`。
- 只测试 tiny LoRA 模型构造与真实 Stage-A/B 样本组 batch；global_step=0、optimizer/scheduler
  未创建、可训练参数仅 LoRA。未运行真实 3B Trainer、backward、训练指标或显存测量。

## 2026-08-31：Trainer 教学骨架（学生接口待实现）

- 按学生决定暂缓 Stage-B replay，当前保持只读取 Search100；Stage-A 冻结配方和全部数据未改动。
- 教师提供 `build_training_arguments()`、两个 YAML 的 `training` 配置及 `--trainer-smoke` 接线；
  `build_trainer()` 只留契约和 `NotImplementedError`，未替学生填写。
- 参考本地 MedicalGPT `training/supervised_finetuning.py` 的 Trainer 接线，复用 HF 原生类；
  配置非重入式 gradient checkpointing、保留 labels、禁用评估/保存/外部日志；本轮没有训练循环。
- 红灯验证：新测试因两个接口不存在产生 `3 failed`。最终 CPU 回归真实结果
  `18 passed, 2 xfailed in 12.25s`；两项 xfailed 仅代表学生接口未实现，不是验收通过。
- 运行命令（仓库根目录 CMD）：`wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh`。
- 环境：现有 `llm-pytorch` 容器，Transformers 5.10.4、PEFT 0.20.0、Accelerate 1.14.0；
  测试禁用 CUDA 和联网，只验证 CPU 接线。没有真实 3B Trainer、backward、显存或训练指标。
- 学生填写后需移除严格 xfail 标记，再验收真实 tiny LoRA Trainer 与 batch；后续正式训练前
  仍需接好 dev 评估、保存/恢复和真实单步检查。

## 2026-08-30：Qwen2.5-3B + LoRA 模型装配验收

- 教师参照 MedicalGPT 已成功 SFT adapter，固定 BF16、Qwen 7 类投影模块、rank 8、alpha 16、
  dropout 0.05，并接好本地底模加载、参数边界检查与统一入口 `--model-smoke`；
- 学生完成 `attach_lora()`：Stage-A 用 `LoraConfig/get_peft_model` 新建 adapter，Stage-B 用
  `PeftModel.from_pretrained(..., is_trainable=True)` 续载，并对缺失 adapter 路径明确报错；
- tiny Qwen 真实完成新建、保存、续载和缺失路径测试；encoder、Dataset、入口与模型完整回归
  为干净的 `15 passed in 6.32s`；
- 真实 Qwen2.5-3B-Instruct BF16 CPU smoke 正常加载434组权重；LoRA 可训练参数
  `14,966,784`、总参数 `3,100,905,472`、占比 `0.4826585%`，没有非 LoRA 可训练参数；
- Stage-B 真实 adapter 尚不存在，因此未伪造 Checkpoint A，也未声称 Stage-B 3B 续训加载通过；
  当前只有 tiny adapter 续载边界通过。

## 2026-08-30：单一 SFT 数据入口 smoke 验收

- 教师参照 MedicalGPT 的参数入口、AutoTokenizer 与 collator 调用链，创建单一
  `MedSearch/sft/train.py`、`stage_a.yaml`、`stage_b.yaml` 和隔离测试；没有复制完整 Trainer 参数层；
- 学生完成 `build_dataloader()`：解析相对/绝对数据路径，复用 `load_samples()`、`SFTDataset`
  与 `build_data_collator()`，并把 batch size、shuffle、worker 数从配置传入 PyTorch DataLoader；
- encoder、Dataset 与统一入口完整回归真实结果为 `12 passed in 10.48s`；
- 正式 CLI smoke 共用同一入口：Stage-A 读取 16,077 条并输出 batch `[2, 683]`、上限 1,024；
  Stage-B 读取 100 条并输出 batch `[2, 1254]`、上限 4,096；两者均正常退出；
- 本轮只加载 tokenizer 和一个 CPU batch，没有加载模型、创建 LoRA/Trainer 或占用 GPU。

## 2026-08-30：统一 SFT Dataset 与动态 padding 验收

- 教师先在 `MedSearch/docs/tests/test_sft_data.py` 建立红灯测试和 `MedSearch/sft/data.py`
  接口骨架；学生完成 `.jsonl/.json` 读取与 Stage-A/Stage-B encoder 分派；
- 动态 padding 未手写重复循环，直接复用 MedicalGPT 已验证的 Transformers
  `DataCollatorForSeq2Seq`：input_ids 使用 tokenizer pad id、attention mask 补0、labels 补 `-100`；
- Dataset/encoder 隔离回归真实结果为 `9 passed in 8.01s`；
- 正式数据 smoke 读取 Stage-A train 16,077 条和 Search100 100 条；抽取首尾样本组成真实 batch，
  Stage-A 长度为 515/409、batch shape `(2, 515)`，Stage-B 长度为 1005/1258、batch shape
  `(2, 1258)`，三张量 shape 一致；
- 当前采用按 `__getitem__` 即时编码的最小实现；只有真实 profiling 证明 tokenizer 成为训练
  瓶颈时才增加预编码缓存，不提前复杂化。

## 2026-08-29：Stage-B Search-R1 编码与长度边界验收

- 学生完成 `MedSearch/sft/preprocess.py::encode_stage_b()`：开头只渲染一次 Qwen user prompt，
  `think/search/answer/EOS` 参与 loss，环境 `information` 保留在 attention 中但 labels 为 `-100`；
- 修复 `step[type]` 将 Python 内置 `type` 误作字典键的问题，并为未知 trajectory type 增加
  明确 `ValueError`；保留 answer 后 `break`，不为已冻结合法轨迹增加重复结构校验；
- 本地 Qwen2.5-3B-Instruct 原生 `max_position_embeddings=32768`；Search100 编码最长约
  3,070 tokens，因此 Stage-B 固定 `max_length=4096`，超长直接报错而不静默截断；
- 教师修正隐藏测试中 prompt 示例 `<answer>B</answer>` 导致的首次匹配误判；Docker 最终结果
  为 `5 passed in 4.88s`，其中包含正式 100 条 Search100 全量 4,096-token 闸门；
- 后续 RL 训练策略确定为 LoRA-GRPO；训练速度、显存和有效 reward 组比例仍必须由真实 smoke
  记录，不能根据模型规模纸面宣称可行。

## 2026-08-29：Stage-A 对话编码与 loss mask 验收

- 学生完成 `MedSearch/sft/preprocess.py::encode_stage_a()`：直接读取冻结数据的 `messages`，
  复用 Qwen2.5 `apply_chat_template`，system/user labels 为 `-100`，assistant 内容和每轮 EOS
  参与 loss；没有引入额外中间 schema、Dataset 或 Trainer；
- 教师维护的隔离测试位于 `MedSearch/docs/tests/`，真实容器运行结果为 `1 passed in 8.28s`；
- 使用本地 Qwen2.5-3B-Instruct tokenizer 全量编码 Stage-A train/dev 共 16,405 条：最短
  48 tokens、最长 1,022 tokens、超 1,024 为0、空 target 为0，三数组长度全部一致；
- 学生能解释多轮中 assistant 必须进入 `accumulated`，否则 `previous_rendered` 会缺失历史并
  破坏下一轮增量切片；能解释 causal LM 在 `forward(labels=...)` 内部对 logits/labels 做移位，
  因此数据预处理不手动右移 labels。严格说该移位由模型 loss 实现，不是 Trainer 本身完成。

## 2026-08-29：Stage-A 最终质量修复

- 全量复核发现 reasoning 1,640 条缺顶层 `id`，以及2条 UltraChat general 样本含原始数据
  自带的 `U+FFFD`；其中一条还包含明显事实性幻觉，不能用标点替换修复；
- reasoning ID 已确定性补为 `medical_o1_{provenance.index}`；两条损坏 general 已整条删除，
  并从同一固定 UltraChat revision 补入2条经结构、主题和 Qwen 1,024-token 检查的样本；
- 总量和配方保持不变：train 16,077、dev 328，single/multi/reasoning/general 为
  10,663/3,281/1,640/821；最终 missing ID、重复 ID、`U+FFFD` 均为0；
- 新增可幂等运行的 `MedSearch/tools/finalize_stage_a.py` 和最小回归测试；
- `MedSearch/README.md` 明确教师/学生职责，下一步正式进入学生编写 SFT 数据入口。

## 2026-08-24：MedSearch SFT 工作区瘦身

- 当前活动根目录只保留 `.gitignore`、`PROJECT_BOOK.md`、正式 `data/` 和一个可恢复的
  `archive/`；不再并列堆放审计说明、抽样目录和一次性脚本；
- Stage-A 活动数据只保留 `stage_a_train.jsonl`、`stage_a_dev.jsonl`、
  `stage_a_manifest.json`；Stage-B 活动数据只保留正式 Search100、医学检索语料与 BM25 索引；
- 清洗、复核、构造、验证脚本及其测试，一次性样本、审计 Markdown、候选判定与失败档案统一
  收进 `MedSearch/archive/data_preparation/`，不参与当前 SFT 编码与训练；
- `PROJECT_BOOK.md` 是当前唯一项目说明来源；旧审计文档不再被活动文档引用。

## 2026-08-23：Stage-B Search100 精品检索轨迹完成

- 从 MedQA/MedMCQA official train 构造 gold-blind 候选；query、检索和最终决策阶段均不读取 gold；
- 真实调用 Search-R1 BM25 医学服务，语料为 MedRAG Textbooks + StatPearls；检索 observation 和稳定
  `document_id` 全部保存在本地审计侧文件；
- 生成 190 条候选（含最初 10 条 pilot），由 `gpt-5.6-sol/xhigh` 按答案、证据、引用和题目质量终审；
- 独立二次复审替换 9 条医学答案正确但存在明显 OCR/拼写噪声的样本，最终 100/100 通过；
- 正式训练集精确保留 100 条：80 条一轮搜索、20 条两轮搜索；顶层字段仅为
  `id/question/options/trajectory`，trajectory 仅含 `search/information/answer`；
- 正式训练文件为 `MedSearch/data/stage_b/search100_training.json`；一次性审计与失败候选
  已在冻结结论和哈希后清理，不进入学生的 SFT 工作区；
- 正式文件 652,019 bytes，SHA-256
  `e9796395e17f6a9c29520cb48758ed9cdbb083c1d231f9562e4d2ef2b25725fb`；
- 清理前自动验证：100 个唯一 ID、字段无审计信息泄漏、answer 均属于 options、轨迹顺序合法；
  当时 53 项测试通过。一次性构建、审计与测试脚本随后按学习边界清理。

## 2026-08-23：Stage-A 医学三池全量清洗

- 数据清洗职责改由教师代理负责，学生从训练搭建开始学习；
- 77,846 条 raw 经确定性规则得到 drop 3,638、review 9,572、keep 候选 64,636；
- 三个 `gpt-5.6-sol` 对 review 做一次性二元终审：accept 7,500、reject 2,072，不留人工队列；
- 目标 Qwen2.5 tokenizer 长度闸门排除 2,494 条，最终 eligible pool 为 69,642 条；
- 三个低成本子智能体完成跨域盲审，修复了多轮 answer-confirmation 过度删除和单轮 false keep；
- 医学三池清洗结果作为下方 65/20/10/5 当前配方的输入；已剪除被替代的 70/20/10
  构建结果，避免学生误用旧 manifest；
- 当前有效统计、字段与冻结配方统一见 `MedSearch/PROJECT_BOOK.md` 和 Stage-A manifest。

## 2026-08-23：Stage-A 加入 5% 通用对话 replay

- 固定 UltraChat 200k 官方 revision `8049631c405ae6576f93f445c6b8166f76f5505a`，只下载并
  校验 `train_sft` 三个 Parquet 分片，共 207,865 条；行数、字节数和官方 SHA-256 一致；
- 确定性规则排除医学、代码、数学、工具动作、时效问法、模板拒答、截断和伪亲历，得到
  121,509 条候选；强模型分层抽检 240 条，accept 151、reject 89，并将 reject ID 固定排除；
- Qwen2.5 真实 chat template 长度闸门保留 59,133 条，排除 62,376 条超 1,024-token 样本；
- 固定 seed `20260823` 重建 65/20/10/5 Stage-A：single 10,663、multi 3,281、reasoning
  1,640、general 821；train 16,077、dev 328，user 重叠 0，强审 reject 命中 0；
- Qwen2.5 Instruct 已有通用指令能力，因此通用数据只作为 5% replay；0%/5%/10% 留作后续
  单变量消融，不把 5% 写成已验证最优值；
- 原 70/20/10 版本已被本配方替代，不再作为当前训练输入。

## 2026-08-23：全量下载验证与首轮风险抽检

- 五个入选文件已全量下载并完成 JSON 解析、记录数与 SHA-256 检查，共 77,846 条记录；
- 固定 seed `20260823` 生成单轮 90 条、多轮 40 条、medical-o1 40 条风险审计清单；
- 三个低成本子智能体只读初筛，共标记 `DROP 21`、`REVIEW 46`、
  `provisional KEEP 103`；该审计集刻意富集风险样本，比例不能外推全库；
- 多轮样本发现缺图和答案机械确认，MedQA-dialog 的 30%～40% 改为待验证上限；
- 当前有效的数据边界和最终选择统一见 `MedSearch/PROJECT_BOOK.md`。

## 2026-08-23：四类 SFT 候选数据各 10 条实物审计

- 固定 Meerkat revision `f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` 与 medical-o1
  revision `fc2c9e8a37b38f38da6d449564a8c350b244aef4`，通过镜像范围读取原仓库文件；
  没有下载全量数据。
- 四类候选各 10 条已完成抽查；冻结数据来源后，一次性样本和独立审计说明已清理。
- ChatDoctor-cleaned 抽样出现 C. difficile 被误判为 ulcerative colitis、持续胸痛被直接
  归为 gastritis 等安全问题，不能全量直接训练。
- MTS-dialog 的现有 target 是 clinical note generation，不是医生多轮回复；第一版排除。
- MedQA-dialog 是 5～17 messages 的 GPT-3.5 合成多轮考试对话，条件保留为小比例数据；
  缺图、答案确认模板和 MedQA 测试污染必须过滤。
- medical-o1 English 条件保留为少量 supervised rationale；需过滤缺图、异常题面，并
  避免把重复的 `Complex_CoT` 与 `Response` 无设计地串接。
- 当前尚未找到足够可靠的普通医学对话主数据源，因此不启动全量清洗或训练。

## 2026-08-23：Meerkat 第二批四个子集各 10 条审计

- 从同一固定 revision 取得 LiveQA、MedicationQA、MedInstruct-52K 和 MedBooks-18-CoT
  各 10 条；宿主机镜像连接超时后，使用现有 `llm-pytorch` 容器访问完全相同的固定 URL，
  文件写回同一工作区，没有改变数据版本。
- LiveQA 适合作为小型单轮质量锚点，但重复问题对应多个答案，需按规范化问题分组；
  MedicationQA 有过短和表格残片 target，只能过滤后保留。
- MedInstruct-52K 是多任务单轮 instruction，不是真实对话；可作为普通医学 assistant
  主体候选，但需过滤 `As an AI`、时效问题、模板答案和输入输出近复制。
- MedBooks-18-CoT 结构完整且论文有 CoT/textbook augmentation 消融支持，推荐作为第一版
  少量 rationale；它不承担普通对话，也不证明临床安全。
- 当前推荐配方为普通 instruction 75%～85%、多轮 5%～10%、rationale 10%～20%；
  该初步配方已被同日最终决策替换。

## 2026-08-23：阶段 A 数据配方冻结

- 第一版停用 ChatDoctor-cleaned、MTS-dialog 和 MedBooks-18-CoT；ChatDoctor 的医学标签
  与分诊噪声不依赖后续 RAG 修复。
- 普通单轮使用过滤后的 MedInstruct-52K 与去重后的 LiveQA/MedicationQA，目标占比
  45%～55%；多轮 MedQA-dialog 提高到 30%～40%。
- reasoning 只使用 medical-o1 English，目标占比 10%～15%。它的 `Complex_CoT` 作为
  SFT target 时仍属于 supervised rationale，不能写成无监督推理。
- medical-o1 第一版候选长度为 CoT 不超过 512 tokens、整条不超过 1024 tokens；使用
  Qwen2.5 tokenizer 统计后再冻结。超长样本整条过滤，不从中间截断推理。
- 总规模目标约 10K；清洗后不足时按真实数量训练，不用已过滤数据补齐。

## 2026-08-23：SFT 数据职责重新冻结

- 阶段 A 改为多数医学单轮/多轮对话加少量 supervised rationale，不再以
  MedQA/MedMCQA 选择题作为普通医学 SFT 主体。
- 阶段 B 固定为约 100 条真实检索支持的 Search100 加普通医学对话 replay；保留 replay
  是为了维持正常回答能力并降低协议过拟合与逢题必搜风险，但具体比例仍是实验超参数。
- 普通对话不强行添加 `<think>`；reasoning 子集只训练可见、可审计的医学推理说明，
  不把它描述为模型真实的内部思维。普通对话与 Agent 样本通过 system/task prompt 区分。
- MedQA/MedMCQA 官方 train 改作 Search100/RL 候选题池；冻结评测资产边界不变。
- 本次只更新项目书与数据审计，没有下载、转换、tokenization 或训练数据。

## 2026-08-21：MedSearch-R1 两阶段 SFT 与数据实物审计（阶段 A 配方已于 2026-08-23 替换）

- 当时先冻结两个 checkpoint 与 Search100 冷启动框架；其中“医学选择题 SFT + 医学 QA
  replay”的旧配方已被 2026-08-23 的“对话型 SFT + 普通医学对话 replay”替换。
- Search100 改为默认冷启动资产，不再以基础模型协议测试失败为前提；一轮/两轮/直答
  比例仍是待实验超参数。
- 本地 MIRAGE MedQA 文件真实为 1,273 条官方 test；MedMCQA 文件真实为 4,183 条
  official dev。二者均冻结为评测，不进入 SFT、Search100 或 RL train。
- MedMCQA dev 的 `cop` 为 1-4；`exp` 有 1,977 条为空，`topic_name` 有 3,760 条为空。
  因此解释和 topic 都不能设为必需字段，也不能为缺失解释伪造真值。
- 第一版统一字段、target 规则、loss mask 和隔离表已合并进
  `MedSearch/PROJECT_BOOK.md`。本次未下载、转换或训练数据，没有实验指标。

## 2026-08-21：MedQA/MedMCQA 官方 train 固定镜像下载与字段打印

- 固定 `openlifescienceai/medqa@153e61cdd129eb79d3c27f82cdf3bc5e018c11b0` 与
  `openlifescienceai/medmcqa@91c6572c454088bf71b679ad90aa8dffcd0d5868`；直连
  Hugging Face 超时后，通过镜像站下载相同仓库、revision 和路径。
- MedQA 文件 5,679,350 bytes、SHA-256
  `d7bac59f5cee804a006172d44f675559dc5160bdc84dbae05aeed9d5aa1b56f3`，PyArrow 读取
  10,178 行；字段为 `id/data/subject_name`，答案字段嵌套在 `data`。
- MedMCQA 文件 85,899,025 bytes、SHA-256
  `b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5`，PyArrow 读取
  182,822 行；Parquet 已把 `cop` 转成 0-3，因此映射是 `"ABCD"[cop]`。
- 两个数据集各 10 条完整字段已检查；一次性打印样本现已清理。MedQA 有 2 个
  精确重复题干且抽样存在缺失图像引用；MedMCQA 有 21,953 条空解释，抽样可见 OCR、
  页码和文本粘连噪声。
- `MedSearch/data/raw/` 已加入局部 `.gitignore`，原始外部数据不会被 `git add -A`
  提交。本次没有转换训练数据或启动训练，没有实验指标。

## 2026-08-20：最新版 veRL/FSDP2 的 3B GRPO 保底架构

- 固定官方 veRL commit `e78efedbe56591cd53ac107150abb7565bb9b8b1`，新增独立
  `Search-R1/modern_verl/` 适配包；Search-R1 XML action、HTTP retrieve、环境 token
  loss mask 和开放域 EM reward 均接入当前 `AgentLoopBase` / custom reward 接口。
- 单卡使用 Qwen2.5-3B-Instruct、LoRA rank `16`、FSDP2、异步 vLLM `0.27.1`、同题
  rollout `2`；没有下载或回退到 1.5B。
- WSL2/Docker 的 vLLM spawn worker 无法重建 CUDA IPC handle；保留 veRL 原有语义，
  通过可复现补丁启用其 CPU shared-memory weight transport。实际日志确认进入 fallback。
- 最终复跑退出码 `0`，完成 `training/global_step:1`：两条轨迹均为 4 turns，step
  `21.549 s`，生成 `13.102 s`，actor update `4.059 s`，权重同步 `3.271 s`。
- 外部 `nvidia-smi` 共采样 `5,420` 点，峰值 `23,117 MiB / 32,607 MiB`；actor 日志
  内部峰值 allocated `14.031 GiB`、reserved `14.789 GiB`。
- 最终 W&B run 为 `lx30qqpq`。两条 rollout 的 EM reward 都是 `0`，因此组内 advantage、
  policy loss、grad norm 都为 `0`；这证明完整训练路径可执行，不证明模型发生有效学习。
- 当前最新版 veRL 的 V1 trainer 还要求额外 TransferQueue 依赖；保底脚本明确使用
  `trainer.use_v1=False` 的稳定调度，但 actor/rollout/AgentLoop/reward 均采用当前接口。
- `data.dataloader_num_workers=0` 已消除 smoke 退出时 DataLoader worker 被杀的噪声；
  上游 W&B `Tracking.__del__` 仍可能在 Ray teardown 后打印 closed-transport traceback，
  发生于 step 和 final validation 之后且进程退出码为 `0`，记录为非训练路径清理告警。

## 2026-08-20：vLLM 单卡 hybrid-engine 权重交换修复

- 根因：旧 sharding manager 在 GPU 上物化 FSDP `state_dict` 后一次性唤醒 vLLM 的
  weights 与 KV cache，导致 Actor、交接张量、vLLM 权重在同一卡上重叠并 OOM。
- 修复后的时序：CPU state-dict → Actor 参数移出 GPU → vLLM 只唤醒 weights → 写入
  最新 Actor 权重 → 唤醒 KV → 生成 → vLLM level-2 sleep → Actor 参数恢复到 GPU。
- 真实 Ray/vLLM 运行连续完成两次交换和两次生成，退出码 `0`；不再出现
  `cumem_allocator.cpp:211`。
- `nvidia-smi` 每 `0.5s` 采样 `777` 点，峰值为 `32058 MiB`；原始记录位于
  `Search-R1/experiments/single_gpu_smoke/vllm_gpu_memory.csv`。
- 权重交换与已有兼容回归共 `10/10` 通过。该方案以 CPU 传输延迟换取单卡显存可行性，
  不声称具备多卡 NCCL 方案的吞吐。

## 2026-08-20：固定双跳样本无训练 Agent 闭环

- 固定子集：train 为 HotpotQA 原始行 `86094`，test 为 2WikiMultihopQA 原始行
  `41745`；随机种子 `20260819`。
- 使用本地 `Qwen2.5-3B-Instruct`、仓库原生 HF rollout、`max_turns=2`、BM25
  top-k `3` 完成无参数更新验证，进程退出码 `0`。
- 真实轨迹依次检索电影导演与 `Jeff Burr birthplace`，最终输出
  `<answer>Aurora, Ohio</answer>`，规则 EM reward 为 `1.0`。
- `ACTIVE_TRAJ_NUM: [1, 1, 1, 0]` 表明轨迹在最终回答后结束；完整证据保存在
  `Search-R1/experiments/single_gpu_smoke/inference.log`。
- 两次 observation 分别从 `471`、`514` tokens 截到 `256`；本题仍答对，但后续实验
  必须记录该截断，不能把一次成功外推为检索信息总是充分。
- 本次未启动外部显存采样器，因此 HF 运行的显存峰值记为“未知”，不以失败运行或
  纸面估算补填。

## 2026-08-20：官方 wiki-18 BM25 检索服务闸门

- 下载并校验官方 `PeterJinGo/wiki-18-bm25-index`：28 个索引文件，合计
  `2,297,611,339` bytes。
- 下载并校验官方 `PeterJinGo/wiki-18-corpus`：压缩文件 `5,123,307,260` bytes，
  SHA-256 `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`
  与 Hugging Face LFS oid 一致，`gzip -t` 通过。
- 发现上游文件虽命名为 `wiki-18.jsonl.gz`，实际是 tar.gz；正确提取出的
  `wiki_dump.jsonl` 为 `14,393,573,105` bytes、`21,015,324` 篇文档，字段为
  `{id, contents}`。
- 修复全局升级后的实际兼容问题：Pyserini 会连带导入 FAISS，旧 `faiss-cpu 1.8.0`
  与 NumPy `2.3.5` ABI 不兼容；已升级到 `faiss-cpu 1.15.0`。
- 作者原有 CPU BM25 server 已启动；真实 `POST /retrieve` 返回 HTTP 200、3 条带正文
  和分数的结果。请求与响应保存在 `Search-R1/experiments/single_gpu_smoke/`。
- 一次混用 Xet 分块缓存与普通 Range 续传产生了哈希不匹配文件，已隔离为
  `.corrupt-90124a`，未解压、未用于服务；待正确资产和服务验收后清理。

## 2026-08-20：vLLM 0.27.1 兼容迁移与训练资产下载

- 保持 Search-R1/veRL 上层同步调用方式不变，新增版本专用兼容层，将旧的
  `generate()`、`sync_model_weights()`、sleep/wake 和 cache 方法映射到 vLLM `0.27.1`。
- 该兼容层明确限定单卡 `TP=1`；由于 WSL2 不支持当前 vLLM V2 runner 所需的 CUDA
  UVA，并且跨进程同卡 CUDA IPC/NCCL 路径不可用，改用 vLLM 官方 public `LLM` 的
  单进程 V1 runner。未修改 Search-R1 的 trainer、rollout 或 GRPO 算法调用方式。
- 兼容层单元测试 `2/2` 通过，Python `compileall` 与 `git diff --check` 通过。
- RTX 5090 真实 smoke 通过并以退出码 `0` 输出 `SMOKE_OK`：初始生成是
  `Paris. The capital of Germany is Berlin`；将内存 actor 的末层 norm 权重置零并同步后，
  输出变为 `!!!!`，证明 actor 权重确实进入 vLLM，而非只验证导入。
- `RotaryEmbedding: Failed to load weights` 已定位为 vLLM 自建的
  `persistent=False` cos/sin 位置缓存不属于 HF `state_dict`；reload 代码保留原缓存，
  不是 actor 参数漏同步。
- Qwen2.5-3B-Instruct 两个 safetensors 分片已完整下载：`3,968,658,944` 与
  `2,203,268,048` bytes；模型位于工作区忽略目录 `models/Qwen2.5-3B-Instruct/`。
- 官方 `PeterJinGo/nq_hotpotqa_train` 数据已完整下载：train `169,615` 行、
  `355,663,891` bytes；test `51,713` 行、`70,370,337` bytes。文件位于
  `Search-R1/data/nq_hotpotqa_train/`，后续 smoke 会从全量文件抽取小子集。
- 没有执行仓库 `scripts/download.sh`：脚本硬编码作者目录、参数名与当前 Python CLI
  不一致，而且默认下载约 64.6 GB Dense Flat 索引；这与当前固定的 CPU BM25 路线不符。

## 2026-08-19：按用户决定直接升级 Docker 全局训练环境

- 容器 `llm-pytorch` 已直接升级为 PyTorch `2.13.0+cu130`、vLLM `0.27.1`、
  Transformers `5.15.0`、Ray `2.57.0`、W&B `0.28.2`、TensorDict `0.14.0`。
- NVIDIA driver `591.74` 声明支持 CUDA 13.1；真实 CUDA tensor 测试通过：RTX 5090、
  compute capability `(12, 0)`、`torch.ones(..., device='cuda') == 1.0`。
- Search-R1 已通过 editable/no-deps 安装；`verl.__file__` 指向工作区
  `/workspace/code_list/some tricks/LLMLeanring/Search-R1/verl/__init__.py`，确认使用本地源码。
- `RayPPOTrainer` 与 `RewardManager` 能导入；`ActorRolloutRefWorker` 在本地
  `verl/third_party/vllm/__init__.py` 主动拒绝 vLLM `0.27.1`：该兼容层只允许
  `0.3.1/0.4.2/0.5.4/0.6.3`。不得只删除版本检查后假定 API 兼容。
- vLLM `0.27.1` 要求 PyTorch `2.13.0`，但其元数据仍安装 TorchAudio `2.11.0`
  CUDA 12.8 build；因 Search-R1 是纯文本任务且二进制不兼容，已卸载 TorchAudio。
- 直接全局升级已如预期破坏旧项目依赖闭包：MedicalGraphRAG、sentence-transformers、
  LangChain 等要求旧 Transformers/NumPy/AnyIO/Packaging。当前阶段接受该代价，不回滚。
- 训练数据没有下载：`Search-R1/data` 中不存在 Parquet。wiki-18 只有约 2.08 GB 的
  `part_aa.incomplete`，而该分片完整大小约 42.9 GB，明显不是“只差一点”；不再续传
  64.6 GB Dense Flat 索引，训练 smoke 只准备少量 NQ 样本。

## 2026-08-19：Docker 与 Search-R1 运行环境审计

- 复用容器：`llm-pytorch`（容器 ID 前缀 `c8f4518d842b`），Linux/WSL2，项目在容器内
  映射为 `/workspace/code_list/some tricks/LLMLeanring/Search-R1`。
- GPU 透传正常：RTX 5090，32 GB；PyTorch `2.11.0+cu128`、CUDA 12.8、BF16 可用；
  根文件系统约 1007 GB，检查时约 912 GB 可用。
- 已有 `transformers 4.44.2`、`datasets 2.16.1`、`pyserini 0.22.1`、
  `faiss-cpu 1.8.0`；缺少 `vllm`、`ray`、`hydra-core`、`omegaconf`、`wandb`。
- 当前 Python 为 `/usr/bin/python` 3.12，并受 PEP 668 保护；直接全局 `pip install`
  被拒绝。后续必须在容器内创建隔离环境，不能使用 `--break-system-packages` 污染 MedRAG。
- 官方旧环境固定 Python 3.9、PyTorch 2.4/CUDA 12.1、vLLM 0.6.3；vLLM 0.6.3
  早于 RTX 5090/SM120 支持，不能在当前硬件上原样运行。不得直接执行旧 requirements。
- `vllm==0.9.0` 是首批可支持 RTX 5090 的版本，但依赖 dry-run 显示它会固定
  `torch==2.7.0`、`transformers>=4.51.1`、`ray>=2.43.0` 等，与容器当前全局环境及
  vendored veRL 的旧接口均不相同；只能放入隔离环境并先做兼容性闸门测试。
- Search-R1 官方的新版 veRL/SGLang 集成仍可作为参考，但当前 veRL 文档注明内置
  `SearchTool` 已移除；不能直接追随 latest，必须固定到仍包含该工具的官方快照。
- 官方 Dense Flat 索引为 64.6 GB，语料压缩包约 5.12 GB；当前 `data/wiki-18`
  只有约 2 GB 的未完成下载，没有可用索引或解压语料。官方 BM25 索引约 2.3 GB，
  更适合作为单卡训练时的独立 CPU 检索服务。
- `Qwen/Qwen2.5-3B-Instruct`、NQ/HotpotQA Parquet、W&B 登录信息均尚未就绪；
  当前没有产生训练 step、W&B run 或实验指标。

## 2026-08-19：Search-R1 源码阅读边界收束

- `fsdp_workers.py` 确认为 veRL 执行基础设施，不继续阅读 device mesh、FSDP、offload、
  vLLM 构建或 Ray RPC 细节。
- 当前掌握标准固定为：知道 actor+rollout worker 负责生成/更新，独立 reference 提供
  KL 对照，GRPO 不创建 critic；能够使用这些接口即可。
- Search-R1 特有的 data、Agent loop、reward、GRPO advantage、actor loss 与 `fit()`
  调度已经形成完整理解链，源码阅读阶段在此收束。
- 后续仅在真实运行出现明确错误时，沿堆栈按需查看 veRL 内部。

## 2026-08-19：路线剪枝——跳过 `_validate()` 深入阅读

- `_validate()` 使用独立 `val_files`；只有配置 `val_data_num` 时才从验证文件内部抽样，
  不是从当前训练 batch 临时切出测试样本。
- 它只生成验证轨迹、计算平均规则 reward 并写日志，不计算 advantage 或更新 actor；
  当前阶段不把它作为 Search-R1 核心机制继续阅读。
- 已保留一个运行阶段待核实问题：多轮路径创建 active 子批次时丢失 meta_info，静态代码
  显示 `do_sample/recompute_log_prob` 会回退到默认 `True`。
- 下一学习点直接进入 actor/rollout/reference worker 的角色与初始化边界。

## 2026-08-19：`RayPPOTrainer.fit()` 完整调度链阅读验收

- 已完整追踪一批题目从 `DataProto`、`n_agent` rollout、old/reference log-prob、
  rule reward、GRPO advantage、information loss mask 到 actor update 的执行顺序。
- 已能解释 `batch.union(final_gen_batch_output)` 是题目 metadata 与完整轨迹 tensor 的
  汇合点；删除后最先在 `_balance_batch()` 读取 `attention_mask` 时失败。
- 已能说明后续 `RewardManager` 缺少 prompts/responses/mask，reward 无法产生，继而
  advantage、ratio、loss 和反向更新均无法执行。
- 已进一步区分直接由 rollout/union 带回的字段与后续派生字段：完整轨迹、mask、
  `old_log_probs` 是直接输入；`ref_log_prob`、scores/rewards、advantages/returns 和
  `loss_mask` 是合并后逐阶段计算得到。
- 本次为源码阅读验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-18：Search-R1 NQ 数据 → batch → uid 阅读验收

- 已读完 `nq_search.py`、`rl_dataset.py`、`ray_trainer.py`、`main_ppo.py`、`qa_em.py` 的指定代码段；已掌握的 reward/GRPO 重复段未重复阅读。
- 已能列出 Parquet 五个顶层字段，并解释 prompt 与 gold answer 隔离是字段协议与防止答案泄漏的共同要求。
- 已纠正 tensor shape：`__getitem__()` 返回单样本 `[max_prompt_length]`，`collate_fn()` 后为 `[batch_size, max_prompt_length]`，再按 `n_agent` 扩展 batch 维。
- 已理解 `n_agent` 在 rollout 前扩展轨迹，而 uid 不参与生成，只在后续 `compute_advantage()` 中提供分组标签。
- 已定位真正的分组代码：`compute_grpo_outcome_advantage()` 通过
  `id2score[index[i]].append(scores[i])` 按 uid 收集同题 rollout，并计算组内均值、标准差和 advantage。
- 已理解 `_read_files_and_tokenize()` 当前不 tokenize、`filter_prompts=True` 当前不生效，以及 `drop_last=True` 只丢弃不完整尾批次。
- 本次是源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-16：GRPO advantage → actor policy loss 阅读验收

- 已能解释 `old_log_prob` 来自 rollout 行为策略，`log_prob` 来自更新中的当前 actor；`ratio` 是逐 token 的新旧策略概率比，不是整条 trajectory 的单一偏好分数。
- 已理解正/负 advantage 分别提高/降低已采样 token 的概率；PPO clip 裁剪 surrogate objective 的有利更新方向，不是把模型真实概率硬锁在区间内。
- 已理解 information 是外部检索 observation：参与 attention 以指导后续生成，但通过 `loss_mask` 排除在 policy loss 之外。
- 已区分 entropy bonus、reference-policy KL 与 `ppo_kl` 监控量：entropy 鼓励分布保持探索性；reference KL 限制当前 actor 偏离冻结的 SFT/reference 模型；`ppo_kl` 监控 current actor 与 rollout old actor 的变化。
- 已通过最小 entropy 判断：分布越均匀 entropy 越高，越集中 entropy 越低。该项是 veRL PPO actor 原有的可选正则，官方快照默认 `entropy_coeff=0.001`，不是本项目新增算法。
- 本次只完成代码阅读和概念验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-14：Search-R1 reward → GRPO advantage 阅读验收

- 已能解释序列级 0/1 outcome reward 写在最后一个有效 response token，是因为结果只能在完整轨迹结束后判断，同时需适配 veRL 的 token-level reward 接口并避开 padding。
- 已区分 PPO/GAE 与 GRPO：GAE 使用 token value 从后向前传播 advantage；GRPO 不使用 value critic，而是对同题多条 rollout 的序列分数做组内标准化，再广播到有效 response token。
- 已正确判断全 1 或全 0 rollout group 的 advantage 为 0；进一步明确这表示该组没有策略梯度信号，是有效样本比例问题，不等于整个 GRPO 训练必然坍塌。
- 本次为源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-13：Search-R1 reward 学习注释准备

- 已为 5 个 reward 主链函数补充输入类型、输出类型、Tensor shape、上游调用位置和下游数据流。
- 验证命令：

```cmd
python Search-R1\scripts\verify_source_notes.py
python -m compileall -q Search-R1\verl\trainer\main_ppo.py Search-R1\verl\utils\reward_score\qa_em.py Search-R1\verl\trainer\ppo\ray_trainer.py
git diff --check
```

- 验证结果：8 个 Python 文件和 1 个 Bash 文件保持 comment-only；3 个 reward + 3 个 action parser 样例通过；Python 编译通过。
- 边界：注释未改变执行逻辑；reward 代码复述已于 2026-08-14 验收，但没有运行训练或产生训练指标。

## 2026-08-13：Search-R1 多轮 rollout 主干阅读

- 已理解 `active_mask` 控制仅对未结束样本生成；search 注入 `<information>` 后继续，answer 标记结束，非法动作按当前 parser 路径处理。
- 已区分 `rollings`（下一轮输入）与 `original_right_side`（最终训练轨迹）。
- 已理解 `max_turns` 后仍有一次 `do_search=False` 的 final rollout，以及 `attention_mask` 与用于排除环境 information token 的 `info_mask`。
- 本次只做源码阅读和协议检查，未运行模型、检索服务或训练。

## 2026-08-12：Search-R1 主干源码学习注释

- 已注释数据处理、dataset、generation、tensor helper、retrieval server、reward、训练入口和 trainer 主链。
- 已建立 `Search-R1/docs/source_code_learning_zh.md` 和标准库验证脚本 `Search-R1/scripts/verify_source_notes.py`。
- 官方 8 GPU 配置不能直接用于单张 RTX 5090；单卡配置必须以后通过真实 smoke 确认。

## MedicalGPT SFT：已验证范围

- 已跑通 Qwen2.5-3B、BF16 LoRA 的 100 条 smoke：rank `8`、alpha `16`、batch `2`、gradient accumulation `4`、max length `1024`、1 epoch、12 steps、seed `20260809`。
- 真实结果：train loss `2.332629`、eval loss `2.008307`、perplexity `7.450692`、runtime `13.9074s`、整卡显存峰值 `17,113 MiB`。
- 结论边界：只证明训练链和 adapter 加载有效；3 条 validation 中 2 条输出变化，不代表医学能力提升。

## Medical GraphRAG / LinearRAG：冻结结论

- 已实现 BM25、Dense、RRF Hybrid、Graph、Qwen3-Reranker Hybrid2 的统一检索入口和四套真实基准评测。
- `MedicalGraphRAG/` 固定在 README 结果对应的 commit `fe89a64`；未验收扩展仅保留在 Git 历史。
- 图检索在四套基准均未胜出。该负结果保留，不再通过无依据调参追逐指标。

## 2026-09-05：RL 数据资产收束

- 后续 RL 只保留 `MedSearch/data/rl/train.parquet`（1,494 条）与
  `MedSearch/data/rl/dev.parquet`（197 条），两者保持独立、不合并，且 ID 不重叠。
- 已删除 `medical_verifiable`、`medical_verifiable_v2`、
  `medical_verifiable_quality_v1` 历史候选目录及其 JSON/JSONL 审核材料；清洗依据、
  来源 revision、隔离规则、人工裁决摘要、文件哈希和使用边界收束到
  `MedSearch/data/rl/MEDICAL_VERIFIABLE_RL_CLEANING.md`。
- 训练只读 train；dev 只作开发期诊断，最终效果仍由冻结的 MIRAGE、MedQA、MedMCQA
  测试集报告。因审核原始记录按整理目标已删除，后续不能重建原人工审核过程。

# 2026-10-07 new06 实际落盘与恢复

- 本轮正式新增16条，853→869；54个真实episode已全部返回，最多两次预算不重置。115题C3全部保存：39E3/26E2/38E1/12E0；26条RL候选暂未训练ready。
- 通过项5次即时导出并真实loader/mask、数量/data hash/manifest hash回读，完整通过未发布0。当前待S2 17、待独立S3 6；准确当前episode材料仅3包，不重审旧首轮已拒轨迹。
- 协作工具thread limit与旧已结束子代pending_init阻塞独立审查；root与调度子代有限spawn/followup/interrupt均未释放。没有NVML报错；不由已接触C3/gold的协调者替代S2盲审，不冒称目标完成。
- 恢复文件：`MedSearch/data/cleaning_v2/sft_20261007_new06/continuation_checkpoint.json`、`RUN_REPORT.md`、`review_tool_blocker_receipt.json`；完整CMD恢复步骤：`MedSearch/data/cleaning_v2/sft_20261007_new06/RECOVERY.md`。释放审查代理slot后先完成现有17S2/6S3并即时发布，适用门0后new06收口，随后自主new07至正式1000；不要恢复旧五批cap/逐批暂停。当前无活动教师，new07未冻结候选。

# 遇到的问题

| 问题 | 当前处理 |
|---|---|
| Search-R1 官方配置面向 8 GPU | 单卡 smoke 使用 3B、小 batch、2 条 rollout、短序列；参数只能通过真实运行确认。 |
| vendored Search-R1 兼容层只接受 vLLM ≤0.6.3 | 已新增 vLLM 0.27.1、单卡 TP=1 兼容层，并通过 CPU 交接与分阶段 wake 修复完整 Ray 路径的瞬时 OOM；真实两轮生成退出码为 `0`。 |
| 全局升级造成旧 MedRAG/GraphRAG 依赖冲突 | 用户明确选择不隔离；保留 `pip check` 结果，当前不回滚。 |
| 单卡 32GB 的 GRPO 资源边界未知 | 不做纸面外推；用真实日志记录 batch、rollout、长度、显存和耗时。 |
| 官方检索 observation 超过 `max_obs_length=256` | 当前按原逻辑截断；固定双跳样本仍得到 reward `1.0`，后续只把长度作为实验配置记录，不研究 retriever 内部。 |
| 多轮验证没有传递 `do_sample=False/recompute_log_prob=False` | 静态路径显示 active 子批次丢失 meta_info；当前先记录，不修改行为，运行阶段再实测。 |
| WSL2/Docker 跨 vLLM spawn 进程的 CUDA IPC 报 `invalid resource handle` | 为当前 veRL 增加环境变量控制的 shared-memory fallback 补丁；默认 native Linux 行为不变。 |
| smoke 的轨迹数小于默认 8 个 AgentLoop workers | 固定 `agent.num_workers=2`，与两条 rollout 等分。 |
| 两条 smoke trajectory 的 reward 都为 0 | advantage/loss/grad_norm 均为 0；保留为 GRPO 同组无差异信号的真实案例，不包装成训练提升。 |

# 下一步

- 唯一核心任务：把已完成的批次候选送过独立复核和全池隔离门槛；当前优先完成 batch6 的 99 条首轮 accept 独立二审，batch7 的 268 条队列随后盲审。
- 完成标准：batch6 的 99 个 ID 与原轨迹一一对应、盲包不暴露首轮结论或 verifier 分数；batch7 结构清单覆盖 316/316 且输入/输出哈希可复验，随后每条由人工按可见证据、动作、答案等价审查；全池 exact/semantic 去重、历史暴露边界与 train/dev/test 检查有可回读 manifest。
- 当前状态：batch6 首轮复核 256/256（accept 99、hold 79、reject 78）；其 99 条盲包已交由独立审查者复核（逐条阅读 99/99，locator/hash 校验收尾），training_ready=false。reserve 8 条独立复核全为 hold。batch7 rollout 为 316/316，结构队列 268 条、排除 48 条，268 条盲审副本已就绪；verifier 标签为 100 accepted、216 rejected，这些标签不计 SFT 准入。已有 393 条跨批二审确认、精确去重后 393 unique；3 个语义近邻待人工裁定；batch8 又生成 1,000 个仅供 provisional 盲审的候选。仍须完成历史暴露核验和 train/dev/test 隔离，不能直接计为最终样本。
- 随后完成 batch7 轨迹人工审核，统一汇总来源重复关系簇和历史暴露边界，再冻结最终 split。若通过审查的不同样本不足 1,000 条，从 HuatuoGPT-o1 精确隔离 reserve 继续抽样、清洗与生成。

## RL 后续待办（SFT 数据目标完成后恢复）

- 建立 `MedSearch/.deps/verl` 干净依赖，锁定 commit、Python、PyTorch、CUDA、vLLM、veRL 与 `transfer_queue` 版本，并验证导入来源。
- 不接 MedSearch 代码，先跑通新版 veRL 官方 V1 单轮最小链路，确认单卡 `SimpleStorage`、rollout 与 actor 可工作。
- 门槛通过后，从协议解析失败案例开始，由用户逐步编写 `protocol.py`；随后依次完成 BM25 客户端、AgentLoop、最小 reward、Checkpoint-B rollout-only 和 1-step GRPO。
# 待补知识

- 新版 veRL V1 在单卡下对 `transfer_queue`、`SimpleStorage` 和 worker 生命周期的实际要求。
- Checkpoint-B LoRA 是否同时正确加载到 actor 与 rollout，以及 step 0 输出的一致性。
- 使用 LoRA 训练时冻结 reference policy 的真实组成；不能确认 reference 为 Base + Checkpoint-B 时先关闭 KL。
- `train_batch_size`、rollout `n`、PPO mini/micro batch 与梯度累积各自控制的维度和单卡实测边界。
- medical verifier 的阈值、人工一致性、显存驻留方式和同组零方差 reward 比例。

# 实验结果总表

| 数据集 / 实验 | 当前最优或关键结果 | 结论边界 |
|---|---|---|
| PubMedQA 封闭基准 | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 documents，不代表全量 PubMed。 |
| NFCorpus | Hybrid2 nDCG@10 `0.384` | 多相关 qrels；Graph 未胜出。 |
| SciFact | Hybrid2：R@10 `0.895`、MRR `0.740`、nDCG `0.772` | Graph 未胜出。 |
| HotpotQA | Hybrid2：R@10 `0.898`、MRR `0.955`、nDCG `0.865` | 每题 2 个 gold；Graph 未胜出。 |
| MedicalGPT 100 条 smoke | eval loss `2.008307`、PPL `7.450692` | 只验证训练管线。 |
| Search-R1 3B GRPO 单步 | step `21.549 s`；GPU peak `23,117 MiB`；W&B `lx30qqpq` | 两条 reward 均为 0，验证训练链路，不代表学习提升。 |

# 失败案例

| 场景 | 原因 | 保留结论 |
|---|---|---|
| PubMedQA 上 RRF Hybrid 低于 Dense | BM25 高位错误稀释 Dense 信号 | 融合不是默认更好。 |
| Graph 在四套基准均未胜出 | 实体传播信号与任务/语料不匹配 | 保留负结果，不包装为提升。 |
| TREC-COVID 未接入 | 24.6% 文档正文为空，部分 qrels 指向空文本 | 不用标题回退伪造正文检索。 |
| MedicalGPT 内部显存统计路径停滞 | Trainer 路径未形成完整产物 | 该次运行不计入；已用外部只读采样完成 smoke。 |
| Search-R1 单步两条 rollout 均答错 | 同题组 reward 全 0，GRPO advantage 与 gradient 为 0 | 保留为“同组无差异则无学习信号”的真实失败样本；扩大实验时监控有效组比例。 |

## 2026-09-09：RL 扩充数据 V1 重做

- 核对确认 `MedSearch/data/rl/medical_verifiable_expansion_v1/` 原先只有 2,000 条自动初筛候选，
  状态为 pending medical second audit，没有 parquet，不能直接用于 RL。
- 保留原候选并以固定 seed `20260902` 发布新候选：train 1,800 条、dev 200 条；随后删除 12 条
  缺失图表/波形依赖题，最终生成
  `train.parquet`、`dev.parquet`、`manifest.json`、`medical_review_queue.jsonl` 和
  `SHA256SUMS.txt`。
- 容器内回读验证通过：veRL 八列 schema、split、ID/规范化题干去重均通过；最终保留 train
  1,789 条、dev 199 条，与当前正式 train 1,494 + dev 197 无规范化题干重叠；Qwen2.5-3B
  tokenizer 为 116–297 tokens。
- 答案沿用官方 Ground-True Answer，没有自行改写；本轮文本-only 审查完成，可进入 RL，
  但不是执业医生的独立临床标注，RL 结果不得解释为临床能力。

## 2026-09-09：扩充数据合并

- 已将审查后的 expansion train 1,789 条、dev 199 条合并到正式 RL 数据；当前正式数据为
  train 3,283 条、dev 396 条，总计 3,679 条。
- 合并前 1,494/197 parquet 已备份到 `MedSearch/data/rl/pre_expansion_backup_20260909/`，
  合并校验和记录在 `MedSearch/data/rl/MERGE_MANIFEST_20260909.json`。
- 合并前不再继续挑选几千条：先用 20、50、200 steps 逐级观察 reward 方差、有效组比例、
  dev-32 和显存/延迟；只有发现题目覆盖或有效组比例不足，才从剩余源池按同一规则扩充。

> 更早的逐日过程、已撤回方案和详细实验配置由 Git 历史、各子项目 README 与 `experiments/` 保留；本文件只维护当前有效状态。


## 2026-09-10：V4修复后42步效果审计

- 审计全部42步：2688条导出中剔除1961条padding，727条有效轨迹，奖励命中66/727=9.08%；旧V2前42步为257/2688=9.56%，筛选与题目不同，不作因果比较。
- 非法动作提示率3.44%，强制作答提示率4.26%；格式交互改善，但不能等同搜索能力提升。
- 每步有效轨迹9–32条，导出同题组均不足8条；不能视为完整8×8实验。初始dev仅保存20/64条，命中3条；无训练后dev结果。
- 未启动训练、未排查终止报错。完整结论与口径：`MedSearch/docs/audits/2026-09-10-v4-42step-analysis.md`。

## 2026-09-11：dev 抽样比较 wiki 与 textbook BM25

- 从冻结 MedSearch/data/rl/dev.parquet 的 396 条题中，用 seed=20260911 固定抽取 20 条；两库均以原题做 BM25 topk=5，完整返回保存在 MedSearch/outputs/corpus_comparison/dev20_top5.json，口径与结论见同目录 README.md。
- textbook 语料约 125,847 个医学片段、18 本书；wiki 索引约 2,101.5 万篇。答案词元 Top-5 命中为 textbook 5/20、wiki 6/20，Top-1 任一答案词元命中为 textbook 8/20、wiki 11/20；该指标只作词面代理，不能宣称医学正确性。
- 快速人工审阅显示 textbook 在尿-阴道瘘、Wernicke 脑病、腹泻补液、食管下括约肌失弛缓等题上提供了更直接的医学关系；wiki 在胸锁乳突肌、上臀神经、血吸虫病等同名实体题上更集中。当前决定 textbook 继续作为 MedSearch 主库，wiki 作为独立 fallback/补充库，不直接混建单一 BM25 索引。


## 2026-09-16：MedAgent DSH 工程与实验文档同步

- 当前推进 MedAgent 工程设计：已统一项目书、逐文件函数规划、README、实验计划与状态清单；历史工程书保留在 `MedAgent/docs/history/`。
- 依据现有报告，DSH 单 Planner 假服务链路通过；本轮未重跑 smoke，未调用付费 API，未编写正式业务代码。
- 当前唯一编码任务：学生实现最小 RoleRunner，并通过实际假服务抓包检查 Planner/Verifier 上下文隔离；详细依赖和后续 P0～P6 顺序见 `MedAgent/ENGINEERING_PLAN_V1_V2.md`。
- V2 保持 200 题构建、每 10 题离线维护、最多 100 条活跃记忆；冻结快照后比较有/无记忆，数据实际数量与隔离仍待核验。
- 本次为文档更新，无准确率、费用或显存实测结果。

## 2026-09-16：新增 MedAgent 工程进度书

- 新建 `MedAgent/PROGRESS.md`，与设计书和实验状态分工：设计书记录计划，工程进度书只记录有代码、命令、日志或抓包证据的完成项，实验指标仍由 `refine-logs/EXPERIMENT_TRACKER.md` 维护。
- 已记录 P0 的固定 DSH 版本、本地 Node runtime、单 Planner 回复链路、配置安全边界及未验证项；README 已加入阅读入口。
- 根据固定版本源码补充 DSH 回复机制：Python SDK 以 stdio JSON-RPC 驱动本地 DSH 子进程，DSH 请求 DeepSeek chat-completions 并解析 SSE，SDK按 session 聚合事件，在 `session.status=idle` 后提取最终文本与 `turn/end`。
- 当前唯一任务不变：学生实现正式 `RoleRunner`，用实际抓包验收 Planner/Verifier 上下文隔离。
# 2026-10-02 §10.14 强制顺序接续检查

- 已重读清洗书 §10.14，并按“结果返回 → 保存账本 → 已通过项先做新增项发布检查并导出 → 回读确认 → 才启动下一任务”执行；未决项独立保存，不等待整批或其他审查者，不重审旧 release。
- 当前 SFT release 指针仍为 682 行，文件 SHA256 `44b1b9296d03b9fe759dd8e17a4bf423d91869ca0e39428175c462f5ed459dd1`。本次只读检查 batch02 的 200 行账本，没有发现已通过但未发布状态；其中已发布 SFT/RL 项不重复计数。
- 401–600 冻结切片已落盘并回读：200 行，SHA256 `712e6dcca7b435947f53c91e2baf3020d1e8ea75eb44a4f142de15ee25a2e058`；源文件 SHA256 `3aabe8b0493ae62ba57e4a00806adc9adbf2befc469f5e7072691a7b6cb5c435`。C1 exact 对照 1,069 条历史行，0 exact conflict、0 invalid question；manifest SHA256 `da80fcb23626a099f57bb87bea96ca848a2ede0626eb9adfa4322aad4b1afffa`。401–420 C1 semantic 已返回并落账：20/20 pass、近邻标记 0，threshold 0.92；semantic manifest SHA256 `f9fe2991bffb9392429adf396e3ed8cfc062e1de5f26d4e9900e2ff879486f49`，逐题 ledger SHA256 `cd5e812078ebba0db66da1900334859a9e931489ef9d68d6aeb2c0b4ec403ad3`。C3 尚未返回；语义通过不等于证据门通过，release 仍 682。
- 目前主 attempt ledger 回读 SHA256 `f6759eb7aff251219b3375fe9de8e5a28e1cc3188e747957d74b424af38b87b7`。Docker 默认受限 shell 的 config/pipe 错误与容器健康无关；授权 Docker 调用正常。已修正 `python -m` 入口和 batch01 C1 文件路径。当前只启动 401–420 的第一条 Q0，逐条保存和回读后再继续。
- 本次 SFT release 仍为 682 行，数量未变化。若后续新增项满足完整内容门，立即执行仅针对新增项的发布检查、导出并回读 ID/数量/hash，再继续下一项。
# 数据清洗接手增量（2026-10-03）

- 按用户节约额度要求细分职责：GPT-6-luna Max 承担代码、批量C3/S2、题干盲查询与确定性入账/真实loader/串行发布；主代理做关键答案等价判断、一次代码审阅和发布后核验；独立轻量Sol做新增项一次S4盲检。盲审上下文分别隔离；正式发布只有一个执行者。
- 接手核实 SFT 703 条。新增项逐条完成真实CPU loader、独立盲S4和发布回读，累计发布到 inc041–083；本检查点746条，接手新增43条，本次续接724条后新增22条。
- CURRENT_SFT_RELEASE.json 指向 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc083_TAKEOVER_40474/release/sft/accepted.jsonl`；SHA256 `6b833f0e444c249bb5ef45a58161f438cac6844191d507bfc044ffcd73ae9f94`。目标未满，还缺254条；没有启动训练。
- batch04 35840、35841、36127 各两次完整episode已耗尽，最后终答与原gold不等价，保存失败原件并进入RL候选，不使用失败轨迹监督。其RL R0–R3仍须完成，未称RL-ready。
- 命令与日志在 `MedSearch/data/cleaning_v2/sft_20261001_rollout_prep/batch04_601_620_20261003/COMMANDS.md` 和 `batch05_621_820_takeover/COMMANDS.md`。B5共200题、C1排除11条，189条Q0已返回；b05_004新增10条均已发布。B6共108题、C1排除5条，103条Q0证据审查已落盘；b06_002的11条原始轨迹中S3等价8条、S2通过7条，最终一次S4通过6条并已发布，39451的S4拒绝结果保留，39763的S2 hold保留。B5新的39条Q1盲查询已冻结，尚未执行检索；旧release不重审。
- B5 retry001已耗尽两次episode的18条失败项，按真实原包/receipt精确补齐审查元数据后封账，发布过的37210/37620未改动。`batch05_621_820_takeover/staging/rl_candidates_from_exhausted.jsonl`保存18条原题/gold/证据关联，无教师轨迹，SHA256 `1566171f32332eba9fa837d2940182f56652c68a5b2e5a0747bd9de9f067c3d5`；`training_ready=false`，R0–R3待执行，未称RL-ready。
- B6剩余11个Q0 E3题已生成b06_003第一轮轨迹：S3等价8条，S2通过7条，其中40054原始invalid_actions=1被S1机器门拒绝；其余6条真实loader和一次独立S4通过，已发布inc078–083。40564的S2拒绝与40091/40134/40528的S3不等价保留并绑定。发布脚本已修复单条非法episode阻塞整批的断言，非法行单独记录，合法行继续；合成回归通过。

## 2026-10-04：B5/Q2 接手收尾检查点

- `status=paused`：昨晚既有工作收尾后按用户要求暂停；不是 blocked 或 completed。本轮未启动新教师、新检索或新批次。
- 当前 `CURRENT_SFT_RELEASE.json` 指向 746 条 release，manifest SHA256 `12bbcbe20e5033edd79f1f91bd7501504e295cab18ff78aec1f376d198f38e35`；实际文件回读为 746 行，SHA256 `6b833f0e444c249bb5ef45a58161f438cac6844191d507bfc044ffcd73ae9f94`。本轮未新增 SFT 行、未启动训练。
- B5 `b05_005` 冻结输入为 `s0_inputs/b05_005.jsonl`，SHA256 `d05bb806051f9313fd32f3d0aae541a07324513f5db4ea94cbc647a31c369d31`。S1 已返回 3 条；S3 决策为 38912/38968 `different`、39014 `equivalent`，决策文件 SHA256 `79f70d5f25ccd2cfc3498e4520acdfdd5a5bb04d94292ebc2acd1b515f5716ec`，绑定 S3 输入 SHA256 `75e34466edaa52be3c63ee6c5b62518bf335e14ab8453bb5ec82a39b67b34668`。39014 的 S2 为 hold/E2：实际可见证据支持 WHO class IV，但不支持终答更具体的 IV-S(A) 亚型；S2 decision SHA256 `5b5908ad9bbf1b70ed2da4f0f69f7684e622780a735b94bd12d8e81df0960398`，已由 `record_reviews --prefix b05_005` 入账，ledger SHA256 `8d6c29e744f1afd1d929ebf36a7f3e3f37bd0d52f74f7f258d08a21fd3509066`。本批 3 条均未放行、不派 S4、不发布；各保留一次既有 episode 预算，按当前暂停要求不执行。
- Q2 `b05_q2_001` 的 30/30 检索均已真实返回，`run_status.json` 全部为 `Q2_returned_pending_evidence_grade`。执行输入 SHA256 `d74f0fb6367441469a741cdae315a1756478315250e74275746f35eb64c89b0e`，execution manifest SHA256 `8c51939f6ac3dda9c116b78865294260f5e4d2a5a7f618f6610ba8b9f042c819`，attempt ledger SHA256 `0c38ff3964ef2bc1256db2f8dcd2a76ef750b0632b3b9f67265903da9b45568d`，`run_status.json` SHA256 `8b45917966627d947b56082bdcda58ade13bcbf8c91f26ef305c3af9c33f50c5`。原始响应 30 份、证据结果 30 份和逐题 summary 30 份均保存在 `batch05_621_820_takeover` 的 `c3_q2*` 目录。
- Q2 两片 C3 复核完成：01 输出经元数据修正后 SHA256 `b93236c73ba1809152e9b9adb4250c45d93ba8a84219b358334a594fa975d15c`，manifest SHA256 `833620619a75e53ad6e9caf066af1dc9ca920671c4e76c6a1602a3a103c69884`；02 输出 SHA256 `a1856875f02dbb26a362ae2a449bb4ac6936fc000eaffc37f4136a68449d0557`，manifest SHA256 `fc356841e0ce287f311918a060f50b1251ee649e94aa7f2bbfb222dc9e093663`。01 只更正 decision.evidence_path/evidence_sha256 两项确定性绑定元数据，原 decision/manifest 已备份，grade、rationale、citations 不变。现有 `record_c3` 无 freeze 入账 reviewed=189、binding_errors=0，ledger SHA256 `db22a77b926a8f2ea6f3bd1b7867b274e9f1fb29858a198106273b0267d041e2`。
- 既有 RL 候选仍未完成 R0–R3，不是 RL-ready；本次不晋级。B6 `freeze_retry.py` 只有草稿、验证未完成且未启用，本轮不运行。teacher、Q2 检索命令及返回记录见 `MedSearch/data/cleaning_v2/sft_20261001_rollout_prep/batch05_621_820_takeover/COMMANDS.md`、`b05_005_dispatch_attempt01.json`、Q2 execution manifest、attempt ledger、`run_status.json` 与逐题原始文件。
- Q2 30 条最终分流报告为 paused，pending=0：19 条 `quarantine_evidence_exhausted`，7 条 `rl_candidate`（R0–R3 pending、training_ready=false），4 条 `sft_candidate_paused`（未生成教师轨迹或标记为可恢复重试，training_ready=false）。候选文件 `MedSearch/data/cleaning_v2/sft_20261001_rollout_prep/batch05_621_820_takeover/staging/b05_q2_001_final_routes.jsonl` SHA256 `422e94f78db74f74a8187658dcf8d1cbcd7f5e661416d9023ccc5f33a9688719`；报告 `staging/b05_q2_001_final_closeout_report.json` SHA256 `aa5ca67882bc9a61eb7b4adf50ad509b00150873ecf19a1008383d1e52aed107`。没有教师调用、S2/S3变更、SFT发布或 RL-ready 声明；旧候选和最后一次 episode 预算留待恢复。
- 当前最终状态为 paused。本检查点未触发新模型或检索调用。


## 2026-10-04：按用户要求精简清洗书与模型分工

- 修改清洗书 §10.6/10.8/10.10–10.15 及顶部入口：6.1 Sol 轻度做简短调度和共用工具修复，Luna Max 做固定输入的批量语义审阅，脚本负责元数据/入账/发布。操作卡由约3,450字符缩至2,294字符，删除重复调度长提示。
- 取消发布前例行二次语义抽检（旧公式导致小增量全量再审）；S2 失败不再送 S3。保留逐题 S2、独立 S3、E3、盲输入、集合隔离及真实 token/mask/loader 验证。允许正常证据推理，不以文风或无关内容不详尽拒绝。
- 明确禁止失败行逐条握手、常驻 Sol 监督、Luna 临场排障或手写 hash/manifest。提前交付采用不可变结果片段，程序绑定元数据；只在批交付和具体异常时介入。
- 本次仅修改并回读文档，未调用教师/审查模型、未发布新增数据；上一核验正式 SFT 为746条，本次未重新计数。现有工具仍需由 Sol 在共用入口适配：元数据自动绑定、各门独立入账、不可变片段合并，以及取消旧语义抽检后的发布条件。不能把规则修改声称为上述代码已实现。新清洗前仅补这些具体接口，不重新医学审查旧数据。
- 成本结论：减少上下文与返工预期有益，但未实测节省额度；官方说明见 https://learn.chatgpt.com/docs/pricing 。取消例行二次复审减少独立检错机会，未声称漏筛率已得到验证。

## 2026-10-04：B5/Q2 四条 SFT 小批试跑

- 续接时正式 SFT release 为 746 条，距离 1,000 还差 254；release 文件 SHA256 `6b833f0e444c249bb5ef45a58161f438cac6844191d507bfc044ffcd73ae9f94`。本次冻结候选 `36557/36717/36867/37075`，均为 C3 E3、首次 S1 额度。
- S1 复用运行中的 `llm-pytorch` 与 SearchR1-Qwen2.5-7B-Instruct revision `74d46a43224daa612b5f27174e1c58b35587ee92`。四条轨迹均返回；verifier 仅作诊断，accepted=1、rejected=3。容器执行 111 秒；权重加载 51.75 秒，报告显存 14.29 GiB。无容器重启、服务预检或重复模型加载。
- 盲输入核对：policy 可见输入文件 `b05_q2_sft001_s0_visible_inputs.jsonl` 只含问题消息；S2 包 `b05_q2_sft001_s2_blind_input.jsonl` 不含 references/gold/verifier 字段；S3 使用独立包比较教师终答和来源参考。未发现 gold 泄漏到 policy 或 S2。
- S2 4/4 fail，等级 E0=1、E1=1、E2=2、E3=0；失败分别涉及无证据猜答、跨谱系关键关系未证、可见证据与解释直接冲突、药物机制关系没有落到教师终答。S3 equivalent=1、not_equivalent=3。只有 S3 答案等价的 `36557` 仍因 S2 无证据而失败。S4 未运行（无 S2/S3 全通过项），本批新增 SFT=0，未路由到 RL；四条均标记一次重试仍可用。
- S2 决策 SHA256 `c2abbaf080b60192878b8b3660751fce3ec7d5db871b069f60adaa33fb779fd7`；S3 决策 SHA256 `b9b6fa8c00b7896a19b09646ce80f6d90ec174e59aa11117fa133ebee3c88a1d`；逐题账本 SHA256 `d47554ce4fc9c218ad7edfc63edebbae836ab0f5b20cd5ef61d73af0993a8d46`。文件与账本回读一致，release 仍为 746 条、原 hash 不变。
- 执行观察（耗时口径更正）：本次续接到写完账本约 47 分钟；教师日志区间 111 秒。原记 S3 约 5 分钟、S2 约 14 分钟来自文件时间与调度记录，包含等待和协调，且两门并行，不能视为模型纯审阅耗时，也不能相加后从总时间扣除来归因。token 用量不可得。一次宿主 Windows Docker API/npipe 权限拒绝发生在进入容器前；按原命令重试后 exit 0，模型正常运行，不能归因为容器/GPU 故障。
- 执行复盘补充：主代理曾查看错误层级的账本而中断执行者，并给出与记录器不符的整包 hash 指令；又多次暂停 S2 审查者，等待失败行的 manifest/账本更新后才允许写下一行。这些协调失误不能归咎于 Luna 的语义判断能力，也不符合 §10.15.B 的批量返回要求。本轮四条均未通过 S2，不存在为立即发布合格项而拆分失败行的必要。当前样本仅支持“端到端执行效率不合格、工具和调度存在具体缺陷”的判断，不能估计审查准确率或纯模型延迟。用户要求先评估 Luna 是否适任；下文所列 attempt 2 暂不启动。
- 发现并修复的流程工具问题：复用审阅包生成器把含 `trajectory_id` 的 S1 回执字典与仅含 path/SHA 的字典整体比较，误触发 attempt 计数断言；现按 path+SHA 识别同一回执。该生成器还曾把简短 review-prep 摘要写到正式 S1 回执同名文件；依据已保存的 dispatch、轨迹和日志恢复正式回执，并将 prep 摘要改存独立文件。通用审阅记录器要求 S2 每题输入行 hash，而初始决策误填整包 hash；已纠正绑定、补 manifest、将决策写入逐题账本。决策文件每条返回后的首次账本更新有延迟，后续需让已完成项在返回后先记账再继续。
- 清洗原则按 §10.6 与 §10.12.C 执行，没有扩大历史排查或重审旧 release；但强制落盘的工程顺序未一次跑通，以上三处工具/接口问题是具体原因。未改清洗书。本批下一动作仅为：对相同四题各生成一次全新 episode（attempt 2、temperature 0.3、seed 43；教师只见冻结题干，不见 gold/S2/S3 失败结论），随后只审新轨迹；发布仍须 S2、S3 通过。
# 2026-10-04 Sol 轻度 + Luna Max 批量清洗（执行中）

## 2026-10-05 new04恢复清洗，端到端收口后暂停

## 2026-10-07 new05恢复清洗

- 最终完成：冻结200题所有最终去向保存回读，18正式SFT、14 E3两次后失败（10轨迹证据/推理不合格，3答案不等价，1动作非法/不完整），36 E2 RL候选待R0-R3，47证据不足不用，15隔离排除、59题目排除、11gold排除。真实教师51episode（32首轮+19末轮），每题最多2次。
- 首轮合格13即时发布；末轮分片10+9独立盲审后独立S3，合格3+2即时发布，835→853。全部18新增项真实loader/观测mask/有效监督token检查及数量hash回读通过，适用待办/完整通过未发布/可重试均0，无环境阻塞。最终报告 `MedSearch/data/cleaning_v2/sft_20261007_new05/RUN_REPORT.md`，逐题原因 `final_batch_outcomes.jsonl`，恢复 `continuation_checkpoint.json`（closed_at_five_batch_expansion_limit）；后续无需重审本批。
- 执行异常真实记录：一名C2子代理CMD调用参数误报文件不存在，明确workdir纠正；主代理在第四Q0未退出时提前打包，IndexError后等待原命令exit0，仅重跑打包成功。冻结前发现跨题引文/OCR断字由审查者修正；S4首次拦1条伪Doc，没有放行。未新增医学审核层、未审旧release、未创建定时任务或训练。所有实际命令见new05/COMMANDS.md，token/精确额度unknown。

- 唯一任务：接续正式835条，从新冻结200题推进共同门、E3真实7B轨迹与独立S2/S3，通过项即时发布，争取1000；受现有第五批预算约束，如不足则报告缺额、不降标准。上批待发布/可重试0。
- 新目录 `MedSearch/data/cleaning_v2/sft_20261007_new05`，源ID7879–8283，history6903，排除16份既有候选，精确隔离通过，semantic真实Docker任务及Q1四个50题包已派发。GPT-6 Luna Max只做隔离语义审查，主代理与脚本串行写账本/正式指针。
- 命令与恢复位置：new05/COMMANDS.md、continuation_checkpoint.json；当前没有新SFT发布，没有训练/定时任务。
- 接续中间状态：C1排除15题；Q1全200为137pass/39quarantine/24reject，交集126题C2全部完成115pass/5reject/6quarantine；115真实Q0/C3全部审完32E3/36E2/34E1/13E0，绑定错误0。E3原生教师首轮四包5/11/13/3；前16返回、S2独立盲审在进行，13native运行、最后3已冻结未生成，不得重复活动包。正式仍835，待发布0；36RL仅候选。本批最大可达867，先收口32E3、不突破第五批冻结范围。
- 首个实际增量：new05_sft02 11条S2 6pass/5fail，独立S3 6equivalent；真实loader均通过，S4伪Doc拦1，其余5合并inc115发布，835→840，hash回读通过。首5条S2 2pass/3fail，独立S3两题在审；第三13native已返回/S2在审；最后3native运行。9条已有最后重试预算；未发布完整合格项0。36RL仅候选，原生教师未见gold/诊断证据/审核结论。
- 首轮最终13发布：inc115五条、inc116两条、inc117五条、inc118一条；均完成真实loader/mask及数量hash回读。32首轮真实episode已审，19失败项（含S4伪Doc、S1动作/不完整、S2证据/推理与S3不等价）合并s0_inputs/new05_retry01.jsonl，最后attempt2原生生成运行中exec33517。待发布0，不重置预算，失败轨迹不作RL监督。第五批范围完成后交付实际缺额，本批理论最高867，不能声称即将凑满1000。

### new04历史闭合记录

- 最终完成：冻结200题最终去向全落盘；15条正式SFT，13条E3两次后失败（11条轨迹证据/推理不合格、1条答案不等价、1条动作非法或不完整）；36条E2 RL候选待R0-R3，28条证据不足不用，13条隔离排除、80条题目排除、15条gold排除。真实7B episode46条，单题最多2次，未扩池。
- 首轮发布10条，最后重试18题发布5条；增量均完成真实loader/观测loss mask/有效监督token检查，正式835数据SHA及manifest SHA回读一致。所有适用门待办、通过未发布、可重试数均0，无环境阻塞。最终文件与逐题原因见 `MedSearch/data/cleaning_v2/sft_20261005_new04/RUN_REPORT.md`、`final_batch_outcomes.jsonl`、`continuation_checkpoint.json`（paused_by_user_after_batch_closure）。本轮没有启动训练或定时任务。
- 执行异常按实际记录：子代理一次CMD读取参数错误纠正；一处引用OCR断字仅修格式、原结果保留；18题Luna Max盲审保存耗时偏长，冻结前引用核对拦下2题信息块外伪Doc。未把这些误判结果发布。精确任务token/额度unknown，不用共享额度冒算成本。命令见new04/COMMANDS.md。

- 用户明确恢复SFT清洗；正式基线820，距1000差180。上一批new03已完成，待审/待发布/可重试0，不重开。
- 新目录MedSearch/data/cleaning_v2/sft_20261005_new04，冻结200题，源ID7548–7877，排除15份既有候选，保护范围及0.92语义阈值保持。精确隔离通过，真实语义检索13对待判断。Q1四包各50题由两名Luna Max分包交付；新上下文不带父对话。程序负责账本/hash/发布，收到冻结交付确认后接收文件，不提前绑定尚在自检的raw。
- 本批终点为合格真实轨迹发布、失败/隔离封账后暂停；不扩本200题之外池，不创建定时任务，不启动训练。最新状态new04/continuation_checkpoint.json及CURRENT_SFT_RELEASE.json；成功命令见new04/COMMANDS.md。
- 接续中间状态：Q1全部200、C2适用107完成，92条实际Q0/C3全部入账（28E3、36E2、23E1、5E0），引用绑定错误0。首批new04_sft01的9条真实7B返回且S2在审，第二批new04_sft02的13条正在生成，最后6条已冻结new04_sft03待同一GPU前批退出后生成。正式仍820；36条RL仅候选，非训练ready。已有门结论直接复用，不重新审阅92题。
- 首轮28题全部完成：S2 14pass/14fail，2条S1动作失败跳过S3，独立S3实际12均equivalent；S4发现2条块外伪造Doc拒绝，实际发布10条（820→830）。真实loader/mask及数量hash回读通过。sft02有1处引用OCR断字抄录格式错误，仅修精确quote为fistulec-tomy并新回执绑定，原始raw保留，不重启语义审查；其他4条未被拖入复审。18题冻结最后一次attempt2，当前run_teacher运行中、无容器报错。命令和修复原因见new04/COMMANDS.md。

## 2026-10-05 new03已经完成并暂停（历史批次）

- 用户明确恢复下一批；正式基线799。本批目录MedSearch/data/cleaning_v2/sft_20261005_new03，冻结200题，源ID7158–7547，排除14份既有候选，继承保护范围。精确隔离通过，语义16对已审查并入账；Q1全部200题、C2适用128题完成，115题通过并已保存真实检索。
- 进行中检查点：C3已入账88/115（E3=28、E2=30、E1=15、E0=15）；首包4条原生7B轨迹完成独立S2/S3，2条通过并发布，正式799→801；数据SHA256 54515785df8cd3d4d237bee541c077602b71c687f7cddba1bf8137e5d827a731，发布inc106_TAKEOVER_BATCH_7370。另2条未通过仍有一次原题重试预算。第二包22条正在生成。RL只保存候选，不计训练ready。最新数以CURRENT_SFT_RELEASE.json及new03/continuation_checkpoint.json为准；本200题端到端收口后暂停。
- 首轮已完成：C3适用115题全部入账（E3=38、E2=33、E1=20、E0=24）；38题首轮真实7B轨迹全部返回、S2/S3收口，发布13条，正式812，inc109_TAKEOVER_BATCH_7350，数据hash ae8844cd730242fbcc03f75ec066c4b8728c0e5414d3b43c1489dc745b4ae8de。25条未准入题在new03_retry01执行最后一次episode，仍在进行中。S2 pass/E2的两条保留原结论，以既有E3门槛记S4 reject并进入重试；只修分流状态，无内容重审。33条RL只保存候选。首轮通过未发布0。
- **本批最终完成并暂停**：第二轮25条盲审完成，独立S3合格8条立即通过机器/真实loader/mask检查并发布。本200题最终去向：21正式SFT、33 RL候选（未训练ready）、44证据不足不用、13隔离排除、59题目排除、13 gold排除、17两次轨迹后未准入。17条具体原因：4动作非法/轨迹不完整、9证据或推理不合格、2轨迹未决、2答案不等价或未决。共63条真实episode，每题最多两次，适用待审/待发布/可重试均0。正式799→820，五次发布增量2+8+1+2+8，最终inc110_TAKEOVER_BATCH_7169；数据hash 4e6e9ab39b1112ba5a337b0f91c211b0af1ffdd17bd173fb86671ea1294cef37，manifest hash 8604c887d2902e9914880617ccf9d524602a38959375c1441b501f5c1a77a1f7。报告new03/RUN_REPORT.md，逐题去向final_batch_outcomes.jsonl（200条，hash a63c47cdf6bde947a7084494b71f75d39e521791ce8350827e7071806d65f7c7）；checkpoint状态paused_by_user_after_batch_closure。CMD子代理路径传参、S2 pass/E2分流和主代理提前接收未冻结raw的偏差已修复留档；没有重启医学审核、旧release审计、扩大冻结范围或第三次生成。无环境阻塞；33 RL候选仍需R0-R3。等待用户明确命令再继续。
- 唯一恢复位置为本批continuation_checkpoint.json与COMMANDS.md；只补缺失门，复用现有工具，不重审旧release、不重新跑已完成批次。按“返回结果→保存账本→合格增量发布→回读→推进下一任务”执行；每题最多两次盲输入7B episode。
- 本批必须完成适用题目的生成/S2/S3/新增机器发布门及失败封账后暂停，不以C3完成冒称清洗完成。E2仅独立RL候选，未经R0–R3不计训练ready；不扩本批之外新池、不创建定时任务。

## 2026-10-04 第二批新候选进行中（目标仍为1000）

- 上一目标回合有实际进展：正式759→781，新增22已发布。1000目标未完成，仍差219；上一批闭合不代表整个目标完成。
- 当前恢复目录 `MedSearch/data/cleaning_v2/sft_20261004_new02`；新增冻结200题，源ID首末5777/7156。排除13份旧候选文件（含前批全部200题及其失败/隔离），复用保护范围，当前SFT781参与新增隔离。冻结时刻及hash见 `run_manifest.json`。
- C1固定语义筛查返回8对，均same_task，已排除；192通过。Q1四包50题全部入账：pass117/reject19/quarantine64。与隔离相交后113进入C2，四包25/30/33/25，独立上下文正在审查。
- C2四包已实际入账：合计104 pass、4 reject、5 quarantine；104题真实Q0检索已保存，C3按20+3、20+10、20+9、20+2四组审核包执行。适用C2待办0、C3待办104，候选均未计入训练数据。
- 2026-10-05用户新增停止边界：审完既有104题的C3证据后暂停，等新命令；禁止本轮新增教师生成或候选批。已绑定原包04共22题：E3 8、E2 7、E1 7，引用绑定错误0；剩余82题分为23/30/29独立包接续。此前子代理额度耗尽中断，已有文件先入账，不重新审核已完成项。
- **2026-10-05暂停收口**：104/104题C3全部实际入账，E3=38、E2=21、E1=27、E0=18，引用绑定错误0。21条E2已保存rl_candidates.jsonl（仅候选，非训练ready）；38条E3待用户恢复后启动教师与S1-S4。本轮没有教师生成或正式新增，正式仍781。报告与恢复位置在new02/RUN_REPORT.md、continuation_checkpoint.json；所有子代理已结束，按用户要求暂停，等待明确恢复命令。
- **2026-10-05用户明确恢复38条收口**：按最新§10.15推进既有38条E3到生成/S1-S4/发布或失败封账，不扩池；收口后再次暂停。已冻结s0_inputs/new02_sft01.jsonl共38题，第一次教师调用已启动，Docker session 90897，日志new02_sft01_teacher.log。旧证据暂停状态已解除；正式基线781。接续先等真实返回，按20+18片完成S2与独立S3，通过项立即发布。
- **38条收口进行中**：第一轮38条原生轨迹已返回并保存，S2共20 pass、16 fail、2 hold；2条S2通过者因动作合法性不满足而停止S3。第一包S3共7 equivalent、1 not_equivalent，7条已立即完成真实loader/mask和发布，正式781→788，inc103_TAKEOVER_BATCH_6097，数据hash a0296335bab189d81e72744e1a4eb09f08dc0a64c690ddf4cf0a7c77954a8f18。第二包10条S3进行中，之后仅按原预算补救本38条，不扩池。
- **第二轮收口进行中**：第二包S3 9 equivalent、1 not_equivalent，9条已发布，正式788→797（本轮累计+16），inc104_TAKEOVER_BATCH_6960，数据hash feed7a6767b80776c2b7fd99b03b6ace5e0372a2324aaa0ced58729c61a7dfba。仅剩余22题进行了第二次盲输入生成，全部原生轨迹已返回；20+2两包S2中，2题已因实质轨迹错误封账，20题待审。各题最多两次，没有扩池。
- **2026-10-05本38题端到端完成并暂停**：第二轮22条S2为4 pass、16 fail、2 hold，4条S3为2 equivalent、2 not_equivalent，2条立即完成loader/mask与发布。正式781→799，累计新增18（7+9+2三个增量），最终inc105_TAKEOVER_BATCH_5887，data hash af5b57cd58bd87f86313c71ebe20e4a5a3d804020a5eda771f0413a64564a48f，manifest hash 704f965988561fa1612a9d78b3c581e1ee0b587811fd3602d4108c577544a117。剩余20题已封账：16轨迹证据/推理失败、2答案不等价、2轨迹未决；全部已用两次，无待审/待发布/可重试积压。21条E2仅RL候选，未新增训练ready RL。本轮episode60条，实际训练ready新增18，未扩池或重审旧数据。报告new02/RUN_REPORT.md，逐题结果final_38_outcomes.jsonl，checkpoint状态paused_by_user_after_38_closure。按用户要求暂停，等待明确恢复命令。
- 当前新批尚无可训练新增项，无已通过未发布积压；正式release仍为781。唯一下一操作是保存返回C2/C3→E3教师轨迹→隔离S2/S3→新增发布，继续推进完整目标。包03未写入、包01非法Unicode转义的文件返回问题及修复记录在本批EXECUTION_NOTES.md；不重新审核旧release。
- 当前接续状态：`MedSearch/data/cleaning_v2/sft_20261004_new02/continuation_checkpoint.json`。不创建定时任务，不重审旧release，不重复前批21条已耗尽生成预算项。

## 2026-10-04 新候选200题已收口：新增22，正式781
<!-- 历史记录；当前下一批见下方2026-10-05 new03，不重启旧批。 -->

- 正式 SFT 基线759，当前新批次 `MedSearch/data/cleaning_v2/sft_20261004_new01` 冻结200个未处理候选；跳过旧候选与已发布项。按清洗书§10.15继续，未扩大旧历史审查。
- C1精确隔离通过；语义近邻4对均判同题并排除，其余196通过。Q1盲审200题已入账：pass136、reject21、quarantine43；与C1相交后132题分四包进入C2。
- C2四包132题全部入账：125通过、3排除、4隔离。125题Q0检索与C3全部完成：E3=43、E2=39、E1=29、E0=14。4处引文绑定错误已由原审查者修复，原始返回保留；未重审其他题。E2只为RL候选，不计入可训练RL。
- 43条E3分5批完成真实教师生成（5/11/9/14/4），首轮新增1/6/3/7/2，共19。24个未通过项统一进行了唯一剩余尝试：S2 pass7/hold11/fail6；3个pass被动作门拦下，仅4个送独立S3，3个equivalent并发布，1个not_equivalent。合计67个真实episode，新增22；其余21个E3题退出本批SFT，未超过两次预算。
- 最终release781，data hash `48a32628188003372abfffc1a53bd8b39144347e1535136cf28a386bc75cb171`，manifest hash `50422599c38bc7077121af8cf370eccff185438f343b20256d6fe2f2d5b09e61`；loader/mask与回读均通过。checkpoint所有适用未决门、待发布、可重试数均0，不能把未适用门的pending当成待做任务。
- 第二轮端到端记录为UTC14:23:07至14:49:09，约26分钟，24个新episode只新增3条；包含生成、审阅等待、协调和发布，不是纯模型耗时。全批起点未显式记录，token unknown。全账户五小时读数本批中途49%→最终83%，周67%→72%，共享额度不能精确归因为本批。
- 技术偏差：复用此前errored代理时，其先处理了旧B6/40326引文绑定，主代理立即停止旧任务；未改变本批冻结输入或release，具体情况在 `RUN_REPORT.md`。不宣称执行过程毫无偏差。
- 默认权限访问Docker被拒绝一次，授权调用后正常；不是NVML或容器故障。修正了新目录证据引用的路径层级，保留已有原始检索与审查包。
- 恢复位置：`MedSearch/data/cleaning_v2/sft_20261004_new01/continuation_checkpoint.json`，状态 `bounded_batch_complete`；39题RL候选保存于 `rl_candidates.jsonl`，尚不训练ready。下一步冻结新的最多200题，排除本批200题及全部旧候选；不重开本批21个终止项，不重审旧release。距SFT目标还差219。

## 本次接续收口：正式 SFT 759

- 从本次恢复前752增至**759（新增7）**，距离1000差241；从早前746累计新增13。当前实际文件：`MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc096_TAKEOVER_39343/release/sft/accepted.jsonl`，SHA256 `3c897a7e5c3d4582dcf2bd4df4571a58f984f12341ba350728c3a36a53dc3356`；manifest SHA256 `6931676df74402ad7a09965777b971b691ce525258d7e99a884ae0d63875e6b7`。末尾checkpoint再次回读数量/data hash/manifest hash通过。
- 实际执行：四批教师分别7/14/5/5，共31条episode，全部容器命令exit0；四批S2结果及时保存，只有有效S1+S2通过项送独立S3。新增分别4/2/0/1，全部做真实训练loader、mask及发布检查后导出。另完成29条Q2检索和20/9证据分片入账，新Q2分级5 E3/9 E2/12 E1/3 E0。未对旧release追加医学复审。
- 当前B6冻结108题：5条隔离门拒绝；103条C3已审（35 E3/51 E2/15 E1/2 E0）。35条E3已收口为19条已发布（含此前）+16条终止失败；17条低证据/冲突按已保存结论退出；51条E2保留为**未完成R0–R3的RL候选**，本轮没有新增正式RL，不能用于训练。详细checkpoint：`MedSearch/data/cleaning_v2/sft_20261001_rollout_prep/batch06_821_928_takeover/sol_luna_resume_checkpoint_20261004.json`。
- 工程返工真实记录：两个Luna存在性预检误报路径缺失，指定CMD相对type后接续；末条S3使用CMD echo写出GBK+字面转义引号，原件已保留`.original_gbk.bak`，只修格式后绑定成功。主代理首次修复也发生CRLF比较失败，修为字节比较；此前入账失败后误继续prepare/smoke造成两个依赖失败调用，随后已改为检查退出码再推进。没有将这些错误算为语料问题。
- 执行限制如实记录：当前发布工具一次命令处理多条但仍逐条创建增量版本，尚未符合§10.15“同时多条合成一个增量”的文件打包要求；数量和既有字节保留检查均通过，没有因此延迟导出。不要声称工具工作流已经完全自动化。
- 成本：账户五小时used 15%→36%、周used61%→65%，共享账户差值不能视为本任务精确成本；token unknown。没有使用额度重置，没有创建自动化。Luna为GPT-6 Luna Max，主调度为本会话6.1 Sol轻度；路径/格式返工仍有改善空间，未宣称节省固定比例。
- **唯一下一动作：当前冻结928题队列不重开耗尽预算；从huatuogpto1尚未处理题冻结下一批最多200条，先执行新增C0–C2/隔离门，再按50题查询、20题证据包推进。** 原先标志拿不准或已隔离题保持原结论。现有51条RL候选保留侧账，后续补R0–R3才能导出；不得作为已清洗RL计数。


- 14条最终重试已收口：S2 6 pass/8 fail，独立S3 2 equivalent/4 not_equivalent；2条真实loader和导出回读通过。正式SFT **758**，本次恢复新增6，距1000差242，release `sft_20261001_joint_run_v1_inc095_TAKEOVER_40054/release/sft/accepted.jsonl`，SHA256 `28fc33be909d54f22995519515b722e3930e93e3f4f33c8770251c3fdfec0050`。12条失败标终止，checkpoint已保存。
- 补检5题首次轨迹 S2 1 pass/4 fail；唯一pass含非法动作，被既有S1机器要求拦下，S3不执行，本次发布0。共享s3pack现把原S1格式/终答/搜索检查前移，避免为不完整轨迹耗一次S3；没有降低标准或新增语义门。5题仍有第二次预算，合并为 `s0_inputs/b06_sol_luna_sft04.jsonl` 正在最后重试。

- 29条Q2已全部C3入账，绑定错误0：新Q2 E3=5、E2=9、E1=12、E0=3（1条conflicting）。5条新E3已冻结 `s0_inputs/b06_sol_luna_sft03.jsonl` 并启动首次教师。14条最后重试已完整返回并保存receipt/账本，`b06_sol_luna_sft02_retry_s2_blind_input.jsonl` 正在整包盲审。当前正式756没有未发布完整通过积压。

- 新接续首批7条完成并收口：S2 5 pass/2 fail，独立S3 4 equivalent/1 uncertain（终答字段null），4条完成真实loader及发布回读。当前正式SFT **756**，本次恢复新增4，距1000差244。指针：`sft_20261001_joint_run_v1_inc093_TAKEOVER_40545/release/sft/accepted.jsonl`，SHA256 `87b6a0eab5e6615692d492606eab18e04cdbe6469a604aee9c3e22321403ae59`，manifest SHA256 `ea5228a6cec1ad51389fb02a0548867e06ab206ba1927c0f151c57af1eb10c14`。
- 已通过未发布积压为0后，B6剩余14条冻结为 `s0_inputs/b06_sol_luna_sft02.jsonl`，正在运行第二次、也是最后一次教师尝试；不得再生成第三次。29条Q2实际检索已全部返回并入账，20/9两片正在C3语义审阅。
- 有限工程问题记录：Luna两次存在性检查误报路径缺失，直接指定CMD `type`相对路径后成功；40326 OCR引文已修且绑定清零；39451旧拒绝文件已同步账本，未重开医学审查；40054保留已知S1非法动作拒绝，未把S2/S3通过误当成可发布。

- 用户再次要求继续后恢复：五小时 used 15%、周 used 61%；没有使用额度重置。40326 引文已按 OCR 原文 `speci- fied` 修正，旧输出留备份，C3 绑定错误清零。
- 已冻结 B6 新 E3 7条 `s0_inputs/b06_sol_luna_sft01.jsonl`，使用已有容器/入口运行首次教师，日志 `b06_sol_luna_sft01.stdout.log` 与 `.stderr.log`，未重启容器。生成返回即 prepare_reviews 保存账本，再隔离 S2/S3，通过项立即发布。
- 同时冻结仍有预算且没有冲突的29条 Q2 输入 `b06_sol_luna_wide02_q2_blind_input.jsonl`，交给新 Luna Max 盲查询上下文。共用冻结入口现在排除 assignment reject/quarantine 和已记录 conflicting，避免实质冲突题继续耗预算。

- 本轮大批候选已全部返回：B6 的 49 条 Q1 审阅 raw 均已落盘；01/02 绑定无错误，03 有 1 处 40326 引文未逐字匹配，已派原审阅者只修引用格式并回写入账，不新增语义复审。新 Q1 原始结果合计 E3=7、E2=11、E1=21、E0=10（含 1 条 conflicting）。保留逐查询结论；账本总体 C3 会复用 Q0/Q1 中更强证据，不能把总体计数当作本轮新增。7 条 E3 是待生成 SFT 候选，11 条 E2 是待 R0–R3 的 RL 候选，未增加正式训练数据。
- 截至收口前最后读取账户：五小时 used 98%、周 used 59%，余量不足以可靠开启下一轮生成。当前没有未发布的完整通过轨迹；下一次直接从 B6 已审候选冻结教师输入续跑，禁止重审这 49 条。未创建定时任务，未使用额度重置。
- 固定批量约定已写入清洗书 §10.15：问题/盲查询默认最多 50 条，完整证据/轨迹默认每包 20 条，合适时可到 50；本次实际为 49 条问题 + 20/20/9 证据分片，并行审阅。保持质量标准，不按条数凑通过。

- 03:57 UTC 收口更新：第二批 13 条完成 S2/S3，5 条通过并完成真实 loader 检查、发布和回读，8 条失败/待定单独入账。两批 20 条教师轨迹共新增 SFT 6 条，正式累计 752，距 1,000 差 248。当前指针指向 `sft_20261001_joint_run_v1_inc089_TAKEOVER_38863/release/sft/accepted.jsonl`，SHA256 `f6553b768f3f8c1069e553272e2c8c8633ca6b986ec4956ab4ce28cee8114bdf`；manifest SHA256 `4b5fcf009800ef194609a8f36dcf0cfc0aaa5cd8d1f6908216d06794e16c1c3e`。第二批 checkpoint 已写入 continue_trial 的对应文件；已通过未发布积压为 0。

- 当前正式 SFT 已由 746 增至 747；新增 medical-o1-verifiable:36867，已完成独立 S2/S3、真实 CPU loader 检查、导出和数量/hash 回读。
- 当前指针：`MedSearch/data/cleaning_v2/CURRENT_SFT_RELEASE.json`；release：`sft_20261001_joint_run_v1_inc084_TAKEOVER_36867/release/sft/accepted.jsonl`，SHA256 `8361ab078f48bfcd88497a4ef4a9cc26566cba44dbda8d3f5352dc8dba4fa4a6`。
- 教师重试第一批 7 条完成：1 条发布，6 条未通过；第二批 13 条已返回并保存 S2：7 pass、4 fail、2 hold，7 条正在独立 S3，尚未计入 release。
- 扩大 B6 候选批量：49 条盲查询与实际 Q1 检索均完成，检索命令 exit 0。证据包拆成 20/20/9，两个 Luna Max 审阅代理并行处理，结果返回后立即按包绑定入账。
- 第二批恢复：`batch05_621_820_takeover/b05_sol_luna_trial02_retry_s3_selected_input.jsonl`；S3 返回后 record_reviews --gate s3，再 publish prepare → 容器 publish smoke → publish publish --machine-s4 → continue_trial finish。不得复核旧 release 或新增语义审查层。
- B6 恢复：`batch06_821_928_takeover/q1_review/input_b06_sol_luna_wide01_01.jsonl` 至 `_03.jsonl`；对应 raw 文件返回后用 batch06.record_c3 --raw-decisions ... --input ... 入账。C3 候选不等于可训练 SFT/RL。
- 本轮账户五小时 used 从 74% 到最近 96%，周 used 从 55% 到 58%；这是共享账户变化，不能全部归因于本任务。原始起始剩余为 26%，并非口述的 35%。不使用免费额度重置。
- 工程开销：修复 raw 审查结果由程序绑定 hash/manifest、S2/S3 单独入账、取消重复语义 S4、扩展 B6 查询入口。首批输入缺 split 导致一次模型加载后失败，已修复并在模型加载前验证；第二批教师运行未重复该错误。保留原日志和输入备份，不伪称稳态吞吐。
# 2026-10-08 SFT 清洗实际暂停检查点

- 当前任务：既有 new06 完整收尾，再处理 new07 一批；本轮已暂停，无活动生成或审核，不扩展 new08，不创建定时任务。
- 正式 SFT：869 → 881，今日新增12条（new06收尾7条、new07新增5条），距1000条还差119条。全部新增通过真实loader、observation loss mask及数量/data/manifest hash回读，没有通过未发布积压。
- 正式文件：`MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc128_TAKEOVER_BATCH_8780/release/sft/accepted.jsonl`；data SHA256 `f16f8077ed8e5e2ac45112d16d843a138ad707983ac3bfcfd1777d434ae5cd94`；manifest SHA256 `975bd49bbf6da16a569145f250794e83c5f77ff5c5aee27bd360d50ac5ff0359`。
- new06：200题完整闭合，累计新增23条，56真实episode，每题最多两次，26 RL候选。本批实际检索为115题，既有内容门均复用；恢复只审未结束episode。最终归宿见new06/final_batch_outcomes.jsonl。
- new07：冻结200题（IDs8669–9044），Q1/C1/C2及107题C3全部实际完成入账。归宿：5条已发布、40条RL候选（尚未R0–R3，不计训练ready）、35条证据不足、15条隔离拒绝、65条题面拒绝或隔离、13条gold拒绝或隔离、1条具体证据/gold冲突隔离、26条环境未闭合。
- new07 已完成17条真实首episode，8条S2通过后独立S3等价5/不等价3。剩14条首episode未执行、12条首episode失败且各剩1次预算，不能把这26条标为清洗完成。
- 环境根因：第二14题原参数初始化失败，有限重试一次仍失败，实际日志为 `Available KV cache memory: -8.53 GiB` / `No available memory for the cache blocks`。两次失败启动均无episode，不消耗题目预算；不将其判成语料质量失败。完整日志`new07/new07_sft02_environment_retry.log`，SHA256 `ad2dd49cee303d73d7f4c41a80f98e4eb7d45a84b9bd0c7cfebdd1401d409f30`。
- 恢复文件：`MedSearch/data/cleaning_v2/sft_20261007_new07/continuation_checkpoint.json`（paused_environment_blocked）、`ENVIRONMENT_BLOCKER.json`、`partial_batch_outcomes.jsonl`、`RUN_REPORT.md`、`COMMANDS.md`。partial 200行已回读，SHA256 `707d49e0acb497ebfd43e55fc1a3e9e31616e810852942b240fab090c9f8fe41`。
- 执行返工：首次S2任务模板漏grade/citations，原盲审者仅补字段、原结论不变，原字节备份保留；空闲代理须followup_task才能启动。后续模板已一次给全。未重审旧release、未开展旧StageA/B排查、未追加审核层。
- 下一步：等待用户恢复环境并下令；复用现有14题冻结输入及12题失败结论，保持各题最多两次episode，合格项即时发布。精确任务token/额度成本unknown，不归算共享账号额度变化。
# 2026-10-09 最新授权：从951继续补足正式SFT 1000

- 2026-10-11 用户追加500条内容复审：从上次未抽查950条内固定种子2026101101抽500，十个50条包已冻结到MedSearch/data/cleaning_v2/audit_20261011_random500。独立Luna Max两组S2各250和一组独立S3共500已派发，每批结果及时落盘；当前尚未完成，正式1000条release不改。先读continuation_checkpoint和实际results文件消费已返回包，不重新派相同任务。只检查题面/证据/关键推理/答案语义，不生成轨迹、不重跑历史隔离或环境排查。

- 2026-10-11 用户指定质量抽查：固定种子20261011随机50条，最终主代理裁定47可用/1证据不足/2待定；建议训练前排除30206、11209、25978。独立原始S2/S3与主代理分歧裁定保留，正式1000条release未修改。真实loader/collator全50通过，最大1762tokens、监督7240tokens，observation/padding不计loss；没有启动训练。报告MedSearch/data/cleaning_v2/audit_20261011_random50/REPORT.md；A档CE-only可进入排除问题项后的pilot，B档仍需要Top-32缓存；抽查不证明全部1000内容正确或训练收益。

- 2026-10-10 收口：正式SFT已达到1000，停止新增SFT生成。本批new13从986增加14；最终文件MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc158_TAKEOVER_BATCH_11183/release/sft/accepted.jsonl；数据SHA256 bea0f504256ba2be03ad7fcd1125d7bc4171486354dcb8496024109d5feb9032，manifest SHA256 af039cee90a703b26de8e78f975d4562a17881a3de0b69dd4e55c8f615c07042。实际回读1000个唯一题目与轨迹ID，新增14条门控及loader/mask回执已核对；既有release审核结论保留。本批200题最终去向全部落盘，RL仅58条候选不计ready，待后续R0-R3。11211错误病原体归因已排除。报告和回读凭据见new13/RUN_REPORT.md、FINAL_EXPORT_READBACK.json；没有正在运行的教师生成任务，没有定时任务。

- 最新正式 **993**，缺7；第三批实际新增2，独立S3三题2等价/1不同，新增真实loader/mask通过，发布回读一致，待发布0。当前 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc155_TAKEOVER_BATCH_11102/release/sft/accepted.jsonl`，数据SHA `245254f7bbd6b0be97b5606c5032d19bad48fd67b990dd6ac095b473dca1cdd9`，manifest `22967dae5dd5534a5e5af673f693c631d4e0226d80e06e952c046bddc6b3cd1d`。19题原最后episode已真实exit0并入账、新LunaMax S2审；最后7未尝试E3冻结new13_sft04进入生成。每题原预算保留，达到1000停止新增SFT生成。

- 最新正式 **991**，差9：new13第二批实际新增5，另2被既有伪Doc门排除。7个内容通过项真实loader/mask通过，最终5发布后数量/hash一致，待发布0。当前文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc154_TAKEOVER_BATCH_10934/release/sft/accepted.jsonl`，SHA256 `8485e0bf927d5eef4676bbd48a899e21a318645b40ebe5f42fd01b19eaf1294d`，manifest `a21d11d3596c73a40aaa5ab34d32795531f1dfcfbc9483970229d47110da13f1`。new13共110题证据审完，38E3/35E2/26E1/11E0，1冲突隔离，34RL候选非训练ready。7条第三批真实生成exit0已送纯LunaMax S2；19首轮未通过合并原最后episode new13_retry_all01（先前first04仅准备未调用），不重置预算。

- 最新正式SFT **986**，差14。new12已收口：实际发布20、RL候选29、两次后不合格23，71实际episodes；本次尾轮7 S2通过、5 S3等价，最终4发布，1被既有伪Doc gate排除。新增真实loader/mask及数量/hash通过，无待审/待发布/可重试。当前 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc153_TAKEOVER_BATCH_10603/release/sft/accepted.jsonl`，SHA256 `d51c96c88e2d4c90ddce053f3c1e222d8069a43dbbb22ed1abee23fa59e76a46`；manifest SHA256 `76a54b5412d4e04ee1ff9df8e9bf0852508855db8e7569ecb8ca6c64eaab7f67`。继续new13，成功7B命令需WSL pin-memory=1和FlashInfer sampler=0，详见new12 COMMANDS。

- 显存阻塞已定位具体来源：宿主Lightroom.exe PID60676占25722204160 bytes（约24GiB），不是Docker或遗留教师。Windows GPU计数器与进程信息交叉确认；已请求用户保存编辑后关闭Lightroom，未强杀。关闭后接续new12原23题，正式仍982，缺18。

- 2026-10-10 最新收口：正式 **982**，距1000差18；本次从968接续新增14，从951累计新增31。最后10571的独立S2/S3及真实loader/mask通过后已立即发布，当前通过待发布0。实际文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc152_TAKEOVER_BATCH_10571/release/sft/accepted.jsonl`，数据SHA256 `63a85005478ce33de0659eea789c31c28276013251550ad68d984e29bf58dadf`，manifest SHA256 `ab8c7f64cbd18ae2b389f0882249d20fde3c3295fe5b3c4c359aa14c6ded3baa`，已回读一致。
- 当前真实阻塞：new12剩余23题原最后episode在模型初始化前失败，一次环境重试同错：GPU空闲6.09GB，既有配置需19.11GB。没有新增episode，不耗尽或重置原题预算；不继续同状态反复加载。冻结输入及错误已保存至new12的 `environment_failure.json`、`new12_retry_tail36_environment_retry.log`、`continuation_checkpoint.json`、`RUN_REPORT.md`。释放GPU后接续原23题，不重审旧数据。29条E2保留RL候选，未计训练ready。所有本轮子代理只做Luna Max独立内容审查。

- 2026-10-10本次再次恢复实际新增13条（7+6），正式981，距1000差19；从951授权累计+30。最新实际文件`MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc151_TAKEOVER_BATCH_10789/release/sft/accepted.jsonl`，SHA256=`1c7b1bd1e3e6d16c6d3c400b974101838b3d8af31ee146d0bb17db8e1df8370c`，manifest SHA256=`781d27cedc813bebddbaceffec9b357af6a27ec53e4be47f980cd4681c27f8cd`。new12剩36教师真实exit0，LunaMax纯S2分20/16为12pass/8fail与9pass/7fail，其中4条证据E2按现有门阻止S3；独立LunaMax S3分别8/10与6/7等价，新增发布检查排除1条伪文档（真实观察标签外文本），其余7+6已真实loader/mask与数量/hash回读通过。两处引用仅标点/OCR格式修正、新不可变结果保留原件；共用binder机械定位真实action/doc，不修改语义结论。首7失败5原attempt2真实exit0，独立S2为2pass/3fail；2题独立S3进行中。其余23未通过已冻结`new12_retry_tail36`并启动原最后attempt2，session41677；不得第三次episode。主代理操作脚本和容器，Luna仅批量语义审阅，无嵌套派发。继续至1000。
- 2026-10-10 new12首增量实际发布2条，正式968，本次恢复累计+17，距1000差32。增量目录`MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc149_TAKEOVER_BATCH_10533/release/sft/accepted.jsonl`，数据SHA256=`efc78d197336aa6437fc7e2bf5bc42a81825e86ce92fb392fe60d1f41e9ff28b`，manifest SHA256=`7f6636c40200be218d35fa4c33a3600d9205bf85f7d551abacae1e92119aeab1`；新增2条真实loader/mask通过，checkpoint数量/hash回读一致。new12 C3全部126审完：43 E3/29 E2/30 E1/24 E0，引用绑定0错误。首7真实教师exit0，独立LunaMax S2为3pass/4fail，独立LunaMax S3为2equivalent/1not_equivalent。因原S2派发schema缺evidence_grade/citations，格式补全保留原结论与文件，用fmt前缀绑定后发布，未重启内容审核。用户提醒§10.15职责边界后撤回Luna协调职责：主代理负责脚本/容器/发布，Luna仅隔离批量语义判断，不嵌套代理。剩36 E3已冻结`new12_sft02`，原成功入口首次episode真实启动session22972；原首7失败5仍各有一次预算。继续到1000，不创建定时任务。
- 2026-10-10 new11完整收口：正式966，本次恢复+15，距1000差34。200题最终15正式SFT、23两次后未过（18证据/推理、5答案不等价）、29 RL候选非训练ready、58弱证据不用、11隔离、43题面不可靠、21gold不可靠；62真实episode，每题最多2次。待审/已通过未发布/可重试均0，报告及最终逐题账本已落盘，状态 `batch_closed_continue_until_1000`。最后增量1条真实loader/mask及数量/hash回读通过，当前目录 `sft_20261001_joint_run_v1_inc148_TAKEOVER_BATCH_10279`，数据SHA256 `46d49bb54ce7e72f1c50fe4d9313a9e1419d590ec982dd7f1ece2d8dd97fbba6`，manifest SHA256 `d0db65552a4ece5d69d099b07935d23f24689d6c0feff918bcb69f1d4e2b1b86`。继续new12，不受旧批后暂停限制。
- 2026-10-10第三增量实际发布6条，正式965，本次恢复累计+14，距1000差35。全38首轮审核完毕：14正式发布、24未通过项合并启动原最后episode `new11_retry_all01`，不重置预算，早先8题准备输入未实际调用。新增真实loader/mask6/6及指针数量/hash回读通过，无已通过未发布积压。当前release目录 `sft_20261001_joint_run_v1_inc147_TAKEOVER_BATCH_10168`，数据SHA256 `111e09a27a313833d42e63f98d4014db4aab64d9592d1935a51c983ee6e57909`，manifest SHA256 `23f2feba14b78d84760f5c5005393ef5887e4cd6eb5464bef77893abf583ab33`。
- 2026-10-10第二增量实际发布3条，正式959，本次恢复累计+8，差41。new11 C3全125已绑定38 E3/29 E2/40 E1/18 E0；剩余25教师首episode真实exit0，全部送纯S2，不以verifier筛选。尾5 S2通过3且独立S3均等价，已先发布，不等待主20；新增真实loader/mask及checkpoint回读通过。当前 `sft_20261001_joint_run_v1_inc146_TAKEOVER_BATCH_10461/release/sft/accepted.jsonl`（在MedSearch/data/cleaning_v2下），SHA256 `c9dcc570cf68df7f7310b1a4e47a1fafad5a65647651b97392c42b2451435595`，manifest SHA256 `b61f529397ba77439a557c07a19bca3eaa2098306888cfe0f922010a27437711`。主20盲审进行中，原末轮尚未调用，已通过待发布0。
- 2026-10-10首个增量实际发布5条，正式956，距1000差44；首13轨迹S2通过8，独立S3等价5/不等价3。新增真实loader和mask 5/5通过，checkpoint数量/hash回读一致，已通过未发布0。当前文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc145_TAKEOVER_BATCH_10236/release/sft/accepted.jsonl`，SHA256 `7793766322fa24c33fec7b713c4dfa74ee8b88e8b4d53b9f25316338d7a75a97`；manifest SHA256 `54845b419b3d52a0a0730c7006a0259255bc1202d9b346440a30a92fd410b008`。默认Docker管道权限拒绝后复用授权升级成功命令，无容器或模型故障，不重查环境。
- 2026-10-10恢复：实际指针仍为951。new11复用已返回13条真实教师轨迹及已有S2原始结论，已绑定8通过/5失败；8条进入隔离的独立S3，不重新审阅。上次审批额度中断未保存的第三主20证据结论本次已实际保存并绑定，引用检查通过；C3当前111/125，37 E3、22 E2、36 E1、16 E0，剩余14正在唯一审阅。以上候选不计正式SFT；先发布完整通过项再启动剩余强证据生成。
- 用户“落盘最后49条轨迹一鼓作气”解除批后暂停，授权必要批次自主迭代至正式1000。基线951；复用清洗书§10.15和既有成功容器入口，每题最多2次真实episode，合格新增即时发布和loader/hash回读。达到1000停止新增SFT生成；不得因凑数放宽E3与独立S2/S3。
- 新批从new11开始；new10全部终态，无待发布或可重试积压，不重审。实际新增以CURRENT_SFT_RELEASE.json为准；候选不计训练ready。不创建定时任务。

## 951时new10完成记录

- 正式 SFT 951 条，距 1,000 差 49。new10 基线933，实际新增18；本次从921接续包含new09积压收口共新增30。实际文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc144_TAKEOVER_BATCH_9783/release/sft/accepted.jsonl`；SHA256 `e3caacbbb84d9a6946bdacc76018302b5f1862eed43111e6904ad892637d492c`。数量与数据/manifest hash回读一致，新增真实loader及loss mask验证通过。
- new10冻结200题最终去向：18正式SFT、22两次后未通过（14轨迹证据/推理不合格、1动作非法或轨迹不完整、7答案不等价）、37 RL候选非训练ready、58证据不足不用、17语义隔离、34题面不可靠、14gold不可靠。63个真实教师episode；每题最多2次。适用待审、通过未发布和可重试均0。
- 协调子代理末尾额度中断，根代理复用已返回S3及真实loader结果先发布1条，再由原独立S3角色完成最后2条检查并绑定，不新增审查层。实际报告、命令和逐题结果见 `MedSearch/data/cleaning_v2/sft_20261009_new10/`。当前状态 `paused_batch_completed`，等待用户命令；不启动new11或定时任务。

## 以下为933阶段记录（历史）

- 最新正式 SFT 933 条，距 1,000 差 67；本次用户恢复基线 921，实际新增 12。当前实际数据 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc140_TAKEOVER_BATCH_9780/release/sft/accepted.jsonl`，SHA256 `ca8f463bc56a1b0e51332c87f80431985a9fca8d1e3bc75cde84f50d87a46f3a`，数量与 hash 回读一致，新增全部真实 loader/mask 验证。
- new09 200 题已收口：26 正式发布、20 两次 episode 后未过、35 RL 候选非训练 ready、58 证据不足不用、13 隔离、30 题面不可靠、18 gold 不可靠。71 个真实教师 episode。实际 `RUN_REPORT.md`、`COMMANDS.md`、`final_batch_outcomes.jsonl` 已写；适用待审、待发布、可重试均 0。
- 用户最新范围为再增加一批。new10 已冻结 200 题（source 9782–10160），基线 933；纯 Q1 与现成语义召回进行中。完成 new10 生成、独立 S2/S3、合格项发布及失败预算收口后暂停；若正式先达到 1,000 则提前停止。不得自动继续 new11，不做 RL 训练或创建定时任务。

## 以下为上次928运行记录（历史）

- 最新真实正式 SFT 928 条，距 1,000 条差 72；用户本次恢复时 921，已实际新增 7。先前恢复基线 881 至今累计实际新增 47。当前文件 `MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc138_TAKEOVER_BATCH_9707/release/sft/accepted.jsonl`，SHA256 `d49abfe968bd2eb1f76b05d8205555c99b7da3de0dc6d966a15fb1f5a0abf873`。新增已真实 loader、loss mask、导出数量/hash 回读，不计候选。
- new09 全 200 的 C1/Q1/C2/C3 已完成并复用，139 真实检索结论为 46 E3/35 E2/35 E1/23 E0。46 第一 episode 全部真实生成及独立 S2/S3 完成，当前已正式发布 21；20 原预算末轮正在真实生成，其余 5 第一 episode 未过保留最后一次预算。无已通过未发布积压。
- 本次用户范围：收口已有 new09，再做 new10 一批，正式达到 1,000 则提前停止 SFT；不无限扩池。new10 已复制现成入口，尚未冻结源或发布基线。new09 实际命令与额度恢复记于 `MedSearch/data/cleaning_v2/sft_20261009_new09/COMMANDS.md`；最新恢复位置由本批 `continuation_checkpoint.json` 记录。

## 下方为先前907闭合记录（历史）

- 当前正式 SFT 为 907 条，距 1,000 条差 93。本次恢复基线 881，累计实际新增 26；new08 单批新增 18，200 题已全部分流闭合，51 个真实教师 episode，每题最多两次。
- 已完成新增项真实 loader、observation/padding loss mask、导出与数量/hash 回读。当前数据：`MedSearch/data/cleaning_v2/sft_20261001_joint_run_v1_inc135_TAKEOVER_BATCH_9326/release/sft/accepted.jsonl`，SHA256 `931f891754ffe31e85d7a55c1bdb4880ad35ff0ca9800286ab8fc2f481d1738e`。
- new08 结果：18 正式 SFT、15 两次后未通过、31 RL 候选（非训练 ready）、64 证据不足不用、46 题面不可靠、16 gold 不可靠、9 语义隔离、1 明确证据冲突；报告见 `MedSearch/data/cleaning_v2/sft_20261008_new08/RUN_REPORT.md`。适用待审/待发布/重试均为 0，不重新审核。
- 当前新批 `sft_20261009_new09` 已冻结 200 题（source 9442–9781），基线 907，纯 Q1 与共用语义召回正在进行。继续沿用已成功 Docker、检索服务及 GPU utilization 0.6，不新增环境预检或内容审查层。额度中断已恢复；用户正式 1,000 目标授权仍有效，不暂停、不创建定时任务。

## 以下为上次恢复记录（历史）

- 用户最新恢复解除下方历史暂停要求，今晚自主迭代至正式1000条SFT后停止；不创建定时任务，不扩RL训练。
- 本次恢复基线881，已新增8条，当前正式889，距1000差111。inc131_TAKEOVER_BATCH_8679；data SHA256 `3d3704eb6e3528a982650b404352ddd143373e31ce2fe4060cbc541e9b3006db`，manifest SHA256 `24554200d7f3cae62e6fa9be5e9053e6fb0aba6696b7af2c82f9aff0c833968c`。逐增量已实际loader/mask验证、导出与回读，没有通过未发布积压。
- new07原26待办全部实际结束并收口：本批累计新增13条、53真实episode，全部200去向保存；18题两次未过（16轨迹、2答案），40RL候选、1证据/gold冲突隔离。原三个环境不完整episode保留历史标记且未重置预算。
- 恢复根因与单变量：GPU当前驱动610.57.01/KMD610.88，占用1206MiB，旧占用11446MiB；显存预算0.5→0.6，实际KV从-8.53GiB恢复为3.55GiB。14首轮、12末轮、10末轮真实返回；服务重启后仅恢复缺失8001/8002，后续复用服务成功回执再加载7B。日志见new07/COMMANDS.md。
- new08已实际冻结200新题，IDs9046–9439、exact isolation pass、baseline889；正在C1语义隔离与四片50题纯题面Q1，待根独立当前C1/C2/C3，盲S2与纯S3角色分开。新08目录 `MedSearch/data/cleaning_v2/sft_20261008_new08`。
- 下方暂停检查点为历史记录，当前唯一动作是完成新08及后续新批的生成、审核、合格即时发布，直到正式1000。已有内容门结论复用，不重审旧release/StageA/B；每qid最多2episode。
# 2026-10-11 SFT用户追加审核：当前状态

- **最新授权已执行剔除：**用户要求正式移除42条并从原数据库补42。真实loader检查958条通过，过滤指针为 `sft_20261001_joint_run_v1_inc159_AUDIT_FILTER42/release/sft/accepted.jsonl`，SHA256 `63e193f1b226c1fe39497d0c4172d6f1b0486338534520c89e58a568a47d24d9`。旧1000版本保留追溯；42条理由单独保存 `sft_20261011_audit_filter42/EXCLUSIONS.md` 和 excluded42.jsonl。补充批 `sft_20261011_new14` 已冻结200新题（11281–11641），执行现有隔离/C2/C3→真实7B→独立S2/S3→即时发布，不重新审旧958，不修改清洗书。
- **替补首次实际发布：**new14_sft01生成7条，S2 4 pass /2 fail /1 hold，4条S3 equivalent；真实loader、loss mask、观察片段屏蔽与文件回读通过后追加4条，正式当时962；历史指针 `sft_20261001_joint_run_v1_inc160_TAKEOVER_BATCH_11281`。new14全部98道C3完成（37E3/23E2/24E1/14E0）。
- **当前替补进度（2026-10-11已收口）：**正式1000条；本次接续新增9条（991→1000），累计替补42（new14 19/new15 17/new16 6）。原42问题ID全部排除，958条字节不变；新42均经盲S2/独立S3/真实引用/loader/mask检查并及时发布。当前inc171_TAKEOVER_BATCH_12251，SHA256 `9eed78360936f183e5f9a676a669c2c0afd5859b474c0a3c7f20341dbec556d4`，数量和manifest hash已回读。实际文件、排除理由、替补ID与回读证明见sft_20261011_audit_filter42/FINAL_REPORT.md。目标完成，已停止新生成和审查；未决与RL候选不算可训练数据，不修改清洗书、不创建定时任务。

- **最新完成：**剩余450条全部审核收口：437 pass / 8 fail / 5 hold。合并三轮1000唯一ID：958 pass / 18 fail / 24 hold，42条建议排除。报告 `MedSearch/data/cleaning_v2/audit_20261011_remaining450/REPORT.md`，全量结论 `full1000_final_decisions.jsonl`，全量排除 `full1000_recommended_exclusions.jsonl`；数量与hash已实际回读。原1000正式release及指针保持不变，958尚不是新正式导出；本轮结束，无活动审查任务，不扩大候选、不启动训练。

- 正式release保持1000条，指针 `MedSearch/data/cleaning_v2/CURRENT_SFT_RELEASE.json` 未变；本轮没有训练、教师重生成或定时任务。
- 原50条+追加500条审核已完成，互不重叠：521 pass / 10 fail / 19 hold，共550条；29条建议排除，不擅改正式训练入口。500条实际结论及hash位于 `MedSearch/data/cleaning_v2/audit_20261011_random500/audit_summary.json`，逐题结论和排除清单均已回读。
- 用户最新授权审核剩余450条。已冻结全部剩余唯一ID，9片各50，目录 `MedSearch/data/cleaning_v2/audit_20261011_remaining450/`。三个隔离GPT-6 Luna Max仅做S2盲审，S2通过后送独立S3；主代理处理文件、账本与明确争议。接续看该目录 continuation_checkpoint.json、RUN_REPORT.md。450完成后汇总1000全量结果，不扩大新数据池。



## 2026-10-13 RL新批次02（执行中）

用户授权弃用旧RL140题，保留原件但退出正式指针，新版正式RL从0重新计算。新批目录：MedSearch/data/cleaning_v2/rl_20261013_batch02。冻结200题，不扩池，完成后暂停。SFT1000题不变。

已完成题面盲审、top10/0.92语义隔离与gold核对；111题进入实际top3/总500学生tokens的Q0证据审查，其余89题已在前门排除或隔离。已修复prompt token统计，已补同账本Q1/Q2接续，尚待本批实际闭环验证。Luna角色限定语义门；没有启动7B或训练。详情以主账本、continuation_checkpoint和RUN_REPORT为准。

### 2026-10-13 RL batch02最终收口：已暂停

新200题已完成：正式RL新增/累计86题（E3/E2各43），reject49、quarantine65，未完成/待发布/环境阻塞0。旧140退出正式指针，原件hash不变；SFT1000及hash不变。实际Q0/Q1/Q2为111/61/33次，205次全部返回、0传输重试，累计预算未重置；7B生成/训练0次。通过项即时发布，真实loader和数量/hash回读通过。

- 正式文件：`MedSearch/data/cleaning_v2/rl_20261013_batch02/release/rl/train.parquet`
- SHA256：`6db75a5138d910070194cc0d7b276ce53db8c57f830dabd703ceb560cd1e237c`
- 报告：同批`RUN_REPORT.md`；品鉴：`PREVIEW_10.md`（E3/E2各5题，非随机）；114条排除/隔离及理由：`excluded.jsonl`。
- 执行偏离/修复见报告和`execution_metrics.json`；清洗书§6/§8已更新实际入口/状态，不恢复旧7/67题。
- 当前`paused_batch_complete`，等待用户命令；不扩池、不定时。


## 2026-10-22 新增46条RL内容复核

按用户要求完成batch02新增46条逐题内容复核：37条未发现新增阻塞，2条明确评分不一致、7条具体疑点，合计建议暂缓9条。报告及逐题理由：MedSearch/data/cleaning_v2/rl_20261022_lunamax_batch02/NEW46_AUDIT_REPORT.md、NEW46_CONTENT_AUDIT.jsonl、NEW46_HOLD_RECOMMENDATIONS.jsonl。46个episode hash/盲prompt IDs/纯问题发布prompt、62条引文、61段观察核对通过。正式release本次未修改，仍312条并包含建议暂缓9条；不得误称312条全量复核通过。未扩池、未生成、未修改清洗书，等待用户后续安排。


## 2026-10-22 Sol更正RL303及每轮100题分工

阶段：MedSearch-R1 RL数据准备。用户要求撤下新增46条复核发现的9条，已完成正式文件更正：312→303；本批保留37（E3=17/E2=20），历史基线266未作医学重审。真实RL loader逐行303和首尾collator批次通过，数量/ID/hash回读一致；SFT1000保持原hash。

当前RL：`MedSearch/data/cleaning_v2/rl_20261022_lunamax_batch02_sol_filter9/release/rl/train.parquet`，SHA256 `a3105a707246abd7c9e92c45c2df34cd9203f0a2ae5fb672d960e0983c1901e1`。回执：`MedSearch/data/cleaning_v2/rl_20261022_lunamax_batch02_sol_filter9/receipt.json`。9条退出理由沿用该目录withdrawals.jsonl，原件不删。

分工已记录：Luna单代理每轮冻结100原题，7B真实盲搜索后候选及时入账staging，awaiting_sol_review暂停；Sol只复核Luna放行增量并最终发布，正常E2与教师失败可保留。Luna书增加§0A和本轮反例，新增Sol书`MedSearch/docs/superpowers/plans/2026-10-22-rl-cleaning-sol-increment-review.md`；旧RULE_SNAPSHOT未改。下一轮选题游标从最近真实冻结批次rl_20261022_lunamax_batch02继续，基线读取CURRENT_RL_RELEASE。

当前已暂停；本次没有启动新100题，等待用户恢复。


## 2026-10-22 单Luna与Sol连续清洗到RL500（进行中）

### 最新完成：2026-10-27，batch27清洗100题并暂停

用户授权按新版书清洗新一轮100题。先复用上一轮Sol已审结论发布5条至p095；新批源19250–19432全部完成：正式新增8、隔离73、拒绝19。31条真实7B episode、39次搜索、5799 response-mask tokens，无未搜索/环境失败。Terra主审，Sol/medium仅抽拟发布3条和未通过2条，均维持原判断；新批8条中3条记录Sol抽检、5条仅Terra主审。inc001/inc002及p096/p097均真实loader和数量/hash回读，当前p097正式369=344正确+25错误，距500还差131正确，training_ready=false；本次对话从356增至369（+13，其中上一批5、本新批8）。旧导出器写死Sol身份，已最小支持Terra主审引用，测试通过且实际导出验证身份区分。文件、评分疑点和数量见`MedSearch/data/cleaning_v2/rl_20261027_terra_batch27/RUN_REPORT.md`及`VERIFICATION.json`。本100题已结束，停止并等待用户继续；无定时任务。

### 最新流程修订：2026-10-27，Terra中等主审＋Sol中等子代理抽检

用户批准将主审/抽检机制写入清洗书：默认主代理`gpt-6.1-terra/medium`，每100题至多新建一个`gpt-6.1-sol/medium`抽检子代理，fork_turns=none且同轮复用。对符合当前配额的拟发布项按包累计抽20%、至少3条（不足全查），每轮另抽2条未通过项；明确争议定点送检，实质漏筛扩查受影响类型，非全量双审。Terra是唯一账本和发布写入者，按包完成抽检及R0–R3后发布，无补抽需求的包不调用Sol。新默认操作卡：`MedSearch/docs/superpowers/plans/2026-10-27-rl-cleaning-terra-sol-sampling.md`；通用书、Luna旧书入口及Sol操作书已同步。19210空答/疑点分轴纠错及既有正反例保留，规则仅对后续授权执行生效，不改旧批冻结/raw。此次仅改文档，正式仍p094共356条，batch26仍暂停，未开启子代理、生成或发布。

### 最新能力验收：2026-10-27，Sol复核Terra的27条C3结果

用户本次要求评估Terra能否代行Sol复核。Sol逐条复核第二包20条及尾包7条，包含全部12条候选及15条未通过项：5条当前完整正确目标的内容判断认可；其余4错答、3空答候选不满足现有配额。发现19210将空答误记为semantic=uncertain的字段问题，19183证据E1/E2有保守边界分歧；未修改原raw、账本或候选。27个实际prompt_ids与固定协议加原题一致，19处引句和episode hash均核对通过。能力报告及逐题结论：`MedSearch/data/cleaning_v2/rl_20261026_lunamax_batch26/sol_review/TERRA_CAPABILITY_REVIEW.md`、同名`.jsonl`。建议可试行Terra主洗与Sol抽查，但尚无同口径耗时/额度数据，不宣称节省比例；本次不改变现行清洗书。正式仍p094共356=331正确+25错误，training_ready=false；保持本批暂停，未启动新批或正式发布。

### 最新接续：2026-10-27，batch26清洗完成并暂停，待Sol验收

本批100题去向已闭合：4条先前已正式发布、12条证据候选待Sol验收、63隔离、21拒绝。48个真实7B episode/72次搜索/9059 response-mask tokens，其中47条有搜索已审、1条未搜索隔离、0环境失败。当前p094正式356=331正确+25错，目标500仍缺144正确；12候选中5答对、4答错、3空答，均未计入正式且不得绕过错答配额。第二/尾包由GPT-6.1 Terra审查并保存raw/账本，候选SHA093adaef27f3db67e6684241a6ae64538fc4501a20b4bfeb0f32873d00abb1a0，完整路径/理由/后续步骤见本批RUN_REPORT和continuation_checkpoint。按照用户本次指示已暂停，不启动下一批，等待Sol验收。以下是历史阶段快照。

用户指定本批洗完暂停，由Sol验收。正式p094为356=331正确+25错，第一包4条已发布。第二包20条由GPT-6.1 Terra一次审实际窗口并保存raw_02，已入账8条E2/E3候选待Sol验收（其中包含错答和空答，当前25条错答额度已满，绝不能直接视作正式新增）。尾包8题generation03 session80385运行中；结束后绑定、审查、保存入账，并写完整候选/去向和接续点，暂停等待Sol。不得继续冻结新一百题。下面为历史快照。

batch26首包20题真实生成/审查完成，7候选主审4retain正确/1reserve/2隔离，inc001及p094实际退出0，真实loader/collator/ID/数量/hash回读通过，正式356=331正确+25错误，剩144正确，无通过待发布积压。第二包generation02 session55782运行中，命令见本批teacher/resume_packet02.md，之后尾03八题；每100原题报告后自动继续500目标。下面为历史阶段快照。

batch26最新：C1已绑定86通过/14重复；C2共86题全部入账48通过/31隔离/7拒绝。teacher-input已生成固定盲输入20/20/8，第一包generation01 session82313运行中，命令见本批teacher/resume_packet01.md。正式p093352=327正确+25错误，剩148正确，无已通过未发布积压。下面为此前阶段快照。

batch25全100题已收口：68真实7B/99检索/11977生成tokens，无未搜索或环境失败；28候选主审8retain正确/10reserve/10隔离，最终8RL/79隔离/13拒绝。三个增量及p091–p093均真实loader/collator/ID/数量/hash回读，尾包全3reserve不生成空发布。SUMMARY/ROUND_REPORTS保存，无未决或待发布通过项。正式352=327正确+25错误，距500剩148正确。batch26已冻结100新原题源19060–19249，C1召回session97386运行，命令已记COMMANDS；连续500授权有效，单一Luna Max子代理只写raw，root机械/主审/发布。下面为历史快照。

第三包已收口：9候选主审3retain正确/1reserve/5隔离，inc003及p093实际退出0，真实loader/collator/ID/数量/hash通过，正式352=327正确+25错误，剩148正确，无通过待发布积压。本轮前三包新增8条，尾8题generation04 session68710运行中，命令见teacher/resume_packet04.md；当前唯一Luna Max只写raw，root机械/正向复核/发布，100题报告后自动继续。下面为历史阶段快照。

第二包20条实际7B和C3完成，9候选主复核1retain正确/5reserve/3隔离；inc002及p092实际退出0，真实loader/collator/ID/数量/hash回读通过。最新正式349=324正确+25错误，剩151正确，无通过待发布积压。第三包generation03 session34071运行中，命令见本批teacher/resume_packet03.md；当前唯一Luna Max子代理/root/rl_luna_20261026只写指派raw，root执行机械/正向主审/即时发布。下面为历史快照，连续500授权有效。

用户继续清洗并询问数量，当前正式指针p091348=323正确+25错误，距500剩152正确；第一包4条通过已实际发布且真实loader/collator/ID/数量/hash回读Exit0，无待发布通过项。本轮68题首审通过分20/20/20/8，第一包收口，第二包generation02 session33073运行中，完整成功模板命令已记本批teacher/resume_packet02.md。接续只处理后48题，不重跑原20；每百汇报后自动继续，SFT不动，不创建自动任务。下面均为历史阶段快照。

batch25最新：C1已绑定87通过/13重复；C2共87题全部入账68通过/19隔离/0拒绝，固定盲输入20/20/20/8已生成。第一包generation01 session6268运行中，完整命令见本批teacher/resume_packet01.md。正式仍p090344=319正确+25错误，剩156正确，无通过待发布积压；下面为此前阶段快照。

batch24全100原题已收口：72真实7B/113检索/14036生成tokens，无未搜索或环境失败；25候选主审9retain正确/7reserve/9隔离，最终9RL/84隔离/7拒绝。4增量及p090真实loader/collator/ID/数量/hash回读通过，正式344=319正确+25错误，未知0，剩156正确；SUMMARY/ROUND_REPORTS保存，无未决或待发布积压。batch25已冻结100新原题源18879–19057，C1召回session7311运行，命令已记COMMANDS。自动连续500授权有效，下面为历史快照。

第三包已收口：7候选主复核1retain正确/3reserve/3隔离；inc003及p089实际退出0，真实loader/collator/ID/数量/hash通过，正式343=318正确+25错误，剩157正确，无通过待发布积压。尾12题generation04 session18287运行中，命令见本批teacher/resume_packet04.md。本轮前三包新增8条；Luna曾报告包03后10题重复，root实际解析固定输入及review证明跨包重复集合为空，已纠正读取混淆并保存receipt，无数据修复、无重绑/重跑。下面均为历史阶段快照。

最新覆盖：第二包20题已全部审查，10候选主复核3retain正确/2reserve/5隔离；inc002及p088实际退出0，真实loader/collator/ID/数量/hash核对通过，正式342=317正确+25错误，剩158正确，无通过待发布积压。第三包generation03 session7791运行中，命令见本批teacher/resume_packet03.md；本轮目前新增7条，旧阶段快照不作为恢复状态。

batch24第一包已收口：20条真实7B/20审查、无未搜索或环境失败，7候选主复核4retain正确/2reserve/1隔离。inc001及p087实际退出0，真实loader/collator/ID/数量/hash通过，正式339=314正确+25错误，剩161正确，无通过待发布积压。第二包generation02 session79836运行中，命令见本批teacher/resume_packet02.md。下面为此前阶段快照，连续500授权不变。

batch24最新步骤：C1已绑定93通过/7重复，C2共93题全部入账72通过/21隔离/0拒绝；teacher-input已生成固定盲输入20/20/20/12，第一包generation01 session18415运行中，完整命令见本批teacher/resume_packet01.md。当前正式仍p086335，无已通过未发布积压。每一包生成后由同一Luna审实际窗口，root复核正向候选并立即发布，全部100原题收口后自动继续。

batch23全部100题已收口：76真实7B轨迹、107检索、13917生成tokens、无未搜索或环境失败。31候选主复核17retain正确/11reserve/3隔离，最终17RL/72隔离/11拒绝；4增量及p086真实loader/collator/ID/数量/hash回读成功。正式335=310正确+25错误，未知0，距500尚缺165正确，training_ready=false；SUMMARY/ROUND_REPORTS已保存，无未决或待发布通过项。下一轮batch24已冻结100原题源18701–18877，语义召回session28803运行；自动连续授权有效，下面为历史阶段快照。

第三包已收口：3retain正确/2reserve/2隔离，inc003及p085实际退出0，真实loader/collator/ID/数量/hash核对通过。最新正式332=307正确+25错误，剩168条完整正确，无待发布通过项。generation04尾16题session8286运行中，完整命令见本批teacher/resume_packet04.md；本轮前三包累计新增14条。18597残缺病原名单、18624辅食建议与所问精确发育年龄不构成有效E2桥接，均降E1隔离，原始审查保留。

最新回执覆盖下方阶段快照：第二包20题真实轨迹及主复核完成，6retain正确/3reserve/1隔离，inc002和p084实际退出0，真实loader/collator/ID/数量/hash通过。正式329=304正确+25错误，剩171条完整正确；无待发布通过项。第三包20题generation03 session68750运行中，命令见本批teacher/resume_packet03.md。18591明确要求病原体及疾病而只答疾病，虽评分接受仍隔离，不修改gold或reward。

最新实物p083：323条=完整正确298+错答25，未知0，training_ready=false，还需177条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p083/release/rl/train.parquet`实际loader/collator/ID/数量/hash回读通过。batch22全100原题收口，68real7B/93search/13610response-mask tokens，新增10条，SUMMARY/ROUND_REPORTS保存。batch23已冻结100新原题源18508–18700，C1为92通过/8重复，C2为76通过/13隔离/3拒绝；真实教师输入20/20/20/16。第一包20条已完成审查，10正向候选主复核5retain正确/5reserve，inc001及p083退出0，无已通过待发布积压。第二包生成session41929运行中，命令及接续记录见teacher/resume_packet02.md。下方旧版本为历史快照，以CURRENT和此段为准；连续500授权有效，每百题汇报后自动继续。

最新实物p078：308条=完整正确283+错答25，未知0，training_ready=false，还需192条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p078/release/rl/train.parquet`实际loader/collator/ID/数量/hash通过。batch21全100收口，77real7B/103search/14421response-mask tokens，0未搜索/0环境失败；29候选主复核15retain正确/9reserve/5隔离，最终15RL/76隔离/9拒绝。SUMMARY/ROUND_REPORTS保存，无未决/待发布积压。batch22已自动冻结100新原题源18312–18507，C1召回session47786运行中。以下p075及更早是历史快照，以CURRENT和此段为准；继续最终500，不能在每100汇报后暂停。

最新实物p075：297条=完整正确272+错答25，未知0，training_ready=false，还需203条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p075/release/rl/train.parquet`实际loader/collator/ID/数量/hash回读通过。batch21 C1绑定92pass/8重复，C2绑定77pass/14隔离/1拒绝，teacher输入20/20/20/17；首包20真实episode/20窗口/0未搜索/0环境失败，4放行候选均主复核retain并inc001发布，无积压。第二包生成session71530运行中。下方p074及更早段为历史，以CURRENT和本段为准；每100题汇报不暂停，继续最终500授权有效。

最新实物p074：293条=完整正确268+错答25，未知0，training_ready=false，还需207条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p074/release/rl/train.parquet`实际loader/collator/ID/数量/hash回读通过。batch20全100原题收口，69真实episode/93search/13731response-mask tokens，1未搜索/0环境失败；30候选主复核13retain正确/10reserve/7隔离，最终13RL/68隔离/19拒绝；SUMMARY/ROUND_REPORTS已保存，无未决/待发布积压。batch21自动冻结新100源18143–18311，C1召回session99289运行中。已知18028错答获奖按既有R1隔离，17958非区分共同神经支配E1隔离，18121题面病因与病理gold粒度疑点隔离；未改评分/gold、未新增审核层。下方p071及更早是历史快照，以CURRENT和此段为准，继续至500。

最新实物p071：285条=完整正确260+错答25，未知0，training_ready=false，还需215条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p071/release/rl/train.parquet`实际loader/collator/ID/数量/hash通过。batch20 C1绑定90pass/10重复，C2绑定69pass/12隔离/9拒绝，teacher输入20/20/20/9。首包20真实episode/20actualC3/0no_search/0环境失败，9候选主复核5retain正确/2reserve/2隔离，inc001已发布无积压；第二包生成session75807。首包曾最后一题慢、manifest19/incomplete，保留等待后完整Exit0，无重跑。下面p070及更早段是历史快照；以CURRENT、此段和coordinator最新恢复字段为准，连续500授权有效。

最新实物p070：280条=完整正确255+错答25，未知0，training_ready=false，还需220条完整正确。路径`MedSearch/data/cleaning_v2/rl_to500_20261022_p070/release/rl/train.parquet`，实际loader/collator/ID/数量/hash回读通过。batch19全部100原题完成，74真实episode/101search/13794response-mask tokens，2未搜索/0环境失败；22候选主复核11retaincorrect/5reserve/6quarantine，最终11RL/76隔离/13拒绝；SUMMARY和ROUND_REPORTS已保存，无未决/已通过未发布积压。batch20已自动冻结100新原题源17940–18140，C1召回session19361运行。下方p069及更早段均为历史快照，以CURRENT和本段为准；每100汇报后自动继续到500。

最新实物p069：278条=完整正确253+错答25，未知0，training_ready=false，还需222条完整正确。`MedSearch/data/cleaning_v2/rl_to500_20261022_p069/release/rl/train.parquet`实际loader/collator/ID/数量/hash通过。batch19全部74episode已生成，72actualC3/2no_search/env0；前3包16候选主审9retaincorrect/4reserve/3Q，inc001-inc003已发布，无积压。尾04实际14题已bind4Exit0，Luna审review/input_04.jsonl（SHA7deb35819b50ac568daf9457168616bc5681c93f001b99d8c11f87859bb8c8c3），等raw04后root收口并report_round，再自动freeze batch20。下面p067/p066段为历史接续快照，CURRENT指针及此段为准。不得在每100汇报后停止。

最新实物为p067：271条=完整正确246+明确错答25，未知0，training_ready=false，距离500还需229条完整正确。路径`MedSearch/data/cleaning_v2/rl_to500_20261022_p067/release/rl/train.parquet`，实际loader/collator/纯问题/数量/ID/hash通过。batch19已C1绑定88pass/12重复、C2绑定74pass/13隔离/1拒绝；teacher输入20/20/20/14。首包20生成/19实际窗口/1未搜索/0环境失败，2候选均主复核retain并发布；无已通过未发布积压。第二包生成session71334运行中。以下p066段为上一轮历史收口，不是当前指针；每100题汇报后继续到500，用户无需再说继续。

**当前发布为`rl_to500_20261022_p066`：269条＝完整正确244＋明确错答25，历史未知0，`training_ready=false`。最终500仍需231条完整正确，新增wrong超额保留储备。** 路径为`MedSearch/data/cleaning_v2/rl_to500_20261022_p066/release/rl/train.parquet`；实际loader/collator、数量、唯一ID及hash回读已完成，hash以`CURRENT_RL_RELEASE.json`为准。batch18全部100原题已收口，71真实episode/97搜索/12041response-mask tokens；28候选经Sol复核15retain正确、8reserve、5quarantine；最终分流15RL/72隔离/13拒绝，无未决和已通过未发布项。batch19已自动冻结100新原题，源17756–17939，C1召回session21427运行中。用户再次明确要求自动连续推进，每100题汇报后继续，不结束任务等待“继续”。不能把当前269条称为已满足整体95%正确的最终训练集。

| 已完成新题轮次 | 原题 | 真实7B题数 | 实际搜索 | 新增正确RL |
|---|---:|---:|---:|---:|
| batch04 | 100 | 62 | 87 | 8 |
| batch05 | 100 | 57 | 77 | 8 |
| batch06 | 100 | 55 | 71 | 10 |
| batch07 | 100 | 63 | 83 | 8 |
| batch08 | 100 | 65 | 87 | 8 |
| batch09 | 100 | 70 | 96 | 12 |
| batch10 | 100 | 63 | 86 | 15 |
| batch11 | 100 | 59 | 83 | 8 |
| batch12 | 100 | 61 | 86 | 6 |
| batch13 | 100 | 62 | 80 | 20 |
| batch14 | 100 | 68 | 97 | 9 |
| batch15 | 100 | 62 | 84 | 9 |
| batch16 | 100 | 58 | 78 | 12 |
| batch17 | 100 | 57 | 74 | 12 |

batch07–17各100题均已完整收口，恢复以来累计新增119条。batch17：21候选经Sol复核，12retain正确、9quarantine；全部100题最终分流12RL、70隔离、18拒绝，生成11345response-mask tokens，无检索0、环境失败0。各轮`sol_review/SUMMARY.md`、不可变`decisions_NN.jsonl`及增量发布原件保留，无已通过未发布积压。本轮没有修改SFT或启动训练。历史303的核验已全部关闭，不再重审或恢复撤回项。

当前唯一清洗智能体仍为`/root/rl_luna_20261024`（gpt-6-luna/max）。batch17全部收口后，batch18已冻结100原题、源17559–17754、基线p062254，root执行既有C1近邻召回。上一轮无待发布积压；17458与正式16594同任务，仅新增ID隔离，旧release不改；17480所问机制/gold错位、17522最大期限范围冲突均按现有规则隔离，不新增审核层或规则修订。root执行共用机械命令，Luna仅保存指派C1/C2/C3判断raw；不改内容标准/预算。本批冻结hash`859811323718a3f9cc9184c77030c8e4ec3c179d049f2b66b9333cef9b9de30d`，旧快照不改。每100题汇报后继续至最终500，除用户暂停或实际阻塞外不停；不创建定时任务。最新恢复入口为`MedSearch/data/cleaning_v2/rl_to500_20261022/coordinator_checkpoint.json`。

2026-10-24仅在现有Luna书增加两句执行提示：`one example`不等于答案身份唯一（15617），语义正确但评分拒绝必须隔离（15465）；旧快照不改。规则新hash为`bcdcba943293c80c9a922903204f6273613fdbeb4d7b2e4feca7445a3fb35dd8`，没有新审核层或额外评分调用。

### 历史接续快照：p017与新100题batch04

**当前p017：109条＝完整正确84＋明确错答25；历史未知0，仍training_ready=false。最终500还需391条完整正确，后续wrong超额留reserve。** 尾34题`rl_20261022_legacy_recheck03`：34真实7B、46搜索、6235tokens；15候选→Sol13retain（正确10/错答3）、1政策reserve、1测试范围quarantine，全部已导出并回读。报告见该run的`sol_review/SUMMARY.md`。误重复teacher-bind shard1仅做机械修复（20ID、27重复引用），原轨迹/预算/内容/已发布hash不改，回执`sol_review/bind_replay_repair.json`；共用bind入口加防重复与已审状态保护，未重跑episode或内容审查。唯一Luna获准从真实新题游标`rl_20261022_lunamax_batch03`冻结新100为`rl_20261022_lunamax_batch04`，按当前p017基线继续，不因历史尾批完成而暂停。以下为旧快照。

**当前p015：累计文件130条＝正确74＋明确错答22＋历史未知34；单题已核验合格96，最终500目标未达，training_ready=false。** 第二历史100题`rl_20261022_legacy_recheck02`已收口：98真实7B、132搜索、18560生成tokens；Luna35候选→Sol保留20（正确15/错答5）、政策暂缓10、内容/评分疑点隔离5。所有通过项已发布并真实loader、数量/hash回读。报告`MedSearch/data/cleaning_v2/rl_20261022_legacy_recheck02/sol_review/SUMMARY.md`；唯一Luna已接续准备好的`rl_20261022_legacy_recheck03`尾34题，之后仍须补新100题各轮直到最终500/错答最多25，不在本百题节点暂停。以下为此前过程快照。

**最新：历史首100题现已完成并接续第二个历史100题。** `rl_20261022_legacy_recheck01`：98个真实7B episode、133次搜索、17153生成tokens；Luna38候选，Sol保留29（正确27、错答2）、按本次最终集合约束暂缓5、具体内容/评分疑点隔离4。所有通过项已导出，p010累计210条＝正确59＋错答17＋历史未知134，training_ready=false；真实loader及数量/hash回读通过。报告`MedSearch/data/cleaning_v2/rl_20261022_legacy_recheck01/sol_review/SUMMARY.md`；同一Luna已接续准备好的`rl_20261022_legacy_recheck02`，用户要求每100题汇报、连续至500，未暂停。下文为历史过程快照。

Docker由用户恢复后，历史recheck01包02已绑定Sol结论并导出p007；3条保留、2条暂缓、2条隔离，实际累计252条，完整正确42、明确错答16、历史未知194。真实loader全部252条及数量/hash回读通过，仍training_ready=false。首40题已收口，唯一Luna Max接续剩余60题，当前协调智能体为`/root/rl_luna_continue`。先前暂停由Docker不可用和额度审批失败引起，现已解除，原目标500/错答最多25继续生效。

100原题，70道真实7B、94次搜索、12655生成tokens；Luna16候选，Sol保留10（正确7、错答3），隔离3、暂缓3。4个增量已发布，累计p005回读281条：正确32、错答15、历史未知234；未达整体目标，training_ready=false。报告见`MedSearch/data/cleaning_v2/rl_20261022_lunamax_batch03/sol_review/SUMMARY.md`。单Luna已接续准备好的legacy_recheck01首100题，不等待用户恢复。以下为本任务起点记录，不作为实时库存。

用户明确授权：一个GPT-6 Luna Max清洗智能体，Sol主代理复核，每100道原题汇报，连续推进到最终500条；整体7B终答错误率≤5%，允许核查及替换历史问题项。执行配置见`MedSearch/data/cleaning_v2/rl_to500_20261022/CAMPAIGN.md`。

已派发/root/rl_luna_cleaner（gpt-6-luna/max）清洗新batch03的100题；历史303库存核实69条有真实episode，234条无真实7B episode。复用69条既有Sol内容结论，25条完整正确、12条正常错答可保留；12条已知内容/reward疑点及20条空答/不完整/不确定移出本次主池。当前正式文件271，其中234仍待补首次真实7B核验，不能宣称已满足整体95%正确。

发布p001已真实loader回读271，hash/ID一致；路径和sha见`CURRENT_RL_RELEASE.json`及campaign/STATUS.json。历史首100题已准备为rl_20261022_legacy_recheck01，复用原C1身份、原题/gold、保留原离线query日志，未重跑已有真实episode。新batch03结束后由同一Luna接续历史核查。当前任务仍运行，100题节点是汇报而非暂停点；不得提前以文件达到500而忽略234未知项。


## 2026-10-27 RL batch32 checkpoint

100 reviewed: 8 published, 72 quarantine/reserve, 20 reject. 40 blind 7B episodes; Sol sampled 3 accepted and 2 rejected, all agreed. Current release p110: 401 = 376 correct + 25 wrong; target 500, 99 remaining. SHA256 ed223ac149d499aa99b4c996fa7d47c1d40479615e2a309f98ecdf7cf2d2a41a, readback and real loader verified; training_ready false. Continue batch33. Full records: MedSearch/data/cleaning_v2/rl_20261027_terra_batch32/RUN_REPORT.md.


## 2026-10-27 RL batch33 checkpoint

100 reviewed, assignment counts {'quarantine': 81, 'reject': 14, 'rl': 5}. 43 blind 7B episodes; Sol sampled 3 positives and 2 failures, all agreed. Current release p112: 406 = 381 correct + 25 wrong; target 500, 94 remaining. SHA256 7b040f51fb46e3f4e67bf9910a704f17855448f7469887ab504e8d88efddcbbf; readback and real loader verified. Continue batch34. Full records: MedSearch/data/cleaning_v2/rl_20261027_terra_batch33/RUN_REPORT.md.


## 2026-10-27 RL batch34 checkpoint

100 reviewed, assignment counts {'quarantine': 73, 'reject': 18, 'rl': 9}. 40 blind 7B episodes, one without search; Sol sampled 3 positives and 2 failures, all agreed. Current release p114: 415 = 390 correct + 25 wrong; target 500, 85 remaining. SHA256 5a8b92597dfc2a1a4636480ff61803a84d55c85b9fb1cec922343f5ed828b31f; readback and real loader verified. Continue batch35. Full records: MedSearch/data/cleaning_v2/rl_20261027_terra_batch34/RUN_REPORT.md.


## 2026-10-27 RL batch35 checkpoint

100 reviewed, assignment counts {'quarantine': 72, 'reject': 21, 'rl': 7}. 40 blind 7B episodes; Sol sampled 3 positives and 2 failures, all agreed. Current release p116: 422 = 397 correct + 25 wrong; target 500, 78 remaining. SHA256 18508689ebdc8cc495b169198034910ff60565d900c8a0fa5c3ae2449df57a75; readback and real loader verified. Continue batch36. Full records: MedSearch/data/cleaning_v2/rl_20261027_terra_batch35/RUN_REPORT.md.


## 2026-10-27 RL batch36 paused

User requested pause until 2026-10-28 03:00 Asia/Shanghai. Packet 1 released 4; packet 2 (13) not started. Current p117: 426 = 401 correct + 25 wrong, 74 remain. Release SHA256 238a4a1eeda93d7b18fb9f9e4f9dd570e810ae8ea2e093be531a8ed08591a6f8; real loader and hash verified. Resume checkpoint MedSearch/data/cleaning_v2/rl_20261027_terra_batch36/continuation_checkpoint.json.


## 2026-10-28 RL batch36 checkpoint

100 reviewed: {'reject': 15, 'quarantine': 77, 'rl': 8}; 33 7B episodes, 2 no-search; Sol checked 3 positive and 2 failed. 21062 evidence E1 corrected to E2; answer incomplete remains quarantine. Formal p118: 430 = 405 correct + 25 wrong; 70 remain. SHA256 1fab6d7638adaca386830c61aa18b49f2a5137208df25087b91a19622f11b738, real loader verified. Continue batch37.


## 2026-10-28 RL batch37 checkpoint

100 reviewed: {'quarantine': 78, 'reject': 15, 'rl': 7}; 35 7B episodes; Sol sampled 3 positives and 2 failures. 21188 downgraded E2 to E1 and quarantined; affected scope checked. Formal p120: 437 = 412 correct + 25 wrong; 63 remain. SHA256 95db31ef53ad636a065c59b8c3a3a383f6ece96f3e2ac30df3006b7ccc13f052, real loader verified. Continue batch38.


## 2026-10-28 RL 500 达成

正式 RL 数据 500 条：7B 正确 475、错答 25（5%），累计文件 MedSearch/data/cleaning_v2/rl_to500_20261022_p155/release/rl/train.parquet；SHA256 74d5d8ab06165361af98eeadc1bb4e07279e86a0a0df5f20392c3eb3beb7d1fc，真实 loader 和回读核验通过。batch47 达标后停止，剩余候选未计入；发布器按顺序选中23072而非预期23074，主审补充复核23072后才完成累计导出，原始轨迹及决策保留。未进行 Sol 最终抽查。
# 第二包第三批：100 题收口（2026-10-29）

100 条原始题中：8 条正式发布、70 条隔离、22 条拒绝。C1 通过 83 条；其中 C2 通过 57 条，隔离 21 条，拒绝 5 条。57 条完成各一次真实 7B 轨迹，1 条没有搜索而隔离，其余 56 条完成窗口审阅。已通过项逐包主审、发布并以真实 loader 和 hash 回读；没有待发布项。

第二包前三批分别发布 7、12、8 条，累计 27 条；累计教师答对 27、错答 0。第一包 500 条独立保留。当前累计文件：`MedSearch/data/cleaning_v2/rl_second500_20261028_p015/release/rl/train.parquet`，SHA256 `b3063b9d1082c371f0e2a7f70d635719abb02cd9f613105aecf21482001cfb73`，指针 `MedSearch/data/cleaning_v2/CURRENT_RL_SECOND500_RELEASE.json`。

复核中，23658 因检索只列出多个运动障碍而缺少鉴别关系隔离；23686 因题面未明确选择含钾或不含钾的阴离子间隙公式而隔离；23739 降为 E2，因窗口未提供宠物接触的鉴别关系，但实际检索支持中间关系、答案正确且原评分接受，符合 RL 条件。

按用户此前“先审原始数据三百条”完成此次范围并暂停；可从第二包指针及本批账本继续。未开启定时任务。
