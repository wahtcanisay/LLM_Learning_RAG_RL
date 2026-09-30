# 当前阶段

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

- batch6 的 Huatuo 候选池已冻结到 `MedSearch/data/medical_verifiable_v2_batch6_20260930/`：1,000 题、seed 20261003，精确隔离 reserve 15,146 条，semantic history 266,592 条；`candidate_batch6.jsonl` SHA256 `8f382f583b3017a22b5a64f3686ef8f8b41d47cf37de8fdef41436a62e8992c7`，与 batch5 及历史题面无精确重叠，`training_ready=false`。gold-blind 问题审核已完成，分片 322/351/327，keep 506、uncertain 420、exclude 74；1,000 条 assignment/输出 ID、题面哈希与必需字段独立完整性审计通过（不代表医学答案审核）；语义近重复筛查进行中。

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

- 唯一核心任务：完成 batch5 的 402 条人工证据、答案和推理审核，再做跨池语义近重复复核、历史训练暴露审计及冻结 split 隔离。逐条使用模型当时可见的 `<information>`；verifier 分数与 source-reference overlap 均不作为自动准入或排除条件。
- 完成标准：402 条人工账本 ID 全覆盖、引用逐字可回查 query/Doc 与 `<information>` 边界；完成 batch3/4/旧 SFT/batch5/batch6 跨池去重、历史训练暴露审计和 train/dev/test 隔离；只有通过全部门槛且隔离后才计入 1,000 目标，目标未满不启动正式训练。
- 当前有 304 条独立确认候选（batch3 112、batch4 101、旧 SFT 91），尚未完成全池语义去重与训练 split 隔离，不能称为最终可训练样本；batch5 另有 402 条待人工轨迹审核，batch6 题目盲审 keep 506 条尚未经过答案/检索轨迹复核。

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
