# 回到RL：当前唯一编码任务

日期：2026-10-14。依据DISTILLATION_PLAN.md §7–8，以及DISTILLATION_ENGINEERING_PLAN.md §5的reward契约、§8的P5/P6验收顺序。工程设计与编码任务书当前合在同一文件。

最新状态2026-10-21：P5运行闭环与checkpoint保存/原生resume/一次动作推理通过，报告grpo_p5_acceptance_20261021.md。训练全错组优势为0，参数未改变，不作为非零学习证据。下一项核心编码进入P6：在V1 _balance_batch之后、_compute_old_log_prob之前接串行失败教师评分；7B冻结、退出后恢复3B。后续仍需接actor合并loss和验收新增字段穿过TransferQueue/microbatch，尚不能启动OPD长训。下文旧任务状态均为历史记录。

最新状态2026-10-20：下文学生reward任务已实现并验收，WSL 17项测试通过，临时本地HTTP实测仅消费final_answer。当前任务已进入P5真实GRPO单步与checkpoint验收，尚未通过；详细状态见reward_acceptance_20261020.md。下文2026-10-14缺口为历史记录。

## 已核对的实际进度

- 两版冷启动定稿：CE epoch3、CE+Top32 epoch8，完整checkpoint哈希在清理前后相同；索引FINALIZED_SFT_CHECKPOINTS.json。清理32个本轮中间产物目标，释放787.231MiB，保留两版完整checkpoint、对应dev完整轨迹及历史评分、配置、日志、结论。
- teacher评分、prepare/restore阶段、build_failure_batch、score_failed_batch、attach_teacher_scores、failure_opd_loss、grpo_with_failure_opd_loss已有；在真实veRL训练循环中尚未被调用。
- veRL本地FSDP源码支持lora_adapter_path，通过PeftModel.from_pretrained(...,is_trainable=True)直接加载已有LoRA。脚本已设置原始3B基座和定稿adapter两个路径、q_proj/v_proj、GRPO、不滤全错组、无reference KL、distillation.enabled=false、rollout n2、训练step1、save_freq1。
- 两版配置解析验收通过，记录grpo_p5_config_preflight_20261014.json；当前正式RL发布correction04共164题，读取指针并校验hash。未执行RL训练，不能把配置通过视为P5完成。

## 当前缺口及理由

MedSearchAgentLoop.run的final_answer传递已按用户授权完成，compute_verifier_outcome_reward仍从完整solution_str抽最后一个answer标签。工程书要求只判定模型最终动作，避免检索文本参与答案抽取，当前缺口剩reward消费。1342的离线capture误拒绝已通过本土化解析及删除重复检查修复，未修改历史评分；根因记录见sft_epoch8_quality_20261014/REPORT.md。

上游AgentLoopWorker._compute_score已把output.extra_fields放入tool_extra_fields。当前veRL原生NaiveRewardManager已将该字段合入extra_info；辅助launcher已改为reward.reward_manager.name=naive，不补丁上游DapoRewardManager。algorithm.adv_estimator=grpo、filter_groups.enable=false保持原配置；当前未启用overlong reward，因此没有移除已启用的奖励分量。实际reward函数的字段消费仍待实现。

## Search-R1解析复用核对（历史方案，2026-10-14）

用户要求有闭合动作即使用第一个完整动作，忽略动作外文本，并优先直接复用Search-R1。实际WSL Docker验证：先加载当前`MedSearch/.deps/verl`，再将`Search-R1`加入模块路径，可直接导入`search_r1.llm_agent.generation.LLMGenerationManager`并调用`postprocess_predictions(None, [text])`，没有模型加载或训练。当前默认模块路径尚找不到`search_r1`。官方方法不使用self，返回动作列表和内容列表；空标签会返回answer及空内容，适配时继续保留现有非空检查。命令及输出在`tmp/check_searchr1_parser_import_20261014.py`、`tmp/searchr1_parser_import_20261014.log`。

用户随后明确由主对话负责，并要求本土化以便MedSearch独立发布，因此没有采用上述跨仓库导入方案，也没有新增Search-R1模块路径。

## 已完成：本土化首个完整动作解析

`rl/protocol.py::parse_action`采用Search-R1相同的`re.search`思路，保留现有ParsedAction接口及非空内容检查，移除动作唯一性、额外协议标签数和尾随文本限制；注释标明原方法来源。`tools/evaluate_agent_rollout.py`移除capture模式下重复解码再解析的检查。未改变生成token前缀、mask或logprob的处理，未修改历史dev产物和评分。

新增两个回归检查，修改前实际失败2项；修改后在现有WSL Docker中运行`python -m pytest MedSearch/tests/test_teacher_trace_export.py MedSearch/tests/test_agent_loop_smoke.py -q`，11 passed，只有现有Ray弃用警告。覆盖首个动作、未闭合/空内容、真实Qwen tokenizer的`>.`同token边界、生成IDs不重编码、环境mask、0/1/2次搜索、logprob前缀。日志`tmp/searchr1_protocol_red_20261014.log`、`tmp/searchr1_protocol_green_20261014.log`。Windows仅从项目根运行真实1342 token检查也通过，解析模块不依赖Search-R1目录。

记录final_answer随后也由用户授权主对话实现；当前唯一学生任务是reward消费。

## 已完成：记录final_answer

文件：`MedSearch/rl/medsearch_agent_loop.py`，函数：`MedSearchAgentLoop.run`。

1. 变量初始化处增加final_answer，默认None。
2. 在现有两个合法answer分支中，保存parse_action返回的cur_action.content，与answered=True同时发生。
3. 在返回AgentLoopOutput的extra_fields里增加final_answer。未被接受的answer保留None。

不从runtime_ids或完整response重新搜索答案；检索文本、纠错提示不决定此字段。沿用已有parse_action，不新增状态类、解析器或通用框架。loop输出及原生manager的字段传递已通过验收；reward函数的字段读取待下一项实现。

输入/输出例子：search→observation→合法answer“Beijing”，输出answered=True、final_answer="Beijing"；只有search、非法answer或检索文本内出现answer标签，输出answered=False、final_answer=None。

## 验收

- 0/1/2次搜索后合法作答，final_answer等于被接受的动作内容。
- 未接受的动作和observation中的answer标签不形成final_answer。
- 原prompt/response IDs、response_mask、response_logprobs保持已有行为。
- 主对话已实现并在WSL运行loop、teacher_trace_export、reward_wiring三个文件，16 passed，1条现有Ray弃用警告。新增loop字段检查在修改前5 failed/1 passed，原dapo manager传递检查失败；修改后均通过。日志tmp/loop_final_answer_red_20261014.log、tmp/reward_field_bridge_red_20261014.log、tmp/loop_reward_fields_green_20261014.log。测试使用假生成/检索和评分回调，实际veRL manager及真实tokenizer；未访问verifier服务、未启动Ray/GPU训练。launcher的bash -n通过。

## 当前唯一学生任务：reward读取final_answer

文件`MedSearch/rl/reward.py`，函数`compute_verifier_outcome_reward`。用extra_info中的final_answer替换全文answer正则抽取；None或空答案直接返回现有失败reward；HTTP请求answer使用该内容。targets、HTTP错误处理、verifier阈值及score/acc映射沿用现有实现。只有这一函数，不同步改其他评分入口。

验收：检索文本含正确answer但最终答案为None时不得调用verifier，返回失败；有最终答案时请求体只包含该答案与现有references。该项通过后进入P5真实GRPO单步、save_freq=1、RL checkpoint保存/重载/动作推理，之后才做P6失败OPD上游接线。

真实单卡教师hook的候选位置已核对为 `verl/trainer/ppo/v1/trainer_base.py` 的_balance_batch之后、_compute_old_log_prob之前。该P6接线在P5验收后实施，当前未修改上游核心代码。
