# MedSearch 蒸馏 + GRPO 工程设计与编码任务书

日期：2026-09-20。状态：工程蓝图；具体实施进度以当前代码和审计记录为准。

2026-10-23 最新状态：run_failure_teacher_phase调度复验通过；辅助distillation/teacher_process.py与真实7B短生命周期进程通过。2条合成轨迹518 token，含4096＋500评分形状，BF16冻结，峰值allocated16.126GiB，进程退出0后显存回到评分前；联合CPU测试28 passed。详情docs/audits/teacher_process_acceptance_20261023.md。没有启动真实学生/Ray/FSDP，P6仍待V1的TransferQueue桥接与_step_once调用、actor loss装配及字段传递。当前唯一学生任务为_compute_failure_teacher_scores(self,batch,score_fn)；用户负责核心代码，辅助进程已由主对话完成。下文旧拟建接口以当前实际函数签名为准。

2026-10-21 最新状态：P5真实纯GRPO单步、checkpoint保存、原生resume及一题搜索→答案推理通过，训练/最终重载均退出0。两条训练轨迹均reward=-1，组优势和梯度为0、LoRA参数未改变；这是运行闭环验收，不是非零学习效果。保存频率1；单卡NO_SHARD重载所需CPU暂存补丁保存在patches/verl_fsdp_checkpoint_cpu_load.patch。报告docs/audits/grpo_p5_acceptance_20261021.md，机器证据outputs/distillation/grpo_p5_20261021/P5_ACCEPTANCE.json。P6仍待实现V1循环的串行冻结教师评分、字段传递和actor合并loss接线；已有辅助函数不代表真实训练已调用。以下日期记录保留历史状态。

2026-10-14 实际进度：冷启动已定稿 CE epoch3（checkpoint375）与 CE+Top32 epoch8（checkpoint1000），索引见 `outputs/distillation/FINALIZED_SFT_CHECKPOINTS.json`。按用户授权清除其他本轮SFT权重与非选中版本完整rollout，保留评分、配置、日志和报告。P5入口已配置原生 `model.lora_adapter_path`，两版Hydra配置解析通过；实际Ray/FSDP/vLLM单步、RL checkpoint重载尚未验收。RL输入读取当前正式发布指针，预检时correction04共164题，旧5756题池不作本次默认入口。失败评分、回填、OPD及合并loss函数已有，尚未接入上游训练循环。回到RL后当前唯一学生编码任务是§5 reward契约的第一步：在 `MedSearchAgentLoop.run` 输出被动作解析器接受的 `final_answer`；之后接通reward消费并完成P5。下文“拟建/尚不存在”的描述保留为初稿历史，不代表当前实现状态。

2026-10-14 后续更新：用户授权主对话完成Search-R1动作解析本土化及loop final_answer输出，WSL相关测试16项通过。辅助launcher改用veRL原生naive reward manager，已验收其tool_extra_fields→extra_info传递，无上游补丁；GRPO与不滤全错组配置保持原样。当前唯一学生任务是compute_verifier_outcome_reward读取final_answer，P5真实GPU更新及RL checkpoint保存/重载/动作推理仍待验收。详见docs/audits/rl_next_task_20261014.md；上段loop字段任务为本次更新前状态。

2026-10-20 更新：用户完成reward读取final_answer，实际veRL manager与临时本地HTTP验收通过，相关17项测试通过；当前进入P5单步，不将其视为已完成。首次真实启动发现ppo_mini_batch_size按题数计，辅助脚本修正为1，V1更新阶段乘n=2后仍为两条轨迹。报告docs/audits/reward_acceptance_20261020.md；较长训练仍须等待RL checkpoint保存/重载/动作推理验收。

2026-10-20 资源状态补充：用户说明同卡脑电模型正在训练，主对话已终止本次GRPO driver，未触碰其他训练。P5已加载基座/定稿adapter并建立FSDP，但rollout、更新、保存/重载仍未验收。小并发下KV比例预算仍初始化失败；脚本512MiB显式KV配置未在停止前验收。待GPU资源合适时继续，无定时重试。

2026-09-29 更新：旧 Stage A/B SFT 路线已退役，相关编码、配置和专属数据已清理。当前离线冷启动只使用 `search_distill` 教师轨迹；下文 A/B 表示蒸馏实验档位，不表示旧 SFT Stage A/B。

方法决策以 [DISTILLATION_PLAN.md](DISTILLATION_PLAN.md) 为准。本文件回答“复用什么、改哪里、每个文件写什么、怎么验收”。本轮只创建文档；下列标为新增的代码文件尚不存在，函数名是拟定接口，不是已实现 API。下游 Agent 应逐项指导学生编码，不能一次生成全套工程。

## 1. 工程边界与结论

保留 MedSearch 的检索服务、协议解析、CPU verifier、LoRA 装配和 veRL 训练循环。新增一组离线轨迹/教师评分工具、一个冷启动 Trainer 和一个在线失败 OPD 接入模块。不要再建立另一套 Agent 框架、检索器或 PPO Trainer。

固定约束：Search-R1 7B BF16 教师；3B 学生；冷启动 A/B；K=32、T=1、冷启动 alpha=0.1；最多搜索两次；4096 上下文预算 + 每次生成最多 500；smoke 每题两条、pilot 四条；GRPO 保留全错组；reference KL 关闭；仅失败轨迹 OPD。在线系数 0.001 和独立损失归一化仍是工程建议，需要在接线任务中明确，不冒充已经验证的配置。

最需要先验证的工程风险是**单卡阶段切换**：veRL 存在教师管理器，不代表默认实现支持我们所需的“释放学生资源后加载 7B 教师”。此项应在大规模轨迹生成和训练之前做小样验证。

## 2. 已有代码审计：复用和必要修改

以下路径均相对 `MedSearch/`。本地 veRL 位于 `.deps/verl/`，审计基准为 `23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`。不要混用根目录 `Search-R1/verl/` 的旧实现。

| 文件/组件 | 处理 | 原因及边界 |
|---|---|---|
| `rl/protocol.py` 的 `parse_action()` | 复用本地入口 | 2026-10-14按用户要求本土化Search-R1首个完整动作提取；保留非空检查，忽略动作外文字，不要求唯一标签；无跨仓库运行依赖 |
| `rl/retriever_client.py`、现有 BM25 服务及索引 | 复用 | 固定当前实际服务、语料和 top-k；保留检索来源，不更换检索方案 |
| `tools/run_cpu_verifier_server.py` | 复用，补输入验收 | 当前 CPU BF16、`/verify` 接口已有；验证长答案截断及空 references，不新增 GPU verifier |
| `rl/reward.py` | 小改 | 保留 verifier 阈值及正确性奖励；改为读取模型最终动作，区别环境异常与答错 |
| `rl/medsearch_agent_loop.py` | 先复用，smoke 暴露具体缺陷才小改 | 已有动作截断、每动作 500 与最终 response 截断；验证两次搜索后仍能作答及 mask |
| `tools/evaluate_agent_rollout.py` | 到 P3 教师轨迹生成时再按需改 | 离线评测是独立入口；若复用为教师生成，直接设置 vLLM 单次 `max_tokens=500`，不抽取共同状态层 |
| `sft/train.py` | 复用模型装配，增加分支 | 已有 `load_base_model()`、`prepare_qlora_model()`、`attach_lora()`、`build_training_arguments()`、`run_training()`、`save_adapter()` |
| `sft/data.py` | 只保留 `search_distill` 数据入口 | 直接保留教师轨迹原始 token ID 和动作 mask；旧 SFT 编码已退役 |
| `.deps/verl/verl/trainer/distillation/` | 复用数学实现 | 现有 forward Top-k 与 sampled-token 蒸馏可参考/接入；不能把其 nested tensor/FSDP 接口直接当 HF Trainer 接口 |
| `run_medsearch_dapo_v9_verifier.sh` | 保留历史入口 | 新增一个 GRPO 入口，复用兼容环境变量；不覆盖旧实验，也不复制出多份 A1/A2/B1/B2 脚本 |
| `scripts/patch_verl_for_wsl.py` | 原样保留 | 已有 SHM 兼容补丁；新蒸馏补丁单独管理，不能混入平台兼容脚本 |

源码显示 AgentLoop 后处理先 `_compute_score()`，再 `_compute_teacher_logprobs()`；后者当前不按失败筛选，教师管理器与 rollout 并行生命周期接线。问题不是“框架没有 reward”，而是**缺少批次级失败门控和单卡串行资源阶段**。

## 3. 目标目录

标记：`[改]` 已有文件小改；`[新]` 拟新增；`[用]` 直接复用；`[产物]` 运行时生成。未列出的历史工程保留原状。

```text
MedSearch/
├─ DISTILLATION_PLAN.md                         [改] 方法决策与边界
├─ DISTILLATION_ENGINEERING_PLAN.md             [新] 本工程任务书
├─ configs/
│  ├─ distillation.yaml                        [新] 共享配置及冷启动 A/B 参数
│  └─ medsearch_agent_distill.yaml              [新] 同一 AgentLoop 的新实验参数
├─ distillation/                               [新] 只放蒸馏专有能力
│  ├─ __init__.py                              空包标识，不做模型加载
│  ├─ data.py                                  原始轨迹验证、Dataset、collator
│  ├─ teacher.py                               BF16 教师评分和缓存
│  └─ losses.py                                Top-k 适配与失败 OPD 损失
├─ rl/
│  ├─ protocol.py                              [用]
│  ├─ retriever_client.py                      [用]
│  ├─ medsearch_agent_loop.py                  [用，失败时小改] 异步 veRL 适配
│  ├─ reward.py                                [改] 统一最终回答与异常语义
│  └─ distillation_hook.py                    [新] 批次评分与资源切换接入
├─ sft/
│  ├─ data.py                                  [改] 只保留 search_distill 路由
│  ├─ train.py                                 [改] 选择 Dataset/Trainer
│  └─ distillation_trainer.py                   [新] CE 与 CE+Top-k 共用入口
├─ tools/
│  ├─ evaluate_agent_rollout.py                 [改] 评测、教师轨迹生成/筛选
│  ├─ run_cpu_verifier_server.py                [用]
│  ├─ check_distillation_inputs.py              [新] 数据隔离/tokenizer/配置预检
│  └─ cache_teacher_scores.py                   [新] 离线评分 CLI
├─ scripts/
│  ├─ patch_verl_for_wsl.py                     [用]
│  ├─ patch_verl_for_distillation.py             [新] 仅必要的版本锁定接线补丁
│  └─ run_medsearch_grpo_distill.sh              [新] 一个入口选择纯 GRPO/OPD
├─ tests/
│  ├─ test_distillation_data.py                 [新]
│  ├─ test_distillation_losses.py               [新]
│  ├─ test_distillation_wiring.py               [新]
│  └─ test_agent_loop_smoke.py 等               [改] 扩充既有协议/长度检查
├─ data/
│  ├─ medical_verifiable_v2_20260919/            [产物] Luna 清洗结果及账本
│  └─ distillation/<dataset_version>/           [产物] 教师轨迹与概率缓存
└─ outputs/distillation/<run_id>/               [产物] 配置、日志、评测、ckpt
```

不新建通用 `BaseTeacher`、服务注册中心、训练框架、工作流引擎或数据库。先一个教师、一个学生、一个检索服务；缓存用 JSONL 元数据 + 分片张量文件即可。

## 4. 数据契约：先对齐，再写训练代码

### 4.1 问题池

消费 Luna 的实际产物；若字段名不同，在读取边界映射一次，不要求重做清洗流水线。统一内部字段：`question_id`、`source`、`source_id`、`question`、`references`、`split`。保留原始答案列表。

原计划训练池目标 5,000，实际冻结池见 `docs/audits/distillation_pools_v2_20260926.md`：冷启动候选 1,000、RL train 5,756、固定 dev 396。问题分组归属在教师生成之前固定。原始 gold 只送 verifier，不进入生成或教师概率评分输入。

2026-09-26 实际固定数据池以 `data/distillation/pools_v2_20260926_devclean/manifest.json` 为准：冷启动 1,000、RL train 5,756、dev 396。新增候选 reject/quarantine 不入池；3,486 条来源 gold 是未经医学专家核验的弱标签。新合并版排除了旧 RL train 与 dev 同题改写的 13 条；旧正式 RL 文件保持不变。以上训练池目标数不再作为必须补齐的门槛。

2026-09-29 教师对冷启动 1,000 题各尝试一次。用户随后将**轨迹筛选**改为仅要求合法最终 `<answer>`；verifier 答错、早先无效动作和零次检索不排除。原始尝试与旧严格筛选仍在 `data/distillation/teacher_cold1000_20260929/`，新训练清单在 `data/distillation/teacher_cold1000_20260929_answered/`，有答案 886 条、无答案 114 条。新规则不改变前述题目池 split，也不把教师答案视为医学正确。

### 4.2 轨迹及其 token

一条轨迹至少保存：

| 字段 | 内容 |
|---|---|
| `trajectory_id / question_id / policy_version` | 唯一身份、问题关联、采样策略版本 |
| `prompt_ids / response_ids` | 初始输入及交错的模型动作/环境文本原始 token IDs |
| `response_mask` | 与 response 等长；模型生成 token 为 1，observation/环境提示为 0 |
| `segments` | 每段的类型、response 起止位置、动作种类；必要时保存本轮实际 prefix |
| `searches` | 查询、检索器/语料版本、返回内容哈希；有文档 ID 时原样保存 |
| `final_answer` | 从模型 answer 动作取得，不能从全文最后一个标签猜测 |
| `status / stop_reason` | 正常结束、协议失败、长度终止、检索故障等，彼此可区分 |
| `verifier_prob / acc` | 有效判定结果；服务故障不得伪造为 acc=0 |
| `model_revision / tokenizer_hash / config_hash` | 可重复评分和恢复所需身份 |

保留 token IDs，不把解码文本重新 tokenize 当成原轨迹。labels 在模型生成位置等于 token ID，其余为 -100；因果位置右移一次：位置 t 的 logits 预测 t+1，禁止缓存和 loss 各 shift 一次。

动作预算和序列预算分开：最多两次 search，每次生成最多 500；第二次 observation 后仍可最终作答。无效动作需有限次数上限，不能无限循环。具体环境提示和无效动作上限由共享状态规则统一。

**长度接线验收**：4096 指作答时上下文预算，500 指新生成上限；engine 必须容纳该次调用的输入与生成。veRL `max_response_length` 是整条交错轨迹的存储/训练边界，不能机械设成 500。先尽量通过 observation 限长让完整轨迹落入预算。若必须做滚动裁剪，必须记录每次真实 prefix，并证明 actor 重算 logprob 与 rollout 使用相同可见上下文；普通全轨迹 teacher forcing 不能悄悄多看已被裁掉的历史。未完成该一致性接线前，超长样本应标记为长度失败，不用尾部截断后冒充完整成功轨迹。

### 4.3 教师概率缓存

冷启动 B：仅缓存有效模型目标位置的 `target_positions`、`teacher_topk_ids[N,32]`、`teacher_topk_logprobs[N,32]`。先将师生未映射到真实 tokenizer token（ID 0–151,664）的输出行屏蔽，在共同 token 集合上归一化并取 Top-32；不对 Top-32 再归一化。按 veRL 当前 forward_kl_topk 的计算和后处理适配，不新增 other/tail 桶。在线 OPD 的教师/学生 logprob 也须采用相同支持集；实际权重的 Top-k 行为留待加载后验证。

在线 OPD：只需失败轨迹实际采样 token 的 `teacher_logprobs`，不要求缓存 Top-32，也不需要教师独立生成答案。与 response 对齐，另存 `opd_mask = response_mask × valid_failure`；成功项可填零，但不能凭填零认定评分完成。

缓存 key 至少包含模型 revision、tokenizer、实际输入 token、位置、K/T 和配置哈希；不同学生 policy_version 的在线批次不可因 question_id 相同而复用。先写临时分片，完整后原子替换并登记 manifest；失败分片不能当成功缓存读取。

## 5. 逐文件职责与拟定接口

以下签名表达职责，编码时根据已安装版本调整类型和参数；不得宣称是上游原生接口。

### `rl/medsearch_agent_loop.py`：复用现有 veRL AgentLoop

2026-09-26 用户决定不新建 `episode.py`、`EpisodeState` 或 `generation_budget()`。现有 loop 已在首次闭合动作处截断生成，YAML 显式配置每动作 500；veRL 的 response/model 长度仍由原配置控制。先用假生成和检索响应验证 0/1/2 次搜索、第二次搜索后最终作答、token/mask 对齐；只有复现具体错误才对原 loop 做最小修正。离线评测不要求与训练 loop 共用状态对象。

### `rl/reward.py`：只判定模型回答

- 现有 `compute_verifier_outcome_reward()`：优先使用结构化 `final_answer/status`，维持已采用的 prob>0.5、reward=2*acc-1。
- `extract_final_answer(segments)`：只识别模型段中的合法 answer，防止 observation 中的 answer 标签污染结果。
- `classify_verification(result, episode_status)`：区分正确、有效失败、环境/服务异常；在线异常批次停止更新并保留诊断，不能静默删掉组内某条后重新算 GRPO。

可把这两个小函数放在已有 reward 文件，不建 verifier SDK。冷启动与评测采用相同输入和阈值。有效但未答出/协议错误属于学生失败；真实服务故障是环境异常。

### `distillation/data.py`：轨迹到训练样本

- `validate_trajectory(record)`：检查身份、token/mask 长度、真实 observation、搜索上限及通过条件。
- `encode_distillation_trace(record)`：从原始 token 构建 input_ids、labels、attention_mask、目标位置；不调用旧选择题 encoder。
- `DistillationDataset.__getitem__()`：读取 A 的轨迹，B 额外按 manifest 取缓存；缺少/错版本缓存立即报错。
- `collate_distillation_batch(samples, pad_token_id)`：padding 序列及稀疏 Top-k 目标，生成有效位置 mask；缓存 key 不转为模型输入。

训练数据只读新派生目录的 `accepted.jsonl`（合法最终答案 886 条）；A/B 使用相同 trajectory_id 顺序和预算。原始 verifier 判定及旧严格筛选理由保存在 `prior_filter_status/prior_filter_reasons`，不作为 A/B 训练过滤条件。无合法最终答案的 114 条只保留记录，不重新分配到 RL；token/mask 错位或不可训练 token 仍须阻止接线，不能靠更宽的答案筛选掩盖数据损坏。

### `distillation/teacher.py`：同一评分器服务两种用途

- `load_teacher(local_path, device)`：从固定本地 checkpoint 加载 BF16 教师，冻结参数并设为 eval；评分前向使用 inference_mode，不加载 verifier。
- `score_tokens(model, prefix_ids, target_ids, mode, k=32)`：输入为同 batch 的 `[B, P]`、`[B, T]` LongTensor，当前 sampled 路径返回 `[B, T]` 教师 logprob；本步批内无 padding，attention_mask 全 1。逐微批 teacher forcing，只使用目标之前上下文，不读未来 token/gold；Top-k 路径后续再接。
- `write_score_shard(records, destination)`：分片存张量和位置索引，计算哈希，写完整性标记。
- `validate_score_cache(manifest, expected)`：检查模型/输入/位置/K/T 一致性及 NaN/Inf。
- `main()`：一次任务文件输入、一次结果文件输出。在线先采用独立短生命周期评分进程，以进程退出明确释放教师资源；暂不做常驻教师服务。

复用 Transformers 加载和 veRL 的数学 helper，首版不复制其多教师服务集群。教师前向也会产生 logits 峰值，少量 sampled-token 输出不等于低显存前向；按小微批实测。

### `distillation/losses.py`：只存项目需要的数学适配

- `forward_topk_loss(student_logits, teacher_cache, target_mask)`：用当前 veRL `fsdp/losses.py` 中的全词表 logsumexp/gather 与 KL helper，补 dense batch/位置适配；同时复现上层 `forward_kl_topk` 的 clamp 等实际行为。保持 T=1；不发明尾部处理。
- `failure_opd_loss(model_output, data, config)`：在原 batch 上计算独立失败蒸馏项，不能覆盖 GRPO response_mask。优先复用 veRL sampled reverse-KL estimator + policy-gradient loss 路径，独立加入 GRPO loss。
  - 2026-10-07 当前单项编码任务先实现二维张量核心 `failure_opd_loss(student_log_probs, old_log_probs, teacher_logprobs, response_mask, opd_mask, config)`，返回 `(loss, metrics)`；后续在 actor wrapper 中适配 model_output/TensorDict。首版使用 `kl_penalty(..., kl_penalty="k1")`，将负 KL detach 后按 opd_mask 置零，调用 `compute_policy_loss_vanilla`，传原 response_mask 与 `seq-mean-token-mean`；成功轨迹仍计入有效序列分母、padding 不计入。无 OPD 位置直接返回带学生计算图的零损失。全局有效序列数由后续 microbatch 接线提供，不使用失败条数；OPD 系数在合并 loss 时添加。本任务不启用原生教师管理器、不改原 GRPO loss。
- `grpo_with_failure_opd_loss(...)`：调用原 `ppo_loss` 获得任务项，追加 `opd_coef * opd_loss`，分别返回日志；系数为零时不要求任何教师数据。

**在线实现前必须写清数学定义**：本地 veRL estimator 读取当前 student logprob，PG 路径会 detach 蒸馏 advantage，再通过策略概率比与 clipping 求导。它不是直接对固定采样 token 的 logp-logq 求均值反传，也不自动等于 DGPO。首版建议采用该现成 PG 形式，独立 OPD clipping/归一化明确记录；不把 teacher KL 混入医学正确性 reward，不把两个 advantage 合并后假定等价。

建议 OPD 每条轨迹按有效生成 token 均值，然后除以完整原 batch 条数，成功轨迹贡献零。失败 mask 不能让框架默认按失败 token 总数归一化，也不能在 microbatch 间改变权重。若本地扩展入口无法保持该口径，先提出最小改动与数值对照，不悄悄换算法。该处属于待落实的工程建议。

### `sft/distillation_trainer.py` 与已有 SFT 接线

- `DistillationTrainer.compute_loss(model, inputs, ...)`：剥离缓存元数据；同一次学生前向计算 CE 与可选 Top-k；A 返回 CE，B 返回 CE+0.1*Top-k。
- 两项采用相同 shifted target mask 和 token 平均口径；避免 HF 默认 CE 与自定义 KD 在梯度累积时分母不一致。
- `sft/data.py`：只接受 `search_distill`，直接编码冻结的教师轨迹；旧 Stage A/B 入口已删除。
- `sft/train.py`：配置校验接受新 stage；`build_dataloader()` 选择新 Dataset/collator；`build_trainer()` 选择新 Trainer；其余加载、LoRA、优化器、保存接口复用。
- LoRA adapter、Trainer 可恢复 checkpoint、合并后的推理模型是不同产物。记录基座与 adapter 哈希；RL 加载冷启动 adapter 必须检查非零权重确实生效。现有 veRL 如果不能直接导入该 adapter，则一次性离线合并到新输出目录并验证 logits，再进行 RL LoRA 初始化；禁止不知情地回到原始 3B。

### `rl/distillation_hook.py` 与版本锁定补丁

- `build_failure_batch(batch_meta, data)`：从完整已评分批次取有效失败项，保留 group_id、trajectory_id 和原始索引；不改变 GRPO 组组成。2026-10-01 按实际 V1 接口明确：`batch_meta` 是 `KVBatchMeta`，`data` 是按其 keys 顺序从 TransferQueue 读取的 TensorDict，包含 `uid/prompts/responses/response_mask/extra_fields`；失败读取 `extra_fields.reward_extra_info.acc == 0`，跳过 tag 中 `is_padding=True` 的合成项。有效样本缺少 acc 应报错，不能默认为失败。输出只携带身份与评分 token，不携带 gold。
- `score_failed_batch(batch, config)`：无失败直接返回；有失败则写评分输入、调用教师进程、验证结果并散射回原 batch。
- `prepare_teacher_phase(runtime)` / `restore_student_phase(runtime)`：封装已核实的 rollout sleep、actor/optimizer offload、模型恢复与权重同步调用；这些是拟建包装函数，不是已经存在的 veRL API。
- `attach_teacher_scores(batch, scores)`：生成 teacher_logprobs/opd_mask/scored_mask；保留原 response_mask、old_log_probs、advantages。

接入点（2026-10-01 源码核对）：当前 `trainer.use_v1=true`、`trainer.v1.trainer_mode=sync`，实际在 `ppo/v1/trainer_base.py`，以 `_balance_batch` 之后、`_compute_old_log_prob` 之前作为批次教师 hook 候选位置；此时 reward 已完成，顺序与 padding 已确定。教师阶段结束后恢复学生，才重算旧概率或更新；同组生成到更新前版本不变。原文 `RayPPOTrainer.fit()` 是旧路径示意，不作为当前修改位置。

上游最小改动清单：

1. `verl/trainer/ppo/v1/trainer_base.py`：可选调用评分 hook；关闭此模式时走原路径。当前不修改旧 `ray_trainer.py`。
2. `verl/workers/engine_workers.py`：在实际 actor loss 装配处选择项目 loss（本地已有 `set_loss_fn`，先验证初始化后不会被覆盖）。
3. batch → TensorDict、microbatch 切分/设备转移链路：确保新增分数字段保留，不因字段白名单被丢弃。

不要直接启用原生在线教师管理器后再加载第二套教师。本方案教师分数由 hook 注入；是否借用原生 distillation config 必须避免触发 `Role.TeacherModel` 的独立 GPU 资源申请及自动全轨迹评分。纯 GRPO 不启动教师。

`scripts/patch_verl_for_distillation.py` 仅实施上述确有必要的接线：检查 commit/目标代码唯一匹配，支持重复执行和检查模式，不匹配立即失败，保存补丁摘要。优先使用已存在的 loss 注入能力；没有稳定公开 hook 的地方才用版本锁定补丁。不要复制整份 RayPPOTrainer 或直接修改旧 Search-R1。

### `tools/` 的入口

- `check_distillation_inputs.py`：`check_split_isolation()`、`check_tokenizers()`、`check_config()`、`write_report()`。核对实际 token→ID、special IDs、输出行边界和样例分词；检查 gold 无泄漏。不能仅凭 vocab_size 判兼容。
- `evaluate_agent_rollout.py`：保留 `load_states()/generate_actions()/run_agent()/write_results()`；到 P3 复用为教师生成时，直接在 vLLM `SamplingParams` 限每次 500，增加模型身份、原始轨迹导出与 teacher-generation 模式；每题一次，输出 accepted/rejected。评测模式不筛选掉错误题再算准确率。
- `cache_teacher_scores.py`：`load_accepted_traces()`、`build_score_jobs()`、`main()`；调用 teacher.py，不重复写模型评分逻辑；A 可不生成 Top-k 缓存。

## 6. 配置、输出与运行入口

`configs/distillation.yaml` 只保存这一实验共享内容：模型/revision、数据与语料 manifest、服务地址、预算、K/T/alpha、候选 OPD 系数、种子、缓存路径。正式 epoch/step/eval 频率留空并要求正式运行显式提供；smoke 允许一个更新。避免复制四份几乎相同配置。

`medsearch_agent_distill.yaml` 仍注册原 MedSearchAgentLoop，填搜索上限、每动作 500、observation 限长及明确长度映射。配置解析后把最终值保存，不能仅保留 YAML 默认值。

`run_medsearch_grpo_distill.sh` 复用 V9 的离线运行、SDPA、SHM、单 GPU、日志与显存采样设置；新增明确的 checkpoint 输入和 `opd_coef` 开关。必要变化：关闭 filter_groups、use_kl_loss/use_kl_in_reward；GRPO clipping 两侧对称作为起始建议，不能继承 DAPO 的 0.28 上界而不说明；选无动态组过滤的 reward 接线并做全错组检查。注意 reward manager 名字本身不等于优势算法。

保留中间、最佳、最后 checkpoint；不得沿用 V9 的 `save_freq=-1`。best 建议按固定 dev verifier accuracy 选择（并保留误判审计）；正式频率在 smoke 后确定。滚动删除策略必须保护 best/last，恢复用 checkpoint 要含 optimizer、scheduler、随机数状态和 sampler 进度；只有 adapter 不代表可精确续训。

每次输出：

```text
outputs/distillation/<run_id>/
  resolved_config.yaml       # 完整生效配置
  manifest.json              # 数据/模型/代码/补丁/语料哈希与种子
  train.log / metrics.jsonl
  gpu_memory.csv             # 同时记录阶段名；主机内存与耗时进入 metrics
  rollouts/                  # 可定位单个 group 与 trajectory
  evaluations/               # 固定 dev 的逐题答案和汇总
  checkpoints/               # 中间及可恢复状态
  checkpoint_index.json      # best/last 指向、指标及路径
```

原始教师导出目录包含 `teacher_attempts.jsonl`、旧规则的 `accepted.jsonl/rejected.jsonl`、`manifest.json`；当前冷启动训练读取新派生目录的 `accepted.jsonl` 和其 `manifest.json`，Top-k 缓存后续另存 `scores/*.pt`。这里只加载项目自己生成的可信张量缓存。保留原 verifier 分数与旧筛选理由，不把有答案比例当任务准确率。

Windows 前台命令使用 CMD；实际 GPU 训练沿用已验证的 Linux/容器环境。下游 Agent 先核对容器名、挂载路径及解释器，再给可直接从 CMD 执行的命令，不猜 docker exec 名字，也不把 Bash 多行语法直接交给 CMD。

## 7. 单卡完整时序和失败处理

```text
固定策略版本生成完整组（smoke 2 / pilot 4）
→ CPU verifier 判定全部轨迹
→ 环境异常：保留批次并停止本次更新，排错后恢复
→ 有效批次：保留完整组 reward 与 old-policy 信息
→ 无失败：跳过教师
→ 有失败：学生资源进入可验证的卸载状态
          → BF16 7B 仅评分失败 token
          → 保存结果，教师进程退出
          → 恢复学生/rollout 状态
→ 原完整组 GRPO advantage + 独立失败 OPD
→ 学生更新 → 记录/保存 → 下一策略版本
```

`empty_cache()` 不等于释放仍活着的模型、optimizer 或 vLLM 进程；要核验实际 GPU 占用与驻留状态。若框架 offload 后仍不能容纳教师，报告阻塞点和实测，再讨论进一步串行化；不要预先量化 7B，也不承诺 32GB 一定成功。

全错组：GRPO 答案优势应为零，但 OPD 有有效梯度；混合组：全组 GRPO，只有失败项 OPD；全对组：不调用教师，不得因空 mask 产生 NaN。teacher 评分失败不能以零分继续更新；保存 batch 和失败原因，不自动跳过难题。

## 8. 编码顺序、验收与下游 Agent 工作方式

每次只交给学生一个核心任务。以下是全项目路线，不是要求一次完成全部。

| 阶段 | 学生编写/修改 | 验收证据 | 完成后才进入 |
|---|---|---|---|
| P0 输入冻结 | 预检脚本、配置，验收 Luna 产物 | 实际数量/隔离报告、模型身份、token 映射、依赖版本 | P1 |
| P1 现有 loop 验证 | 显式配置已定长度；具体缺陷才改原 loop | 假生成/检索响应下 0/1/2 次搜索、token/mask 对齐，第二次搜索后能回答；不新增 episode 状态层 | P2 |
| P2 单卡可行性 | teacher sampled 评分、资源 hook 最小接线 | 一小批学生 rollout→判定→7B BF16 评分→恢复学生前向；阶段显存、主机内存和耗时实测 | P3 |
| P3 冷启动数据 | 教师生成/筛选和缓存 CLI | 少量真实题每题一次；accepted 可追溯；无 gold；缓存逐 token 位置核对 | P4 |
| P4 A/B 冷启动 | Dataset、collator、Top-k loss、Trainer | tiny 一步有限梯度；A/B 用相同轨迹；保存/重载一致；真实 3B 单步通过 | P5 |
| P5 GRPO 对照 | 新脚本、SFT checkpoint 导入 | 两个 rollout、完整组优势、无 reference、checkpoint 真正保存；纯 GRPO 不加载教师 | P6 |
| P6 失败 OPD | loss、hook、版本补丁 | 全错/混合/全对三种组的梯度与调用次数；真实单卡完整一个更新通过 | P7 |
| P7 小规模比较 | A0/A1/A2、B0/B1/B2 | 相同 dev/检索/预算记录，保留 best/intermediate/last；再确定正式训练预算 | 后续实验 |

三个新测试文件只验证实际风险：

- `test_distillation_data.py`：gold 泄漏、mask/shift、缓存错位、跨划分重复、teacher 筛选；不写只复述字典赋值的测试。
- `test_distillation_losses.py`：小词表手算与当前 veRL 对照、目标位置 shift、成功项零梯度、全错非零梯度、空失败组有限值、microbatch 拆分前后归一化一致。Top-k 数值比较使用上游相同后处理，不拿完整 KL 作必然相等的断言。
- `test_distillation_wiring.py`：新增 tensor 随 batch 切片正确对齐；原 GRPO mask 不变；teacher 异常阻止更新；alpha/opd_coef 为零时回到对应基线；冷启动 adapter 实际生效。

CPU 假对象测试只能证明协议和接线；显存、CUDA offload、Ray worker 生命周期和真实权重更新必须由 GPU smoke 证明。smoke 不证明准确率提升。

### 下游 Agent 可直接使用的指导约定

1. 先读本文件与方法计划，核对当前代码和学生已完成项；报告本次只做哪一个函数/模块。
2. 解释输入输出与一条样例，给函数签名、伪代码及验收命令；让学生先实现，不直接覆盖答案。
3. 阅读学生实际代码与日志；定位具体错误，必要时提供局部修正。不能凭“跑通了”完成验收。
4. 一个任务通过后记录命令、产物与结果，再给下一个任务；不提前创建所有空文件/TODO。
5. 不能改动已确定的教师、长度、数据隔离、搜索预算，不能暗中启用 DAPO/teacher 答对过滤或新尾部算法。

**P2 当前状态（2026-09-27）：** `score_tokens(..., mode="sampled")` 的因果对齐与共同词表数值测试、`load_teacher(local_path, device)` 的小 Qwen2 本地 BF16 加载/冻结/eval/前向测试均已通过。固定 revision 的真实 7B 权重已下载并核对哈希，在现有 WSL Docker 中 BF16 短动作 sampled 评分通过，见 `docs/audits/searchr1_7b_teacher_smoke_20260927.md`。学生已写 `prepare_teacher_phase` / `restore_student_phase`；真实 3B→CPU→7B 评分→3B 短前向的辅助资源探针通过，见 `docs/audits/phase_switch_smoke_20260927.md`，但该探针模拟 veRL 自动卸载、使用轻量 runtime，未启动 Ray/FSDP/vLLM 或 verifier。按用户决定，Top-k 模式和非 sampled 模式的处理留到 B 档接线。学生真实 rollout→判定→教师评分→恢复学生前向的小批链路仍待验收，P2 尚未全部通过。

## 9. 尚待落实而非阻止本次目录规划的事项

- 候选教师 revision `74d46a43224daa612b5f27174e1c58b35587ee92` 的 tokenizer 对照、权重分片哈希与短动作 BF16 sampled 评分已验收；最终教师选择、实际搜索轨迹和 Top-k 行为仍需核对。Luna 两批审计与最终数据池见 `docs/audits/distillation_pools_v2_20260926.md`；pending 来源 gold 为弱标签，不称医学专家复核。
- 在线 OPD 具体 estimator/clipping/归一化按第 5 节建议明确后实施；不宣称与 DGPO 数值等价。
- 4096 + 500 与 veRL 的 response/model 长度字段须在 P1 用实际配置和 smoke 核对；不再回退 2048，也不为此新增预算状态层。
- RL 导入 SFT adapter 和单卡资源切换须实测；没有证据前不声称已经支持。
- 正式 epoch、训练步数、保存/评测频率在完整 smoke 后确定。

本轮产出为设计文档和源码审计结论，没有新增训练实现或实验指标。
