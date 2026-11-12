# MedAgent 工程进度书

> 维护规则：只把有代码、命令、日志或抓包证据的内容记为完成。设计文档写“准备做什么”，本文件写“实际做了什么”。实验指标继续记录在 `refine-logs/EXPERIMENT_TRACKER.md`。

## 当前状态

- 当前阶段：P3-c，继续 SearchAgent 接线。开发占位默认用 `Qwen2.5-3B-Instruct` + Top32 SFT LoRA `checkpoint-1000`，正式 benchmark 使用之后产出的 RL+OPD 权重。权重由调用方指定，接线共用。
- Top32 SFT 单题真实推理已验收。`SearchAgent` 实际生成一次搜索和一次终答，固定证据进入第二轮，两次都在对应动作闭标签停止。下一项验收是接入真实 BM25 并检查实际投递证据。
- 已完成：固定 DSH 版本、本地开发 runtime、单 Planner smoke、RoleRunner fake backend 边界、Plan/SafetyResult 契约、fake Search → Verify → Retry 闭环。
- 后续集成项：真实 DSH 双角色抓包、错误/挂起行为与线程清理；不在本次单元验收中冒充完成。
- 付费状态：尚未调用真实 DeepSeek API；接入真实凭据前必须提醒用户。
- Evidence 规范化和 retrieve 假响应已验收。2026-11-10 普通回归共 66 项通过，其中 7 项覆盖具体 SearchAgent.run 的 fake 控制流。语义返工最多一次。

## 总进度

| 阶段 | 内容 | 状态 | 主要证据 |
|---|---|---|---|
| P0-a | 固定 DSH 上游版本与职责边界 | 完成 | `UPSTREAM.lock`、`docs/decisions/dsh-boundary-validation.md` |
| P0-b | Python SDK → DSH runtime → 假 DeepSeek 单角色回复 | 完成 | 本机抓包、`turn/end`、usage=17/9/26 |
| P0-c1 | 正式 RoleRunner 的 fake backend 边界 | 完成 | `tests/test_role_runner.py`；9 tests passed |
| P0-c2 | Planner/Verifier 真实 DSH 独立上下文 | 后续集成 | 需要双请求抓包，不由 fake 测试代替 |
| P0-d | 401、503、挂起与真实线程清理 | 后续集成 | 需要真实 DSH 请求次数和进程清理证据 |
| P0-e | 一次短真实 DeepSeek 调用 | 未运行 | 会产生 API 费用，接入前提醒用户 |
| P1-a | Plan/SafetyResult 与 validate_plan | 完成 | `tests/test_contracts.py`；13项新测试，全回归26 passed |
| P1-b | 确定性 preflight | 延后 | 当前无审核后的医学规则集与风险评测，不构建脆弱关键词系统 |
| P2-a | fake Search → Verify → Retry 闭环 | 完成 | tests/test_workflow.py；9项新测试，全回归35 passed |
| P2-b | 预算账本与调用次数约束 | 延后 | 当前不接真实 API、不跑批量实验；保留设计位置，产生实际成本前再实现 |
| P2-c | 公共审计 trace | 完成 | redact、start、append、finish、replay；全量56 tests passed |
| P2-d | 完整 workflow 状态机 | 等待依赖 | DIRECT/EMERGENCY/Generator 具备后再实现；当前不写空迁移表 |
| P3 | retrieval / 本地 SearchAgent / verification | P3-c 当前任务 | Evidence 规范化与检索假响应检查通过；真实 BM25 只读请求待服务可用时记录 |
| P4～P6 | evaluation、V1/V2 实验 | 未开始 | 见工程架构书与实验书 |

## 已完成记录

### 2026-11-11：Top32 SFT 单题 GPU 接线验收通过

- 在 `llm-pytorch` 中执行 `python tmp\docker_medsearch.py --log MedAgent\.runtime\request_action_live_sft_20261111_retry.log env VLLM_WSL2_ENABLE_PIN_MEMORY=1 python MedAgent/.runtime/request_action_live_sft_20261111.py`，退出 0。第一次启动时辅助脚本缺少多进程主入口保护，退出 1。加上保护后重跑同一题，首次错误原日志保存在 `.runtime/request_action_live_sft_20261111.log`。
- 使用本地 Qwen2.5-3B-Instruct 和 Top32 `checkpoint-1000`，LoRA 实物 SHA256 与定稿清单一致。BF16，温度 0，采样 seed 42，最长上下文 2048，每次动作最多 500 tokens，最多生成两次。`gpu_memory_utilization=0.40`，日志记录模型加载占用 5.82 GiB，配置 KV cache 为 6.5 GiB。
- 历史 dev 样例 `medical-o1-verifiable:18322` 的首轮 202 input tokens 与存档逐项一致，真实生成 96 tokens 并保留 `</search>`，耗时 5.685 秒。回填一段存档证据后，第二轮 997 input tokens，真实生成 66 tokens 并保留 `</answer>`，耗时 3.269 秒。含模型启动耗时 82.936 秒，终答非空，与历史终答的文本不同。
- 检索使用固定存档 observation，模型实际生成的搜索词与历史一致。验收范围是本地模型接线、搜索动作解析、证据回填和终答闭标签。机器报告 `.runtime/request_action_live_sft_20261111.json`，真实 BM25 和答案质量待后续验收。

### 2026-11-11：Docker 中验收 `request_action`，未通过输入检查

- 在现有 `llm-pytorch` 容器运行 CPU 验收。Transformers 为 5.10.4，vLLM 为 0.27.1。`python tmp\docker_medsearch.py --log MedAgent\.runtime\request_action_regression_20261111.log env CUDA_VISIBLE_DEVICES= PYTHONPATH=MedAgent python -m unittest discover -s MedAgent/tests` 退出 0，66 项通过。历史 SFT 回放也退出 0，续写前缀字段为 `true`。两项仍用注入回复，覆盖已有控制流。
- 专项探针读取本地 Qwen2.5-3B tokenizer 和 `medical-o1-verifiable:18322` 的历史原始消息。正确首轮模板与历史 token 逐项一致，共 202 tokens。当前直接传字符串只得到 21 tokens，原用户提示未保留。单换消息列表得到 199 tokens，缺助手起始标记。第二轮用当前字符串模板得到 21 tokens，历史对应请求为 998。
- 复验经过完整 `SearchAgent.run()`。`SearchTask.query` 使用原始 SFT 用户消息中的完整搜索提示词和原题，第二轮进入 `request_action` 前仍保留提示词、搜索动作和证据。经当前 `apply_chat_template(context)` 后，两次传给生成替身的输入均为 21 tokens，均缺完整提示词。两轮替身按历史动作返回搜索和终答，复验定位的是实际输入转换，未测模型生成效果。
- 把套模板移入 `run()` 后复验，Docker 中 66 项运行、59 项通过、7 项报错。`__init__` 尚未设置 `self.tokenizer`，所以 `run()` 第 103 行在首次生成前抛出 `AttributeError`。同一 tokenizer 实测 `tokenizer=False` 返回 `BatchEncoding`，正确参数 `tokenize=False` 返回字符串，随后 `encode()` 得到与历史首轮逐项一致的 202 个 token。`request_action` 目前仍在重复套模板。日志为 `.runtime/request_action_recheck_20261111.log` 和 `.runtime/request_action_regression_recheck_20261111.log`。
- 再次复验确认 `run()` 已用 `tokenize=False`，`request_action` 已改用 `encode()`。测试和回放调用方已传入新增的 `tokenizer` 参数。构造函数第 98 行写成 `self.tokenzier`，`run()` 读取 `self.tokenizer`，Docker 原样回归 59 通过、7 个相同 `AttributeError`。用真实 tokenizer 的专项探针仅在测试对象上临时补正确属性后完成两轮回放，首轮 202 个 token 与历史一致，第二轮保留助手续写和证据，实际 997 tokens，历史 998，未达到 token 逐项一致。两次 usage 均与真实输入列表长度一致，历史终答复现。未修改生产代码，未运行 GPU 推理。日志为 `.runtime/request_action_tokenizer_regression_20261111.log` 和 `.runtime/request_action_tokenizer_recheck_20261111.log`。
- 应用户要求修正生产代码的 `self.tokenizer` 拼写并更新分词注释。随后 Docker 原样回归 66 项通过，历史回放和真实 tokenizer 双轮探针均退出 0，无测试临时属性。首轮 202 tokens 与历史逐项一致。第二轮 997 对 998，token 对齐有两处替换，证据回填与历史终答回放通过。差异见 `.runtime/request_action_token_boundary_20261111.log`。Docker `nvidia-smi` 读取时 GPU 利用率 78%，已用 11,005 MiB，本轮未启动真实 SFT 推理。
- 当前 tokenizer 返回 `BatchEncoding`，`TokensPrompt.prompt_token_ids` 应接其 `input_ids` 列表。用记录参数的生成替身复验，当前代码把输入 token 数报成 2。真实 vLLM 的 `SamplingParams(stop=[...])` 默认不保留停止字符串，实际接线需显式保留闭标签。
- 专项命令 `python tmp\docker_medsearch.py --log MedAgent\.runtime\request_action_acceptance_20261111.log env CUDA_VISIBLE_DEVICES= python MedAgent/.runtime/request_action_acceptance_20261111.py` 退出 1。当前修复目标是首轮正确应用模板并展开 token 列表，后续沿同一助手前缀续写。此次只加载 tokenizer 和请求类型，未加载权重，未启动 GPU 推理。用户实现原样保留。

### 2026-11-10：SFT 权重核验与 P3-c CPU 回放

- `FINALIZED_SFT_CHECKPOINTS.json` 的 Top32 项指向 `MedSearch/outputs/distillation/sft_epoch_comparison_20261013/training/checkpoint-1000`。Windows 用 `certutil -hashfile` 回读 adapter 和 config，均与清单一致。adapter SHA256 为 `8f51a6a6929e1d3292d8e2fcaa98269013a23968e966bc3ce1a3067c0c2e5144`，本地基座两个权重分片均在。adapter 配置中的基座路径来自训练容器，后续显式传本机基座目录。
- `python scripts\check_sft_action_replay.py` 绑定定稿 adapter 和历史 rollout 的 hash。样例 `medical-o1-verifiable:18322` 的 2 次动作和 1 次回放检索得到历史终答。原 SFT 第二次输入为 998 tokens，包含第一次生成片段和实际 observation。修复前重建 `<search>` 会丢掉动作前缀，回放字段为 `false`。
- `SearchAgent.run()` 共用一份内部 `context` 做模型续写和结果回传。每轮保留原始回复到已匹配的动作闭标签，搜索时追加实际 observation，作答时返回含最终动作的上下文。两次搜索的回归核对了续写、终答、闭标签后的截断与内部结果。`python -m unittest discover -s tests` 共 66 项通过，历史回放退出 0，续写前缀字段为 `true`。
- 历史 rollout 的 stop 为 `</search>`、`</answer>`，保留闭标签。回放读取已保存 token，替代本轮模型请求与检索请求，未加载 3B 或启动 GPU 任务。真实单动作服务、真实 BM25、正式 Verifier 及端到端三路由仍待验收。
- 开发回放和之后的 SFT 单题 smoke 单列为工程记录。正式 benchmark 等 RL+OPD checkpoint 实际保存并通过重载和动作验收，再冻结权重身份、模板、检索语料、动作预算和独立测试集。

### 2026-09-16：固定并构建 DSH 开发运行时

**做了什么**

- 固定官方 `dsh-v0.1.5-rc.2`，commit `fb2c4b9e698e30edb738bca4cf0618587db7d203`，许可证 MIT。
- 源码保存在被忽略的 `.deps/deepseek-harness/`。
- 官方 PyPI 在该版本只有占位包，GitHub Release 没有匹配 wheel，因此使用同 commit 的 Python SDK 源码和 Node 开发 runtime。
- Windows 单文件 EXE 在官方构建脚本的 production install 阶段因缺少 `lefthook` 失败；Node runtime 已生成并足够完成本地开发 smoke。正式分发问题仍未解决。

**安全边界**

- `dsh_smoke/no_tools.patch.yml` 禁用 `persistent-bash` 与 `persistent-pwsh`。
- 关闭 `session-log-deepseek` 上传。
- `.runtime/effective-config.yml` 已证明上述最终配置实际生效。

### 2026-09-16：跑通一套 Planner 回复链路

实际链路：

```text
Python smoke / 未来 RoleRunner
  → DeepSeekHarness.run(prompt, session_id)
  → Python SDK 启动本地 DSH Node 子进程
  → stdin 发送 JSON-RPC initialize 与 session/prompt
  → DSH 建立 session、写入 user message、启动 agent turn
  → DSH DeepSeek adapter 组装 /chat/completions 请求
  → 本地假 DeepSeek 服务返回流式 SSE
  → DSH 解析 SSE，形成 assistant/message 与 turn/end 事件
  → stdout 返回 JSON-RPC notifications 与 session.status=idle
  → Python SDK 聚合事件，提取 final_response 和 finish_reason
  → Python 校验 Planner JSON，并读取最终 usage
```

**实测结果**

- Planner 输出：`route=SEARCH`，`search_query=aspirin evidence`。
- `finish_reason=completed`。
- 事件包含 `assistant/message` 与 `turn/end`。
- usage：输入 17、输出 9、总计 26 tokens。
- 假服务捕获到 DSH 的实际 messages，目标仅为 `127.0.0.1:8765`。
- 更新后的 CMD 单元测试：4 tests passed。

**这一步证明了什么**

DSH 可以作为角色执行器：接收 Python 提交的角色输入，管理一次模型 turn，调用 provider，解析流式响应，并把文本、事件和 usage 返回给 Python。

**这一步没有证明什么**

- 尚未证明 Planner 与 Verifier 的上下文互不泄漏。
- 尚未证明 401、503、重试和挂起清理符合预算要求。
- 尚未接入真实 DeepSeek，也没有真实费用、延迟或模型质量数据。
- 尚未实现正式 Python Harness、医学状态机、3B 搜索循环或 V1 实验。

### 2026-09-17：正式 RoleRunner fake backend 单元验收

- 学生实现 `medagent/contracts.py`、`prompts.py` 与 `role_runner.py`：支持 Planner/Verifier 独立 system prompt、session/finish/JSON 校验、usage 提取、外层 deadline 和 finally close。
- 教师编写 `tests/test_role_runner.py`，覆盖两个角色成功路径、未知角色、坏 JSON、session 错配、非 completed、backend 异常、缺失 usage 与协作式 fake 超时。
- Windows 项目虚拟环境实测共 13 tests passed：既有 DSH smoke 4 项、新 RoleRunner 9 项；运行时间 0.074s。
- 本次只证明 fake backend 下的 Python 控制边界。超时测试证明抛出 TimeoutError、调用 close 且不返回迟到结果；真实 DSH 关闭后线程/子进程是否退出仍留待集成验证。
- `Usage.known` 已改为必须显式传入，消除了未知 token 被默认标为已知的问题。

### 2026-09-18：Plan 与 SafetyResult 契约验收

- 学生实现 Route、RiskLevel、SafetyResult、Plan 与 validate_plan；覆盖必需字段、枚举、列表、查询和风险下限。
- 教师新增 `tests/test_contracts.py` 13项测试：三路由、所有必需字段、类型、SEARCH/non-SEARCH query、风险降级、紧急一致性、输入不可变和冻结结果。
- 本地 Python 全回归 26 tests passed，0 failed，运行时间 0.068s。Windows interop 本次报 Exec format error，纯 Python 测试改由本地 Python 运行；这不是代码失败，也不代表 Windows DSH 集成已复验。
- 下一项：实现数据驱动的 `preflight()`；规则内容与匹配引擎分离，具体医学规则仍需后续审查和版本化。

### 2026-09-18：确定性关键词预检延后

- 用户决定暂不实现数据驱动关键词 preflight：当前没有经过审核的医学规则集、否定/转述语义处理和风险测试集，继续扩展容易形成复杂且有漏洞的筛选机制。
- SafetyResult/risk_floor 契约保留，未来可由经过验证的安全组件提供；本阶段不宣称已有完整医学安全预检。
- 工程主线进入 fake Search → Verify → 最多一次 Retry → PASS/ABSTAIN 闭环。

### 2026-09-18：fake Search → Verify → Retry 闭环验收

- 学生实现工作流契约与 run_search_cycle()，任务索引采用 0/1；attempts 表示实际搜索调用次数，初值为 0，每次调用后增加。
- 教师新增 tests/test_workflow.py 9项测试，覆盖首次 PASS、首次 ABSTAIN、RETRY 后 PASS/RETRY/ABSTAIN、一次搜索上限、空 retry hint、接口关键字和最大两次调用边界。
- 本地全量回归 35 tests passed，0 failed，运行时间 0.081s。
- 本阶段只验证 fake SearchAgent/Verifier 的控制流，不代表真实 3B 检索质量或 DSH Verifier 集成已完成。
- 下一项唯一任务：实现最小预算账本，先约束 reserve/settle、余额、幂等结算和超限拒绝。

### 2026-09-20：预算账本延后

- 用户决定当前不实现预算账本：现阶段使用 fake backend，尚未产生真实 API 成本，也未开始批量实验。
- `budget.py` 的设计位置保留；接入付费 API、开放多层重试或开始批量评测前重新评估。
- 不把“没有预算控制”描述为已解决；真实调用前仍需人工确认费用边界。
- 下一项唯一任务：定义 TraceEvent，并实现 trace.py 的白名单 redact()；通过后再实现 TraceStore。

### 2026-09-20：TraceEvent 与 redact 验收

- TraceEvent 按用户决定只保留当前字段，不增加 schema_version、event_id、timestamp、duration_ms 或 model_version。
- 教师新增 tests/test_trace.py 5项测试；trace 专项5项通过，全量回归40项通过。
- 项目书和工程书已同步当前字段，模型版本保留在运行 manifest，不在单个事件重复记录。
- 下一项唯一任务：实现 TraceStore.start(manifest)，创建不覆盖的 run 目录并写入 manifest.json。

### 2026-09-20：TraceStore.start 验收

- 学生实现运行目录与 manifest.json 写入；教师补全明确异常类型和同实例重复启动保护。
- start 专项5项通过，全量回归45项通过。
- 下一项唯一任务：实现 TraceStore.append(event)，追加 events.jsonl 并拒绝 run_id 不匹配和重复 sequence。

### 2026-09-22：TraceStore.append 验收

- 学生实现 TraceEvent 追加写入 events.jsonl、run_id 一致性检查和 sequence 去重。
- append 专项4项通过，全量回归49项通过。
- 下一项唯一任务：实现 TraceStore.finish(result)，写入 answer.json 与 summary.json，且不改写 events.jsonl。

### 2026-09-22：TraceStore.finish 验收

- 学生实现 answer.json 与 summary.json 独占写入，并在写入前完成输入检查和两部分序列化。
- finish 专项4项通过，全量回归53项通过；events.jsonl 保持不变。
- 下一项唯一任务：实现离线 replay(path)，读取现有运行产物且不调用模型。

### 2026-09-22：trace replay 验收并进入 workflow

- replay 按用户决定定位为纯离线读取器，不重复执行写入侧已完成的 run_id、sequence 和对象类型校验。
- replay 专项3项通过，全量回归56项通过；公共审计 trace 阶段完成。
- 下一项唯一任务：实现项目书既定 RunState 与合法迁移表，开始正式 workflow 编排。

### 2026-09-22：初期工程汇报与状态标注

- 项目书和工程书新增当前状态矩阵，逐模块标记已实现、部分实现、延后、等待依赖和待实现。
- 新增 `docs/reports/INITIAL_PROGRESS_REPORT_2026-09-22.md`，汇总已有证据、未证明事项和下一阶段。
- 完整 workflow 状态机等待 DIRECT/EMERGENCY/Generator 具备后实现；当前主线进入 P3 retrieval/SearchAgent。

### 2026-09-22：待实现工程规格审查与收敛

- 对照 `../MedSearch/DISTILLATION_ENGINEERING_PLAN.md`，核对现有五个业务文件、检索服务协议与 `MedSearch.rl.protocol.parse_action`；仅修订文档，没有改业务代码、测试或 MedSearch 外部资产。
- 发现并修订：预算延后却强制 invoke、额外 structure 调用、重复证据校验、提前会话/记忆参数、函数数量目标、Generator 无逐函数规格，以及项目书/README 仍派发 P0 的冲突。
- 工程书补齐待实现函数的输入输出、内部步骤、失败边界和最小验收；明确外部错误直接暴露，不追加传输重试、动作修复、备用模型或存储回退。保留语义返工一次、证据来源、超时、额度和评测隔离要求。
- 记录待接线差异：Runner 的 Verifier 仅 PASS/RETRY，正式业务需 ABSTAIN/retry_query_hint；Planner smoke 缺完整 Plan 字段；workflow 的 max_retries 实际为总尝试次数。本轮不改已实现部分。
- 同步 PROJECT_BOOK、README、AGENTS 与实验计划的执行边界；V2 数据规模和研究对照保持原定目标，简化实现手段，不新增实验结果。
- 本次复核命令：在 MedAgent 目录执行 `python -m unittest discover -s tests -v`；实际结果 `Ran 56 tests in 0.111s ... OK`。只证明现有 Python 测试，不代表真实 DSH/GPU 服务已验收。
- 下一项唯一任务：Evidence + normalize_evidence 的纯函数验收。

## DSH 一套回复的具体机制

1. **Python 配置执行器**：`DeepSeekHarness` 接收 provider、model、workspace、`dsh_home`、patch、base URL 和 key。它把 base URL/key放入子进程环境，不在 Python 内直接发模型 HTTP 请求。
2. **启动本地 runtime**：SDK 用 `subprocess.Popen` 启动 DSH CLI。当前开发环境显式使用 Node carrier；生产目标应使用匹配的 runtime 可执行文件。
3. **初始化**：SDK 通过换行分隔的 JSON-RPC 发送 `initialize`，告诉 DSH 当前 workspace、provider、model 和输出上限。
4. **提交 prompt**：`run()` 创建指定 `session_id`，订阅该 session 的通知，再发送 `session/prompt`。文本先标准化为 DSH content block。
5. **确认入队**：SDK 等待带相同 message ID 的 `agent/inbox/spliced`，避免把订阅前或其他 session 的事件误当成本次回复。
6. **模型请求**：DSH agent loop 根据 session 上下文形成 messages；DeepSeek adapter 序列化请求，自动带 `stream=true` 和 `stream_options.include_usage=true`，然后请求 `<baseURL>/chat/completions`。
7. **消费 SSE**：adapter 逐块解析 provider SSE，把文本、结束原因和 usage 翻译成 DSH stream chunks。agent loop 最终落成 `assistant/message`，并以 `turn/end` 标记本轮结果。
8. **返回事件**：DSH 经 stdout 发出 JSON-RPC notifications。SDK reader 线程按 request ID 分发响应，并按 session ID 过滤事件。
9. **判断完成**：SDK一直等到该 session 发出 `session.status=idle`，再从最后一个 `assistant/message` 拼接文本，从最后一个 `turn/end.data.reason.kind` 取得结束原因。
10. **Python 接管业务**：我们的代码再解析 JSON、执行 Planner/Verifier schema 校验、结算预算并决定下一状态。DSH 的 `completed` 只表示执行完成，不表示医学答案正确。
11. **关闭与超时**：正常关闭先发 `shutdown`，随后等待、terminate，必要时 kill。SDK 的 request timeout 不能覆盖 `Session.run()` 整段等待，因此正式 `RoleRunner` 仍要保留 Python 外层 deadline，并拒绝迟到结果。

## P3-c 早期接线记录

2026-09-23 文档决策：Evidence 只保留 text、可选 title 和 score，废弃检索证据 ID、哈希补 ID 和按 ID 去重/冲突规则；原文及实际可见内容随结果传递，具体关联方式待后续接线讨论。任务占用/隔离可用启动时独立随机 run_id，不与检索数据绑定。本次仅统一现行文档，不代表检索实现或新测试已完成，用户的 retrieval.py 草稿保留。

2026-09-24 验收：`normalize_evidence(response)` 的检索测试与现有测试合计 58 项通过；`retrieve` 的假响应检查覆盖单次 POST 请求体、空召回、坏 JSON 和 HTTP 500 区分。固定本地服务未做真实只读请求，P3-b 只记为代码与假响应已验。重定向特殊处理按用户决定后置，若实际服务出现需求再加。

2026-09-24 模型核查（历史记录）：`MedSearch/models` 仅有外部基线；当时扫描本机项目目录未找到 MedSearch 自训权重或 adapter。V9 运行日志显示 `save_freq=-1`，对应运行目录只有日志与 rollout，不能把已有训练记录称为可加载 checkpoint。当时新增 `ScriptedActionRequester` 占位；截至 2026-09-25，它已移到 `tests/test_search_agent_placeholder.py`，生产模块仅保留 ActionResponse 和具体 SearchAgent。占位测试按顺序返回标签文本、用量标 unknown；59 项回归不证明真实 3B 服务或具体 run 循环已验收。

2026-09-25 代码/文档复核：具体 SearchAgent 已调用单参数 `request_action(context)`，复用 MedSearch 解析器并执行检索和回填；成功/空候选两条返回路径均提供 context。此前独立假回复 smoke 已跑通 search→假检索→answer；同级 MedSearch 路径处理后，普通测试命令无需手设 PYTHONPATH，最近 59 项通过。真实 BM25、3B 服务和医学效果未验。

2026-09-26：SearchAgent 的初始 context 只用 query，原题与反馈留在外层任务中；成功和空候选路径的 evidence 已统一为 list。新增 6 项 fake 测试，覆盖 search→回填→answer、两次检索后回答、空召回、非法动作、第三次 search 不检索、检索超时向上传递；全量 65 项通过。workflow 同名 Protocol 尚未按具体类决定收敛。默认 max_tries=3 是动作生成上限（最多 2 次检索），workflow 的 max_retries=2 是外层总尝试次数。精确可见证据、公开留档及真实服务仍待验；context 不能直接写入公开 trace，usage 当前也未累计。

2026-09-26 的下一项计划是补 SearchAgent 的输入与回填，再核验 checkpoint、模板、stop 和单动作请求。接续入口见本文件顶部的当前状态。

## 后续真实 DSH 集成验收（不是当前派工）

正式 backend 接入时，用两个唯一标记运行 Planner 和 Verifier：

- Planner 请求必须包含 Planner 标记。
- Verifier 请求必须包含 Verifier 标记。
- Verifier 请求不得包含 Planner 标记或 Planner 历史。
- 两个角色必须有不同的 system prompt、session ID、结果和结束事件。
- 证据以假服务实际抓到的 messages 为准，不能只检查最终返回文本。

## 更新模板

后续每次完成一个小任务，追加：

```text
日期 / 阶段：
完成内容：
学生实现：
教师验收：
运行命令：
结果与指标：
证据位置：
发现的问题：
下一项唯一任务：
```
