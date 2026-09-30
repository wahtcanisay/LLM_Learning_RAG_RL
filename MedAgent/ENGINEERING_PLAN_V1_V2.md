# MedAgent 工程架构与函数规格：DSH 版 V1 / V2

> 更新：2026-09-25，按当前代码区分已有接口与待完成要求。本文是编码规格，不是实现完成证明。
> 目标与冻结决策见 [PROJECT_BOOK.md](PROJECT_BOOK.md)，实际完成项见 [PROGRESS.md](PROGRESS.md)，实验口径见 [EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)。
> 本轮仅改文档；已实现代码、接口和测试不重构。下列“拟定”接口只在对应任务开始时实现，不提前生成空文件。

## 0. 当前进度与本次边界

| 文件/模块 | 实际状态 | 下一项必要工作 |
|---|---|---|
| `contracts.py` | 已有 Usage/RoleResult、Plan/Safety、Search/Verification/Cycle、TraceEvent/ReplayResult、Evidence、validate_plan | 统一 SearchResult 实际返回类型；新增字段只随消费者加入 |
| `prompts.py` | 已有 planner/verifier smoke system prompt 与 get_system_prompt | 服务接线时补正式字段和动态输入 |
| `role_runner.py` | fake backend 下 run、deadline、extract_usage 已实现 | 正式 DSH factory、双角色抓包、错误与挂起后的退出验证 |
| `workflow.py` | fake Search → Verify → 最多一次返工通过；同名 SearchAgent Protocol 仍在 | 按具体类决定收敛并验收接线；完整三路由等待对应策略和 Generator |
| `trace.py` | redact、TraceStore.start/append/finish、replay 已实现 | 调用方接线；不扩字段、不重写存储或回放器 |
| `retrieval.py` / `search_agent.py` | retrieval 假响应及具体 SearchAgent 基本 fake 循环已验 | 核验真实 BM25、checkpoint 与 3B 单动作服务；随后验精确可见片段 |
| `verification.py` | 未实现；仅有 fake Verifier | 确定性证据检查、正式 Verifier 数据接线 |
| `config.py` | 未实现 | 首个真实服务接入时加载一份配置 |
| `policy.py` | preflight 延后，输出策略待冻结 | 无审核规则不写关键词安全引擎；Generator 前冻结公开输出规则 |
| `budget.py` | 延后 | 付费调用或批量实验前实现最小运行额度；不恢复通用事务账本 |
| `generator.py` | 未实现，等待输出策略 | 冻结自由改写/原文编排选择后，按第 4.8 节实现 |
| `evaluation.py` / `cli.py` | 未实现 | P4 做串行评测和实际可用入口 |
| `memory.py` | V2，未实现 | V1 冻结后按第 5 节推进 |

截至 2026-09-26，已新增 retrieval.py 和 search_agent.py；最近普通回归为 65 tests，OK（2026-09-22 的历史结果为 56，2026-09-25 为 59）。其中 6 项覆盖具体 SearchAgent.run 的 fake 控制流；仍未进行真实 DSH 抓包、模型推理或 GPU 实验。历史单 Planner → DSH → 假服务 smoke 证据见初期汇报。

当前完成的是局部执行边界、fake 控制流和 trace 文件读写。真实 3B/BM25、正式多角色 DSH、完整三路由和医学效果均未验收。总体学习路线仍处阶段 5，MedAgent 子项目当前为 P3，不重新从 P0 派工。

## 1. 对旧设计的审查与取舍

此前架构从 pi 路线收敛为 Python 主导 harness、DSH 执行隔离角色、本地 MedSearch 独立动作循环：业务控制和可替换性由 Python 保持，DSH 不注册检索工具。Evidence 去 ID 是因为当前没有可靠的检索溯源数据库，强造 ID/哈希映射不会增加可信度；保留原文、实际可见片段和运行语料身份更直接。预算、完整状态机与医学预检按真实调用和规则条件后置，先验收 P3 检索循环，不把未来目标误记为已完成。

旧工程书有函数名清单，但尚不足以直接指导实现。主要问题及本轮处理：

| 原问题 | 为什么会制造额外工作 | 本轮规格 |
|---|---|---|
| 状态表写预算延后，调用链却强制 budget.invoke | retrieval 未接通先写预留/结算/重试框架 | P3 本地调用显式限次与超时；付费前补最小额度检查 |
| 固定 51 个函数/方法的目标数量 | 容易为了清单创建转发函数、空模块 | 删除数量目标；按有消费者的功能建函数 |
| 强制 structure 角色，再 check、review、resolve | 一份答案重复处理、多一次模型调用与独立协议 | 默认一次 Verifier 返回审查与声明映射；额外 structure 仅在实测不能满足时另提方案 |
| contracts.validate_search 与 verification.check_evidence 重复检查来源/跨度 | 同一规则维护两份 | HTTP 形状归 retrieval；业务证据关系归 verification |
| 全局 RunContext 提前包含会话摘要、记忆、预算、历史 | 当前单题任务被扩成会话管理平台 | P3 一题一轮外层任务；用户多轮会话及摘要延后到有具体样例时 |
| 通用依赖诊断、多个重试层、自动修复/降级 | 真实配置错误被隐藏，故障路径多于主流程 | 边界报错并保留阶段；当前无自动传输重试、无动作修复调用 |
| generator 在表中但无函数规格 | 执行者自行发明输出策略、兼容层或验证循环 | 补齐职责、函数、准入和验收；未冻结的输出政策不得猜 |
| 旧 workflow 签名/Verifier 协议与现有代码不一致 | 可能另建适配框架或重写已验收闭环 | 第 3 节记录真实接口和集成差异 |
| V2 参数和 memory 类型提前进入 V1 | 为未来图/记忆承担当前复杂度 | V2 启动后才增加参数、字段与命令 |
| README/项目书末尾还要求 P0 | 重复已完成任务 | 同步为 P3 的 Evidence 规范化 |

不是删除所有异常处理。外部输入校验、超时、有限调用、证据来源、未知用量标记、无 gold 泄漏和失败留痕必须保留。简化的是预想中的框架和恢复机制。

## 2. 调用链、复用与建文件时机

当前 P3：
```text
已有 workflow.run_search_cycle
    → SearchAgent.run(task)
        → 本地 3B 单动作请求 → 已有 MedSearch parse_action
        → retrieval.retrieve → observation → 再生成或返回候选
    → EvidenceVerifier.verify(result)
        → 引用检查 → 可选的一次 DSH 语义审查
    → 原 workflow 决定结束或唯一一次返工
```

完整 V1 才增加 Planner 路由、policy/Generator、总额度与 CLI/evaluation 接线。DSH 只执行隔离角色；不给 DSH 注册医学检索工具。业务控制归 workflow，3B 内部动作循环归 search_agent；不得各写一套循环。

| 复用资产 | 实际位置 | 接入要求 |
|---|---|---|
| 搜索动作解析 | `../MedSearch/rl/protocol.py:parse_action` | 只读导入轻量模块，不复制正则、不导入训练框架；在运行环境验证包可见性 |
| 检索 HTTP 协议 | `../Search-R1/search_r1/search/retrieval_server.py:retrieve_endpoint` | 请求 queries/topk/return_scores；响应 result 的单 query 列表 |
| BM25 启动参考 | `../MedSearch/tools/run_medrag_bm25_server_8001_fixed.py` | 只核对部署和语料身份，不在 MedAgent 启动、重建或修改索引 |
| DSH 配置与边界证据 | `UPSTREAM.lock`、`dsh_smoke/no_tools.patch.yml`、`docs/decisions/dsh-boundary-validation.md` | SDK/runtime 同版本；正式代码不导入 smoke 的运行入口 |
| 现有 fake 控制流与 trace | `medagent/workflow.py`、`medagent/trace.py` | 通过已有接口接线，不改为通用状态机/事件总线 |

保留平铺的 `medagent/` 包。P3 先按任务增加 retrieval/search_agent/verification/config；Generator、policy、budget 在其触发条件满足时建；evaluation/cli 在 P4 建；memory 在 P5 建。不新建插件注册中心、通用模型客户端、DI 框架、异常类树或持久化任务队列。

配置只从一份 `configs/v1.toml` 开始，新增字段必须有当前读取者。依赖在真实接入时写 `pyproject.toml`，按现有环境复用；不虚构 DSH 的可安装发行包。safety_rules/pricing/v2 配置各在对应功能启用时添加，不提前放空文件。测试使用已有 unittest，不按旧规划重命名已完成测试。

## 3. 数据流与现有接口约束

### 3.1 已有接口，保持原样

- `RoleRunner(config, backend_factory).run(role, prompt, session_id, timeout_s) -> RoleResult`；factory 当前接收 `config` 与 `system_prompt` 关键字参数。
- `SearchAgent.run(task: SearchTask) -> SearchResult` 与 `Verifier.verify(result: SearchResult) -> VerificationResult` 是 workflow 的调用契约。search_agent.py 已有具体 SearchAgent；workflow.py 仍有同名 Protocol。2026-09-25 决定统一为具体类、不另建 LocalSearchAgent，Protocol 清理尚未落实，本轮只记录差异。
- `run_search_cycle(question, initial_query, searchagent, verifier, max_retries=2)` 中 **max_retries 实际是总尝试次数 1 或 2**，不是额外返工次数。接线时尊重现状，不在本轮改名。
- `SearchTask` 当前字段是 question/query/attempt/verifier_feedback；attempt 为 0/1。现有初次调用反馈为 None，尽管注解为 tuple；新增调用方把 None 当无反馈，不据此宣称整个契约已统一。
- `SearchResult` 当前必填字段为 attempt/candidate_answer/evidence/context，evidence 注解及两条返回路径均为 `list[Evidence]`，context 注解为 `str | None`。`VerificationResult` 当前只有 verdict/failure_codes/retry_query_hint。不要恢复 evidence_ids 或为检索数据造 ID。

真实角色集成前必须消除的协议差异：Planner smoke prompt/Runner 目前只要求 route/search_query，正式 validate_plan 还需要 risk_level/constraints/reason_codes；Verifier Runner 目前只接受 PASS/RETRY，业务闭环还需要 ABSTAIN 和非空 retry_query_hint。届时在已有提示/校验处作最小统一，不能用默认值捏造风险或返工 query，也不另造 legacy/new 两套协议。本轮不改这些已实现代码。

### 3.2 已有 Evidence 与后续证据传递

已有 frozen dataclass 放在 contracts.py：

| 字段 | 含义与要求 |
|---|---|
| text: str | document.contents 原文；规范化阶段不删改、不截断 |
| title: str 或 None | 服务实际返回的 document.title；缺失则为 None，只用于显示，不作唯一标识 |
| score: float 或 None | 服务真实 score，缺失标 None；不是概率 |

`evidence_id`、`source_id`、`corpus_ref` 三个 Evidence 字段及缺 ID 时的内容哈希回退均废弃。按响应顺序保留结果，标题可能重复或缺失，不能拿标题冒充文档 ID。语料身份属于运行配置，记在 manifest；暂不建 url/医学实体/原文 offset 等无可靠来源字段。

当前 SearchResult 用 evidence 保存原文，用 context 保存模型工作上下文；尚无独立 visible_evidence、stop_reason 或 actions 字段。下一步先验证确切回填内容、停止原因和公开动作可被调用方取得，再按 Verifier/trace 的实际需要确定最小承载方式，不强制一次新增四个字段或 Observation 类型。每次 run 的证据必须自包含，不依赖共享可变 ledger；候选证据集合**不冒充已证明的逐声明支持关系**。

context 不是可直接公开的 trace：它包含任务文本、回填正文和候选，应只在受控调用间传递。留档时沿用现有 TraceStore 白名单，公开动作只投影动作类型、query、条数、停止原因与必要用量；原始生成前缀和隐藏推理不得写入。精确可见片段与原文的关联、脱敏和产物接线尚待实现，当前没有 SearchAgent 自动留档。

语义审查时才补 `Claim`（candidate_answer 的 start/end）与获准声明结果；start/end 是 Python 字符串半开区间，切片必须非空且落在原答案中。支持它的证据内容必须随结果传给 Verifier/Generator，具体关联和引用格式留待接线时讨论，不预设 ID、列表序号、索引或标题作为引用键。模型只能选原答案片段，不能把补写的新事实当原声明。查询已召回但未实际给模型看的内容不能用于声称候选已有证据支持。

任务归属由启动时生成的独立随机 run_id（确有独立子任务时才另用 task_id）及本次局部结果维持；调用方按任务传递和核对，不能凭一个随机字符串代替上下文隔离。它们与检索文档无关，不给每条 Evidence 生成随机 ID。已有 run_id、call_ids 和 Trace 实现保留。

UserRequest/RunResult/FinalAnswer/EvalCase 等只在完整编排或评测消费者出现时定义。RunResult 至少保存完成/失败状态、答案、失败阶段、调用数和 known/unknown 用量；不新增庞大的通用 RunContext。同一份 labels 只归 evaluator，禁止传入角色或搜索输入。

## 4. V1 待实现函数规格

以下签名为拟定接口；与已有接口的接线遵循第 3 节。每节同时规定内部动作、失败边界和最小验收。内部小步骤不自动意味着要再拆一个函数。

### 4.1 contracts.py：仅新增实际需要的数据

- `Evidence`：第 3.2 节 dataclass 已有；构造数据的边界校验放在 normalize_evidence，不再为每字段写 validator。
- 后续 SearchResult/VerificationResult 扩充仅随 P3 使用点加入；已实现 validate_plan 不重写。
- 不新增旧清单中的 `validate_search`、`validate_verification`、`validate_answer` 三套公共入口：相应规则分别由 verification 与 generator 在其外部结果入口校验一次。

验收：真实形状的响应按顺序保留原文、可选标题和分数，不依赖 document.id；已有回归保持通过。不写只验证 dataclass 字段赋值的镜像测试。

### 4.2 retrieval.py：一个固定 BM25 服务

**`normalize_evidence(response) -> list[Evidence]`**
- 输入是已经解码的单 query HTTP 响应；只接受 `{"result": [[{"document": {"contents": "...", "title": "..."}, "score": 1.2}]]}`。title 和 score 可缺失；document.id 即使返回也忽略，不检查、不转换、不补造，不猜字段别名。
- 检查 result 恰好一组、document 为对象、contents 为非空字符串；当前固定检索服务返回的可选 title/score 直接保留，不为未出现的类型异常另加检查；`result=[[]]` 返回空列表。
- 按第 3.2 节生成 Evidence，保持排名和原文；缺失或重复 ID、重复标题均不影响结果，不按 ID/标题去重或报冲突。缺必需字段/坏类型明确报错，不能转成空召回。
- 无网络、无写盘、不修改输入；验收覆盖正常/空召回/可选字段缺失/忽略 ID/同名不同正文/坏响应。

**`retrieve(query, config, timeout_s, top_k=3) -> list[Evidence]`**（当前参数顺序）
- 校验非空 query、正整数 top_k 及已配置上限；向配置中的固定 `/retrieve` POST 一次，body 为 `queries=[query], topk=top_k, return_scores=true`。
- 使用明确超时及响应大小上限；服务 URL 来自固定配置，不能让模型提供；调用 normalize_evidence 后最多返回 top_k 条。当前不另写重定向处理，若实际固定服务出现重定向再处理。
- HTTP/超时/JSON 失败向上传递并保留阶段；不重试、不更换后端、不返回伪空列表。空召回是合法结果，与服务故障分开。
- 验收：假 HTTP 捕获到正确 body，一次请求、空响应与 500/坏 JSON 可区分；之后另做一次真实只读请求并记录语料身份。

不建 `register_evidence`、EvidenceLedger 或 ID→Evidence 字典。检索结果用有序列表传递，多轮按实际返回顺序追加；当前不增加去重机制，正文观点是否冲突归语义审查。

### 4.3 config.py：只加载当前依赖

**`load_config(path) -> dict`**
- 用 tomllib 读取一个 TOML，按当前启用模块检查必需项、有限正数上限、URL 和已知字段；缺失/拼错即报错，不自动寻找备用服务。
- P3 仅配置 BM25 URL/语料身份/top_k/query 与响应限长、3B checkpoint/服务/模板身份、输出与上下文上限、超时和搜索次数；3B 服务形状未核验前，不填虚构可运行默认值。
- DSH 接入时再加入锁定版本、runtime/patch 路径和角色输出上限；密钥从环境取，保存的生效配置不含凭据。
- 不建 variant 深合并、热更新、配置注册器或 `check_dependencies` 诊断框架。实际文件缺失在加载点报错，服务可用性由该服务的一次 smoke 证明。

验收：一份最小配置可加载，缺关键字段/负上限/未知字段报错；生效配置可不带密钥写入 manifest。实验基线只用显式少量开关，不复制多份近似配置。

### 4.4 search_agent.py：SearchAgent 单一实现

架构决定（2026-09-25）：统一使用具体类 `SearchAgent`，不新增 LocalSearchAgent。当前具体类已存在，workflow 中同名 Protocol 仍在，待用户接线时收敛；fake agent 可继续提供 run(task)。

当前代码：`SearchAgent(request_action, retrieve, retrieve_config, retrieve_timeout_s)`，`run(task, max_tries=3)`；每次调用 `request_action(context)`，context 只从 task.query 初始化，符合当前只给模型 query 的决定。原题和反馈仍留在外层任务中，不自动拼入模型上下文。默认最多 3 次动作生成、2 次检索，最后一步返回 search 时不执行检索；未解析动作/额度结束均返回空候选，尚无明确停止原因。当前 top_k 固定为 3，usage 未累计，token 限制和标签转义未实现。下面是待完成要求，不能视为现状。

**`request_action(context: str) -> ActionResponse`**（当前注入调用契约）
- 对核验过的本地单动作生成端点请求一次，返回原始生成文本与 Usage；模型/模板/stop/max_new_tokens 和超时由注入 callable 持有的生效配置提供。真实服务若要求 messages，由该请求函数转换，不另维护不一致的三参数接口。
- 实施前先取一份真实请求/响应并冻结端点协议。端点尚不存在时记录依赖阻塞，不导入 veRL 来充当推理服务，不写多提供商兼容客户端。
- 不尝试补标签、重新生成或切备用模型；网络故障与截断分别记录，缺 usage 标 unknown。原始生成只在当前调用内解析，不进入公开 trace。

**回填格式化要求**（尚未实现；需要复用时再提取 format_observation）
- 按排名生成 information 文本，可展示服务返回的标题，不添加证据 ID；只使用已返回正文，转义正文内动作/information 标签，避免它们成为协议分隔符。
- 使用实际模型 tokenizer 限长；先保留完整片段，不够再截尾，记录实际可见片段及截断位置；空召回给明确空结果文本，不捏造证据。
- 保留确切回填文本和实际可见片段，原 Evidence 全文用于核对；承载方式按第 3.2 节随消费者确定，不预先增加 Observation 类型或引用键。Verifier 只据可见片段审查候选，不能把字符数称 token 数，也不能把未经核对的 context 当作可见证据清单。

**`SearchAgent.__init__(request_action, retrieve, retrieve_config, retrieve_timeout_s)`**（当前接口）
- 保存实际依赖与配置，不联网、不加载训练模型；tokenizer 只在实际限长接线时加入。普通构造函数注入足够，不创建后端工厂层。

**`SearchAgent.run(task) -> SearchResult`**
- 接收包含原题、当前 query、attempt 和验证反馈的 SearchTask；模型初始 context 只用 query。原题供外层 Verifier 核对，attempt 供 Python 计数，反馈保留在任务中但不自动扩充模型输入。维护局部 context、有序证据列表、搜索次数与公开动作。
- 调用 request_action 后复用 `MedSearch.rl.protocol.parse_action`。解析规则已有：一个完整非空 search/answer，拒绝额外动作与尾部文本；允许的前缀不进入公开日志。
- search 时执行一次 retrieve、合并证据、追加 observation；answer 时返回候选和实际可见证据。当前先使用最多 2 次检索，检索额度耗尽后仍允许一次 answer-only 生成；因此正常路径最多 3 次动作生成。
- 非法动作或额度耗尽后再 search：以可观察的停止原因结束，不多一次模型“纠错”；承载字段接 trace 时确定。上下文超限同样明确结束，先靠回填预算控制，不写滚动摘要器。
- 无有效候选时 candidate_answer 为空，由 Verifier 转成 ABSTAIN；传输/服务异常抛出给外层作为 FAILED，不能装成医学证据不足。
- 外层返工重新调用 run(task)，新结果证据自包含；旧轮次或其他任务的正文不自动进入新候选的可见证据。完整 V1 额度跨两次 run 累计，不能在返工时重置。

**`SearchAgent.answer_fixed(task) -> SearchResult`**（P4 的 B1 才加）
- 固定 query 执行一次检索，然后同一 3B 做一次 answer-only 生成；复用已有请求/observation/解析函数。返回 search 动作即失败，不偷偷升级为动态搜索。

验收：脚本化假回复覆盖 search→answer、两次 search 后仍能 answer、非法动作、超限、服务失败；核对真实请求次数和传给模型的 observation。真实 3B 单题日志另验，fake 通过不能证明模型会遵循协议。

### 4.5 verification.py：一次硬检查与一次语义审查

**`check_evidence(search) -> tuple[str, ...]`**
- 查候选是否非空、可见片段是否来自本次结果并符合实际展示范围；有 Claim 映射后再检查原答案跨度和声明引用是否来自实际可见正文。具体引用关联及检查方式待接线时确定，不按标题识别证据。
- 返回固定少量失败码，如 EMPTY_ANSWER、INVALID_SPAN；引用相关失败码随实际协议确定，废弃按证据 ID 查存在性/冲突的规则。不做 LLM 调用，也不声称内容来自本次结果就等于语义支持。

**`EvidenceVerifier.__init__(question, runner, config)`**
- 保存当前题和检查模式 structural/semantic；实例按题创建，不共享别题历史；不接 gold。

**`EvidenceVerifier.verify(result) -> VerificationResult`**
- 先 check_evidence。空答案、越界片段或不属于本次可见内容的引用直接 ABSTAIN；不让模型 PASS 覆盖硬错误。正文观点冲突由语义审查判断。
- structural 仅返回结构级结果，必须在运行记录中标 `check_level=structural`；旧 CycleStatus.VERIFIED 在该基线不解释为医学语义验证通过。
- semantic 通过独立 Verifier session 发起一次调用，输入原题、原候选、实际可见证据、公开动作和已有反馈。同一次返回 verdict/failure_codes/retry_query_hint，以及进入 Generator 所需的原答案声明跨度与证据映射；不先发 structure 调用。
- 返回 JSON 在此完成业务校验：PASS 无失败码；RETRY 必须有具体非空 query；ABSTAIN 不带下一查询。语义 PASS 的声明必须通过跨度/引用硬检查；没有获准内容不能进入 Generator。
- 缺证据或可修复支持不足可以建议 RETRY；最终是否还有返工机会只由现有 workflow 判断。Verifier 不读取/消耗返工次数，不执行搜索。
- 坏 JSON/缺字段/服务异常报协议或执行故障，不自动转 PASS，不调用第二个模型修复，不建立独立 `resolve_verdict` 策略层。

本地检索 smoke 可继续用 fake Verifier；真实语义集成前才统一第 3.1 节的 Runner/prompt 差异与新增 Claim 字段。验收：不属于本次可见内容的引用挡住 PASS、同名标题不混淆正文、RETRY query 正确传给旧 workflow、ABSTAIN 可贯通、合法跨度确实来自原答案；模型语义判断质量需独立审核。

### 4.6 role_runner.py 与 prompts.py：只补集成缺口

**`create_backend(*, config, system_prompt)`**（新增）
- 按现有 backend_factory 调用形状接固定 DSH SDK/runtime、patch、home/workspace、provider/model；每次 run 一个 backend，复用既有关闭责任。
- 不并行加另一套 DeepSeek HTTP 客户端、长驻 backend 池、session 缓存或兼容多版本事件解析。
- 验收必须是 Planner/Verifier 两次实际 DSH 请求抓包，验证不同 system/session、无前一角色历史；另验鉴权/服务错误请求次数和挂起后本机线程/子进程退出。

**`build_role_prompt(role, inputs) -> str`**（新增）
- 按该角色固定字段白名单序列化当前输入；Planner 看问题和真实安全输入，Verifier 看候选/可见证据/反馈，Generator 看获准内容。
- 固定 system 与动态内容分开，未知 role/缺必需字段直接报错；不递归扫描全文脱敏，不自动汇总历史，不接 memory_view，V2 时再扩。
- 不传 gold、全量其他角色消息或隐藏推理；文档内指令仅作为证据文本。

已有 get_system_prompt/run/run_with_deadline/extract_usage 不在本轮改。正式业务接入时按第 3.1 节统一字段，避免新增第二层同义 validator。当前 extract_usage 只取最终 assistant/message，不能宣传为多尝试总费用；付费前检查底座是否重试和实际 usage 覆盖范围。若无法观察所有底层尝试，就标费用未知并停止扩大实验，不写“估计已结清”。

### 4.7 policy.py 与 budget.py：有触发条件才实施

**policy.py**

- `preflight(request, rules) -> SafetyResult`：延后到审核规则与否定/转述测试齐全。只实现这套具体规则和其版本，命中返回风险下限；不建可插拔规则引擎。未运行预检不能填一个 LOW 冒充安全通过，未完成前仅支持明确限定的研究 smoke。
- `make_public_context(plan, search, verification) -> dict`：在 Generator 策略冻结后，投影获准原声明、引用和不确定性；弃答/紧急分支不带未经验证候选。
- `fallback_answer(kind, reason_codes) -> FinalAnswer`：仅为已冻结的紧急/弃答提供固定文案和研究用途说明；基础设施失败返回故障状态，不伪装医学结论。不让模型参与回退。
- 原 `resolve_plan` 暂不另建；合法路由由现有 validate_plan 校验，坏 Planner 在 workflow 终止，禁止自动改判 SEARCH。

验收：紧急风险不被降低、公开上下文不含未批准候选、无模型时固定紧急/弃答路径可结束。医学规则召回率需有标注样例单独评测。

**budget.py**（付费调用或批量评测前；P3 单题本地不依赖）

- `check_budget(usage, limits, next_max_tokens, now) -> None`：调用前检查实际已用调用数/搜索数、总 deadline，以及已知 token + 本次最大输入输出是否超限；不足直接停止。单题串行，不预留 reservation，不做锁或事务。
- `record_usage(usage, call_result) -> None`：调用点唯一一次累计尝试数和已知 usage；失败尝试也记次数，缺失标 unknown。若 token 总额是硬限制，unknown 后禁止继续模型调用，不能按零结算。
- 调用方在一个明确位置做前检和事后记录；不做通用 `invoke` 装饰器、幂等结算框架、三类自动重试计数。外层返工次数继续只归 workflow。
- 付费前冻结输入/输出限制、尝试可见性及价格来源；只观测到 token 时只报告 token，不伪造人民币成本。底座内部重试必须禁用或证明纳入总限制，不能叠加 Python 自动重试。

验收：恰好到上限后的下一调用被拒、异常不漏计次数、两次外层搜索共享总量、未知 usage 不算零。以后确有并发预留需求再重新设计，而非先恢复旧五函数账本。

### 4.8 generator.py：补齐遗漏，输出策略先于实现

前置决策：PROJECT_BOOK 第 5.7 节仍未选定“自由改写”或“获准原文编排”。这不是执行者可以自行补的细节。模块开始前必须固定一份合法输入/输出样例、是否可改写、证据内容传递和引用展示格式，以及最终新事实如何审核；两种方案均不要求给检索文档设 ID。没有这些只继续 P3，不建两套后端等选择。

**`generate(public_context, runner, config) -> FinalAnswer`**
- 仅接 make_public_context 的获准字段，通过独立 Generator session 一次调用；不重新检索、不读原始内部轨迹、不替 Verifier 创造声明。
- 返回答案、引用与正常/弃答/紧急标识；紧急内容顺序、用途说明与不确定性按冻结政策执行。

**`validate_output(payload, public_context) -> FinalAnswer`**
- 校验结构、引用属于获准集合、分支必需内容和禁止泄漏字段。
- 若选择原文编排，模型只编排获准原文片段，由这里核对内容来自获准集合；若选择自由改写，必须先明确新事实检查办法与成本，普通 JSON/字符串检查不能证明无新事实。
- 失败直接结束该生成分支；已冻结的紧急/弃答可用固定文案，正常答案验证失败不可原样公开。无循环纠错、自动换模型或悄悄采用另一种政策。

验收：一份获准内容正常输出、伪造引用拒绝、被拒声明不出现、紧急/弃答在生成失败时仍有确定结果。自由改写若未完成支持性验收，不能称正常输出政策通过。

### 4.9 workflow.py 与 trace.py：接线，避免再建一层 Runtime

现有 run_search_cycle 和 trace 函数保持原样。待完整分支具备后新增：

- `run(request, services, config) -> RunResult`：创建本次 trace、局部运行数据和总额度；执行预检/Planner/路由；SEARCH 调现有 cycle，其他路由按冻结政策走 Generator；最终调用 TraceStore.finish。services 是装配好的普通依赖，不是容器框架。
- `transition(context, target, cause) -> TraceEvent`：只有实际出现多分支状态后才加入；用一个允许边表检查前后状态，写现有 TraceEvent，终态不再推进。不写状态机基类、回调注册或恢复引擎。
- 原计划 `finish(context,...)` 暂不单独实现：先在 run 一个出口完成投影、生成、产物保存；只有实际重复才抽函数。

技术故障在最外层记录 stage/error 并终止 FAILED；不把每一层都包 try/except 后返回空值。trace 写失败同样暴露，不另写内存队列/备用日志系统；残留不完整产物由既有 replay 的实际行为呈现，不承诺恢复执行。

验收：SEARCH 接原闭环；DIRECT/EMERGENCY 在其政策完成后分别验；返回最终结果前产物存在且可 replay；故障保留真实原因，不宣称已验证真实远端取消。完整状态图仍是项目目标，不是 P3 必须先填完的空表。

### 4.10 evaluation.py 与 cli.py：只支撑已完成基线

**evaluation.py（P4）**

| 拟定函数 | 输入 → 输出与内部要求 | 最小验收 |
|---|---|---|
| load_cases(path) | JSONL → 带 id/source/split/question/options/gold 的记录；拒绝重复 ID 和缺失必需字段 | 有效小集、重复 ID 报错 |
| make_request(case) | 记录 → 仅 question/允许 options；按白名单建新对象 | gold/qrels/风险真值不进入推理 |
| validate_splits(cases, history_manifest, dedup_report) | 核对现有 ID/规范化/近重复报告及数据用途；返回清单与未覆盖项 | 已知交叉题被发现；无语义报告不得称全面隔离 |
| run_evaluation(cases, config, services) | 同题同顺序逐题调用实际基线，保存含失败的逐题结果 | 一题服务故障仍有记录，不能删除后改分母 |
| score_answer(prediction, gold, task_type) | 选择题 ID/预先冻结 EM → 分数；弃答/执行失败计总体错误 | 小表手算一致；不以宽松 F1 代医学正确 |
| summarize(results, labels) | 汇总答案、完成率、检索数、耗时、已知 token；无标签/用量标 unknown | 无 qrels 不出 Recall/MRR，无安全标注不出安全率 |
| write_report(report, out) | 保存 predictions.jsonl/metrics.json/report.md 与失败样例 | 不覆盖旧实验，保存实际分母/配置身份 |

首轮不实现所有未来指标插件；费用只在使用量/单价足够时计算。实验诊断请求失败保留记录；配置或身份错误在批次前停止，不能把整批配置错当模型正确率。

**cli.py**
- `parse_args(argv)`：用 argparse，仅注册已经有实现的 run/evaluate/replay；check-config 若只是一行加载可直接分支调用，不写诊断平台。V2 命令以后加入。
- `main(argv=None) -> int`：加载配置、直接装配依赖、调用入口；技术失败非零退出，付费端点要求显式 live 选择，无凭据不调用。不得默认退回假模型后把实验标真实。

验收：无参数/错误参数清晰退出；本地 fake 单题产物可 replay；真实 5～10 题 smoke 与模型身份另行记录。当前 CLI 尚未实现，不能把未来命令写成已可运行。

## 5. V2：保留研究目标，缩小实现手段

只在 V1 冻结后新增 memory.py。不改 200 道构建题/每 10 题离线维护/最多 100 条活跃记忆、审核后发布、在线只读、评测不回流这些已确定实验边界；可用题不足如实缩减。M1/M2 是维护策略，B4/B5 是表示/检索差别，不能混为同一变量。

三层 Query/Interaction/Insight 图用普通节点/边记录与 ID 映射即可；小规模向量检索先用数组相似度排序和一跳邻接表。无实测瓶颈不引入图数据库、向量服务、增量索引、后台维护进程或插件系统。

| memory.py 拟定函数 | 基本功能与内部要求 | 失败/验收 |
|---|---|---|
| propose_updates(batch, related, maintainer) | 一批完整 build 公开轨迹 → 带来源、适用条件、反例的候选；本地 Qwen 仅提议 | 坏输出记录失败，不自动激活；验候选能追到真实来源 |
| select_memories(current, approved, policy, build_usage, limits) | 审核/版本过滤后按冻结 M1 或 M2 规则选最多 100 个唯一条目；同分按稳定 ID 顺序 | 不读 eval 访问记录；不足不凑数，未超容量不报淘汰效果 |
| validate_memory(snapshot, approval, limits) | 查节点/边引用、来源、审批、版本、角色、单条长度与总字节；来源图同样计字节 | 悬空边/未审核/超容量拒绝；不做医学“真理校验器” |
| build_snapshot(records, approval, encoder, config, out) | 编码选定条目、形成三层来源关系，调用 validate_memory；写节点/边/向量/ID 对照/manifest | 新目录发布且最后写完整 manifest；中断产物不可加载，不实现断点恢复 |
| load_snapshot(path, expected_hash) | 一次读取和校验完整性/身份，返回只读数据 | 损坏报错停止该记忆运行，不自动降级后冒充 B5 |
| retrieve_memory(query, role, risk, snapshot, encoder, config) | 角色/风险过滤后向量 top-1/2；graph 模式最多一跳，限扩展数并返回来源路径 | 无合格条目返回空；不写访问统计或修改快照 |
| project_memory(hits, role, token_counter, limit) | 按角色挑选允许字段、以完整条目限制 token，保留适用条件/来源 | 无 gold/原始隐藏推理/新医学事实规则；无适配条目则空 |

`evaluate_memory_pairs(control, treatment, labels)` 放在 evaluation.py，按 case/seed/config 对齐结果，输出错→对、对→错和成本；缺失运行显式报告，失败不剔除。不要在 memory.py 再建评测系统。

到 P5 才扩 config、prompt 参数和 workflow 的记忆读取点；trace manifest 记录快照身份，RoleRunner 不管理写入。V2 的数据隔离、来源审批、总字节上限数值和 encoder 身份在第一次 10 题构建前冻结。不存在经验先验保证，不能宣称有记忆一定更准。

## 6. 基线和失败的统一口径

- B0：DeepSeek direct 离线能力参照，不等于部署 DIRECT 政策，不作 B1 检索单因素对照。
- B1：固定一次检索 + 同一 3B 候选；B2：动态搜索但无语义验证返工；B3：语义审查 + 最多一次返工。B1/B2 保留最低结构/引用检查，仅标结构通过。
- B3 无返工消融按现有 workflow 使用总尝试次数 1；允许一次返工使用 2，避免把“retries=0”原样传给当前接口。
- 传输自动重试与动作纠错首版均为 0；语义返工仍最多一次，二者不是同一件事。未来要加恢复实验，应作为显式新变量，不因故障“兜底”悄悄加入。
- 检索空结果可记录并交给后续判定；HTTP 故障为执行失败。模型坏协议必须记录协议停止原因，即使返回空候选交给 Verifier，也不能仅留“证据不足”；当前这项归因尚未实现。未知成本不作 0。
- P3 局部接线不启用未验收的 DIRECT/EMERGENCY，不得因此宣称完整 V1 或医学安全能力已完成。

## 7. 实施顺序和下游执行约定

| 阶段 | 本次只实现的能力 | 进入下一步的证据 |
|---|---|---|
| P3-a 已完成 | Evidence + normalize_evidence | 正常、空、可选字段缺失、忽略 ID、同名不同正文、坏结构的输入输出；按序保留原文 |
| P3-b 代码与假响应已验 | retrieve 固定只读 HTTP | 请求体、空召回/服务错误已用假响应检查；真实端点只读请求待服务可用时记录 |
| P3-c 基本 fake 循环已验 | 具体 SearchAgent 的成功、空召回、限次、坏动作及检索超时已用假服务验收；再接真实服务 | 实际模板/stop/checkpoint、BM25 请求、精确可见片段与一题真实动作日志 |
| P3-d | 正式 DSH factory、统一角色协议、EvidenceVerifier | 独立 messages/清理证据、真实候选/证据审查；付费前额度准入 |
| V1 完整接线 | 冻结 DIRECT/Generator 与安全输入，policy/budget/run | 三路由、有限返工、明确失败和 trace 接线；未满足不能进入正式 V1 |
| P4 | evaluation/cli、B0～B3 | 先 5～10 题 smoke，再冻结集合；失败/成本同口径 |
| P5/P6 | memory 离线构建与配对评测 | 10 题审核后扩大，冻结快照、不回流、真实有/无记忆结果 |

每次只给学生一个核心函数/能力：解释一条输入输出、必要内部步骤、失败行为和验收命令；学生实现后检查代码与日志。不要为了提前满足路线表生成所有 TODO，不代写整套工程。

新增保护分支之前，执行者必须能指出：哪一个真实输入/依赖会触发、错误应在哪层终止、已有边界为什么没覆盖。不能回答就不加。规范外的自动重试、兼容协议、默认修复、备用模型、存储回退都不属于隐含授权需求。

已有回归命令（CMD）：
```cmd
cd /d "D:\code_list\some tricks\LLMLeanring\MedAgent"
python -m unittest discover -s tests -v
```

新测试只补实际行为风险，不为每个函数再写一份同义规格；不把测试数量当工程进度。真实服务测试单独保存请求、版本和输出，不用 fake 结果代替。运行产物沿用 manifest.json/events.jsonl/answer.json/summary.json；模型/代码/配置/语料身份记 manifest，不重复扩 TraceEvent。
