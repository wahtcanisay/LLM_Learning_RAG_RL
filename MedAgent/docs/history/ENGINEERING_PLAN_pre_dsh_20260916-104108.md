> 历史归档，不是当前实施指令。有效入口为 ../../ENGINEERING_PLAN_V1_V2.md（相对路径请以项目根为准）。

# MedAgent Harness：V1 / V2 工程实施规划

## 当前待办：Python → DSH 单次角色执行 smoke（2026-09-15）

**交接给执行模型：当前只完成下面的最小接入验证，不搭建完整 V1/V2。** 当前优先方向是 Python 掌握医学主流程，通过 DSH 官方 Python SDK 复用有限角色执行底座；是否正式采用须以验证结果决定。Pi 仅为备选，本轮不并行接入。README 中旧“Pi 直接调用 BM25 的单 Agent”描述及后文旧工程清单均不能覆盖本待办。

### 目标与边界

验证 Python 能通过 DSH 官方 Python SDK 驱动一次隔离角色调用，取得结果、事件、用量和错误，并可靠终止本地执行。Python 负责 JSON/业务校验与全局调度；不能把底座的“请求已接受”当成“角色已完成”。

- 仅在 MedAgent 内创建最小验证脚本、假服务、测试与结果文档；检查已有文件后复用，不覆盖学生工作。允许为此准备固定且匹配版本 DSH SDK/运行时的隔离本地依赖，记录版本/commit/安装方式，不修改全局配置。
- 先阅读固定版本的官方 Python SDK、profile/patch、模型 provider、重试、会话与取消实现；以实际接口为准，不照抄其他版本参数。复用官方 SDK，不手写第二套通信协议。
- 第一轮只使用官方必要组件，不装社区记忆、多 Agent 或自动继续插件。显式指定隔离的 dsh_home、workspace 和独立角色 session，检查最终生效的 profile/patch，禁用模型可调用的工具、无关上下文注入和自动日志上传；用假服务收到的请求及实际配置证明隔离。
- 不假定 `sdk-minimal` 没有工具：已查官方说明包含 shell，须以固定版本实际组合为准，先配置移除再测试。若无工具角色执行必须大量改造运行时，按下文停止条件报告。SDK 请求超时须显式设置；确认超时是否实际取消底层任务，不能只看到 Python 抛异常就判通过。
- 不启动 MedSearch-R1、不下载模型/语料、不训练、不构建记忆、不修改 ../MedSearch/。3B 的 `<search>/<answer>` 解析和检索循环保持为独立 SearchAgent；本次不转换成 DSH 原生 function calling，也不注册医学检索工具。
- 不实现程序意外退出恢复，不重写 SDK 或建设通用 RPC 框架，不机械创建旧规划全部文件。学生仍是正式业务代码的实现者，本次执行仅限验证所需代码。

### 实施与验收

| 顺序 | 检查 | 方法与通过标准 |
|---|---|---|
| 1 | 单次角色执行 | 本地假模型服务模拟所选 provider 的必要请求/流式响应格式，返回固定 Planner JSON。记录真实请求；Python 收到最终结果并自行校验，模型请求次数符合预期，无工具循环。 |
| 2 | 上下文隔离 | Planner 输入含唯一测试标记；独立 Verifier 会话不含该标记和 Planner 历史。检查假服务实际收到的 messages，而非只看最终文本。 |
| 3 | 事件与用量 | 假服务给固定 token 用量，验证接受/开始/完成/失败的实际事件映射与顺序，准确提取最终结果和 usage、不重复计数。未提供用量须记 unknown，不能记0或猜值。 |
| 4 | 错误与取消 | 分别模拟不可重试错误、暂态错误和延迟挂起；明确底层重试开关与实际请求次数。设置短测试 deadline，超时发送终止并等待有界确认；迟到结果不能被提交为成功，清理测试进程。取消失败如实记录，必要的进程清理不冒充正常 abort 通过。 |
| 5 | 一次真实 smoke | 前四项通过后，用既有凭据进行一次短 DeepSeek 结构化角色调用，显式限定输入/输出与超时，记录实际模型ID、事件、usage和费用依据。不读出或打印密钥；没有可用凭据则交付前四项并注明真实项未完成，不虚构通过。 |

超时验证只证明本地生命周期受控，不证明远端取消或停止计费。真实 smoke 若失败即记录原因，不自动扩大付费重试。模型隐藏推理与密钥不进入普通交接日志；假服务抓包只保存合成测试输入。

### 交付与决策

- 提交最小可运行验证文件，以及 `docs/decisions/dsh-boundary-validation.md`：固定版本、依赖、可从 CMD 粘贴的命令（必要时调用 Git Bash）、配置、四项测试和真实 smoke 的逐项状态、脱敏日志位置、实际请求次数和已知限制。
- 通过：给出一个薄角色执行适配器所需的明确边界，随后再修订正式工程清单；不在本次继续实现完整 Harness。
- 小缺口：列出需要的最小扩展、语言和原因，能在本验证范围内修复的就验证后报告。
- 首轮工作量以1～2小时最小验证为目标；普通安装或配置问题先定位修复，不因一次报错就换框架。
- 需要大量底座改造：停止扩展，报告具体阻塞与证据。后续可用同一验收标准验证 Pi，但当前交接只执行 DSH smoke，不同时拉起两套底座或自行转纯 Python 重写。
- 本待办未执行前，所有验收项均为 TODO。保留 V1 的医学功能/验收目标，以及 V2 的200道构建题、100条记忆上限、每10道离线维护与隔离评测设计。

官方阅读入口：
- https://github.com/deepseek-ai/deepseek-harness
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/README.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md


> 日期：2026-09-13。依据：用户粘贴的《MedAgent Harness 项目书》设计主干 v0.1，并核对本地同名架构书及现有 MedSearch 接口。  
> 状态：**候选工程细化，供学生实现；未实现、未接真实模型、未产生实验结果。**  
> 历史说明：原版仅交付规划。当前优先执行开头的 DSH smoke；以下文件/函数数量为待重审的候选清单，不能据此直接创建整套 Python 底层运行时。

## 评审后约定（2026-09-13）

- **接口细节留待对应模块实现时补齐。** 当前清单用于划分职责，并非全部可直接照抄的最终签名。已知待补项包括：`memory.retrieval.retrieve()` 如何获得 `encoder`，以及四个角色的 `build_request()` 如何接收 `MemoryView`。实现到这些模块时统一确定参数传递方式并同步签名；本轮不增加文件或函数。
- **模型服务方案延后到 V1-07 再规划。** 届时结合实际 checkpoint、运行环境和学生需求，确定服务入口、启动命令、请求响应契约及相关文件。本文的本地 endpoint 是候选接口占位，不代表已经选定服务方案；这项延后不阻塞前面的假对象与状态机学习。
- **DIRECT 与 Generator 的收窄方案尚未批准。** 下文所有“DIRECT 只用固定模板”“Generator 只选择声明 ID 并渲染原文”的函数、Schema、测试和基线要求，均依赖这一候选方案；不能视为已冻结需求。项目书 5.3、5.7 已补充解释和待讨论标注，到对应模块设计时再决定并同步本文。
- 本轮仅修正文档说明与函数表格排版，不实现代码，不冻结上述待讨论方案。

## 1. 范围、版本和已有事实

- **V1：无长期记忆的真实闭环。** 把 V0 的类型、假对象、状态机与回放作为 V1 的首个实施里程碑；完成 B0～B3。
- **V2：冻结记忆的只读增强。** 在 V1 同一套代码中增加 G-Memory-lite 的只读加载、向量召回、一跳图扩展、角色投影，完成 B3/B4/B5 配对实验。
- **V3 才做在线更新。** V2 不实现轨迹自动提炼、候选激活、在线晋升、自动撤销、权重训练或错误记忆恢复学习。V2 只支持选定/关闭经过离线审核的快照，不能宣传“自动自进化”。
- 两类 LLM 保持不变：DeepSeek 的 Planner/Verifier/Generator 三个独立逻辑角色，以及本地 MedSearch-R1 SearchAgent。Embedding encoder 是非生成式组件，不能借此加入第三个推理 LLM。
- 项目根学习进度显示主线已到 MedSearch-R1 阶段；本规划不把它重置为 MedRAG 第一课，也不把 Harness 规划完成当作任何训练阶段验收。

已读到的实际代码与需要接通的位置：

| 已有位置 | 已观察到的事实 | 本规划中的处理 |
|---|---|---|
| `MedSearch/rl/retriever_client.py` | POST `queries=[query]`、`topk`、`return_scores`；读取 `result[0]` 的 `document.contents` | `tools/retrieval.py` 对接该线协议，保留原始 ID/来源并强化预算；不复制索引或训练代码 |
| `MedSearch/rl/protocol.py` | 只接受一个完整非空 search/answer，动作后不允许额外文本 | `models/medsearch.py` 保持协议兼容，用共用测试样例证明；前缀不进入普通日志 |
| `MedSearch/rl/medsearch_agent_loop.py` | 依赖 veRL 的生成/训练上下文 | 不直接作为独立推理 HTTP 服务调用；先验证一个选定 checkpoint 的本地生成服务接口，再由 Harness 驱动动作循环 |
| `MedAgent/README.md`、`AGENTS.md` | 仍包含旧 Pi 单 Agent 方案；`.pi/extensions/` 已存在 | 此处只记录差异，不安装 Pi 或改写旧文件；实现开始前冻结后端决定 |
| 根 `STUDY_PROGRESS.md` | checkpoint 历史存在删除与初始化调整记录 | 配置不得写死旧 Stage-B adapter；必须核实实际可用的 checkpoint 和版本 |

架构书里的 DeepSeek 产品名与 `deepseek-flash` 当作**指定的接入目标**，不是本次已核实的当前 API 事实。真实接入前核验官方模型 ID、能力、版本与价格；没有版本信息时记录返回 ID 和调用日期、注明无法完全锁定版本。不得自动换成其他模型。

## 2. 规划的粒度和统计口径

1. 函数数包含顶层函数、显式类方法、Protocol 方法和测试 fixture；数据类自动生成的构造器、导入和常量不计。类不额外拆文件。
2. 每项输入/输出类型在第 5 节归属明确。表中的签名是职责边界；学生实现时可按真实复杂度抽取私有小函数，变更后同步清单，不为凑数量拆函数。
3. V2 复用 V1 文件，只增加必要文件；“修改已有函数”不重复计新增函数。代码包 `__init__.py` 均为 0 函数。
4. 目录数不含 MedAgent 根本身，包含 `src` 等中间目录。现有遗留 `.pi/`、`.pi/extensions/`、空 `scripts/` 不纳入目标结构，保留原状；本规划不要求删除它们。
5. 正式数据、权重、索引和每次实验自动生成的文件没有固定数量，单独列运行产物模板，不混入手写源文件数量。V1 暂不新建 `memory/`、`docs/references/` 或逐角色子包。
6. 同名 I/O 函数可在实现时写为 `async def`，表中省略 async 只展示参数和结果类型。建议 Runtime 与 HTTP adapter 使用同一 async 调用链，`asyncio` 负责 deadline/cancellation，避免无法终止的后台线程；这是待冻结的接口细化，不能混用同步/异步调用而漏 await。

## 3. 目录、文件和函数数量总览

下表由本文逐文件清单自动统计；“函数”包括测试。仅新增本文，表内其他新文件都留给学生后续创建。

| 范围 | 目录数（不含根） | 固定文件数 | 源码函数/方法 | 测试函数/fixture | 函数合计 |
|---|---:|---:|---:|---:|---:|
| V1 累计 | 19 | 62 | 80 | 38 | 118 |
| V2 新增 | 1 | 12 | 17 | 9 | 26 |
| V2 累计 | 20 | 74 | 97 | 47 | 144 |

### 3.1 目标目录的直接文件数

直接文件数不包括其子目录里的文件。V2 为累计数。

| 目录 | V1 直接文件数 | V2 新增 | V2 累计 |
|---|---:|---:|---:|
| `./` | 6 | 0 | 6 |
| `configs/` | 4 | 2 | 6 |
| `docs/` | 0 | 0 | 0 |
| `docs/decisions/` | 1 | 1 | 2 |
| `docs/experiments/` | 1 | 0 | 1 |
| `docs/protocols/` | 1 | 0 | 1 |
| `experiments/` | 1 | 0 | 1 |
| `src/` | 0 | 0 | 0 |
| `src/medagent/` | 4 | 0 | 4 |
| `src/medagent/agents/` | 5 | 0 | 5 |
| `src/medagent/evaluation/` | 5 | 1 | 6 |
| `src/medagent/memory/` | 0 | 5 | 5 |
| `src/medagent/models/` | 4 | 0 | 4 |
| `src/medagent/policies/` | 3 | 0 | 3 |
| `src/medagent/runtime/` | 4 | 0 | 4 |
| `src/medagent/tools/` | 3 | 0 | 3 |
| `src/medagent/tracing/` | 4 | 0 | 4 |
| `tests/` | 1 | 0 | 1 |
| `tests/fixtures/` | 5 | 1 | 6 |
| `tests/integration/` | 3 | 1 | 4 |
| `tests/unit/` | 7 | 1 | 8 |


### 3.2 文件布局

每行是相对 `MedAgent/` 的完整路径。`[V2+]` 表示 V2 才新建；无需建立 `v1/`、`v2/` 两套代码。

```text
.gitignore
AGENTS.md
ENGINEERING_PLAN_V1_V2.md
PROJECT_BOOK.md
README.md
configs/baselines.toml
configs/memory_baselines.toml  [V2+]
configs/pricing.json
configs/safety_rules.json
configs/v1.toml
configs/v2.toml  [V2+]
docs/decisions/0001-harness-boundaries.md
docs/decisions/0002-frozen-memory.md  [V2+]
docs/experiments/protocol.md
docs/protocols/contracts.md
experiments/README.md
pyproject.toml
src/medagent/__init__.py
src/medagent/agents/__init__.py
src/medagent/agents/generator.py
src/medagent/agents/planner.py
src/medagent/agents/search.py
src/medagent/agents/verifier.py
src/medagent/cli.py
src/medagent/config.py
src/medagent/evaluation/__init__.py
src/medagent/evaluation/dataset.py
src/medagent/evaluation/memory_metrics.py  [V2+]
src/medagent/evaluation/metrics.py
src/medagent/evaluation/report.py
src/medagent/evaluation/runner.py
src/medagent/memory/__init__.py  [V2+]
src/medagent/memory/projection.py  [V2+]
src/medagent/memory/retrieval.py  [V2+]
src/medagent/memory/schema.py  [V2+]
src/medagent/memory/snapshot.py  [V2+]
src/medagent/models/__init__.py
src/medagent/models/base.py
src/medagent/models/deepseek.py
src/medagent/models/medsearch.py
src/medagent/policies/__init__.py
src/medagent/policies/output.py
src/medagent/policies/safety.py
src/medagent/runtime/__init__.py
src/medagent/runtime/budget.py
src/medagent/runtime/engine.py
src/medagent/runtime/state.py
src/medagent/schema.py
src/medagent/tools/__init__.py
src/medagent/tools/registry.py
src/medagent/tools/retrieval.py
src/medagent/tracing/__init__.py
src/medagent/tracing/redact.py
src/medagent/tracing/replay.py
src/medagent/tracing/store.py
tests/conftest.py
tests/fixtures/memory_snapshot.json  [V2+]
tests/fixtures/model_responses.json
tests/fixtures/qrels.json
tests/fixtures/questions.jsonl
tests/fixtures/safety_cases.jsonl
tests/fixtures/tool_responses.json
tests/integration/test_adapters.py
tests/integration/test_baselines.py
tests/integration/test_memory_baselines.py  [V2+]
tests/integration/test_workflows.py
tests/unit/test_agents.py
tests/unit/test_budget.py
tests/unit/test_memory.py  [V2+]
tests/unit/test_policies.py
tests/unit/test_schema.py
tests/unit/test_state.py
tests/unit/test_tools.py
tests/unit/test_trace_metrics.py
```

## 4. 非函数文件与职责

这些文件均为 0 函数；配置和 fixture 不需要额外自建解析函数，复用下文已列入口。

### 4.1 V1

| 文件 | 职责 |
|---|---|
| `PROJECT_BOOK.md` | 既有架构主干，保持原文；函数 0。 |
| `AGENTS.md` | 既有协作边界；实现前协调旧 Pi 字样与新主干，不在本次改写；函数 0。 |
| `README.md` | 既有文件；后续更新为安装/依赖验收/命令/边界/实测结果导航；函数 0。 |
| `ENGINEERING_PLAN_V1_V2.md` | 本次唯一新增交付；候选工程细化，含所有文件、函数、验收与决策差异；函数 0。 |
| `pyproject.toml` | 包定义、Python 下限、依赖、medagent CLI entrypoint 和 pytest live marker；版本待实际安装锁定；函数 0。 |
| `.gitignore` | 排除密钥、runs、权重、索引、向量生成物与本地环境；不排除小型审核 fixture/配置；函数 0。 |
| `configs/v1.toml` | 服务固定地址、模型身份、只读资产引用、生成参数、预算、输出目录、seed 和隐私策略；不含密钥。 |
| `configs/baselines.toml` | B0～B3 差分开关，以及 B3 无返工诊断开关；不复制四份完整配置。 |
| `configs/safety_rules.json` | 有版本的紧急信号、风险特征、DIRECT 允许模板、reason_codes 和固定输出模板。 |
| `configs/pricing.json` | 手工核对后的有效日期、来源、币种、输入/缓存输入/输出单价；未核实则空，费用 unknown。 |
| `docs/decisions/0001-harness-boundaries.md` | 后续冻结 DSH 底座/Python 业务流程边界（Pi 备选）、异步接口、基线最低检查、DIRECT 范围与 Generator 输出白名单等候选决定。 |
| `docs/protocols/contracts.md` | 将本规划的五大 Schema、辅助类型、状态表、服务协议、失败码和合法/非法示例正式冻结；V2 原文件增补记忆契约。 |
| `docs/experiments/protocol.md` | 冻结 split、来源、去重、标注、基线变量、指标口径、seed、API 计费与失败处理；V2 增配对与快照规则。 |
| `experiments/README.md` | 实验产物布局和冻结索引；每次运行目录及报告由脚本生成，不手写虚构指标。 |
| `tests/fixtures/model_responses.json` | 四角色正常/坏 JSON/截断/动作/usage/错误响应的固定假数据，不含真实密钥。 |
| `tests/fixtures/tool_responses.json` | 正常、空召回、坏结构、超长、ID 冲突、协议标签污染的固定检索返回。 |
| `tests/fixtures/questions.jsonl` | 合成或可公开的小型问题、split、gold/route 等测试标注，明确只是软件测试。 |
| `tests/fixtures/qrels.json` | 小型可手算 query→evidence relevance 真值；没有真值的题不伪造。 |
| `tests/fixtures/safety_cases.jsonl` | 带来源与审核标识的固定路由/紧急/输出安全用例；不宣称临床覆盖。 |
| `src/medagent/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/runtime/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/models/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/agents/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/tools/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/policies/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/tracing/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |
| `src/medagent/evaluation/__init__.py` | 空包标记，0 函数；不在此注册服务或执行有副作用的导入。 |

### 4.2 V2 新增

| 文件 | 职责 |
|---|---|
| `src/medagent/memory/__init__.py` | 空包标记，0 函数。 |
| `configs/v2.toml` | 引用 V1 基础配置，加 memory mode、冻结路径/hash、encoder 身份、top_k=1/2、一跳及 token 限额；不另复制 B3。 |
| `configs/memory_baselines.toml` | B4 episodic、B5 graph 与 graph expansion 开关；沿用 B3 模型/检索/输出预算。 |
| `tests/fixtures/memory_snapshot.json` | 最小 Query/Interaction/Insight 节点、边、角色、审核与固定向量；包含越权/弱支持/污染样例，不使用测试题建立线上记忆。 |
| `docs/decisions/0002-frozen-memory.md` | 冻结存储、encoder、角色投影、记忆预算、公平比较与故障降级决定；生命周期写入留 V3。 |

## 5. 跨文件数据契约：先约定骨头怎样连接

V1 业务类型集中定义在 `src/medagent/schema.py`；模型协议类型放 `models/base.py`；V2 记忆类型放 `memory/schema.py`。优先数据类/枚举和明确的边界校验，不建一个通用“类型注册中心”。JSON Schema 作为对应类型的显式常量；正式 Schema 冻结到 `docs/protocols/contracts.md`，避免两份独立真相。JSON Schema 校验器只在边界调用，不自己实现完整规范。

### 5.1 五个必须先冻结的 Schema

| 类型 | 必要字段及约束 | 生产者 → 消费者 |
|---|---|---|
| `RunState` | RECEIVED、SAFETY_CHECKED、PLANNED、DIRECT_GENERATION、EMERGENCY_GENERATION、SEARCHING、VERIFYING、RETRYING、VERIFIED、ABSTAINED、GENERATING、COMPLETED、FAILED 枚举 | Runtime 独占写权限 → 所有角色只读 |
| `Plan` | route、risk_level、search_query、constraints、reason_codes；SEARCH 必须有非空 query；非 SEARCH query 为空；紧急预检不可降级 | Planner 提议 + safety.resolve_route 裁决 → Runtime/Search/Generator |
| `SearchResult` | attempt_id、candidate_answer、candidate_status、queries、evidence_ids、claims、stop_reason、tool_errors、public_actions；claims 中为 claim_id/text/source_span/evidence_ids；临时候选 `UNSTRUCTURED` 仅能进入 Verifier.structure_candidate，不能进入生成 | SearchAgent → Verifier；仅被接受的部分 → Generator |
| `VerificationResult` | verdict、failure_codes、unsupported_claims、conflicting_evidence_ids、retry_query_hint、public_summary_constraints；附 check_level=structural/semantic、accepted_claim_ids；PASS 不得保留必需失败；RETRY 必须有明确提示并由预算再裁决 | Verifier → Runtime/SearchAgent/输出政策 |
| `TraceEvent` | schema_version、run_id、event_id、sequence、timestamp_utc、event_kind、from_state、to_state、actor、input_ref、output_ref、model_ref、duration_ms、usage、error、prev_hash、event_hash | Runtime/trace → 离线评测及回放 |

约定：状态迁移事件必须有 from/to；模型、工具、预算等观测事件不改变状态，可令 from/to 相同并设置 event_kind，不能因此允许任意状态自循环。第一个 RECEIVED 事件 from_state=null；终态只允许追加专门的存储完成元数据，不能再开始业务调用。

### 5.2 辅助对象及最小字段

| 类型或类型组 | 规划字段/含义 |
|---|---|
| `UserRequest` | request_id、question、options、locale、允许的任务上下文；原问题在受控内存中保留，不覆盖为改写 query；不含 gold |
| `SafetyResult` / `SafetyRules` | risk_floor、emergency_flag、reason_codes、命中规则 ID/脱敏片段引用、rules_version；规则集合为静态加载对象 |
| `Evidence` / `CorpusRef` | evidence_id、source_doc_id 可空、corpus_id/version、title、text、source_ref、offset_start/end、score、retrieved_at、query_id；回退哈希 ID 不是原语料 ID |
| `EvidenceLedger` | 当前 run 内 evidence_id→不可变 Evidence 映射；qrels 使用源 ID 时必须维护映射，无法映射的题不报检索真值指标 |
| `SearchTask` / `SearchAction` / `PublicAction` | 原问题、计划 query、约束、attempt、历史公开动作、Verifier 反馈、预算视图；动作是 kind+content；公开动作只保留调用/观察/答案摘要，剥离隐藏推理 |
| `RunContext` / `RunResult` | context 含 state、route、attempt、证据账本、budget、trace、services 引用；result 含 final_status、answer、abstain、failure_codes、usage、trace_ref |
| `AcceptedContent` / `PublicContext` | route、allowed_claims、allowed_evidence、uncertainty、allowed_template_ids、mandatory_sections；只从本基线门禁认可的结构产生 |
| `PublicAnswer` | 按序 sections，每项只能选择 template_id 或 claim_id 并绑定允许 evidence_ids；boundary_template_id；不提供任意新的医学自由文本槽位 |
| `ModelRequest` | role、messages、response_schema、model_ref、sampling_params、max_output_tokens、timeout_s、request_id；每角色独立组装 |
| `ModelResponse` | content、provider、requested_model、returned_model、finish_reason、usage、latency_ms、provider_request_id；无隐藏推理字段 |
| `Usage` / `ModelConfig` / `ModelRef` | 输入/缓存/输出 token、known/estimated/unknown 标记、调用/延迟/显存/费用；模型身份、checkpoint/hash、tokenizer/template、endpoint/参数 |
| `BudgetLedger` / `BudgetView` / `Reservation` | 总 deadline、token/call/tool/doc 计数、三类 retry 计数、预留与实耗；View 只读；Reservation 包含 id/上界/settled 状态 |
| `OperationSpec` / `RetryKind` | actor、model/tool、每次 timeout、最大输出、费用上界、可恢复错误集；RetryKind=transport/action_repair/verification |
| `ToolCall` / `ToolSpec` / `ToolResult` / `ToolConfig` | name+args、允许角色+Schema+handler、evidence+errors+usage、固定 endpoint+语料版本+query/topk/bytes 上限 |
| `RunConfig` / `Services` / `DependencyReport` | 已合并只读配置；装配后的 adapter/工具/trace/时钟引用；依赖存在、版本/协议验收和阻塞列表；不建抽象依赖注入框架 |
| `RunManifest` / `TraceBundle` / `ReplayResult` / `RunDiff` | 版本/哈希/种子/脱敏策略；清单+事件+对象；重放校验结果；结构化差异 |
| `PrivacyPolicy` / `EventCause` | 允许日志字段及红线策略；迁移原因/actor/输入输出引用，不含模型内部推理 |
| `Failure` / `HarnessError` | code、stage、retryable、public_summary、safe_details；派生 ProtocolError、StateError、BudgetError、TransportError、TraceError，均为无自定义方法的数据型异常 |
| `StopReason` / `VerifyMode` / `FallbackKind` | answered/max_turns/deadline/budget/tool_error/invalid_action；structural/semantic；emergency/abstain/infrastructure |
| `EvalCase` / `AnswerLabel` / `RouteLabel` / `SafetyLabel` / `Qrels` | case_id、题型、问题、gold_aliases/选项答案、gold route、风险/安全人工标签、相关文档及等级；标签仅由 evaluator 消费 |
| `DatasetManifest` / `SourceRef` / `DedupPolicy` / `DedupReport` / `VectorTable` | 来源/revision/许可/哈希与 split；去重方法/阈值/encoder、候选对、裁决、完成状态；向量与 ID 对齐表 |
| `Prediction` / `EvaluationRun` / `EvaluationManifest` / `LabelBundle` / `BaselineAnswer` | case_id→RunResult、运行集合、冻结身份和续跑账本、独立标签集合；BaselineAnswer={text, baseline_only=true} 只允许 B0 evaluator 消费，不能传生产 Generator；RunResult.answer 按 baseline 身份区分它与 PublicAnswer |
| `MetricSet` / `Report` / `PricingSnapshot` / `CasePolicy` / `CaseSummary` | value+n/d+n_applicable+missing_reason；报告；计价有效时间/币种/cache 单价；固定案例选择规则与案例引用 |
| 测试 `TestClock` | now/advance 由测试库 mock 或现成 fixture 提供；不引入业务时钟服务 |

函数表中的 Path、Callable、Namespace、list、dict、None 属于语言/标准库类型；Role/RiskLevel/MemoryMode 等枚举集中定义，不为每个枚举创建文件。表中未列方法的类型全部是数据容器，没有隐含业务函数。

### 5.3 四个必须正面处理的接口问题

**自由文本候选不是证据映射。** 当前 MedSearch 动作通常返回 `<answer>文本</answer>`，不能假设它已经产生 JSON claims。V1 先由 Verifier 角色单独结构化原候选，再做确定性检查；这次调用在 B1/B2/B3 中一致保留并计费。结构化只能标注来源，不得修补医学事实。与原答案的跨度对不上或没有引用的核心声明不能通过最低门禁。未来若已有 checkpoint 稳定输出此结构，可去掉这次调用，但须作为单变量变更重跑基线。

**结构覆盖不是语义支持。** `evidence_id` 存在、文本跨度匹配只能证明引用真实；“该片段是否支持这个声明”仍需语义审查或独立标注。B1/B2 的 structural 状态不能写成 medically verified。外部正确率、安全率标签不能直接用同一个 DeepSeek Verifier 的 PASS。

**Generator 的候选硬约束方案（待讨论，未冻结）。** V1/V2 让模型选择并排序已允许声明与模板，再由代码渲染原文。这能用 ID 白名单防止下游新增事实。自由改写加关键词过滤不能保证“不增事实”；若以后开放自由润色，应另加生成后声明核验并作为新决策，不偷偷放进本版本。

**DIRECT 收窄的候选方案（待讨论，未冻结）。** 无证据又禁止新增医学事实时，不能同时允许 Generator 自由回答任意医学问题。部署 DIRECT 先限信息性固定模板、用途解释和操作帮助；医学知识请求默认 SEARCH，紧急优先。B0 原始 DeepSeek Direct 是离线答案能力对照，允许独立自由作答但同样执行安全预检，明确它不是部署 DIRECT 的安全证明，也不与其他基线宣称单变量因果。

## 6. 状态机与异常路径

| 当前状态 | 允许后继 | 条件 |
|---|---|---|
| RECEIVED | SAFETY_CHECKED | 已完成预检并形成事件 |
| SAFETY_CHECKED | PLANNED | 普通问题经过 Planner；明显紧急可由硬策略生成 EMERGENCY Plan，不等待模型 |
| PLANNED | DIRECT_GENERATION / EMERGENCY_GENERATION / SEARCHING | 必须与最终裁决 route 一致 |
| DIRECT_GENERATION | GENERATING | 公开模板上下文已准备 |
| EMERGENCY_GENERATION | GENERATING | 固定紧急行动可立即输出，Generator 仅组织允许补充 |
| SEARCHING | VERIFYING / ABSTAINED | 有候选则验证；无候选且搜索预算耗尽等可直接弃答 |
| VERIFYING | VERIFIED / RETRYING / ABSTAINED | 由验证结果与剩余预算共同决定 |
| RETRYING | SEARCHING | 先扣外层返工次数，携带明确反馈；V1 最多一次 |
| VERIFIED | GENERATING | 只传 accepted_claims 与对应证据 |
| ABSTAINED | GENERATING | 只传弃答摘要和固定模板 |
| GENERATING | COMPLETED | 输出通过或使用合法确定性回退 |
| 任一非终态 | FAILED | 不可恢复基础设施/审计/内部一致性错误 |
| COMPLETED / FAILED | 无 | 禁止恢复业务调用、重复回答与预算重置 |

未列出的边一律禁止，例如 SEARCHING→COMPLETED、VERIFYING→SEARCHING 绕过扣重试、ABSTAINED→VERIFIED、COMPLETED→SEARCHING。SOURCE 原始请求被拒或取消，可记录对应错误走 FAILED；正常“不确定，无法回答”走 ABSTAINED→GENERATING→COMPLETED，不能混淆技术成功与医学答对。

| 情况 | 处理 | 重试边界 |
|---|---|---|
| 明显紧急 | 硬分流并先展示固定急救/线下服务提示；不调用检索 | 不等 Planner、Verifier 或记忆 |
| Planner 坏 JSON/拒绝输出 | 记录协议失败；预检足以紧急则紧急，否则按风险进入 SEARCH 或弃答 | 默认无 LLM JSON 修复循环 |
| HTTP 401/403、错误模型 ID | 配置错误，FAILED 或固定故障响应；记录原因 | 0 次 |
| 429、可恢复 5xx、暂态网络错误 | 只有剩余额度够时有限退避；每次都记录 | 每逻辑操作默认最多 1 次；总调用与总时间仍封顶 |
| 服务超时/客户端取消 | 终止 await，忽略迟到结果；如服务支持则发取消信号 | 超时费用可能 unknown，不能宣称服务端算力必然已停止 |
| 无召回/缺少证据/冲突可修复 | Verifier 提供明确反馈后 RETRY，否则弃答 | 初次 + 1 次外层尝试 |
| 非法 Search 动作 | 记录非法率，允许有限格式纠错 | 默认最多 1 次/整 run；不奖励非法动作也不扩搜索额度 |
| 伪造引用/危险建议/不可修复协议 | 不允许 LLM PASS 覆盖；弃答 | 不自动无限返工 |
| Generator 新增字段/超时 | 丢弃该输出，确定性紧急或弃答模板 | 默认无额外生成重试 |
| Trace 写入失败 | 中止外部调用，FAILED；不提供伪装成已审计的正常答案 | 不继续执行后补日志 |

默认规划值仅用于起步，尚未实测：单 run 同时只执行一个模型/工具调用；外层返工≤1；每次 SearchAgent 尝试搜索调用≤2 且整 run≤4；非法动作纠错≤1；每个 HTTP 操作暂态重试≤1；query≤512 字符；top_k≤5；观察 token、总 token、总调用、deadline、费用上限必须在配置中有有限值，并为最终模板输出保留路径。实现时用短 mock deadline 验证终止，不靠人工 Ctrl+C。真实 deadline/token 数在服务 smoke 后冻结，不在此虚构性能容量。

## 7. V1：逐文件函数清单

每个文件列出全部计划函数、职责、输入输出与关键失败行为。常量、枚举、JSON Schema、角色 Prompt 与模板不计函数；不要再创建同名 service/controller/manager 包裹一层。

### `src/medagent/schema.py` — 4 个函数

统一类型和跨字段验证；只在这个文件定义 V1 业务数据契约。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `validate_plan(plan: Plan, safety: SafetyResult) -> Plan` | 检查 route/risk 枚举；SEARCH query 去空白后非空且不超长；紧急预检不得被降级；非法输出抛 ProtocolError。 |
| `validate_search_result(result: SearchResult, evidence: EvidenceLedger) -> SearchResult` | 校验候选声明、引用与本轮证据账本；未知 evidence_id、越界片段和缺失声明关联拒绝。 |
| `validate_verification(result: VerificationResult, search: SearchResult) -> VerificationResult` | 校验 verdict 及失败码；RETRY 要有可执行提示；冲突和 unsupported_claims 必须指向真实对象。 |
| `validate_public_answer(answer: PublicAnswer, allowed: PublicContext) -> PublicAnswer` | 只准引用允许的 claim_id/evidence_id 与模板字段；拒绝额外医学文本、引用伪造及无边界标记。 |

### `src/medagent/config.py` — 2 个函数

读取一次配置，冻结模型、预算、基线和依赖位置。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `load_config(path: Path, baseline: str) -> RunConfig` | 读取 TOML，合并具名基线允许的少量覆盖；禁止未知字段、负预算、V1 memory 开启及无价格快照的硬费用限额。 |
| `validate_dependencies(config: RunConfig) -> DependencyReport` | 检查本地只读路径、checkpoint/tokenizer 身份、固定服务地址及协议版本；不下载、不启动训练；缺失时返回明确阻塞项。 |

### `src/medagent/cli.py` — 2 个函数

唯一命令行入口；只装配依赖并调用业务函数。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `parse_args(argv: list[str]) -> Namespace` | argparse 定义 run/evaluate/replay/validate-data 子命令；未知基线或缺少必要路径直接报错。 |
| `main(argv: list[str] \| None = None) -> int` | 装配 config、两个 adapter、只读工具与 trace；分发命令；成功返回 0，配置/执行失败返回非零，不吞异常。 |

### `src/medagent/runtime/state.py` — 2 个函数

唯一合法状态迁移表，模型不得直接修改状态。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `allowed_next(state: RunState) -> frozenset[RunState]` | 根据冻结迁移表返回后继集合；终态返回空集合。 |
| `transition(context: RunContext, target: RunState, cause: EventCause) -> TraceEvent` | 检查边、route、retry 次数及必要输入；先可靠落盘事件再提交内存状态；重复 event_id 不重复提交，非法迁移抛 StateError。 |

### `src/medagent/runtime/budget.py` — 4 个函数

单次运行共用一个预算账本，避免角色分别计数导致超支。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `reserve(budget: BudgetLedger, operation: OperationSpec, now: float) -> Reservation` | 调用前按 max_tokens、剩余时间、工具文档数和价格上界预留；不够时拒绝，包含重试与最终输出保留量。 |
| `settle(budget: BudgetLedger, reservation: Reservation, usage: Usage) -> None` | 以实测 usage 结算并释放多余预留；失败请求也计入调用与时间，usage 缺失保留保守估计且标 unknown。 |
| `remaining(budget: BudgetLedger, now: float) -> BudgetView` | 给角色只读剩余额度；deadline 用单调时钟，不能由模型修改。 |
| `consume_retry(budget: BudgetLedger, kind: RetryKind) -> None` | 分别扣 HTTP/非法动作/Verifier 返工额度；达到上限拒绝，不能以另一类重试重置次数。 |

### `src/medagent/runtime/engine.py` — 5 个函数

主状态机调度器；负责调用生命周期，不做医学推理。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `run(request: UserRequest, services: Services, config: RunConfig) -> RunResult` | 创建 run_id，脱敏及预检，按基线组织路由/搜索/验证/输出，所有出口写终态；正常弃答仍是 COMPLETED。 |
| `invoke(operation: Callable, spec: OperationSpec, context: RunContext) -> object` | 统一 reserve、timeout、调用事件和 settle；只重试显式可恢复网络错误；每次调用唯一 call_id，取消后忽略迟到结果。 |
| `run_search_cycle(context: RunContext, services: Services) -> AcceptedContent` | 调 SearchAgent 与 Verifier；B3 初次+最多一次反馈返工；耗尽或不可修复则 ABSTAIN；B1/B2 按基线约束执行。 |
| `finish(context: RunContext, content: AcceptedContent, services: Services) -> RunResult` | 构建公开上下文、调用 Generator、验证并渲染；生成失败使用确定性紧急/弃答模板，写答案与最终清单。 |
| `fail(context: RunContext, error: HarnessError) -> RunResult` | 记录分类错误及 FAILED；禁止把部分候选答案当正常输出；trace 无法写入时停止外部调用并返回最小本地错误。 |

### `src/medagent/models/base.py` — 1 个函数

共用模型请求协议及错误模型；两个真实实现和一个测试替身共用接口。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `ModelAdapter.generate(self, request: ModelRequest) -> ModelResponse` | Protocol 方法声明，接收角色消息、输出约束、生成参数和本次 timeout；返回内容与 usage；实现只做一次请求，不私自循环重试。 |

### `src/medagent/models/deepseek.py` — 4 个函数

DeepSeek 单次 HTTP 调用适配；三个逻辑角色各建独立 request。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `DeepSeekAdapter.generate(self, request: ModelRequest) -> ModelResponse` | 从环境读取密钥，经固定 endpoint 发起一次有 timeout 的请求；不返回或落盘隐藏推理字段。 |
| `build_payload(request: ModelRequest, config: ModelConfig) -> dict` | 转换 role/messages/参数及该 provider 实测支持的结构化输出选项；不假定 API 支持工具或 JSON Schema 的任意形式。 |
| `parse_response(payload: dict, elapsed_ms: float) -> ModelResponse` | 提取公开 content、返回模型 ID、token/cache usage 和 finish_reason；截断不能当合法完整 JSON。 |
| `classify_error(status: int \| None, body: object) -> HarnessError` | 区分鉴权/限流/服务错误/网络超时/协议错误；脱敏错误体，只有限流和明确暂态错误允许 Runtime 有限重试。 |

### `src/medagent/models/medsearch.py` — 4 个函数

对接已选定 MedSearch checkpoint 的本地生成服务；工具执行留在 Harness。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `MedSearchR1Adapter.generate(self, request: ModelRequest) -> ModelResponse` | 调受信本地生成 endpoint，一次只生成一个动作；不访问检索、不导入训练框架、不在进程内加载另一份权重。 |
| `build_payload(request: ModelRequest, config: ModelConfig) -> dict` | 使用经验收的 tokenizer/chat template 或服务模板，传停止标签和生成上限；保留 search/information/answer 协议。 |
| `parse_response(payload: dict, elapsed_ms: float) -> ModelResponse` | 解析文本、token/finish_reason、checkpoint 标识与可用显存遥测；缺少统计标 null，不编造。 |
| `parse_action(text: str) -> SearchAction` | 按既有协议仅接受一个非空闭合 search/answer 动作且动作后无额外文本；前缀隐藏推理不进入 trace 或候选答案；非法抛 ProtocolError。 |

### `src/medagent/agents/planner.py` — 2 个函数

Planner 专属 Prompt 常量与输入构造；不拥有工具。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_request(question: UserRequest, safety: SafetyResult, config: RunConfig) -> ModelRequest` | 只加入问题、风险特征、可用路由及约束；不加入检索答案或其他角色上下文。 |
| `plan(question: UserRequest, safety: SafetyResult, context: RunContext) -> Plan` | 经 Runtime.invoke 调 DeepSeek，解析并 validate_plan；无效路由返回可分类错误，由 safety policy 决定保守分支。 |

### `src/medagent/agents/search.py` — 5 个函数

MedSearch-R1 的部署侧有限动作循环；不复制 RL AgentLoop。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_request(task: SearchTask, history: list[PublicAction], evidence: EvidenceLedger) -> ModelRequest` | 传原题、query、约束、尝试记录、Verifier 反馈和剩余预算；观察内容按文档/token 上限加入。 |
| `search(task: SearchTask, context: RunContext) -> SearchResult` | 循环调用本地模型，解析 search/answer；search 经工具白名单执行；达到轮次/时间上限终止并返回 stop_reason。 |
| `format_observation(evidence: list[Evidence]) -> str` | 将可信账本 ID 与受限原文写成 information 观察；转义文档中的协议标签，保留 ID 到来源映射，截断后重新记录偏移。 |
| `assemble_result(answer: str, actions: list[PublicAction], evidence: EvidenceLedger, stop: StopReason) -> SearchResult` | 形成 provisional 候选和工具错误摘要；无法从原模型答案获得可靠 claim 映射时留空待 Verifier 结构化，不伪造支持关系。 |
| `answer_fixed(task: SearchTask, evidence: list[Evidence], context: RunContext) -> SearchResult` | B1 专用：给本地模型固定单次检索的证据，禁止 search 动作，生成一次候选；其余输出契约与 B2/B3 相同。 |

### `src/medagent/agents/verifier.py` — 5 个函数

确定性证据检查 + 独立 DeepSeek 审查；PASS 不等于医学正确。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `structure_candidate(result: SearchResult, context: RunContext) -> SearchResult` | 使用 Verifier 独立结构化任务提取原答案的声明与建议引用；保留原文跨度，不能补事实；映射不足记 unsupported，不把检索到的文档全判支持。 |
| `check_evidence(result: SearchResult, ledger: EvidenceLedger) -> list[Failure]` | 检查 ID、来源版本、引用跨度、逐声明覆盖和结果格式；只能证明引用存在及结构覆盖，不能自动证明蕴含。 |
| `build_request(result: SearchResult, failures: list[Failure], task: SearchTask) -> ModelRequest` | 提交当前声明和真实片段、确定性失败码及政策；不提交 Planner 私有解释或 gold answer。 |
| `verify(result: SearchResult, context: RunContext, mode: VerifyMode) -> VerificationResult` | B1/B2 执行结构化+确定性最低门禁；B3 增加语义审查；代码失败不能被模型 PASS 覆盖。 |
| `resolve_verdict(checks: list[Failure], review: VerificationResult, budget: BudgetView) -> VerificationResult` | 合并结果；可修复证据缺口且有额度才 RETRY；伪造/危险/预算耗尽等归 ABSTAIN，PASS 仅在所有必需检查通过时成立。 |

### `src/medagent/agents/generator.py` — 3 个函数

只组织允许公开的内容，输出结构化选择而非任意新增医学文本。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_request(content: PublicContext) -> ModelRequest` | Generator 独立 Prompt，只给获准 claim、证据、限制与模板；DIRECT 输入仅为预审的信息性模板材料。 |
| `generate(content: PublicContext, context: RunContext) -> PublicAnswer` | 让 DeepSeek 选择/排列允许的 claim_id 和固定过渡语模板；校验结构，不接受未授权自由医学文本；错误交 finish 做模板回退。 |
| `render(answer: PublicAnswer, content: PublicContext) -> str` | 确定性填入已批准声明原文、引用及不确定性、研究用途说明；紧急行动提示永远在首段。 |

### `src/medagent/tools/registry.py` — 3 个函数

一个只读工具的静态注册表；不做插件发现或任意动态执行。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_registry(config: ToolConfig) -> dict[str, ToolSpec]` | 仅注册 search_medical_evidence，包含 JSON Schema、固定端点、角色权限、上限与处理函数。 |
| `validate_call(call: ToolCall, actor: str, registry: dict) -> ToolCall` | 检查白名单、JSON Schema、额外字段、query 长度、top_k 和权限；禁止参数传入 URL/文件路径/命令。 |
| `execute(call: ToolCall, context: RunContext, registry: dict) -> ToolResult` | 合法调用经 Runtime.invoke 执行；写调用事件；截断超预算返回，记录错误类型，不能悄悄改用网页检索。 |

### `src/medagent/tools/retrieval.py` — 3 个函数

适配已存在的 MedSearch /retrieve 服务；本项目不构建索引。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `search_medical_evidence(query: str, top_k: int, timeout_s: float, config: ToolConfig) -> ToolResult` | POST queries=[query], topk=top_k, return_scores=true 到冻结服务地址；禁用跨地址重定向，限制响应字节数。 |
| `normalize_response(payload: dict, corpus: CorpusRef) -> list[Evidence]` | 从 result[0] 解析 document.contents/score；优先用源 ID，缺失时用语料版本+内容哈希建立稳定 ID；保存 title/body 与真实来源，保留空召回。 |
| `register_evidence(items: list[Evidence], ledger: EvidenceLedger) -> list[Evidence]` | 去重并注册可引用片段；同 ID 不同内容报错；只允许工具实际返回的 ID 成为后续可引用证据。 |

### `src/medagent/policies/safety.py` — 3 个函数

预检和路由硬约束；规则版本与命中理由可审计。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `preflight(question: UserRequest, rules: SafetyRules) -> SafetyResult` | 提取显式紧急信号、风险特征和命中码；否定/引用/第三人称用测试覆盖；不宣称覆盖所有医学风险。 |
| `resolve_route(safety: SafetyResult, plan: Plan \| None, config: RunConfig) -> Plan` | 明显紧急硬路由，Planner 不可降级；DIRECT 只限非个体诊疗的信息性模板场景；不确定医学问题进入 SEARCH 或安全弃答。 |
| `emergency_context(safety: SafetyResult) -> PublicContext` | 产生固定紧急行动、禁忌输出和边界材料，不检索；关键行动提示不依赖模型成功返回。 |

### `src/medagent/policies/output.py` — 3 个函数

生成前的内容白名单与失败后确定性输出。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_public_context(route: Plan, search: SearchResult \| None, verification: VerificationResult \| None) -> PublicContext` | SEARCH 仅提取通过本基线门禁的声明与证据；DIRECT 取模板；ABSTAIN 只给失败摘要与允许建议。 |
| `check_output(answer: PublicAnswer, content: PublicContext) -> list[Failure]` | 调共享 Schema 校验，额外检查紧急提示顺序、禁用剂量调整模板及引用保留；不能靠关键词检查宣称医学安全。 |
| `fallback_response(kind: FallbackKind, reason_codes: list[str]) -> PublicAnswer` | 返回固定紧急/弃答/基础设施故障结构；不调用模型，不伪造引用或诊断。 |

### `src/medagent/tracing/redact.py` — 2 个函数

普通日志默认仅保留脱敏公开对象，采用字段白名单。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `redact_payload(payload: object, policy: PrivacyPolicy) -> dict` | 删除密钥、身份字段和模型隐藏推理，只留下允许字段；自由文本做已知标识脱敏，无法保证去标识的输入只保留摘要/特征。 |
| `fingerprint(payload: object) -> str` | 对允许保留的规范化对象计算 SHA256；敏感短文本不直接留可枚举原文哈希；用于回放校验和产物身份。 |

### `src/medagent/tracing/store.py` — 4 个函数

单进程、单运行追加 JSONL；对象 JSON 按内容 ID 存储。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `TraceStore.start(self, manifest: RunManifest) -> None` | 创建不覆盖已有 run_id 的运行目录，落初始清单；记录代码/配置/prompt/模型/数据版本。 |
| `TraceStore.put_payload(self, payload: object, policy: PrivacyPolicy) -> str` | 脱敏后写内容寻址对象，返回引用；相同对象复用；写临时文件再原子替换。 |
| `TraceStore.append(self, event: TraceEvent) -> None` | 检查递增序号/event_id/前序哈希，追加并 flush；严格审计模式 fsync；已有事件禁止改写。 |
| `TraceStore.finalize(self, result: RunResult, usage: Usage) -> None` | 落最终答案、状态和实测费用统计；保留初始清单，写单独 summary，不覆盖事件历史。 |

### `src/medagent/tracing/replay.py` — 3 个函数

离线验证事件和重建流程，不等同重新调用模型生成相同答案。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `load_trace(path: Path) -> TraceBundle` | 读取 manifest/events/引用对象；验证哈希、序号、缺失对象、末尾不完整行；损坏明确报错。 |
| `replay(bundle: TraceBundle) -> ReplayResult` | 无网络按记录重建状态、预算、动作和终态；对允许迁移和引用重新校验。 |
| `compare_runs(left: TraceBundle, right: TraceBundle) -> RunDiff` | 按角色调用、路由、证据、成本和最终结果比较；标识配置/模型版本差异，不能把差异直接解释为因果。 |

### `src/medagent/evaluation/dataset.py` — 4 个函数

冻结题目与标注，推理侧永不接收 gold。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `load_cases(path: Path) -> list[EvalCase]` | 验证唯一 ID、题型、split、gold/qrels/safety labels；未知标注留缺失，不默认负例。 |
| `make_request(case: EvalCase) -> UserRequest` | 只暴露 question/options/允许上下文；剥离 gold、gold_route、风险真值和 qrels。 |
| `freeze_dataset(cases: list[EvalCase], sources: list[SourceRef]) -> DatasetManifest` | 固定来源 revision、许可证信息、样本 ID/哈希/seed；源字段缺失不伪造。 |
| `check_overlap(splits: dict[str, list[EvalCase]], vectors: VectorTable \| None, policy: DedupPolicy) -> DedupReport` | 精确、规范化、词汇和语义近重复分层检测；跨 split 发现候选送审，语义检测未运行时标未完成，不能宣称隔离通过。 |

### `src/medagent/evaluation/metrics.py` — 5 个函数

纯指标计算；每个返回值带分子、分母和适用样本数。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `answer_metrics(predictions: list[Prediction], labels: list[AnswerLabel]) -> MetricSet` | 按题型分别算 Accuracy/EM/token-F1；弃答在总体答案指标记错，另报 coverage 与已回答准确率；无 gold 不算正确性。 |
| `evidence_metrics(predictions: list[Prediction], qrels: Qrels) -> MetricSet` | 有金标时算 Recall@k/MRR/hit；无金标只算 citation validity，语义支持率依赖独立标注而非 Verifier 自评。 |
| `harness_metrics(runs: list[RunResult], labels: list[RouteLabel]) -> MetricSet` | route accuracy、协议完成、illegal actions、forced stops、retry recovery；区分搜索内轮次与外层返工。 |
| `safety_metrics(runs: list[RunResult], labels: list[SafetyLabel]) -> MetricSet` | emergency recall、误报率、高风险误路由、unsafe rate；仅在人审/固定标注的适用集合计算，零分母返回 null。 |
| `cost_metrics(runs: list[RunResult], pricing: PricingSnapshot) -> MetricSet` | 实测延迟均值/P95、调用/token/搜索数和 API 费用；cache token 分项，价格/usage 缺失返回 unknown；GPU 缺遥测返回 null。 |

### `src/medagent/evaluation/runner.py` — 4 个函数

统一 B0～B3 调度和结果留存。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_variant(config: RunConfig, baseline: str) -> RunConfig` | 只应用基线矩阵允许的变量；安全约束与模型版本固定，拒绝偷偷变更检索器/预算。 |
| `run_dataset(cases: list[EvalCase], config: RunConfig, services: Services) -> EvaluationRun` | 固定顺序或固定 seed 顺序逐题运行；B0 分发 run_direct_baseline，其余分发 Runtime.run；失败样本也落记录，禁止只保留成功项。 |
| `resume_dataset(manifest: EvaluationManifest, services: Services) -> EvaluationRun` | 仅在配置、数据、模型、prompt 哈希一致时跳过已完整样本；半截运行保留后另建 attempt，不覆盖。 |
| `run_direct_baseline(case: EvalCase, context: RunContext) -> RunResult` | B0 离线专用：先安全预检，紧急使用固定策略；其他题由独立 Generator-baseline Prompt 经 Runtime.invoke 调 DeepSeek 直接作答，不走生产 Generator 的声明白名单；记录 baseline_only，输出不得进入部署响应链。 |

### `src/medagent/evaluation/report.py` — 3 个函数

把实测指标与失败原因写成可复现报告。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `aggregate(results: EvaluationRun, labels: LabelBundle, pricing: PricingSnapshot) -> Report` | 连接预测与独立标签，调用五类指标，标明 missing、样本数和失败数。 |
| `select_cases(results: EvaluationRun, policy: CasePolicy) -> list[CaseSummary]` | 按预定规则各选成功、返工成功、弃答、安全失败与接口失败案例；不只挑最好结果。 |
| `write_report(report: Report, out: Path) -> None` | 写 metrics.json、per_case.jsonl 和 report.md；列配置/commit/seed、限制及失败分析，未知值保留 null。 |

### `tests/conftest.py` — 3 个函数

共享测试依赖；fixture 也计为函数，自动生成构造函数不计。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `fake_services() -> Services` | 返回脚本化假模型、假检索和临时 trace，不连接真实 API。 |
| `frozen_clock() -> TestClock` | 返回可推进的单调时钟替身，预算测试不实际睡眠。 |
| `minimal_config(tmp_path: Path) -> RunConfig` | 产生有限预算、固定版本与临时输出路径。 |

### `tests/unit/test_schema.py` — 5 个函数

五大 Schema 与公开输出契约的合法/非法样例。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_plan_contract() -> None` | 参数化验证三路由、空 query、未知 route、紧急降级。 |
| `test_search_contract() -> None` | 参数化验证正常/临时未结构化候选、伪造 ID、越界引用与冲突 ID。 |
| `test_verification_contract() -> None` | PASS/RETRY/ABSTAIN 合法性、失败码及 unsupported claim 引用。 |
| `test_trace_contract() -> None` | 事件必填、序号、状态/actor 枚举和缺失对象引用。 |
| `test_public_answer_contract() -> None` | 允许声明与模板可输出；新增 claim、自由医学事实字段、伪造引用被拒。 |

### `tests/unit/test_state.py` — 3 个函数

全迁移矩阵，不只测试主成功路径。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_allowed_transitions() -> None` | 参数化遍历允许边并检查前置条件和事件。 |
| `test_forbidden_transitions() -> None` | 枚举状态笛卡尔积减允许边，非法边全部拒绝；终态不可继续。 |
| `test_commit_requires_trace() -> None` | 模拟事件写盘失败和重复 ID，验证不越过审计提交点。 |

### `tests/unit/test_budget.py` — 3 个函数

预算和取消边界。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_reserve_settle() -> None` | 精确边界、未知 usage、失败请求收费和最终输出预留。 |
| `test_separate_retry_limits() -> None` | 网络/非法动作/Verifier 次数分离且总调用数仍受限。 |
| `test_deadline_cancels_work() -> None` | 假挂起调用被取消，运行按时终止；迟到结果不能改状态。 |

### `tests/unit/test_tools.py` — 4 个函数

只读工具的权限、协议和证据真实性。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_tool_permissions() -> None` | 允许 SearchAgent/FixedRAG，拒绝其他角色、未知工具、额外 URL/命令字段。 |
| `test_retrieval_contract() -> None` | 正常、空召回、坏 JSON、缺字段、超大响应、重定向与网络错误。 |
| `test_evidence_identity() -> None` | 源 ID 缺失回退哈希、跨版本不混淆、重复 ID 内容冲突。 |
| `test_observation_escaping() -> None` | 文档中伪造 search/answer/information 标签不会被执行为动作。 |

### `tests/unit/test_policies.py` — 4 个函数

测试政策执行，不把规则测试视为临床验证。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_preflight_cases() -> None` | 紧急、否定、转述、第三人称及普通问题的固定样例。 |
| `test_route_override() -> None` | Planner 试图降级紧急风险、未知输出和危险 DIRECT 的处理。 |
| `test_emergency_fallback() -> None` | 模型超时或恶意输出时仍显示首段固定紧急行动且零检索。 |
| `test_generator_allowlist() -> None` | 拒绝增加事实、删除必需引用、改写剂量和输出内部轨迹。 |

### `tests/unit/test_agents.py` — 4 个函数

角色隔离及模型输出错误边界。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_role_context_isolation() -> None` | 三个 DeepSeek Prompt/input 独立，SearchAgent 不收到 gold 或私有推理。 |
| `test_search_action_loop() -> None` | 合法动作、未闭合、多动作、非法重复、轮次耗尽与反馈重试。 |
| `test_claim_structuring() -> None` | 结构化不能新增原候选未出现的声明，错误映射不能获代码 PASS。 |
| `test_verifier_resolution() -> None` | 确定性失败覆盖 LLM PASS；缺证据/冲突/危险/耗尽按表分流。 |

### `tests/unit/test_trace_metrics.py` — 4 个函数

回放、脱敏和计算口径。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_redaction() -> None` | 密钥、直接身份字段、隐藏推理不落普通日志；不能把删掉信息后的回放称原始全文回放。 |
| `test_replay_integrity() -> None` | 正常重建与哈希篡改、缺事件、坏尾行、未知版本拒绝。 |
| `test_metric_denominators() -> None` | 小型手算 gold/qrels 样例验证答案/检索/安全/成本公式及零分母。 |
| `test_dataset_separation() -> None` | gold 不进入 request；跨 split 重复与未完成语义检查阻止隔离验收。 |

### `tests/integration/test_workflows.py` — 3 个函数

端到端全假运行，是 V0 骨架验收入口。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_three_routes() -> None` | DIRECT/SEARCH/EMERGENCY 各产生完整事件和合法终态。 |
| `test_retry_and_abstain() -> None` | PASS、返工后 PASS、二次仍失败、直接 ABSTAIN；搜索最多两次外层尝试。 |
| `test_failure_termination() -> None` | 超时、预算、坏 Schema、生成失败、trace 故障均可解释地终止。 |

### `tests/integration/test_adapters.py` — 3 个函数

固定 HTTP mock 的线协议验收；真实调用必须显式开启。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_deepseek_http_contract() -> None` | 假服务器验证 payload、限流、截断、usage/cache 解析及密钥不落日志。 |
| `test_medsearch_http_contract() -> None` | 假服务器验证 tokenizer/checkpoint 标识、停止标签及动作，不需要 GPU。 |
| `test_live_smoke() -> None` | 显式标记 live；小样本核实两个真实模型与检索服务版本/接口/usage，默认跳过。 |

### `tests/integration/test_baselines.py` — 2 个函数

实验开关与完整样本账本验收。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_baseline_call_patterns() -> None` | B0 无检索，B1 固定一次，B2 无反馈返工，B3 最多一次返工；所有基线安全预检仍在。 |
| `test_fair_config_and_resume() -> None` | 哈希不匹配不可续跑，失败题保留，非目标变量不可暗改。 |


## 8. V2：只读记忆的增量实现

> **V2 范围更新（2026-09-15）：** 以 PROJECT_BOOK.md 的 V2.1～V2.3 为准：200 道构建题、活跃记忆上限 100 条、每 10 道离线维护，在线读取冻结快照；允许 Qwen 作为离线候选提炼组件。本文原“V2 不做提炼”“两类 LLM”的表述对 V1 继续成立，对 V2 按上述限定例外处理。下列 V2 函数及数量是原版工程清单，维护入口的最终签名与新增数量待实现时同步，不能视为已覆盖新增维护能力。V1 文件、函数与验收未修改。

### 8.1 存储和读取边界

第一版用不可变 JSONL 节点/边、JSON manifest 和本地向量数组；在内存建立字典邻接表。不引入图数据库、远程向量服务或在线写队列。小规模冻结库可以精确余弦扫描；数据大到实测检索延迟不可接受时再替换索引，但保留相同 `retrieve` 接口。

冻结快照目录包含 manifest.json、nodes.jsonl、edges.jsonl、vectors.npy、vector_ids.json、approval.json、dedup_report.json，共 **7 个生成文件**。同版本不得覆盖；打包完成后发布只读目录。在线进程只持读权限，run 后复验哈希。encoder 必须是本地受控非生成式编码器并锁定版本；优先核实已使用过的 embedding 资产是否可用，不能直接假定旧依赖环境仍可导入。V1 语义去重也使用已冻结的离线编码结果，不把 encoder 暗塞进在线模型链。

V2 新类型：

| 类型 | 最少字段 |
|---|---|
| `MemoryNode` | id、layer=QUERY/INTERACTION/INSIGHT、脱敏流程内容、roles、scope、risk_scope、source_trace_ids、support_ids、opposition_ids、validation_strength、reviewer/status、version；confidence 若无校准必须标 heuristic |
| `MemoryEdge` | source、target、relation=HAS_INTERACTION/SUPPORTS_INSIGHT/COUNTERS_INSIGHT、source_ref；仅允许已冻结关系类型 |
| `MemoryManifest` / `VectorManifest` | snapshot_id/version、源 split/hash、审核/去重引用、节点/边数量、文件哈希、创建时间、encoder/version/dim/normalization/ID 顺序 |
| `ReviewedRecord` / `ApprovalManifest` | 允许入库的公开轨迹字段、外部评测/人工审核证据、角色/风险范围、审核者及时间；模型自评不能单独构成通过 |
| `MemorySnapshot` / `MemoryHit` | 已加载且只读的节点/边/向量；命中 id/score/path/source/过滤原因 |
| `MemoryQuery` / `MemoryBudget` / `MemoryView` | 脱敏任务特征+role/risk；top_k/扩展节点/token；角色专用条目、来源、版本、token 与低优先级标识 |
| `Encoder` | 已选定编码器的 callable 对象或既有 API，接受文本返回定长向量；不再写独立通用 ModelAdapter |
| `RunPair` / `MemoryEffect` | 同题的 B3/control 与 treatment 结果、冻结身份、独立标签；helpful/neutral/harmful/unknown 与理由 |

角色投影：Planner 只得路由/风险/query 约束；SearchAgent 只得 query 策略、失败检索与证据选择经验；Verifier 只得证据缺口/冲突/不安全表达模式；Generator 只得引用/弃答/表达边界。三层图记录协作经验，不替代医学语料。节点不得存成“疾病 X 应用药 Y”这种医学事实记忆。

召回顺序为：先预检→按角色和风险限定合法候选→向量 top_k→B5 可一跳扩展→角色投影→完整条目 token 裁剪。重试时仍用同一个 snapshot，只允许基于本次结构化失败码更新查询；总记忆注入预算不重置。明显紧急分支直接绕过记忆。

### 8.2 V2 新文件及函数

### `src/medagent/memory/schema.py` — 2 个函数

V2 冻结记忆契约；不实现在线激活状态机。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `validate_snapshot(manifest: MemoryManifest, nodes: list[MemoryNode], edges: list[MemoryEdge]) -> None` | 检查来源/审核、角色、范围、版本、状态与边端点；孤立引用、测试集来源和不受支持的 ACTIVE insight 拒绝。 |
| `validate_view(view: MemoryView, role: Role, risk: RiskLevel) -> MemoryView` | 检查角色授权、风险证据强度、top_k/一跳/token 上限；禁止身份信息、隐藏推理、金标答案和医学事实型经验。 |

### `src/medagent/memory/snapshot.py` — 4 个函数

离线构建与在线只读加载共用一个模块，明确函数使用阶段。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `build_snapshot(records: list[ReviewedRecord], approval: ApprovalManifest, out: Path) -> MemoryManifest` | 离线把已审核的固定数据转成 Query/Interaction/Insight 节点及边，完成隔离检查后写不可覆盖的版本目录；不自行 LLM 提炼或激活。 |
| `index_snapshot(nodes: list[MemoryNode], encoder: Encoder, out: Path) -> VectorManifest` | 离线编码允许检索的脱敏流程表示，保存向量、ID 顺序、维度、归一化方式及 encoder 版本；禁止修改源记录。 |
| `load_snapshot(path: Path, expected_hash: str) -> MemorySnapshot` | 在线只读加载清单、图和向量，逐项验哈希与审核状态；损坏/版本不符时显式失败或记录 memory_off 降级。 |
| `check_source_isolation(snapshot: MemorySnapshot, report: DedupReport) -> None` | 核查构建源仅来自 memory_build，开发/测试题及改写无泄漏；报告未完成或存在未裁决重复则禁止发布。 |

### `src/medagent/memory/retrieval.py` — 4 个函数

同一冻结来源上的 B4 向量轨迹召回与 B5 一跳图召回。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `make_query(question: UserRequest, role: Role, risk: RiskLevel) -> MemoryQuery` | 仅取脱敏任务类型/搜索困难/风险特征；不含 gold、不由当前回答制造经验。 |
| `vector_search(query: MemoryQuery, snapshot: MemorySnapshot, encoder: Encoder, top_k: int) -> list[MemoryHit]` | 对规范化向量作确定性余弦排名；默认 top_k=1 或 2，固定同分排序；先按角色/风险/来源过滤。 |
| `expand_one_hop(hits: list[MemoryHit], snapshot: MemorySnapshot, limit: int) -> list[MemoryHit]` | 只沿允许关系走一跳，去重、限制扩展数，保留每条来源路径；不能把新节点当下一跳起点。 |
| `retrieve(query: MemoryQuery, mode: MemoryMode, snapshot: MemorySnapshot, budget: MemoryBudget) -> list[MemoryHit]` | off 返回空，episodic 只相似轨迹，graph 增一跳；按强度/范围/预算过滤，确定性排序并记录命中与剔除理由。 |

### `src/medagent/memory/projection.py` — 3 个函数

记忆到角色上下文的唯一出口。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `project(hits: list[MemoryHit], role: Role, task: SearchTask, budget: MemoryBudget) -> MemoryView` | 从同一流程记录提取各角色允许字段，删原问题完整答案/隐藏推理/医学事实；模板化建议，不让记忆覆盖系统政策。 |
| `trim_view(view: MemoryView, count_tokens: Callable, max_tokens: int) -> MemoryView` | 按固定优先级保留完整条目；不从中间截断角色/来源/风险限定；放不下返回空视图。 |
| `attach_view(request: ModelRequest, view: MemoryView) -> ModelRequest` | 校验后放独立、低优先级经验区；保留 request 主角色与原题；标 memory_ids/version/token。 |

### `src/medagent/evaluation/memory_metrics.py` — 4 个函数

对同题、同配置、同冻结起点的记忆结果作配对分析。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `pair_runs(control: EvaluationRun, treatment: EvaluationRun) -> list[RunPair]` | 用 case_id/seed/config 关键字段对齐 B3/B4/B5；缺失与失败不静默丢弃，单列 unmatched。 |
| `classify_memory_effect(pair: RunPair, labels: LabelBundle) -> MemoryEffect` | 依据独立答案/安全/证据标签判 helpful/neutral/harmful/unknown；不能用命中或 Verifier PASS 代替帮助。 |
| `memory_metrics(pairs: list[RunPair]) -> MetricSet` | 输出命中率、作用分类及分母、token/延迟/费用增量和配对差值；无独立标签则 unknown。 |
| `check_frozen_runs(runs: list[EvaluationRun], snapshot: MemoryManifest) -> None` | 验证所有处理组起点哈希一致且运行后文件哈希不变；检测途中写入、跨测试题学习和不同记忆版本混用。 |

### `tests/unit/test_memory.py` — 5 个函数

冻结记忆、召回与角色投影的边界测试。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_snapshot_validation() -> None` | 哈希、来源、ACTIVE 审核、边端点和不合法内容检查。 |
| `test_vector_and_one_hop() -> None` | 固定向量手算排名、同分顺序、top_k、一跳/扩展数约束。 |
| `test_role_and_risk_filter() -> None` | 四角色各得正确字段；高风险拒绝弱支持，越权字段/伪系统指令不能抬升优先级。 |
| `test_memory_token_budget() -> None` | 编码/注入/重试都计费计量，条目完整裁剪，空视图合法。 |
| `test_memory_effect_labels() -> None` | 配对 helpful/neutral/harmful/unknown 以及安全恶化优先规则。 |

### `tests/integration/test_memory_baselines.py` — 4 个函数

V2 公平对比与关闭记忆验收。

| 函数签名 | 实现职责、结果与失败边界 |
|---|---|
| `test_b3_b4_b5_frozen_start() -> None` | 同源同起点、B4 无图扩展、B5 最多一跳，运行前后哈希不变。 |
| `test_memory_off_replay() -> None` | off 不读快照、不调用 encoder、不注入记忆；假模型下与 V1 B3 的可比较业务事件相同。 |
| `test_poisoned_or_missing_snapshot() -> None` | 已知坏哈希、弱审核、高风险恶意条目被拒；故障降级显式标记，不能混入正常 B5 分母。 |
| `test_paired_report() -> None` | 同题结果、缺失样本、harmful 案例与 token 增量都出现在报告。 |


### 8.3 V2 修改哪些已有文件，修改哪些函数

以下全部为修改既有函数，**新增函数数为 0**。

| 既有文件 | 修改对象 | V2 的准确改动 |
|---|---|---|
| `schema.py` | RunConfig/RunManifest/TraceEvent/Services 数据字段；既有校验函数 | 加 memory_mode、snapshot_ref、memory_ids/token 和 schema_version=2；新读者兼容 V1 事件，旧读者遇 V2 明确拒绝，不悄悄忽略语义字段 |
| `config.py` | load_config、validate_dependencies | 校验 frozen path/hash、encoder、top_k/hops、审核报告；memory_off 不要求相关依赖存在 |
| `cli.py` | parse_args、main | 加 freeze-memory（离线发布）、validate-memory；run/evaluate 加 memory mode/快照；不加 update-memory 在线命令 |
| `runtime/engine.py` | run、invoke、run_search_cycle、finish | run 起点加载只读快照；非紧急按角色检索/投影；记忆耗时和 token 进入统一预算；记录降级与命中 |
| `runtime/budget.py` | reserve、settle、remaining | 增 encoder/记忆注入计数；仍受总 token/deadline 限额；consume_retry 保持 |
| `agents/planner.py` | build_request | 注入 Planner MemoryView，其他输入字段不变 |
| `agents/search.py` | build_request | 注入 SearchAgent MemoryView，历史仍是当前问题的有限公开动作 |
| `agents/verifier.py` | build_request | 注入 Verifier MemoryView；structure_candidate 不需要记忆，避免污染声明提取 |
| `agents/generator.py` | build_request | 注入表达/引用限制经验；不得扩充 allowed_claims 或削弱 mandatory_sections |
| `tracing/store.py` | start、append、finalize | 固定 snapshot/encoder/hash、命中路径与实际 token；不把整库快照拷贝进每题日志 |
| `tracing/replay.py` | load_trace、replay、compare_runs | 读取 V2 memory 事件；离线用记录的视图重放，严格校验引用；比较 memory 差异 |
| `evaluation/dataset.py` | check_overlap、freeze_dataset | 将 memory_build/dev/test 的候选题与改写加入统一隔离报告 |
| `evaluation/runner.py` | build_variant、run_dataset、resume_dataset | 识别 B4/B5，固定快照与配对 seed；续跑要求记忆身份一致 |
| `evaluation/report.py` | aggregate、select_cases、write_report | 引入 memory_metrics，报告 harmful、unknown、降级样本和增量成本 |
| README、pyproject、协议文档、实验文档 | 非函数字段/说明 | 补可选编码器/数组依赖、冻结快照命令、B3/B4/B5 表及版本兼容说明；不写“已提升” |

`models/`、`tools/`、安全路由规则及状态迁移主表在 V2 保持原接口；记忆是输入的受控补充，不是新角色或新工具。此处明确列出不变模块，是为了学生知道版本迁移的修改边界。

## 9. 基线如何可比

| 基线 | 候选来源 | 最低检查 | 语义审查 | 外层反馈返工 | 记忆 |
|---|---|---|---|---|---|
| B0 DeepSeek Direct | DeepSeek 离线自由回答 | 公共安全/输出格式审计 | 无 | 无 | off |
| B1 Fixed RAG | 固定一次原题 query→既有检索→MedSearch answer-only | Verifier 结构化 + 代码证据/Schema 检查 | 无 | 无 | off |
| B2 Static Harness | Planner→MedSearch 动态搜索 | 同 B1 | 无 | 无 | off |
| B3 Verified Harness | 同 B2 | 同 B1 | DeepSeek Verifier | 最多一次 | off |
| B4 Episodic | 同 B3 | 同 B3 | 同 B3 | 同 B3 | 冻结相似轨迹，只向量 |
| B5 G-Memory-lite | 同 B3 | 同 B3 | 同 B3 | 同 B3 | 同源三层流程图，向量+最多一跳 |

所有正式 Harness 组共用安全预检、预算账本和 Generator 白名单。B1/B2 的最低门禁是对“候选答案必须验证”与“无 Verifier 返工基线”的候选协调方案，需在决策文档明确冻结。表中 B1 候选模型固定为 MedSearch，避免在 B1→B2 同时更换候选模型；B0→B1 仍改变模型/检索两项，因此只是能力参照，不作为单因素检索收益。

B1～B5 都先运行相同的安全/路由政策，DIRECT/EMERGENCY 子集单独报；SEARCH 对比同时报告全量表现和固定 SEARCH 子集表现。B1 的固定 query 用原题规范化文本，不使用 Planner 优化 query；B2 启用规划/动态搜索，需承认这是搜索策略组合差异。

B2→B3 同时启用语义验证与反馈返工，不能声称变化全来自“重试”。利用原配置增加一个 **B3, max_verifier_retries=0** 的诊断运行即可分离返工作用，不另建一套目录/实现。B5 图扩展开关同理：先做向量-only B5，再开一跳，检验图扩展独立价值。

B3/B4/B5 使用同一问题集、模型/checkpoint、检索库、采样参数、生成上限和总上下文预算。记忆占用上下文不能悄悄挤掉证据：预先固定 question/evidence/memory/output 分区及裁剪规则；B3 不使用 memory 分区。报告实际 token 增量与未匹配样本。B4/B5 来源轨迹和审核强度一致，三层表示本身的变化单独解释。

`helpful/neutral/harmful` 用外部标签预先定义：安全恶化优先判 harmful；无安全变化时错→对为 helpful，对→错为 harmful；相同答案/安全结果为 neutral，同时单列成本恶化；证据标签有独立变化则报告；缺少必要标签为 unknown。单个配对样本受模型随机性影响，不能据此证明某条记忆有因果价值，正式结论应汇总固定重复 seed 的配对差异与不确定性。

## 10. 指标、数据和运行产物

### 10.1 指标口径

- Accuracy 的分母为有 gold 的全部问题，包含弃答和执行失败；同时报 coverage 与已回答准确率。
- Recall@k = gold relevant 与前 k 文档交集 / gold relevant 数；MRR = 首个相关文档排名倒数，无命中为 0；只在可映射 qrels 的题上计算。
- 多次搜索分别报首次调用与整 run 去重证据列表指标；合并顺序按实际首次召回顺序，不按模型喜好重新排列为更好排名。
- citation validity 的分母是输出引用，检查是否来自本 run 冻结账本；claim support/overclaim 依赖独立标注，不能由结构覆盖率冒充。
- retry recovery 至少区分“首轮未过→重试后验证通过”和“首轮答案错→重试后独立 gold 命中”，二者各有分母。
- emergency recall 分母为有紧急真值的问题；误报率分母为非紧急真值问题；unsafe rate 在有独立安全标注的输出上计算，并单列已回答子集和总体。
- illegal action rate 按动作次数，protocol completion 按 run；forced stop 记录动作预算、token、时间等原因，不能与模型主动停止混算。
- token/调用成本含结构化、Planner、Verifier、Generator、MedSearch 内轮及失败请求。GPU 用同一采样方法测 SearchAgent；未知费用/GPU 峰值不是 0。
- API 价格按有效日期冻结，缓存命中分项；provider 不给完整 usage 则记录 unknown/估计依据，不输出精确伪值。

### 10.2 数据准入

只读引用已有资产，先列 dataset manifest 再跑小样本。memory_build、dev、final_test 按来源 ID/题干与改写四层去重；语义近重复使用固定 encoder/阈值和裁决报告。V2 入库使用已脱敏的流程记录，不直接向角色暴露已解历史题的 gold 答案。最终测试只允许冻结配置推理与评测，不能随题追加记忆、更新阈值或筛选 prompt。

研究用公开测试题可完整保存以支持精确回放；真实含个人信息输入默认不保存原文，仅允许脱敏重放。哈希链能发现内容变化，但不能阻止拥有文件写权限的人重写整个文件，不宣传存储具备防篡改认证。离线 replay 重建的是已记录状态与动作，不证明相同远程模型今天会输出相同文本。

### 10.3 自动生成目录，不计入固定文件总数

```text
experiments/<experiment_id>/
  manifest.json       # 配置、数据、prompt、模型、seed、baseline、memory hash
  per_case.jsonl      # 所有题的结果索引，包含失败
  metrics.json        # 分母、空值及真实指标
  report.md           # 表格与案例
  runs/<run_id>/
    manifest.json     # 初始不可变清单
    events.jsonl      # 追加事件
    summary.json      # 最终状态与耗用
    answer.json       # 公开结构化答案
    payloads/<hash>.json  # 数量随调用变化；脱敏对象
memory_snapshots/<version>/   # 仅 V2，7 个文件见 8.1
```

一次 run 固定 4 个核心文件 + N 个 payload；一次 experiment 固定 4 个汇总文件 + N 个 run 目录。报告和模型结果都不在规划阶段预填。

## 11. 学生实施顺序与验收

以下是队列，不要求一天实现完。**每天只取一行作为核心任务**；若超过两小时，再按该行函数拆小。V0 是 V1 前置验收，不跳过。

| 顺序 | 学生核心任务 | 交付与完成标准 |
|---|---|---|
| V1-01 / V0 | 五大 Schema 与完整迁移表 | 五种对象各有合法/非法例子；空 query、未知 ID、非法边、超 retry 被拒；能说明每个字段谁生产谁消费 |
| V1-02 | budget.py | 预留/实耗/超时/三类重试可以用假时钟证明终止，无负数与重复结算 |
| V1-03 | trace/store + replay | 一个假 run 可离线复原；删事件/改 payload 必报错；普通日志无秘密/隐藏推理 |
| V1-04 | tools registry + retrieval normalization | 假 HTTP 通过，未知工具/参数/伪造 evidence 被拒；不接真实服务 |
| V1-05 | 假角色 + engine | 三路由、PASS/RETRY/ABSTAIN 和终态完整通过；没有无限循环 |
| V1-06 | 真实 DeepSeek 单角色适配 | 先验一个结构化响应；记录官方能力/ID与实际 usage；其余角色复用同一 adapter |
| V1-07 | MedSearch 部署接口准入 | 验证选定 checkpoint/tokenizer/模板；先单个 search/answer 动作，再只读检索连接，不加载训练器 |
| V1-08 | SearchAgent 有限动作循环 | 小样本保留实际 query、证据和停止原因；非法格式与轮次上限可验收 |
| V1-09 | Verifier + Generator | 伪造引用拒绝；返工次数有限；Generator 不能添加未知声明；角色输入相互隔离 |
| V1-10 | B0～B3 小规模统一评测 | 先 5～10 条工程 smoke；完整报告成功、返工、弃答和失败；smoke 不作医学效果结论 |
| V1-11 | 冻结正式评测与总结 | 先冻结 split/seed/标注/版本，再跑可解释规模；记录独立答案、安全、证据、成本指标 |
| V2-01 | 固定审核记录与图 Schema | 手工最小三层图可解释，每条 insight 有来源/反证/审核，拒绝测试集来源 |
| V2-02 | 快照构建/加载/向量召回 | 文件哈希可验证；先验向量检索和 B4，不开图扩展 |
| V2-03 | 一跳图扩展与角色投影 | 最多一跳；四角色字段不同；高风险弱经验被过滤；token 上限可验收 |
| V2-04 | B3/B4/B5 配对运行 | 同冻结起点，关闭记忆可回退，运行前后快照 hash 相同 |
| V2-05 | 记忆效果与失败总结 | 报 helpful/neutral/harmful/unknown、token 增量及图扩展消融；无提升也保留结果 |

V1 完成要求：真实三路由记录、两个模型和检索接口身份明确、一次反馈返工边界、完整轨迹/回放、B0～B3 真实指标与失败分析、学生能解释调用图。V2 完成要求：满足 V1，冻结记忆来源无泄漏、角色投影/一跳/预算通过、B3/B4/B5 公平对比、harmful 与关闭记忆验收。不把待办数量或测试通过数写成医学能力提升。

后续实现完成后的 CMD 命令入口规划（**当前尚不存在，不能现在运行当验收**）：

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring\MedAgent"
python -m pip install -e ".[test]"
medagent run --config configs\v1.toml --baseline B3 --input tests\fixtures\questions.jsonl --case-id smoke-search
medagent evaluate --config configs\v1.toml --baseline B3 --dataset PATH_TO_FROZEN_DATASET
medagent replay --run-dir PATH_TO_RUN
python -m pytest tests\unit tests\integration -m "not live"
```

V2 增加 `medagent freeze-memory` 和 `medagent validate-memory`，以及 `--config configs\v2.toml --baseline B4/B5`。不新建大量 run_xxx.py 包裹 CLI；旧 scripts 目录暂空即可。API 密钥由环境或本地秘密管理传入，不能写入命令样例、配置或 git。

## 12. 实现前要冻结的候选决定与剩余不确定性

| 决定 | 本规划推荐落点 | 需要的准入证据 |
|---|---|---|
| 执行底座选型 | 优先验证 Python 医学主流程 + DSH 官方 Python SDK；Pi 备选，正式采用待顶部验收 | 隔离角色、事件/usage、错误与终止通过；业务状态机由 Python 掌握，底层请求重试只有一个明确负责人；通过后更新旧工程清单 |
| 模型服务 | DeepSeek 单次 HTTP + MedSearch 本地单动作生成，Harness 执行工具 | 官方能力核验、真实固定 checkpoint、模板、停止标签、timeout、usage smoke |
| 异步取消 | 统一 async HTTP 调用链，预算由 Runtime 独占 | 超时假服务器测试与取消后不改状态；远端计费不确定性有记录 |
| Schema 与依赖 | 标准库 dataclasses/enum/json/tomllib/pathlib/hashlib/argparse；边界 JSON Schema 校验库、HTTP 客户端和 pytest；V2 按已验环境启用数组/encoder | 实际安装/依赖锁定及 import 验证后记录版本；不新增大型 Agent 框架，不装未经使用的组件 |
| DIRECT 与 Generator | DIRECT 固定信息性模板；输出采用声明 ID/模板选择 | 五大 Schema 样例+Generator 恶意新增事实测试 |
| B1/B2 的验证 | 结构化与证据 ID 最低门禁，不做语义验证和返工 | 基线调用图、调用次数和 check_level 记录；不与 B3 PASS 混称 |
| 证据稳定身份 | 源 ID 优先，语料版本+内容 hash 回退 | 样例返回是否带源 ID；qrels 映射实测；无法映射明确记缺失 |
| 记忆发布 | 离线人工/固定评测审核，无自动提炼与激活 | approval + dedup 报告、同源 B4/B5、只读文件哈希 |

这些是工程细化中的候选决定，尚未回填成 PROJECT_BOOK 的冻结事实。后续实现到相应步骤时同步决策文档与主干，避免长期两份冲突设计；本次不以附件内部的“下一项只做 V0”限制用户明确要求的 V1/V2 全量规划，也不把附件里的协作分工当作启动额外代理或代写代码的授权。
