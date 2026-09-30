# MedAgent Harness 项目书

## 当前实现状态（2026-09-25）

> 状态定义：**已实现**表示已有代码并通过本地测试；**部分实现**表示仅 fake/局部路径通过；**延后**表示主动保留、等待触发条件；**待实现**表示尚无对应业务代码。详细证据见 [初期工程汇报](docs/reports/INITIAL_PROGRESS_REPORT_2026-09-22.md) 和 [工程进度书](PROGRESS.md)。

| 架构模块 | 当前状态 | 已有证据 | 未完成或等待条件 |
|---|---|---|---|
| DSH 执行底座 | 部分实现 | 固定 `dsh-v0.1.5-rc.2`；源码 SDK + 同 commit Node runtime；单 Planner 假服务 smoke | Planner/Verifier 真实 DSH 隔离、401/503/挂起清理和付费 API 调用待集成验证 |
| Python RoleRunner | 部分实现 | fake backend 下角色 prompt/session、结果解析、usage、deadline/close 已通过测试 | 真实 DSH backend 与进程退出行为待验证 |
| Planner 契约 | 已实现 | `Plan`、`SafetyResult`、`validate_plan()` 已通过测试 | 真实 Planner 输出接入仍在 P3 |
| Search→Verify→Retry | 已实现（fake） | PASS/RETRY/ABSTAIN、最多两次外层 SearchAgent.run 调用及终态测试通过 | 具体 SearchAgent 与正式 Verifier 的完整集成待验；次数不等于内部检索次数 |
| 公共审计 trace | 已实现 | `redact()`、`TraceStore.start/append/finish`、`replay()` 已通过测试 | 尚未接入完整 workflow；当前 replay 只读取写入侧生成的产物 |
| 预算账本 | 延后 | 设计位置保留 | 接入付费 API、多层重试或批量实验前重新评估 |
| 确定性安全 preflight | 延后 | `SafetyResult.risk_floor` 契约保留 | 等待审核后的医学规则集与风险测试集 |
| 完整 workflow 状态机 | 等待依赖 | 项目书状态图保留；当前搜索闭环可独立运行 | DIRECT、EMERGENCY、Generator 具备后再实现多路由 `run()` 和必要状态约束 |
| SearchAgent / retrieval / verification | 部分实现（P3） | Evidence、检索规范化及 HTTP 假响应已验；具体 SearchAgent 的 6 项 fake 循环测试通过 | 真实 3B/BM25、精确可见证据与留档待验；verification 尚未实现 |
| Generator、evaluation、CLI | 待实现 | 无正式业务代码 | P3/P4 按工程书推进 |
| V2 memory | 待实现 | 仅项目与实验设计 | V1 冻结后进入 P5/P6 |

2026-09-26 本地普通回归为 **65 tests passed**，其中 6 项覆盖具体 SearchAgent.run 的 fake 控制流。该结果不证明医学正确性、真实 DSH 多角色隔离或真实模型质量。2026-09-22 的 56 项和 2026-09-25 的 59 项结果保留为历史记录。

> 状态：设计主干 v0.3（2026-09-22，待实现范围收敛）  
> 用途：求职作品、工程复现与医学 Agent 研究；不提供临床诊断或治疗建议。

## 一、项目定位

项目名称暂定为：

> **MedAgent Harness：可审计、自进化的医学证据搜索 Agent 运行框架**

本项目构建一套可展示、可解释、可测试的医学专用 Agent Harness。自行实现医学业务流程、
验证返工、预算、证据与记忆治理；优先验证复用 DSH 的有限执行能力，不要求从零重写模型通信和底层事件循环。
项目负责协调模型适配、消息协议、工具权限、安全分流、轨迹记录、实验评测与回放，并区分复用与自研部分。

在线系统使用两类 LLM（V2离线Qwen例外见V2节）：

1. **DeepSeek-V4.1-Flash**：承担 Planner、Verifier 和 Generator 三个逻辑角色；API 调用时使用
   官方当前模型 ID `deepseek-flash`，模型版本必须写入每次运行清单。
2. **MedSearch-R1**：项目已有的本地医学 SearchAgent，负责搜索决策、query 改写、证据阅读和
   基于证据形成候选答案。

三个 DeepSeek 角色可以共享同一基础模型，但必须使用独立的角色 Prompt、输入字段、输出 Schema、
上下文和记忆视图。不能因为它们使用同一个模型，就把三个角色合并成一次不透明调用。

## 二、求职展示目标

项目最终应证明以下能力：

- 能独立设计 Agent Harness，而不依赖大型多 Agent 框架替代核心控制逻辑；
- 能分别接入外部角色执行器与本地 SearchAgent，并统一预算、结果和审计协议；
- 能用类型化协议和显式状态机组织多角色协作；
- 能实现搜索动作解析、权限限制、参数校验、超时、预算和有界错误恢复；
- 能处理医学高风险请求，并保存可审计的安全分流证据；
- 能实现有上限的 `Search → Verify → Retry` 闭环；
- 能用真实指标比较无检索、固定检索、Agent 搜索、验证返工和长期记忆；
- 能复现失败轨迹，并说明性能、延迟、token 和 API 成本之间的权衡；
- 能实现受反馈约束、可撤销的经验记忆，而不是把“自我总结”直接称为自进化。

## 三、协作和文档治理

本文件是项目的“脊柱文档”，负责维护稳定主干：项目目标、非目标、组件职责、数据契约、安全边界、
状态机、实验基线、阶段顺序和完成标准。

后续协作分工如下：

- 当前 Codex 负责维护主干一致性，发现设计冲突、缺失验收和范围膨胀；
- Astra 用于补充更细致的模块设计、异常分支、Schema、测试矩阵和实现建议；
- 其他模型可以用于逐个指导学生编写代码，但不能绕过项目书的接口和验收要求；
- 学生是关键代码的主要实现者和最终解释者；代理只有在学生明确要求代写时才直接实现代码。

Astra 或其他模型提出的新内容应先作为“候选设计”记录。只有满足以下条件后才能并入主干：

1. 说明解决什么问题；
2. 说明输入、输出、失败行为和测试方式；
3. 不破坏已经冻结的模型数量、安全边界和可比实验；
4. 如改变主状态机、数据集、奖励、检索后端或核心指标，必须单独记录设计决策；
5. 未经实验验证的判断必须标记为假设，不能写成已证明结论。

## 四、总体架构

```text
User Request
    │
    ▼
Preflight Safety Gate
    │
    ▼
Planner / Router（DeepSeek-V4.1-Flash）
    ├── EMERGENCY ──► Emergency Policy ──► Generator ──► Final Response
    │
    ├── DIRECT ─────► Generator ───────────────────────► Final Response
    │
    └── SEARCH
           │
           ▼
    MedSearch-R1 SearchAgent
           │ candidate answer + evidence + trace
           ▼
    Verifier（DeepSeek-V4.1-Flash + deterministic checks）
           ├── PASS ─────► Generator ──────────────────► Final Response
           ├── RETRY ────► SearchAgent（bounded loop）
           └── ABSTAIN ──► Generator ──────────────────► Safe Refusal

所有步骤 ──► Trace Store ──► Evaluation / Replay ──► G-Memory-lite
```

安全预检在完整方案中位于 Planner 之前，但当前等待审核规则和风险样例，不实现关键词/结构化特征/LLM 分类的组合引擎。P3 仅验收限定研究问题的检索链，不能声称预检已通过或完整医学安全能力已经完成。

## 五、核心组件

### 5.1 Harness Runtime

职责：创建并推进运行状态、调度模型和工具、检查结构化输出、执行超时/重试/预算、生成不可变事件序列，
并支持离线回放。Runtime 不负责医学推理，也不把业务规则藏进任意 Agent Prompt。

### 5.2 角色执行与搜索边界

DeepSeek角色经Python RoleRunner调用DSH；RoleRunner提供 `run(role, prompt, session_id, timeout_s) -> RoleResult`，负责隔离配置、完整结束、JSON/角色Schema校验及公开事件/usage提取。system提示由角色定义生成，不能只依赖用户传session_id实现隔离。

MedSearch-R1是独立SearchAgent，保持 `<search>/<answer>` 动作协议和3B生成→检索→回填循环，不注册为DSH原生模型工具。不强行让“单次DeepSeek角色调用”和“多轮SearchAgent任务”共享一个generate接口。

完整 V1 两条路径都接入 Python 额度与 trace；P3 本地单题先用显式调用上限和超时，不依赖尚未实现的预算包装器。SDK/runtime保持同commit；普通日志不得保存密钥、可识别医疗隐私与隐藏推理。DSH底层状态不等于医学RunState，业务状态只由Python workflow改变。

### 5.3 Planner / Router

Planner 接收原始问题和安全预检结果，输出：

```json
{
  "route": "DIRECT | SEARCH | EMERGENCY",
  "risk_level": "LOW | MEDIUM | HIGH | EMERGENCY",
  "search_query": "",
  "constraints": [],
  "reason_codes": []
}
```

原始问题始终保留，`SEARCH` 必须包含可执行 query，`EMERGENCY` 不进入普通医学答案链路。自由文本解释
不能代替结构化字段，路由错误必须单独评测。

> **待讨论：DIRECT 可以回答哪些问题（2026-09-13）。** 工程规划提出，DIRECT 只回答“这个系统能做什么”等固定说明，普通医学知识问题则走 SEARCH。这会缩小直接回答的范围：例如“什么是抗体”这样的医学知识题，是否必须检索，需要在设计路由时具体决定。原项目书并未冻结“DIRECT 只能用固定模板”。当前只标记问题，待实现 Planner / Router 时结合实际问题样例讨论，不默认采用工程规划的收窄方案。

### 5.4 Emergency Policy

紧急分支采用固定策略约束 Generator：

- 明确建议立即联系当地急救或线下医疗服务；
- 不输出诊断结论，不建议用户自行更改处方药剂量；
- 不因检索耗时延迟紧急提示；
- 补充信息只能放在紧急行动建议之后；
- 输出包含研究/信息用途边界。

这不是医疗器械级安全系统。项目只能报告测试集指标，不能声称临床可用。

### 5.5 MedSearch-R1 SearchAgent

P3 的 SearchTask 保留原题、query、attempt 和 Verifier 反馈供 workflow 传递；SearchAgent 的初始模型上下文只使用 query，不重复拼入可能较长的原题。原题仍由外层用于验证最终候选。输出候选答案、真实返回且可见的证据及公开动作/停止原因。候选证据集合不等于已证明的逐声明支持；声明—证据映射在同一次语义 Verifier 调用中产生并由代码核对原答案跨度，不强制增加 structure 模型调用。完整 V1 的约束和总额度随实际调用方接入。SearchAgent 不决定最终输出是否合法，第一版只连接已有 MedSearch 医学检索服务。

架构决定（2026-09-25）：统一使用具体类 SearchAgent，通过 `run(task)` 供 Python workflow 调用，不新增 LocalSearchAgent。search_agent.py 已有具体类并复用 MedSearch 动作解析；workflow.py 中同名 Protocol 仍在，收敛尚未落实。

当前 SearchResult 为 attempt/candidate_answer/evidence/context；evidence 两条返回路径均为 list。context 从 query 开始，符合当前只用 query 的决定；原题和反馈不自动拼入模型上下文，context 也不等于可公开 trace。默认一次 run 最多生成 3 次动作、执行 2 次检索；workflow 最多调用 run 两次，即最多一次语义返工，两层次数不可混用。fake 测试已覆盖成功回填、空召回、限次、非法动作和检索超时；精确可见片段、停止原因、用量与留档仍待验收，真实 3B 单动作服务和可加载 checkpoint 尚未接通。最小接口与下一步要求见工程书第 3、4.4 节。

### 5.6 Verifier

Verifier 同时使用确定性检查和 DeepSeek 审查：格式是否完整、证据是否来自本次实际可见内容、核心声明是否有证据、
证据是否冲突、是否超出证据范围，以及是否存在危险或过度确定的医学建议。

```json
{
  "verdict": "PASS | RETRY | ABSTAIN",
  "failure_codes": [],
  "unsupported_claims": [],
  "retry_query_hint": null,
  "public_summary_constraints": []
}
```

首版 `RETRY` 最多一次，稳定后才能增加。达到上限仍不通过时必须 `ABSTAIN`。

### 5.7 Generator

Generator 只接收允许公开的结构化内容，不接收无界的全部内部轨迹。它负责整理已验证答案、保留证据和
不确定性、不增加上游没有提供的新医学事实，并根据路由生成正常回答、紧急提示或安全弃答。

> **待讨论：Generator 能否用自己的话整理答案（2026-09-13，2026-09-23 更新）。** 本节冻结的是“不增加上游没有提供的新医学事实”，尚未冻结必须逐字输出。候选方案是仅编排获准原文，或允许受约束的改写并检查最终输出；留到 Generator 模块设计时讨论。上游获准内容及其支持证据必须传到 Generator，具体传递、关联和引用展示方式尚未确定，不预设声明/证据 ID、序号、索引或标题为引用键。本轮不选择方案；相关 Schema、函数和测试暂按候选看待。

“合规总结”仅指遵循项目输出政策，不代表取得法律、医疗器械或临床合规认证。

## 六、显式状态机

```text
RECEIVED
  → SAFETY_CHECKED
  → PLANNED
  → DIRECT_GENERATION | EMERGENCY_GENERATION | SEARCHING
  → VERIFYING
  → RETRYING | VERIFIED | ABSTAINED
  → GENERATING
  → COMPLETED | FAILED
```

每次状态变化形成事件，当前只包含 `run_id`、`sequence`、actor、前后状态、`call_ids`、usage、
输入输出引用和错误。模型版本放在运行 manifest 中，不在单个事件重复保存。状态迁移由 Runtime 控制，Agent 只能提出结构化动作。任务启动时可生成独立随机 run_id；确有独立子任务时才另设 task_id，配合局部上下文和归属检查维持任务边界，与检索数据无关。已有 Trace 字段和实现保留，不要求每条 Evidence 带 ID。

## 七、工具与权限边界

首版工具只有固定地址的只读医学检索；用明确的请求/响应字段检查、query 限制、超时、返回文档预算与调用记录约束，不搭通用工具注册/Schema 框架。传输自动重试与动作纠错首版为 0，语义验证返工最多一次；未来恢复实验再显式增加。禁止 shell、任意文件读写、任意网页访问和临床系统访问。

## 八、自进化设计：G-Memory-lite

### 8.1 定义

“自进化”第一阶段不改变模型权重，而是让 Harness 跨任务积累、检索、验证和撤销协作经验。任何提升都
必须通过固定测试集和无记忆基线证明。

### 8.2 三层记忆

- **Query Graph**：记录脱敏问题表示、任务类型、风险、路由、搜索成本、Verifier 和外部评价状态；
- **Interaction Graph**：记录 Planner 计划、工具调用、必要的实际可见证据内容、候选答案、Verifier 失败码、重试和结果；
- **Insight Graph**：记录经验证的流程性经验，如安全路由、搜索策略、常见证据缺口和修复办法。

它们描述 Agent 协作经验，不是医学实体知识图谱。Interaction Graph 不保存模型隐藏思维链，Insight 也
不能从轨迹中创造新的医学事实。

### 8.3 角色定制记忆

- Planner 只获取路由、安全和 query 约束经验；
- SearchAgent 只获取搜索策略、失败 query 和证据选择经验；
- Verifier 只获取常见证据缺口、冲突和不安全表达；
- Generator 只获取表达边界、引用保留和弃答格式。

禁止把全部历史轨迹同时塞给所有角色。

### 8.4 记忆生命周期

```text
NEW TRAJECTORY → QUARANTINED → CANDIDATE → ACTIVE → RETIRED | REJECTED
```

记忆不能仅凭 LLM 自评进入 `ACTIVE`。支持证据按强度排序：程序可验证答案与证据、固定评测器、人工
审核、多次独立成功轨迹；LLM 自评只能产生候选。每条记忆保存来源、支持和反对轨迹、角色、范围、
置信度、版本、状态和撤销原因。

### 8.5 检索和更新边界

- `top_k` 从 1 或 2 开始，图扩展最多一跳；
- 先验证向量召回，再验证图扩展是否有独立价值；
- 高风险问题不使用弱验证 Insight；
- 记忆写入与当前回答分离，全部提炼结果可回溯；
- 支持关闭全部记忆并回放无记忆基线；
- 不保存可识别个人身份的医疗信息。

## 九、相关性失败风险

Planner、Verifier 和 Generator 共用 DeepSeek-V4.1-Flash，可能同时提出、认可并润色同一个错误。因此
Verifier 不能只做主观语言评分，Schema、实际可见内容边界、引用和预算必须由代码检查；有标准答案时必须使用
程序化指标；Verifier 的 PASS 也不能直接等同于医学正确。

## 十、评测设计

### 10.1 基线

| 编号 | 系统 | 搜索 | Verifier 返工 | 长期记忆 |
|---|---|---|---|---|
| B0 | DeepSeek Direct | 否 | 否 | 否 |
| B1 | Fixed RAG | 固定单轮 | 否 | 否 |
| B2 | Static Harness | MedSearch-R1 | 否 | 否 |
| B3 | Verified Harness | MedSearch-R1 | 是 | 否 |
| B4 | Episodic Memory | MedSearch-R1 | 是 | 简单相似轨迹 |
| B5 | G-Memory-lite | MedSearch-R1 | 是 | 三层受控记忆 |

一次只改变一个主要变量，尽量固定模型、测试集、检索后端、上下文和生成预算。

### 10.2 指标

- 答案：Accuracy、Exact Match/F1、支持性和过度断言比例；
- 证据：Recall@k、MRR、evidence hit、citation validity、claim-evidence support；
- Harness：route accuracy、协议完成率、retry recovery、非法动作率、强制停止率；
- 安全：emergency recall、误报率、unsafe answer rate、高风险错误路由率；
- 成本：平均/P95 延迟、调用数、搜索轮数、token、API 费用、SearchAgent 显存峰值；
- 记忆：hit rate、helpful/neutral/harmful rate、启用前后差异、撤销错误记忆后的恢复。

### 10.3 数据隔离

记忆构建集、开发集和最终测试集严格隔离；最终测试题及改写不得进入记忆。保存精确、规范化、词汇和
语义近重复报告。记忆实验从同一冻结初始状态开始，不允许只展示最优结果。

## 十一、阶段路线

### V0：Harness 骨架

不调用真实模型，先跑通类型、fake Search/Verify 有限闭环、执行错误和事件回放。预算账本与完整多路由状态机已按触发条件延后，不作为当前 P3 的前置空壳；相关功能启用时再补额度停止及合法/非法迁移验收。

### V1：无长期记忆的真实闭环

接入 DeepSeek-V4.1-Flash、MedSearch-R1 和只读检索服务。DIRECT、SEARCH、EMERGENCY 有真实记录，
建立 B0～B3，并分析成功、返工成功、弃答和安全失败案例。

### V2：有容量上限的离线记忆维护与在线只读增强

> 2026-09-15 按用户讨论更新：采用“200 道构建题、最多 100 条活跃记忆”的首轮配置。
> 本节是 V2 当前范围，优先于旧工程规划中 V2 仅加载人工固定数据的说明；V1 的角色、接口和验收保持不变。
> 以下为待实施方案，不代表数据已划分、记忆已构建或效果已验证；细节遇到实际 dev 问题再调整。

#### V2.1 数据与首轮规模

- 先冻结在线模型/checkpoint、Prompt、检索器及预算。只读引用既有 RL dev；2026-09-09 合并清单记录其为 396 条，实施前复核当前文件行数、哈希及重复情况。
- 固定 seed=20260915，按题型/来源在可用数据中划分 200 道 `memory_build`，其余作为 `memory_eval`；如仍为 396 条且隔离检查无剔除，则余下 196 条。隔离不足时如实减少数量，不补入测试题凑数。
- 保存独立的 ID、split、来源、哈希及去重清单，不修改 MedSearch 原 dev。构建题、改写及语义近重复不得进入评测侧；检查与已有 SFT/RL 数据的交叉使用记录。旧 dev 已用于模型开发，这轮结果只称内部开发评测，不称全新独立测试。
- 200 道题是构建问题数，不是保证产生 200 条记忆。生成一次完整 Harness 公开交互轨迹流；旧 RL 轨迹缺少其他角色，只能辅助调试，不能替代该轨迹流。

#### V2.2 离线维护与容量

- 每完成 10 道构建题的轨迹处理，离线维护一次，最多 20 批；先处理 10 道检查流程，再继续至 200 道。首轮固定无记忆的构建轨迹，便于不同维护策略复用同样输入。
- V2 允许增加本地 Qwen3 中小模型作为离线维护组件，候选为 Qwen3-4B-Instruct-2507，具体部署在实施时核验。它仅提出候选经验和合并建议，不接管在线 Planner/Verifier/Generator，也不改变 MedSearch-R1。此为项目“两类 LLM”约束的 V2 离线限定例外，不能宣称整个 V2 仍只有两类 LLM。
- 每批只读公开动作、必要证据片段、失败码、独立结果反馈和相关旧记忆，不读隐藏思维链。程序检查格式、来源、兼容性和容量；提炼后的适用条件与反例经人工或固定独立评测审核后才可发布。答对或 Qwen 自评均不能单独证明经验正确；金标答案不写入在线经验。
- `max_active_memories=100` 是起始容量，尚非经实验验证的最优点。计数单位为一个唯一、可注入的经验/代表案例条目 ID；角色投影不重复计数，Query/Interaction 来源节点不算额外可注入条目。所有随快照装载的节点、边、向量、来源摘要均计入总字节统计与总容量约束，不能借来源节点无限扩张。
- 另设单条长度与快照总字节上限，具体值在实现时按真实长度确定。只保留合格条目，不凑满 100 条；先排除失效/有害条目、合并明确重复、按角色/失败类型保留代表，再在同类同质量条目内按 LRU 淘汰。维护仅发生在构建阶段，评测访问不修改下一批记忆排序或快照。
- 记录实际候选数、活跃数、拒绝/合并/淘汰数。若从未超过 100 条，不能声称已验证容量淘汰效果；若候选没有可区分的使用历史，LRU 仅是确定性时间顺序基线，需明确口径。
- 来源轨迹按实验记录保留，退出在线索引不等于删除历史证据。每批产出不可变快照，在线任务仅加载指定版本；不做逐题自动激活、在线权重更新或程序意外退出后的恢复。

#### V2.3 配对评测与外部验证

- 必做首轮比较：M0/B3 无记忆、M2/B5 维护后冻结记忆，在同一 `memory_eval` 题集上独立运行。模型、采样参数、搜索/生成预算、encoder、top-k 和图扩展配置固定，原题答案不得流入推理输入。
- 在相同已审核候选、100 条容量及相同检索设置下，补 M1 简单 LRU 对照，才能评价复杂维护相对简单淘汰的价值。M1/M2 是维护策略标签，不等于原 B4/B5 的向量/图检索区别；原 B4/B5 对比保留为后续表示消融，避免同时改变维护与检索方式。
- 首轮优先在 200 道构建题处理完后评测全部剩余题，直接比较最终快照与无记忆。需要曲线时，从剩余题中预先固定 50 道 `probe`，只在构建量 0/50/100/200 时观察同样题目；一旦据 probe 调整方案，最终确认仅使用剩余未参与调参题，并分开报告。每 10 题维护不要求每批全量评测。
- Accuracy/EM 按题型预先固定规则，弃答与失败计入总体分母；开放答案不以宽松 token-F1 冒充医学正确。另报对→错、错→对、覆盖率、引用有效率、独立证据支持审核、搜索次数、token、维护/推理成本和记忆大小。无 qrels 时不声称有真值检索指标。
- 不预设正确率线性或单调上升，不只保存最优快照。首轮一个固定 seed 验证流程；正式结论再按预算做配对重复并报告样本数与不确定性。较少记忆也可能有用，记忆条数不等于效果。
- 外部验证首选 MIRAGE：优先将本轮最终冻结记忆直接测在未参与构建/调参且已做交叉去重的保留题上，检验迁移。沿用题目和答案评分，但记忆增强属于自定义扩展，不直接宣称符合官方 zero-shot 榜单协议。
- 如另做同域构建实验，可在尚未使用的 MIRAGE 候选池中预先划 200 道 build 与独立 eval，重建一个上限 100 条的快照；这是另一个实验，不能混称跨域迁移。既有最终保留题不自动转用途，需先列清单并明确用途调整；已用于构建的题永久退出本轮测试。
- MedAgentsBench 作为可选难题补充，可采用相同隔离思路；先核对与 MIRAGE、旧 RL 数据重叠，不能因仓库名字不同就视为独立数据。若可用独立样本不足则降低规模。首轮不同时跑两个新 benchmark，先完成 RL dev 内部验证，再选一个外部集合。
- 无记忆对照、构建轨迹与维护成本分开记账；评测期间快照不变、测试轨迹不回流。故障恢复仍由独立的假服务测试验证，QA 准确率不代替运行可靠性。

完成标准：200 道构建题（或经核验后的实际数）来源明确；活跃记忆不超过 100 条且有审核与来源；有隔离题上的有/无记忆配对结果、失败案例、真实成本和数据用途说明。未比较 LRU 时不声称维护策略优于 LRU，未做外部验证时不声称跨数据集泛化。

### V3：受控记忆更新

实现隔离、候选、激活、版本和回滚。未验证轨迹不得影响在线运行；注入错误记忆后，系统应能检测、
停用并恢复。只有真实实验支持时才使用“自进化提升”表述。

## 十二、工程目录与底座

有效目录、数据契约、每文件函数清单和实施顺序统一维护于 [ENGINEERING_PLAN_V1_V2.md](ENGINEERING_PLAN_V1_V2.md)，本文件不再保留第二份会漂移的目录树。

正式Python包采用 `MedAgent/medagent/`，DSH依赖只在role_runner接入，现有dsh_smoke是验证资产；不同时创建旧src/medagent层级或另一套DeepSeek客户端。Pi保留为必要时的替代候选，不并行接入两套底座。

## 十三、当前冻结决策

- 主目标是求职展示用的自定义医学 Agent Harness；
- 在线使用 DeepSeek-V4.1-Flash 和 MedSearch-R1；V2允许离线Qwen维护，不改变在线角色；
- DeepSeek 分别承担 Planner、Verifier、Generator；MedSearch-R1 是唯一 SearchAgent；
- 使用 DIRECT、SEARCH、EMERGENCY 三种路由；
- SearchAgent 候选答案必须验证，失败只允许有限返工，随后弃答；
- Evidence 只保留 text、可选 title、可选 score；废弃检索证据 ID、缺 ID 补哈希及按 ID 去重/冲突规则，document.id 可忽略，title 不是唯一标识。原文和实际可见片段随结果传递，SearchAgent → Verifier → Generator 的具体关联方式待接线时讨论；任务随机 ID 独立于检索证据；
- Generator 不补充未经支持的医学事实；
- 自进化首先指受控长期记忆，不等于在线修改权重；
- G-Memory 是方法来源，首版实现医学安全约束下的 G-Memory-lite；
- 首版只有只读本地医学检索工具；
- `../MedSearch/` 继续作为外部只读模型、数据、索引和训练资产依赖。

## 十四、待实现规格的补充边界

逐文件、逐函数规格统一见工程书，不再派发“补全所有 Schema、错误枚举和异常转移”的开放任务。当前仍需在对应模块开工前冻结：3B 真实服务协议、正式角色输出差异、DIRECT 范围与 Generator 改写政策、安全输入来源；V2 再冻结 encoder 和快照容量数值。

已实现 trace 不重复设计；未知接口不靠兼容层猜测；新增恢复机制须指出真实故障和必要性。规格补充必须说明输入、输出、内部动作、失败行为与最小验收。完整目标保留，延后项不得成为 P3 隐含前置依赖。

## 十五、当前唯一任务

P3-c 的 SearchAgent.run 基本 fake 循环已验；初始模型上下文只用 query，不把原题/反馈拼入。下一步核实本地 3B 单动作服务的真实请求/响应、模板、stop 和 checkpoint，接入真正的 `request_action`；回填限长与精确可见片段在真实模板明确后验收。可加载 checkpoint 与真实服务仍是外部未完成依赖。P3-a 的 Evidence 规范化与 P3-b 的只读检索假响应检查已通过；真实 BM25 端点只读请求待服务可用时记录，不用付费 API 替代。

现有 RoleRunner、fake 闭环和 trace 不重写。正式 DSH 双角色抓包、错误与挂起清理仍是后续真实角色集成准入，完整 workflow 等实际分支具备后再编码。付费前提醒并完成额度边界；V1 冻结之前不开始 200 题记忆构建。

---

## 十六、参考与声明

- G-Memory: *Tracing Hierarchical Memory for Multi-Agent Systems*, NeurIPS 2025；借鉴三层记忆与角色定制，
  但论文未在医学 QA 上验证；
- DeepSeek-V4.1-Flash 的 API 能力、模型 ID 和价格以运行当日官方文档为准；
- MedSearch-R1 的训练、检索与历史实验以 `../MedSearch/` 的真实配置和日志为准；
- 所有准确率、安全率、成本、延迟和改进比例必须来自真实实验，不得从论文结果外推。

