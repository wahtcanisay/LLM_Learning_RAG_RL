"""MedAgent 的业务数据契约。"""

from dataclasses import dataclass
from enum import Enum

# ============================================================================
# P0 / P1：DSH 执行结果与 Planner 校验协议
#
# 这一部分负责：
# 1. 描述 DSH 返回结果的结构；
# 2. 描述 Planner 输出的 route、risk_level、search_query 等字段；
# 3. 校验 Planner 输出是否符合既定协议；
# 4. 确保 Planner 不能绕过 Python 侧的风险下限。
#
# 相关对象：
# Usage、RoleResult、Route、RiskLevel、SafetyResult、Plan
# 相关函数：
# validate_plan()
#
# 这一部分只负责数据结构和输出校验
# ============================================================================

@dataclass(frozen=True)
class Usage:
    """模型调用的 token 统计。"""
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    known: bool

@dataclass(frozen=True)
class RoleResult:
    """角色执行后的统一结果。"""
    role: str
    session_id : str
    payload: dict[str, object]
    finish_reason: str
    event_types: tuple[str, ...]
    usage: Usage

class Route(str, Enum):
    """Planner 允许选择的后续处理路线。"""

    DIRECT = "DIRECT"
    SEARCH = "SEARCH"
    EMERGENCY = "EMERGENCY"

class RiskLevel(str, Enum):
    """安全策略使用的有序风险等级。"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"

@dataclass(frozen=True)
class SafetyResult:
    """预检结果和风险下限。"""
    risk_floor : RiskLevel
    rule_ids : tuple[str, ...]
    rules_version : str

@dataclass(frozen=True)
class Plan:
    """经过校验的 Planner 计划。"""
    route: Route
    risk_level: RiskLevel
    search_query: str
    constraints: tuple[str, ...]
    reason_codes: tuple[str, ...]


def validate_plan(
    payload: object,
    safety: SafetyResult,
) -> Plan:
    """Validate and normalize one Planner output.

    Input:
        payload (dict): Planner 返回的对象
            route: 路由名称，取值为DIRECT/SEARCH/EMERGENCY
            risk_level: 风险等级，取值为 LOW/MEDIUM/HIGH/EMERGENCY
            search_query: 搜索查询词
            constraints: 字符串约束列表
            reason_codes: 字符串原因码列表

        safety (SafetyResult): Python 侧安全预检结果
            risk_floor: Planner 不得降低的最低风险等级
            rule_ids: 命中的安全规则 ID 元组
            rules_version: 安全规则版本

    Output:
        Plan: 校验后的不可变计划对象。
            route: Route枚举
            risk_level: RiskLevel枚举
            search_query: 已校验的查询词
            constraints: 字符串元组
            reason_codes: 字符串元组
    """
    # 基础类型
    # region payload 和 safety 类型校验
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    if not isinstance(safety, SafetyResult):
        raise TypeError("safety must be a SafetyResult")
    # endregion

    # 必需字段
    # region 必需字段校验
    required_fields = (
        "route",
        "risk_level",
        "search_query",
        "constraints",
        "reason_codes",
    )
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"missing field: {field}")
    # endregion

    # 字符串枚举
    # region route 和 risk_level 类型校验
    raw_route = payload["route"]
    raw_risk_level = payload["risk_level"]
    if not isinstance(raw_route, str):
        raise TypeError("route must be a string")
    if not isinstance(raw_risk_level, str):
        raise TypeError("risk_level must be a string")
    # endregion

    # region route 和 risk_level 取值校验
    valid_routes = {item.value for item in Route}
    valid_risks = {item.value for item in RiskLevel}
    if raw_route not in valid_routes:
        raise ValueError("invalid route")
    if raw_risk_level not in valid_risks:
        raise ValueError("invalid risk_level")
    # endregion

    route = Route(raw_route)
    risk_level = RiskLevel(raw_risk_level)

    # 字符串列表
    # region constraints 和 reason_codes 校验
    constraints = payload["constraints"]
    reason_codes = payload["reason_codes"]
    if not isinstance(constraints, list) or not isinstance(reason_codes, list):
        raise TypeError("constraints and reason_codes must be lists")
    if any(not isinstance(item, str) for item in constraints):
        raise TypeError("constraints must contain only strings")
    if any(not isinstance(item, str) for item in reason_codes):
        raise TypeError("reason_codes must contain only strings")
    # endregion

    # 路由和查询词
    # region search_query 校验
    search_query = payload["search_query"]
    if not isinstance(search_query, str):
        raise TypeError("search_query must be a string")
    if route is Route.SEARCH:
        if not search_query.strip():
            raise ValueError("SEARCH requires a non-empty search_query")
    elif search_query != "":
        raise ValueError("non-SEARCH routes require an empty search_query")
    # endregion

    # 不得低于安全下限
    # region 风险等级和紧急路由校验
    if not isinstance(safety.risk_floor, RiskLevel):
        raise TypeError("safety.risk_floor must be a RiskLevel")
    risk_order = {
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.EMERGENCY: 4,
    }
    if risk_order[risk_level] < risk_order[safety.risk_floor]:
        raise ValueError("plan risk level is below the safety floor")
    if route is Route.EMERGENCY and risk_level is not RiskLevel.EMERGENCY:
        raise ValueError("EMERGENCY route requires EMERGENCY risk level")
    if (
        safety.risk_floor is RiskLevel.EMERGENCY
        and route is not Route.EMERGENCY
    ):
        raise ValueError("EMERGENCY safety floor cannot be downgraded")
    # endregion

    # 转为不可变元组
    return Plan(
        route=route,
        risk_level=risk_level,
        search_query=search_query,
        constraints=tuple(constraints),
        reason_codes=tuple(reason_codes),
    )


# ============================================================================
# P2-a：Search → Verify → Retry 工作流协议
#
# 这一部分负责：
# 1. 描述 SearchAgent 的搜索任务和搜索结果；
# 2. 描述 Verifier 对候选答案的判定；
# 3. 描述 PASS、RETRY、ABSTAIN 三种验证结果；
# 4. 描述最多一次重试后的最终状态。
#
# 相关对象：
# VerificationVerdict、CycleStatus、SearchTask、SearchResult、
# VerificationResult、SearchCycleResult
#
# 这一部分描述工作流数据
# ============================================================================

class VerificationVerdict(str, Enum):
    """Verifier 对候选答案的判定。"""

    PASS = "PASS"
    RETRY = "RETRY"
    ABSTAIN = "ABSTAIN"

class CycleStatus(str, Enum):
    """搜索验证循环的最终状态。"""

    VERIFIED = "VERIFIED"
    ABSTAINED = "ABSTAINED"

@dataclass(frozen=True)
class SearchTask:
    """SearchAgent 的一次搜索输入。

    question: 原始问题。
    query: 当前搜索词。
    attempt: 搜索次数，从 0 开始。
    verifier_feedback: 上一次验证反馈，首次搜索为空。
    """

    question: str
    query:str
    attempt: int
    verifier_feedback: tuple[str, ...]

@dataclass(frozen=True)
class SearchResult:
    """SearchAgent 返回的一次搜索结果。

    attempt: workflow 的搜索尝试编号，从 0 开始。
    candidate_answer: 当前候选答案。
    evidence: 本次检索返回的 Evidence，按检索顺序保留原文。
    context: 模型续写使用的内部上下文，可含 <think>。
    """

    attempt: int
    candidate_answer: str
    evidence: list["Evidence"]
    context: str | None

@dataclass(frozen=True)
class VerificationResult:
    """Verifier 对搜索结果的判定。

    verdict: PASS、RETRY 或 ABSTAIN。
    failure_codes: 验证失败原因。
    retry_query_hint: RETRY 时的下一次查询词，其他情况为 None。
    """

    verdict: VerificationVerdict
    failure_codes: tuple[str, ...]
    retry_query_hint: str | None

@dataclass(frozen=True)
class SearchCycleResult:
    """一次搜索验证循环的最终结果。

    status: VERIFIED 或 ABSTAINED。
    search: 最后一次搜索结果。
    verification: 最后一次验证结果。
    attempts: 实际搜索次数，只能是 1 或 2。
    reason: 结束原因。
    """

    status: CycleStatus
    search: SearchResult
    verification: VerificationResult
    attempts: int
    reason: str

# ============================================================================
# P2-c：公共审计 Trace 协议
#
# 这一部分负责：
# 1. 记录一次运行中的公开状态转移；
# 2. 标记执行角色、调用编号、token 使用和证据引用；
# 3. 为后续 TraceStore 写入 events.jsonl 提供统一数据结构；
# 4. 通过白名单投影，阻止密钥、完整 prompt、完整回答和隐藏推理进入公开记录。
#
# 相关对象：
# TraceEvent、Usage
#
# 相关函数：
# redact()
#
# TraceEvent 只描述“发生了什么”，不负责执行动作或改变工作流状态。
# redact() 只处理顶层字段，只保留调用方明确允许公开的字段。
# ============================================================================


@dataclass(frozen=True)
class TraceEvent:
    run_id: str                     # 标记事件属于哪次运行
    sequence: str                   # 保证统一运行内的时间顺序，与run_id定位唯一事件
    actor: str                      # 区分 Planner/SearchAgent/Verifier/Generator
    state_before: str               # 记录状态转移起点
    state_after: str                # 记录状态转移终点
    call_ids: str| None             # 记录具体模型或者检索调用
    usage: Usage | None             # token使用统计
    input_refs: tuple[str, ...]     # 输入证据/公开对象
    output_refs: tuple[str, ...]    # 输出证据/候选答案/结果对象
    error: str | None               # 错误信息

# ============================================================================
# ReplayResult：Trace 回放结果
#
# 这一部分负责：
# 1. 汇总一次运行的 manifest、事件、答案和摘要；
# 2. 为回放、审计和结果检查提供统一数据结构；
# 3. 允许未完成运行暂时缺少 answer 或 summary。
#
# ============================================================================

@dataclass(frozen=True)
class ReplayResult:
    manifest: dict[str, object]                 # 记录本次运行的初始清单
    events: tuple[dict[str, object], ...]       # 记录运行期间的公开事件
    answer: dict[str, object] | None            # 记录最终答案，未完成时可以为空
    summary: dict[str, object] | None           # 记录运行摘要，未完成时可以为空


# ============================================================================
# P3-a：Evidence 检索证据契约
#
# 这一部分负责描述 BM25/检索服务返回后，规范化阶段保留下来的最小证据。
#
# 旧 evidence_id/source_id/corpus_ref 字段废弃；按检索顺序传递原文。
# ============================================================================

@dataclass(frozen=True)
class Evidence:
    text: str                   # document.contents 原文
    title: str | None = None    # document.title，服务未提供时为空
    score: float | None = None  # 检索排序分数
