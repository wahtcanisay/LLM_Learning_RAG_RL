"""MedAgent 角色执行边界。

raw_result 是 backend.run(...) 返回的 DSH RunResult 类对象。为了便于
理解，它可以近似表示成下面这样的 JSON 结构：

{
  "session_id": "planner-001",
  "final_response": "{\\"route\\": \"SEARCH\\", \"search_query\\": \"aspirin evidence\\"}",
  "finish_reason": "completed",
  "events": [
    {
      "type": "assistant/message",
      "data": {
        "message": {
          "content": [
            {
              "type": "text",
              "text": "{\\"route\\": \"SEARCH\\", \"search_query\\": \"aspirin evidence\\"}"
            }
          ]
        },
        "usage": {
          "inputTokens": 17,
          "outputTokens": 9,
          "totalTokens": 26
        }
      }
    },
    {
      "type": "turn/end",
      "data": {
        "reason": {
          "kind": "completed"
        }
      }
    }
  ],
  "notifications": []
}

上面是协议的 JSON 化示意，实际返回值是 Python RunResult 对象，不是普通
JSON 字典。字段含义如下：

- session_id：DSH 实际执行的会话 ID。
- final_response：模型最终输出的原始文本。本项目要求它是一个 JSON
  对象字符串，但 DSH 只负责传输，不负责校验这个业务协议。
- finish_reason：执行结束原因，例如 "completed" 或 "error"。
- events：当前根会话的事件流；assistant/message 通常包含模型文本和
  usage，turn/end 包含本轮结束原因。
- notifications：更完整的 DSH 通知流。本文件当前不把它暴露到
  RoleResult 中。

本文件负责把 DSH 层的 raw_result 转换成 MedAgent 层的 RoleResult。
协议说明放在文件开头，具体字段的使用规则则写在对应代码旁边。
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from .contracts import RoleResult, Usage
from .prompts import get_system_prompt


def run_with_deadline(
    backend: Any,
    prompt: str,
    session_id: str,
    timeout_s: float,
) -> Any:
    """在工作线程中运行 backend，并在调用方设置的期限内返回。

    DSH 的 backend.run 是同步调用，直接调用可能让整个 harness 无限
    等待。工作线程让当前函数可以在超时时间到达后抛出 TimeoutError；
    外层 RoleRunner.run 的 finally 负责关闭 backend。
    """

    # 用一个容量为 1 的队列，把工作线程的成功结果或异常传回调用线程。
    result_box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        """捕获 backend.run 的结果或异常，交给外层线程处理。"""

        try:
            result = backend.run(prompt, session_id=session_id)
            result_box.put((True, result))
        except Exception as error:
            result_box.put((False, error))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # join 只等待指定时间，不会让当前线程无限阻塞。
    thread.join(timeout_s)

    # region 执行超时校验
    if thread.is_alive():
        # 这里不能安全地强制杀掉 Python 线程；由外层 finally 关闭 backend。
        raise TimeoutError(f"role execution exceeded {timeout_s} seconds")
    # endregion

    # 线程已结束时，队列中应该恰好有一个结果或异常。
    # region 工作线程执行结果校验
    ok, value = result_box.get_nowait()
    if not ok:
        raise value
    # endregion
    return value


def extract_usage(events: list[dict[str, Any]]) -> Usage:
    """从最终的 assistant/message 事件中提取 token 使用量。

    DSH 把 usage 放在事件流中，而不是直接放在 final_response 上。倒序
    查找是为了优先读取本轮最后一个 assistant/message 事件。缺少 usage
    时返回 known=False，避免把“未知”误记成 0 token。
    """

    unknown = Usage(None, None, None, known=False)

    for event in reversed(events):
        # turn/end 等事件不承载模型 token 使用量。
        if event.get("type") != "assistant/message":
            continue

        data = event.get("data")
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        if raw_usage is None:
            return unknown
        # region usage 类型校验
        if not isinstance(raw_usage, dict):
            raise ValueError("assistant usage must be an object")
        # endregion

        # DSH 事件适配器使用 camelCase 字段名。
        input_tokens = raw_usage.get("inputTokens")
        output_tokens = raw_usage.get("outputTokens")
        total_tokens = raw_usage.get("totalTokens")

        # 在边界处拒绝坏数据，避免记录出误导性的实验指标。
        # bool 是 int 的子类，因此要显式排除 bool。
        # region token 数值校验
        for name, value in (
            ("inputTokens", input_tokens),
            ("outputTokens", output_tokens),
            ("totalTokens", total_tokens),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        # endregion

        # total_tokens 可能由后端省略；known 只表示核心输入/输出值齐全。
        known = input_tokens is not None and output_tokens is not None
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            known=known,
        )

    # 没有任何 assistant/message 事件时，usage 仍然是未知。
    return unknown


class RoleRunner:
    """协调一次 planner 或 verifier 的 DSH 调用。"""

    def __init__(self, config, backend_factory):
        """保存共享配置和可注入的 backend 构造函数。

        backend_factory 让测试可以注入 fake backend，生产环境则可以构造
        真实 DSH backend。当前最小实现仍然是每次 run 创建一个 backend。
        """

        self.config = config
        self.backend_factory = backend_factory

    def run(
        self,
        role: str,
        prompt: str,
        session_id: str,
        timeout_s: float,
    ) -> RoleResult:
        """执行一个角色，校验输出协议，并返回项目层的 RoleResult。

        校验分为三层：调用方输入、DSH 执行元数据、角色专属 JSON。这样
        “backend 执行完成”不会被误认为“角色业务结果合法”。
        """

        # 在启动 backend 或工作线程前，先检查调用方参数。
        # region 调用参数校验
        if role not in {"planner", "verifier"}:
            raise ValueError(f"unknown role: {role!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        # endregion

        backend = None
        try:
            # 固定 system prompt 按角色选择，并在 backend 创建时注入；
            # 它和本轮用户 prompt 是两个不同层次的输入。
            system_prompt = get_system_prompt(role)
            backend = self.backend_factory(
                config=self.config,
                system_prompt=system_prompt,
            )

            # raw_result 是模块开头所描述的 DSH RunResult-like 对象，
            # 此时还不是 MedAgent 自己的 RoleResult。
            raw_result = run_with_deadline(
                backend=backend,
                prompt=prompt,
                session_id=session_id,
                timeout_s=timeout_s,
            )

            # 返回会话必须和请求会话一致，否则结果可能属于其他执行。
            # region session_id 校验
            if raw_result.session_id != session_id:
                raise ValueError("returned session_id does not match request")
            # endregion

            # completed 只表示 DSH 正常结束，不代表 JSON 或医学内容正确。
            # region finish_reason 校验
            if raw_result.finish_reason != "completed":
                raise RuntimeError(
                    f"role did not complete: {raw_result.finish_reason!r}"
                )
            # endregion

            # DSH 给出的是文本；先解析成 JSON，再执行角色专属字段校验。
            # region JSON 响应校验
            try:
                payload = json.loads(raw_result.final_response)
            except json.JSONDecodeError as error:
                raise ValueError("role response is not valid JSON") from error

            if not isinstance(payload, dict):
                raise ValueError("role response must be a JSON object")
            # endregion

            if role == "planner":
                # Planner 协议：SEARCH 必须有查询词；其他路线必须为空。
                # region Planner 路由校验
                route = payload.get("route")
                if route not in {"DIRECT", "SEARCH", "EMERGENCY"}:
                    raise ValueError("planner route is invalid")
                # endregion

                # region Planner 搜索词校验
                search_query = payload.get("search_query")
                if not isinstance(search_query, str):
                    raise ValueError("planner search_query must be a string")
                if route == "SEARCH" and not search_query.strip():
                    raise ValueError("SEARCH requires a non-empty search_query")
                if route != "SEARCH" and search_query != "":
                    raise ValueError(
                        "non-SEARCH planner routes require an empty search_query"
                    )
                # endregion

            else:
                # Verifier 协议：PASS 不能带失败原因；RETRY 可以带原因列表。
                # region Verifier 判定校验
                verdict = payload.get("verdict")
                if verdict not in {"PASS", "RETRY"}:
                    raise ValueError("verifier verdict is invalid")
                # endregion

                # region failure_codes 校验
                failure_codes = payload.get("failure_codes")
                if not isinstance(failure_codes, list):
                    raise ValueError("failure_codes must be a list")
                if any(not isinstance(code, str) for code in failure_codes):
                    raise ValueError("failure_codes must contain only strings")
                if verdict == "PASS" and failure_codes:
                    raise ValueError("PASS requires empty failure_codes")
                # endregion

            # 对外返回精简、稳定的项目层结果：
            # - final_response 已变成 payload；
            # - events 已提取成 usage；
            # - event_types 保留轻量的执行轨迹摘要。
            return RoleResult(
                role=role,
                session_id=session_id,
                payload=payload,
                finish_reason=raw_result.finish_reason,
                event_types=tuple(
                    event.get("type", "") for event in raw_result.events
                ),
                usage=extract_usage(raw_result.events),
            )
        finally:
            # 无论成功、协议校验失败、backend 报错还是超时，都释放 backend。
            # 否则 DSH 子进程可能在失败路径上继续存活。
            if backend is not None:
                backend.close()
