from __future__ import annotations

import argparse
import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class Harness(Protocol):
    def run(self, prompt: str, *, session_id: str) -> Any: ...
    def close(self) -> None: ...


class RoleDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int | None


@dataclass(frozen=True)
class RoleResult:
    session_id: str
    payload: dict[str, Any]
    finish_reason: str
    usage: Usage | None
    event_types: tuple[str, ...]


def validate_planner(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("planner response must be a JSON object")
    route = payload.get("route")
    if route not in {"DIRECT", "SEARCH", "EMERGENCY"}:
        raise ValueError("planner route must be DIRECT, SEARCH, or EMERGENCY")
    query = payload.get("search_query", "")
    if not isinstance(query, str) or (route == "SEARCH" and not query.strip()):
        raise ValueError("SEARCH requires a non-empty search_query")
    return payload


def extract_usage(events: list[dict[str, Any]]) -> Usage | None:
    """Read finalized assistant usage once; streaming samples are not added."""
    for event in reversed(events):
        if event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        raw = data.get("usage") if isinstance(data, dict) else None
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("assistant usage must be an object")
        input_tokens = raw.get("inputTokens")
        output_tokens = raw.get("outputTokens")
        total_tokens = raw.get("totalTokens")
        values = (input_tokens, output_tokens)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("usage token counts must be non-negative integers")
        if total_tokens is not None and (
            not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens < 0
        ):
            raise ValueError("totalTokens must be a non-negative integer")
        return Usage(input_tokens, output_tokens, total_tokens)
    return None


def run_planner(harness: Harness, prompt: str, session_id: str, deadline_seconds: float) -> RoleResult:
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            outcome.put((True, harness.run(prompt, session_id=session_id)))
        except BaseException as error:
            outcome.put((False, error))

    thread = threading.Thread(target=worker, name=f"dsh-{session_id}", daemon=True)
    thread.start()
    thread.join(deadline_seconds)
    if thread.is_alive():
        harness.close()
        thread.join(2.0)
        raise RoleDeadlineExceeded(f"DSH role {session_id!r} exceeded {deadline_seconds:.3f}s")

    ok, value = outcome.get_nowait()
    if not ok:
        raise value
    if value.finish_reason != "completed":
        raise RuntimeError(f"DSH role did not complete: {value.finish_reason!r}")
    payload = validate_planner(json.loads(value.final_response))
    events = list(value.events)
    return RoleResult(
        session_id=value.session_id,
        payload=payload,
        finish_reason=value.finish_reason,
        usage=extract_usage(events),
        event_types=tuple(str(event.get("type", "")) for event in events),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated DSH Planner smoke")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--dsh-home", required=True)
    parser.add_argument("--session-id", default="planner-smoke-001")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--deadline-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)

    from deepseek_harness import DeepSeekHarness

    root = Path(__file__).resolve().parent
    workspace = Path(args.workspace).resolve()
    dsh_home = Path(args.dsh_home).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    dsh_home.mkdir(parents=True, exist_ok=True)
    harness = DeepSeekHarness(
        provider="deepseek-official",
        model=args.model,
        max_tokens=256,
        cwd=str(workspace),
        dsh_home=str(dsh_home),
        profile="sdk-minimal",
        patches=(str(root / "no_tools.patch.yml"),),
        base_url=args.base_url,
        api_key="fake-smoke-key",
        request_timeout_seconds=5.0,
        shutdown_timeout_seconds=1.0,
        env={"DSH_SYSTEM_PROMPT": "Return only the requested JSON. Do not call tools."},
    )
    try:
        result = run_planner(
            harness,
            '{"task":"Return a SEARCH plan for aspirin evidence."}',
            args.session_id,
            args.deadline_seconds,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    finally:
        harness.close()


if __name__ == "__main__":
    raise SystemExit(main())
