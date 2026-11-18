import re
from dataclasses import dataclass

# 沿用 Search-R1 llm_agent/generation.py::postprocess_predictions 的首个完整动作解析。
ACTION_PATTERN = re.compile(
    r"<(search|answer)>(.*?)</\1>",
    re.DOTALL,
)

@dataclass(frozen=True)
class ParsedAction:
    kind: str
    content: str


def parse_action(text: str) -> ParsedAction | None:
    """取第一个完整动作，忽略动作外文本；空内容不执行。"""
    match = ACTION_PATTERN.search(text)
    if match is None:
        return None

    kind = match.group(1)
    content = match.group(2).strip()

    # search 和 answer 都不允许内容为空
    if not content:
        return None

    return ParsedAction(kind=kind, content=content)
