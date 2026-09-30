import re
from dataclasses import dataclass

""" 正则, 提取<answer>/<search>动作 """
ACTION_PATTERN = re.compile(
    r"<(search|answer)>(.*?)</\1>",
    re.DOTALL,
)
""" 找出文本中出现的所有协议标签 """
PROTOCOL_TAG_PATTERN = re.compile(
    r"</?(?:search|answer)>"
)

@dataclass(frozen=True)
class ParsedAction:
    kind: str
    content: str


def parse_action(text: str) -> ParsedAction | None:
    matches = list(ACTION_PATTERN.finditer(text))

    # 必须恰好有一个完整动作
    if len(matches) != 1:
        return None

    match = matches[0]
    kind = match.group(1)
    content = match.group(2).strip()

    # search 和 answer 都不允许内容为空
    if not content:
        return None

    # 一个合法动作应当只有一个开标签和一个闭标签
    tags = PROTOCOL_TAG_PATTERN.findall(text)
    if len(tags) != 2:
        return None

    # 动作结束后只能有空白，不能继续生成其他内容
    trailing_text = text[match.end():]
    if trailing_text.strip():
        return None

    return ParsedAction(kind=kind, content=content)