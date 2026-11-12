import json
import re
from urllib.request import Request, urlopen


_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def parse_action(text: str) -> tuple[str | None, str]:
    match = _ACTION_RE.search(text)
    return (match.group(1), match.group(2).strip()) if match else (None, "")


def truncate_to_first_action(text: str) -> str:
    match = _ACTION_RE.search(text)
    return text[: match.end()] if match else text


def format_passages(passages: list[dict]) -> str:
    chunks = []
    for index, item in enumerate(passages, start=1):
        title, _, body = item["document"]["contents"].partition("\n")
        chunks.append(f"Doc {index}(Title: {title}) {body}\n")
    return "".join(chunks)


def retrieve(query: str, search_url: str, topk: int, timeout: float) -> str:
    payload = json.dumps({"queries": [query], "topk": topk, "return_scores": True}).encode("utf-8")
    request = Request(search_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return format_passages(body["result"][0])


def append_environment_tokens(
    runtime_ids: list[int], response_mask: list[int], environment_ids: list[int], response_length: int
) -> tuple[list[int], list[int]]:
    available = max(0, response_length - len(response_mask))
    environment_ids = environment_ids[:available]
    return runtime_ids + environment_ids, response_mask + [0] * len(environment_ids)
