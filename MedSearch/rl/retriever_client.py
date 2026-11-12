"""HTTP client for the Search-R1 compatible ``/retrieve`` service."""

import json
from typing import Any
from urllib.request import Request, urlopen


def format_passages(passages: list[dict[str, Any]]) -> str:
    """格式化返回结果"""
    chunks = []
    for index, item in enumerate(passages, start=1):
        title, _, body = item["document"]["contents"].partition("\n")
        chunks.append(f"Doc {index}(Title: {title}) {body}\n")
    return "".join(chunks)


def request_retrieval(
    query: str,
    search_url: str,
    topk: int,
    timeout: float,
) -> dict[str, Any]:
    """送入query到采用searchR1协议的HTTP中"""
    payload = json.dumps(
        {"queries": [query], "topk": topk, "return_scores": True}
    ).encode("utf-8")
    request = Request(
        search_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_observation(response_body: dict[str, Any]) -> str:
    """得到retreiver_server的返回结果，清洗并作为str返回"""
    passages = response_body["result"][0]
    results = format_passages(passages)
    return "<information>" + results.strip() + "</information>"



def retrieve(
    query: str,
    search_url: str,
    topk: int,
    timeout: float,
) -> str:
    """Retrieve one query and return an observation ready for AgentLoop."""
    response_body = request_retrieval(query, search_url, topk, timeout)
    return build_observation(response_body)


def encode_observation(observation: str, tokenizer: Any, max_tokens: int) -> list[int]:
    """Fit a retrieval observation in the budget without leaving its tag open."""
    token_ids = tokenizer.encode(observation, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return token_ids
    closing_ids = tokenizer.encode("</information>", add_special_tokens=False)
    if max_tokens <= len(closing_ids):
        raise ValueError("observation budget cannot fit the closing tag")
    return token_ids[: max_tokens - len(closing_ids)] + closing_ids
