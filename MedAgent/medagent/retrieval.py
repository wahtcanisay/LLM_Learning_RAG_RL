
from .contracts import Evidence
import json
from urllib.request import Request, urlopen

def normalize_evidence(
        response: dict[str, object],
) -> list[Evidence]:
    """
    应返回response结果如下：
    {
      "result": [
        [
          {
            "document": {
              "title": "...",
              "text": "...",iwe
              "contents": "..."
            },
            "score": ...
          }
        ]
      ]
    }
    这一part负责讲检索结果整理为Evidence的list
    """
    # response外壳校验
    # region 
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    result = response.get("result")
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("result must contain exactly one query group")
    # endregion

    items = result[0]                   # 第一条query对应的整组文档, 这里测试中只发送一条query

    # items校验
    # region
    if not isinstance(items, list):
        raise ValueError("query result must be a list")
    if not items:
        return []
    # endregion

    evidence_list = []

    for item in items:
        # item类型校验
        # region 
        if not isinstance(item, dict):
            raise ValueError("result item must be an object")
        # endregion

        document = item.get("document")

        # 返回文章内容校验.1
        # region 
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        # endregion
        text = document.get("contents")

        # 返回文章内容校验.2
        # region 
        if not isinstance(text, str) or not text:
            raise ValueError("document.contents must be a non-empty string")
        # endregion
        
        title = document.get("title")
        score = item.get("score")

        evidence = Evidence(
            text=text,
            title=title,
            score=score,
        )

        evidence_list.append(evidence)

    return evidence_list
        
def retrieve(
        query: str,
        config,
        timeout_s: float,
        top_k: int=3,
):  
    """向配置的 BM25 服务检索证据。

    Input:
        query: 搜索词。
        top_k: 本次最多返回条数。
        config: 服务地址及检索上限配置。
        timeout_s: 请求超时秒数。
    Output:
        按检索顺序返回最多 top_k 条 Evidence；无结果时返回空列表。
    """
    # 发送前校验
    # region
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0 or top_k > config["max_top_k"]:
        raise ValueError("top_k must be within the configured limit")

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be a number")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if (
        isinstance(config["max_response_bytes"], bool)
        or not isinstance(config["max_response_bytes"], int)
    ):
        raise TypeError("max_response_bytes must be an integer")
    if config["max_response_bytes"] <= 0:
        raise ValueError("max_response_bytes must be positive")

    # endregion

    # 构造检索请求内容
    payload = {
        "queries" : [query],
        "return_scores": True,
        "topk" : top_k,
    }

    body = json.dumps(payload).encode("utf-8")

    # 创建 POST 请求
    request = Request(
        config["retrieval_url"],
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # 发送请求并限制响应读取大小
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read(config["max_response_bytes"] + 1)
    if len(raw) > config["max_response_bytes"]:
        raise ValueError("retrieval response exceeds max_response_bytes")

    # 解析响应并整理为 Evidence
    raw_response = json.loads(raw.decode("utf-8"))
    evidence_list = normalize_evidence(raw_response)

    # 返回本次要求的数量
    return evidence_list[:top_k]
    
