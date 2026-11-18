from .contracts import SearchResult, VerificationResult, VerificationVerdict, RoleResult
import json

def check_evidence(search: SearchResult) -> tuple[str, ...]:
    """检查候选答案是否为空。

    Args:
        search: 本轮搜索结果。

    Returns:
        空答案或纯空白时返回 ``("EMPTY_ANSWER",)``，否则返回空元组。
    """
    if not search.candidate_answer.strip():
        return ("EMPTY_ANSWER",)

    return ()


def build_verifier_prompt(question: str, search: SearchResult) -> str:
    """组装供 Verifier 审查的输入。

    Args:
        question: 用户的原始问题。
        search: 本轮搜索结果，提供候选答案和按检索顺序排列的证据正文。

    Returns:
        包含原题、候选答案和证据正文的 JSON 字符串。
    """
    payload = {
        "question": question,
        "candidate_answer": search.candidate_answer,
        "evidence": [item.text for item in search.evidence],
    }

    return json.dumps(payload, ensure_ascii=False)

class EvidenceVerifier:
    def __init__(self,
                 *,
                 question:str,
                 runner,
                 session_id:str,
                 timeout_s:float,
                 ):
        """保存当前问题及 Verifier 调用参数。

        Args:
            question: 用户的原始问题
            runner: 外层创建的 RoleRunner 实例，负责执行 Verifier 角色调用
            session_id: 外层分配的 Verifier 会话 ID
            timeout_s: 每次调用的超时时间，单位为秒
        """
        self.question = question
        self.runner = runner
        self.session_id = session_id
        self.timeout_s = timeout_s

    def verify(self, search: SearchResult) -> VerificationResult:
        """审查本轮候选答案，返回供 workflow 使用的判定。"""
        # 空答案直接弃答，省去一次模型调用。
        failure_codes = check_evidence(search)
        if failure_codes:
            return VerificationResult(
                verdict=VerificationVerdict.ABSTAIN,
                failure_codes=failure_codes,
                retry_query_hint=None,
            )

        # 用原题、候选答案和本轮证据发起一次审查。
        prompt = build_verifier_prompt(
            question=self.question,
            search=search,
        )

        # 成功时返回RoleResult
        result = self.runner.run(
            role="verifier",
            prompt=prompt,
            session_id = self.session_id,
            timeout_s = self.timeout_s,
        )

        # 将 RoleRunner 已校验的 JSON 转成业务结果。
        payload = result.payload

        return VerificationResult(
            verdict=VerificationVerdict(payload["verdict"]),
            failure_codes=tuple(payload["failure_codes"]),
            retry_query_hint=payload["retry_query_hint"],
        )
