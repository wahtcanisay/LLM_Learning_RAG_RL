from typing import Protocol

from .contracts import (
    SearchTask,
    SearchResult,
    VerificationResult,
    SearchCycleResult,
    VerificationVerdict,
    CycleStatus,
)

# 预留接口设计
class SearchAgent(Protocol):
    def run(self, task:SearchTask) -> SearchResult:
        ...

class Verifier(Protocol):
    def verify(self, result:SearchResult) -> VerificationResult:
        ...

def run_search_cycle(
        question: str,
        initial_query: str,
        searchagent: SearchAgent,
        verifier: Verifier,
        max_retries: int = 2,
) -> SearchCycleResult:
    """执行搜索、验证和按反馈重试，返回本轮循环的最终状态。"""

    # 限制重试次数，避免搜索循环无限进行。
    # region max_retries 校验
    if max_retries <= 0 or max_retries > 2:
        raise ValueError("max_retries must be 1 or 2")
    # endregion

    # 保存首次搜索所需的查询和验证反馈。
    query = initial_query
    feedback = None
    attempts = 0

    for attempt in range(max_retries):
        # 将当前查询和上轮失败反馈封装成一次搜索任务。
        seach_task = SearchTask(
                question=question,
                query=query,
                attempt=attempt,
                verifier_feedback=feedback,
            )
        # 执行搜索并记录实际尝试次数，再交给验证器判定。
        search_result = searchagent.run(task=seach_task)
        attempts += 1
        verification = verifier.verify(search_result)

        # 通过验证后立即结束循环。
        if verification.verdict == VerificationVerdict.PASS:
                    return SearchCycleResult(
                        status=CycleStatus.VERIFIED,
                        search=search_result,
                        verification=verification,
                        attempts=attempts,
                        reason="verified",
                    )
        
        elif verification.verdict == VerificationVerdict.ABSTAIN:
            # 验证器无法确认时停止搜索，并保留当前结果。
            return SearchCycleResult(
                status=CycleStatus.ABSTAINED,
                search=search_result,
                verification=verification,
                attempts=attempts,
                reason="verifier_abstained",
            )

        elif verification.verdict == VerificationVerdict.RETRY:
            # 最后一轮仍要求重试时，直接以重试耗尽结束。
            if attempt == max_retries - 1:
                return SearchCycleResult(
                    status=CycleStatus.ABSTAINED,
                    search=search_result,
                    verification=verification,
                    attempts=attempts,
                    reason="retry_exhausted",
                )
            
            # 重试必须提供新的非空查询，并把失败原因传给下一轮。
            retry_hint = verification.retry_query_hint

            # region retry_query_hint 校验
            if retry_hint is None or not retry_hint.strip():
                raise ValueError("RETRY requires a non-empty retry_query_hint")
            # endregion
            query = retry_hint.strip()
            feedback = verification.failure_codes
            
        else:
            # 未知判定不能被静默忽略。
            # region verification verdict 校验
            raise ValueError("unsupported verification verdict")
            # endregion
        
