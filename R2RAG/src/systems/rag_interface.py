"""
Abstract interface for RAG systems following MMU-RAG challenge requirements.

学习定位：这是 R2RAG 所有具体系统共同遵守的“协议层”。

调用链可以先记成：
    HTTP API -> RAGInterface -> RAGRouterLLM -> VanillaRAG / VanillaAgent

这里不负责检索或生成，而是统一两种调用方式：
1. /run 使用 run_streaming()，边处理边返回中间步骤和答案片段；
2. /evaluate 使用 evaluate()，在内部消费完整流，再整理成一次性评测结果。

因此后续无论增加 BM25、Dense 还是医学检索后端，只要上层系统仍实现
RAGInterface，API 层就不需要知道具体检索器的内部细节。
"""

from abc import ABC, abstractmethod
from typing import Callable, List, AsyncGenerator, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel


class CitationItem(TypedDict):
    """
    单条引用的结构化格式。

    学习重点：最终答案中的引用不只有 URL，还保留原文 text、短编号 sid
    和 chunk_idx。这样既能在界面展示来源，也能在静态评测时取回模型实际
    看过的上下文。
    """

    url: str  # 原始文档地址，是引用的稳定来源标识。
    icon_url: Optional[str]  # 前端展示网站图标时使用，与检索相关性无关。
    date: Optional[str]  # 文档日期可能缺失，因此是 Optional。
    title: Optional[str]  # 当前部分检索结果没有标题时允许为 None。
    text: Optional[str]  # 真正送入生成模型的证据文本，也会被 evaluate() 收集。
    sid: Optional[str]  # 答案中显示的短编号，如 1、2 或 1_1、1_2。
    chunk_idx: Optional[int]  # 在原文中的分块序号；整篇文档没有分块时为 None。


# MMU-RAG Challenge Request/Response Models
# 下面四个 Pydantic 模型构成 API 与 RAG 系统之间的数据契约。
# 它们的作用类似 MedRAG 中统一的 snippets/context 结构：先约定字段，再让
# 不同实现围绕同一结构交换数据，避免 API 层依赖某个具体 RAG 类。
class EvaluateRequest(BaseModel):
    """静态评测请求：问题文本 query 加上评测样本 ID iid。"""

    query: str  # 待回答的问题。
    iid: str  # 样本 ID；不会参与检索，只用于让输出与输入一一对应。


class EvaluateResponse(BaseModel):
    """静态评测返回：把完整流式执行结果压缩成一个普通对象。"""

    query_id: str  # 原样返回请求中的 iid，便于离线评测对齐答案。
    generated_response: str  # 将多个流式答案片段拼接后的最终文本。
    citations: List[str]  # 从 CitationItem 中提取出的 URL 列表。
    contexts: List[str]  # 模型生成答案时实际使用的证据文本列表。


class RunRequest(BaseModel):
    """动态 /run 请求；在线交互只需要一个自然语言问题。"""

    question: str


class RunStreamingResponse(BaseModel):
    """
    /run 的单个流式事件。

    一次请求会产生多个 RunStreamingResponse，而不是只返回一个：
    - 搜索、重排或思考过程放在 intermediate_steps；
    - 可展示的答案 token/片段放在 final_report；
    - 最后一条事件把 complete 设为 True，并附带 citations/metadata。

    is_intermediate 与 complete 表达的是两个不同维度：前者区分“中间过程还是
    最终答案”，后者表示“整条流是否结束”。所以答案片段通常是
    is_intermediate=False、complete=False，直到终止事件才 complete=True。
    """

    intermediate_steps: Optional[str] = None
    final_report: Optional[str] = None
    is_intermediate: bool = False
    complete: bool = False
    # extra field on top of the OpenAI format
    citations: Optional[List[CitationItem]] = None
    error: Optional[str] = None
    # Additional metadata (e.g., models_used, timing info, etc.)
    metadata: Optional[dict] = None


class RAGInterface(ABC):
    """
    所有 RAG 系统的抽象基类。

    子类必须给出 name 和 run_streaming()；evaluate() 已在基类实现，因此
    VanillaRAG、VanillaAgent 和 RAGRouterLLM 不需要分别重复静态评测逻辑。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回系统标识，供健康检查、日志或多系统 API 注册使用。"""
        pass

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request for the /evaluate endpoint.

        学习重点：evaluate() 没有另一套生成链路。它把静态请求转换成 RunRequest，
        复用同一个 run_streaming()，再消费所有事件。这保证在线运行与离线评测
        执行的是相同 RAG 系统，减少“两套入口结果不一致”的风险。

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with query_id, generated_response, citations, and contexts
        """
        # 评测样本的 iid 只用于最终对齐；RAG 执行阶段真正需要的是问题文本。
        run_request = RunRequest(question=request.query)

        # run_streaming() 返回“生成器工厂”而不是立即执行的生成器。
        # 这里先 await 完成策略选择/准备，再通过 stream_func() 真正开始消费事件。
        stream_func = await self.run_streaming(run_request)

        # generated_response 需要逐片段累加；引用通常在结束事件中一次性给出。
        generated_response = ""
        citations: List[CitationItem] = []
        error_msg: Optional[str] = None

        try:
            # async for 会一直消费到生成器结束，或遇到 complete=True 主动停止。
            async for response in stream_func():
                # 中间推理不能混进最终答案，只拼接非中间事件的 final_report。
                if response.final_report and not response.is_intermediate:
                    generated_response += response.final_report

                # 引用不是逐 token 累加；后出现的完整引用列表覆盖先前值即可。
                if response.citations:
                    citations = response.citations

                # 流可以用结构化 error 字段报告异常，而不一定直接抛出异常。
                if response.error:
                    error_msg = response.error

                # complete 是协议层的明确终止信号，避免继续等待无意义事件。
                if response.complete:
                    break

        except Exception as e:
            # 兜底捕获“消费异步流期间”抛出的异常，让评测仍能生成可对齐的结果。
            error_msg = str(e)
            generated_response = f"Error processing query: {error_msg}"

        # Handle error cases
        if error_msg and not generated_response:
            generated_response = f"Error processing query: {error_msg}"

        # 静态评测不直接返回 CitationItem，而是拆成来源 URL 与证据正文两列。
        # 这正好支持分别评估引用质量与答案所依赖的上下文。
        citation_urls: List[str] = []
        contexts: List[str] = []

        for citation in citations:
            if citation.get("url"):
                citation_urls.append(citation["url"])
            if citation.get("text"):
                contexts.append(citation.get("text") or "")

        return EvaluateResponse(
            # query_id 必须使用原始 iid，不能用自然语言问题替代。
            query_id=request.iid,
            # strip() 清理流式拼接边界；完全没有答案时给出明确占位文本。
            generated_response=generated_response.strip()
            if generated_response
            else "No response generated",
            citations=citation_urls,
            contexts=contexts,
        )

    @abstractmethod
    async def run_streaming(
        self, request: RunRequest
    ) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request for the /run endpoint.

        这是子类必须实现的核心接口。返回值是一个“无参数可调用对象”，调用后
        才得到 AsyncGenerator。RAGRouterLLM 可以因此先选择 simple/complex
        分支，再把被选中系统的事件原样转发给同一个 API 流。

        Args:
            request: RunRequest containing the question

        Returns:
            A callable that returns an AsyncGenerator yielding RunStreamingResponse objects
            following the MMU-RAG streaming format
        """
        pass
