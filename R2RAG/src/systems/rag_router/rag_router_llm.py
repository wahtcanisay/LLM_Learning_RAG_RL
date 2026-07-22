"""
R2RAG 的动态路由入口。

学习链路：RunRequest -> QueryComplexityLLM -> 二选一：
    simple  -> VanillaRAG（单轮查询扩展、检索、重排、生成）
    complex -> VanillaAgent（多轮检索、证据充分性判断、停止控制）

这个类本身不检索文档，也不生成答案；它只负责“应该使用哪条 RAG 路径”，
然后把被选中路径产生的流式事件透明地转发给 API 层。
"""

from typing import AsyncGenerator, Callable
from systems.rag_interface import (
    RAGInterface,
    RunRequest,
    RunStreamingResponse,
)
from systems.vanilla_agent.vanilla_agent import VanillaAgent
from systems.vanilla_agent.vanilla_rag import VanillaRAG
from tools.classifiers.llm_query_complexity import QueryComplexityLLM
from tools.logging_utils import get_logger


class RAGRouterLLM(RAGInterface):
    """使用 LLM 查询复杂度分类器，在单轮 RAG 与迭代 Agent 之间动态路由。"""

    def __init__(self):
        # 两个策略对象在路由器初始化时同时创建，但每个问题只会执行其中一条路径。
        # rag_simple_query 适合一次检索即可回答的事实型或单一问题。
        self.rag_simple_query = VanillaRAG()
        # rag_complex_query 适合多部分、多视角或需要逐步补证据的问题。
        self.rag_complex_query = VanillaAgent()
        # 分类器只判断路由标签，不负责改写问题或评估检索结果。
        self.query_complexity_model = QueryComplexityLLM()
        self.logger = get_logger('RAGRouterLLM')

    @property
    def name(self) -> str:
        # API 健康检查和多系统注册使用的稳定名称。
        return "rag-router"

    def _inter_resp(self, desc: str):
        """把路由阶段的状态包装成统一的中间流事件。"""
        # 同一份描述既写入结构化日志，也发送给调用方，方便观察路由过程。
        self.logger.info(f"Intermediate step | {desc}")
        return RunStreamingResponse(
            intermediate_steps=desc,
            # 这条消息用于展示过程，不属于最终回答文本。
            is_intermediate=True,
            # 路由完成不等于整条 RAG 流完成，后面还有检索和生成。
            complete=False
        )

    async def run_streaming(
        self, request: RunRequest
    ) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        为当前问题构造路由后的异步事件流。

        注意：外层方法返回 route_and_stream 函数，而不是在这里直接跑完整流程。
        这与 RAGInterface 的契约一致，API/评测层调用返回的函数后才开始消费事件。
        """

        async def route_and_stream():
            # 第一条事件让用户知道系统正在做复杂度判断。
            yield self._inter_resp(f"Checking query: {request.question}...")

            # 分类器返回 PredictionResult；路由器真正使用的是 is_simple 布尔值。
            # await 表明复杂度判断可能触发一次 LLM 推理，不能当作普通字符串规则。
            complexity = await self.query_complexity_model.predict(request.question)

            if complexity.is_simple:
                # 简单问题：只调用单轮 VanillaRAG，避免复杂 Agent 的多轮成本。
                yield self._inter_resp("simple\n\n")
                stream_func = await self.rag_simple_query.run_streaming(request)
            else:
                # 复杂问题：交给 VanillaAgent 反复补充证据，直到满足停止条件。
                yield self._inter_resp("complex\n\n")
                stream_func = await self.rag_complex_query.run_streaming(request)

            # 路由器不修改子系统事件：搜索状态、答案片段、引用和 complete 终止事件
            # 都从被选中分支原样向上游转发。这种写法也叫异步流的“代理/委托”。
            async for response in stream_func():
                yield response

        # 返回生成器工厂；此时 route_and_stream 中的代码尚未真正执行。
        return route_and_stream
