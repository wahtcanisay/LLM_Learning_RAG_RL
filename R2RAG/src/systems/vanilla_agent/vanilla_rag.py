"""
R2RAG 的“简单问题”单轮路径。

完整数据流：
    RunRequest.question
        -> 生成多个 query variants
        -> 对每个变体执行 Web 检索并合并候选
        -> Qwen3-Reranker 重排（可配置跳过）
        -> 按证据预算截断、重新编号
        -> 组织带引用的生成 Prompt
        -> 流式返回 reasoning / final answer / citations

它与 VanillaAgent 会复用搜索、重排和生成组件，但这里没有证据充分性审查，
也没有“改写查询后继续下一轮”的循环。一次查询扩展与检索完成后就进入生成。
"""

import asyncio
from typing import Any, AsyncGenerator, Callable, Optional
from systems.rag_interface import EvaluateRequest, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, get_default_llms, inter_resp, search_w_qv
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker
from tools.web_search import SearchResult
from tools.docs_utils import truncate_docs, update_docs_sids


class VanillaRAG(RAGInterface):
    """
    单轮 RAG 系统，被 RAGRouterLLM 用于处理 is_simple=True 的问题。

    “Vanilla”并不代表没有查询扩展或重排；它表示没有 VanillaAgent 那种跨轮次
    保存 query_history、累积证据并判断是否继续搜索的控制循环。
    """

    def __init__(
        self,
        # ---------- 默认本地生成模型 / vLLM 配置 ----------
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 25_000,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        # ---------- 检索、上下文与查询扩展配置 ----------
        retrieval_words_threshold: int = 18000,
        # Actually only supports default model + Qwen3 /nothink style
        enable_think: bool = True,
        # k_docs 当前保留在配置/日志中；实际 query-variant 搜索由 search_w_qv() 执行。
        k_docs: int = 30,
        cw22_a: bool = True,
        num_qvs: int = 3,
        search_engine: str = "clueweb22b",  # "clueweb22b" or "brave_jina"
        skip_rerank: bool = False,
        # ---------- 外部 OpenAI 兼容生成服务（可替代默认本地 LLM） ----------
        alt_llm_api_base: Optional[str] = None,
        alt_llm_api_key: Optional[str] = None,
        alt_llm_model: Optional[str] = None,
        # alt model like gpt-oss should be configured in alt_llm_reasoning_effort,
        alt_llm_reasoning_effort: Optional[str] = None,
        # ---------- 外部重排服务（可替代默认本地 reranker） ----------
        alt_reranker_api_base: Optional[str] = None,
        alt_reranker_api_key: Optional[str] = None,
        alt_reranker_model: Optional[str] = None,
        # ---------- 启动前连通性检查与 Web 文档切块配置 ----------
        pre_flight_llm: bool = False,
        pre_flight_reranker: bool = False,
        chunk_max_words: int = 300,
        chunk_overlap_words: int = 50,
        # 测试/嵌入场景可以直接注入已构造的客户端，优先级高于 alt 和默认本地模型。
        preset_llm: Optional[object] = None,
    ):
        """
        Initialize VanillaRAG with LLM server.

        学习时可把这些参数分成三组：
        1. 生成模型资源：max_model_len、max_tokens、gpu_memory_utilization；
        2. 检索上下文：num_qvs、search_engine、chunk 大小、证据总词数；
        3. 服务选择：preset_llm、alt_*、默认本地 vLLM。

        __init__ 只保存配置和创建日志对象，并不会在这里执行检索或生成。

        Args:
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            max_tokens: Maximum tokens to generate
            skip_rerank: Whether to skip reranking step (default False)
            alt_llm_api_base: Alternative LLM API base URL
            alt_llm_api_key: Alternative LLM API key
            alt_llm_model: Alternative LLM model name
            alt_llm_reasoning_effort: Alternative LLM reasoning effort setting
            alt_reranker_api_base: Alternative reranker API base URL
            alt_reranker_api_key: Alternative reranker API key
            alt_reranker_model: Alternative reranker model name
            pre_flight_llm: Whether to perform pre-flight check for LLM
            pre_flight_reranker: Whether to perform pre-flight check for reranker
        """
        # 保存默认本地 LLM 的服务参数；真正取得客户端发生在 get_active_models()。
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.retrieval_words_threshold = retrieval_words_threshold
        self.enable_think = enable_think
        self.k_docs = k_docs
        self.cw22_a = cw22_a
        self.num_qvs = num_qvs
        self.search_engine = search_engine
        self.skip_rerank = skip_rerank
        self.alt_llm_api_base = alt_llm_api_base
        self.alt_llm_api_key = alt_llm_api_key
        self.alt_llm_model = alt_llm_model
        self.alt_llm_reasoning_effort: Any = alt_llm_reasoning_effort
        self.alt_reranker_api_base = alt_reranker_api_base
        self.alt_reranker_api_key = alt_reranker_api_key
        self.alt_reranker_model = alt_reranker_model
        self.pre_flight_llm = pre_flight_llm
        self.pre_flight_reranker = pre_flight_reranker
        self.chunk_max_words = chunk_max_words
        self.chunk_overlap_words = chunk_overlap_words
        self.preset_llm = preset_llm

        self.logger = get_logger("vanilla_rag")
        # 这两个属性目前用于表示潜在客户端类型；活跃对象由 get_active_models() 返回。
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

        # 初始化日志记录本次实验的主要变量，后续对比延迟/效果时需要这些配置证据。
        self.logger.info("Initialized VanillaRAG",
                         max_tokens=self.max_tokens,
                         retrieval_words_threshold=self.retrieval_words_threshold,
                         enable_think=self.enable_think,
                         k_docs=self.k_docs,
                         cw22_a=self.cw22_a,
                         num_qvs=self.num_qvs,
                         search_engine=self.search_engine,
                         alt_llm_api_base=self.alt_llm_api_base,
                         alt_llm_model=self.alt_llm_model,
                         alt_reranker_api_base=self.alt_reranker_api_base,
                         alt_reranker_model=self.alt_reranker_model)

    @property
    def name(self) -> str:
        """返回 API 注册和健康检查使用的系统名称。"""
        return "vanilla-rag"

    async def get_active_models(self):
        """
        根据配置选择本次请求实际使用的生成模型客户端和重排器。

        生成模型优先级：preset_llm > 外部 alt LLM > 默认本地 LLM。
        重排器优先级：外部 alt reranker > 默认本地 reranker。

        返回二元组 (llm, reranker)。当用户只提供外部 LLM、没有配置外部重排器时，
        当前实现会返回 (alt_llm, None)；若 skip_rerank=False，后面调用 rerank()
        会失败。因此实验配置必须保证“需要重排时存在 reranker”。
        """
        # preset_llm 主要用于测试、依赖注入或上层已统一管理模型服务的场景。
        if self.preset_llm:
            alt_llm = self.preset_llm
        elif self.alt_llm_api_base and self.alt_llm_model:
            # 外部服务只要兼容 OpenAI Chat API，就可用 GeneralOpenAIClient 封装。
            alt_llm = GeneralOpenAIClient(model_id=self.alt_llm_model,
                                          api_base=self.alt_llm_api_base,
                                          api_key=self.alt_llm_api_key,
                                          # Cerebras only use this for GPT-OSS, for Qwen3, use /nothink in system prompt
                                          reasoning_effort=self.alt_llm_reasoning_effort,
                                          max_retries=3)
        else:
            alt_llm = None

        if self.alt_reranker_api_base and self.alt_reranker_model:
            # 重排服务单独配置，因为生成模型和 reranker 可能运行在不同端口/设备上。
            alt_reranker = GeneralReranker(model_id=self.alt_reranker_model,
                                           api_base=self.alt_reranker_api_base,
                                           api_key=self.alt_reranker_api_key)
        else:
            alt_reranker = None

        # 两个外部服务都齐全时，完全绕过默认本地模型管理器。
        if alt_llm and alt_reranker:
            return alt_llm, alt_reranker
        # 只注入外部 LLM 时不自动启动默认 reranker；调用方需配合 skip_rerank。
        if alt_llm and not alt_reranker:
            return alt_llm, None

        # 没有外部 LLM 时，取得项目默认的本地 Qwen LLM 和 Qwen reranker。
        llm, reranker = await get_default_llms()
        if alt_reranker:
            # 允许“默认本地生成模型 + 外部重排器”的混合部署。
            return llm, alt_reranker
        return llm, reranker

    async def pre_flight_models(self) -> None:
        """
        可选的模型连通性检查。

        LLM 只生成 1 个 token，reranker 使用两条人工文档；目的是尽早暴露服务、
        端口或模型加载问题，不是评估答案质量。两个开关默认均为 False。
        """
        # 这些类型仅在启用预检时需要，因此放在函数内部避免普通导入路径额外加载。
        from openai.types.chat import ChatCompletionMessageParam
        from typing import List
        from tools.reranker_vllm import _dummy_search_result as _search_result

        llm, reranker = await self.get_active_models()
        if self.pre_flight_llm:
            self.logger.info("Performing pre-flight check for LLM")
            test_messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            async for chunk in llm.complete_chat_streaming(test_messages, max_tokens=1):
                # 能收到任意流式 chunk 就证明客户端至少可以连接并发起生成。
                self.logger.info("Pre-flight LLM response received",
                                 response=chunk)

        if self.pre_flight_reranker:
            self.logger.info("Performing pre-flight check for Reranker")
            test_query = "Where is the capital of China?"
            test_docs = [
                _search_result("1", "The capital city of China is Beijing."),
                _search_result("2", "The capital city of China is Shanghai."),
            ]
            ranked_docs = await reranker.rerank(test_query, test_docs)
            # 这里只记录排序后的 sid，不把预检结果混入真实请求。
            self.logger.info("Pre-flight Reranker response received",
                             ranked_doc_ids=[doc.sid for doc in ranked_docs])

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        构造一次单轮 RAG 的异步响应流。

        外层函数只返回 stream 生成器工厂；真正的模型准备、检索、重排和生成，
        会在 API 或 evaluate() 调用 stream() 并开始 async for 后执行。
        """

        async def stream():
            try:
                # 预检被放入后台 task，不阻塞当前请求主链路。默认开关关闭时它只会
                # 取得活跃模型后快速返回；开启时预检与真实请求可能并发访问服务。
                asyncio.create_task(self.pre_flight_models())

                # 统一得到本次请求实际使用的生成客户端和重排器。
                llm, reranker = await self.get_active_models()

                # ---------- 第 1 步：查询扩展与检索 ----------
                # inter_resp() 产生中间事件，让调用方看到搜索状态但不混入最终答案。
                yield inter_resp(f"Searching: {request.question}\n\n", silent=False, logger=self.logger)
                # docs = await search_clueweb(request.question,
                #                             k=self.k_docs, cw22_a=self.cw22_a)
                # search_w_qv() 先让 LLM 生成 num_qvs 个检索表达，再搜索、合并并去重。
                # qvs 是实际使用的查询列表，docs 是多路查询召回的候选结果。
                qvs, docs = await search_w_qv(request.question, num_qvs=self.num_qvs, enable_think=self.enable_think, logger=self.logger, cw22_a=self.cw22_a, search_engine=self.search_engine, preset_llm=llm, chunk_max_words=self.chunk_max_words, chunk_overlap_words=self.chunk_overlap_words)
                # total_docs 记录重排/截断前候选规模，便于后续分析过滤比例。
                total_docs = len(docs)
                qvs_str = "; ".join(qvs)
                yield inter_resp(f"Searched: {qvs_str}, found {len(docs)} documents\n\n",
                                 silent=False, logger=self.logger)

                # 搜索工具也可能返回 SearchError 等对象；只有 SearchResult 能进入重排和生成。
                docs = [r for r in docs if isinstance(r, SearchResult)]

                # ---------- 第 2 步：候选文档重排 ----------
                if not self.skip_rerank:
                    yield inter_resp("Reranking documents...\n\n", silent=False, logger=self.logger)
                    # reranker 使用原始用户问题，而不是某个 query variant，按最终信息需求排序。
                    docs = await reranker.rerank(request.question, docs)
                # ---------- 第 3 步：控制证据预算并建立引用编号 ----------
                # retrieval_words_threshold 是所有保留文档的累计词数预算，不是单篇长度。
                docs = truncate_docs(docs, self.retrieval_words_threshold)
                # 截断后重新分配 sid，使 Prompt 中 [1]、[2]... 与最终 citations 一致。
                docs = update_docs_sids(docs)
                reranked_docs = len(docs)
                yield inter_resp(f"""Search returned {total_docs}, identified {reranked_docs} relevant, truncated to {len(docs)} web pages.\n\n""", silent=False, logger=self.logger)

                # ---------- 第 4 步：构造 Prompt 并流式生成 ----------
                yield inter_resp("Starting final answer\n\n", silent=False, logger=self.logger)
                # build_llm_messages() 把原问题与带 sid 的证据组织成聊天消息，并要求引用来源。
                messages = build_llm_messages(
                    docs, request.question, self.enable_think, model_id=llm.model_id)
                async for chunk in llm.complete_chat_streaming(messages):
                    if chunk.choices[0].finish_reason is not None:
                        # finish_reason 表示模型 token 流结束；完整 RAG 流还需发送引用终止事件。
                        break

                    delta = chunk.choices[0].delta
                    # 不同 OpenAI 兼容服务可能使用 reasoning_content 或 reasoning 字段。
                    reasoning_content = hasattr(
                        delta, 'reasoning_content') and delta.reasoning_content
                    reasoning_content = reasoning_content or (
                        hasattr(delta, 'reasoning') and delta.reasoning)
                    if reasoning_content:
                        # 思考 token 属于中间过程，silent=True 表示主要用于内部流/日志处理。
                        yield inter_resp(reasoning_content, silent=True, logger=self.logger)
                    elif hasattr(delta, 'content') and delta.content:
                        # 普通 content 才是用户可见答案；此时尚未 complete，因为还有后续片段。
                        yield RunStreamingResponse(
                            final_report=delta.content,
                            is_intermediate=False,
                            complete=False
                        )
                    # 没有 reasoning/content 的空 delta（例如纯角色信息）直接忽略。

                # ---------- 第 5 步：整理引用并发出流终止事件 ----------
                # 引用只从最终保留并真正放入 Prompt 的 docs 构建，不能使用截断前候选。
                citations = [
                    CitationItem(
                        url=r.url,
                        icon_url=to_icon_url(r.url),
                        date=str(r.date) if r.date else None,
                        sid=r.sid,
                        title=None,
                        text=r.text,
                        chunk_idx=r.chunk_idx,
                    )
                    for r in docs if isinstance(r, SearchResult)
                ]
                # 这条事件通常没有 final_report；它的职责是交付引用、模型元数据和 complete。
                yield RunStreamingResponse(
                    citations=citations,
                    is_intermediate=False,
                    complete=True,
                    metadata={
                        # 当前实现查询变体生成与最终回答共用同一个 llm 客户端。
                        "answer_model_id": llm.model_id,
                        "query_variants_model_id": llm.model_id,
                    },
                )

            except Exception as e:
                # 任一阶段失败都转换成合法的终止事件，避免 SSE 客户端永久等待 complete。
                self.logger.exception("Error in run_streaming")
                yield RunStreamingResponse(
                    final_report=f"Error processing question: {str(e)}",
                    citations=[],
                    is_intermediate=False,
                    complete=True,
                    error=str(e)
                )

        # 与 RAGInterface 约定一致：返回函数本身，而不是 return stream()。
        return stream


if __name__ == "__main__":
    # 下面是手工联调入口，会真实启动/连接模型和搜索服务；阅读代码时不要误认为单元测试。
    import asyncio

    async def main():
        """Simple test execution for VanillaRAG."""
        print("Testing VanillaRAG with LLM server...")

        # Initialize VanillaRAG
        rag = VanillaRAG(
            api_key=None,  # Optional API key
            max_tokens=4096
        )

        try:
            # Test 1 验证 evaluate() 能消费流并整理成一次性响应。
            print("\n=== Testing Evaluate Method ===")
            eval_request = EvaluateRequest(
                query="What is artificial intelligence?",
                iid="test-001"
            )

            eval_response = await rag.evaluate(eval_request)
            print(f"Query ID: {eval_response.query_id}")
            print(f"Response: {eval_response.generated_response}")
            print(f"Citations: {eval_response.citations}")

            # Test 2 直接观察中间步骤、最终答案片段、引用和错误事件。
            print("\n=== Testing Streaming Method ===")
            run_request = RunRequest(
                question="Explain the concept of machine learning in simple terms."
            )

            stream_func = await rag.run_streaming(run_request)
            print("Streaming response:")

            print_type = 'intermediate'
            async for response in stream_func():
                if response.is_intermediate:
                    if response.intermediate_steps:
                        if print_type != 'intermediate':
                            print_type = 'intermediate'
                            print(
                                f"\n[THINK] {response.intermediate_steps}\n\n")
                        print(response.intermediate_steps, end="", flush=True)
                else:
                    if print_type != 'final':
                        print_type = 'final'
                        print(f"\n[FINAL] {response.final_report}\n\n")
                    if response.final_report:
                        print(response.final_report, end="", flush=True)
                    if response.citations:
                        print(f"\n\nCitations: {response.citations}")
                    if response.error:
                        print(f"\n\nError: {response.error}")

        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # 直接执行本文件时才进入事件循环；被项目 import 时不会运行。
    asyncio.run(main())
