import asyncio
import json
from datetime import datetime, timezone
from openai.types.chat import ChatCompletionMessageParam
from typing import Any, AsyncGenerator, Callable, List, NamedTuple, Optional, Tuple
from systems.rag_interface import RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.model_config import get_model_config
from systems.vanilla_agent.rag_util_fn import build_llm_messages, build_to_context, get_default_llms, inter_resp, reformulate_query, search_w_qv
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker, _dummy_search_result as _search_result
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult
from tools.docs_utils import atruncate_docs, calc_tokens, calc_tokens_str, update_docs_sids


class QueryHistoryItem(NamedTuple):
    """一轮搜索的压缩状态。

    Agent 不把上一轮的全部文档再次塞进控制提示词，而是只保留查询、
    有用文档数量和摘要，帮助下一轮避免重复搜索并判断还缺什么证据。
    """
    query: str
    doc_count: int
    summary: str


class VanillaAgent(RAGInterface):
    """用有限轮数实现“动态查询 + 迭代检索”的最小 Agent。

    这里的核心不是发明一个新的 Retriever，而是让 LLM 在每轮检索后
    做两个控制决策：当前证据是否足够，以及如果不够下一次搜什么。
    得到足够证据后，另一次 LLM 调用才负责生成最终答案。
    """

    def __init__(
        self,
        context_length: int = 25_000,  # LLM context length in tokens
        docs_review_max_tokens: int = 4096,
        answer_max_tokens: int = 4096,
        num_qvs: int = 5,  # number of query variants to use in search
        max_tries: int = 5,
        cw22_a: bool = True,
        search_engine: str = "clueweb22b",  # "clueweb22b" or "brave_jina"
        alt_llm_api_base: Optional[str] = None,
        alt_llm_api_key: Optional[str] = None,
        alt_llm_model: Optional[str] = None,
        alt_llm_reasoning_effort: Optional[str] = None,
        alt_reranker_api_base: Optional[str] = None,
        alt_reranker_api_key: Optional[str] = None,
        alt_reranker_model: Optional[str] = None,
        pre_flight_llm: bool = False,
        pre_flight_reranker: bool = False,
        chunk_max_words: int = 300,
        chunk_overlap_words: int = 50,
        preset_llm: Optional[object] = None,
    ):
        """
        Initialize VanillaAgent with LLM server.
        """
        self.context_length = context_length
        self.docs_review_max_tokens = docs_review_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.num_qvs = num_qvs
        self.max_tries = max_tries
        self.cw22_a = cw22_a
        self.search_engine = search_engine
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

        self.logger = get_logger("vanilla_agent")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

        self.logger.info("Initialized VanillaAgent",
                         context_length=self.context_length,
                         docs_review_max_tokens=self.docs_review_max_tokens,
                         answer_max_tokens=self.answer_max_tokens,
                         num_qvs=self.num_qvs,
                         max_tries=self.max_tries,
                         cw22_a=self.cw22_a,
                         search_engine=self.search_engine,
                         alt_llm_api_base=self.alt_llm_api_base,
                         alt_llm_api_key=self.alt_llm_api_key,
                         alt_llm_model=self.alt_llm_model,
                         alt_llm_reasoning_effort=self.alt_llm_reasoning_effort,
                         alt_reranker_api_base=self.alt_reranker_api_base,
                         alt_reranker_api_key=self.alt_reranker_api_key,
                         alt_reranker_model=self.alt_reranker_model)

    @property
    def name(self) -> str:
        return "vanilla-agent"

    def _format_query_history(self, query_history: List[QueryHistoryItem]) -> str:
        """Format query history into a readable string for the prompt.

        Args:
            query_history: List of QueryHistoryItem containing query, doc_count, and summary

        Returns:
            Formatted string describing the search history
        """
        if not query_history:
            return ""

        history_str = "\n=== SEARCH HISTORY ===\n"
        history_str += "Here is the complete history of our previous searches and what we found:\n\n"

        for i, entry in enumerate(query_history, 1):
            history_str += f"Search #{i}:\n"
            history_str += f"  Query: \"{entry.query}\"\n"
            history_str += f"  Result: Found {entry.doc_count} useful document(s)\n"
            history_str += f"  Summary: {entry.summary}\n\n"

        history_str += "=== END SEARCH HISTORY ===\n\n"
        history_str += "IMPORTANT: Do not repeat these queries in <new-query>. Consider what information we already have vs what is still missing.\n\n"

        return history_str

    async def get_active_models(self):
        if self.preset_llm:
            alt_llm = self.preset_llm
        elif self.alt_llm_api_base and self.alt_llm_model:
            alt_llm = GeneralOpenAIClient(model_id=self.alt_llm_model,
                                          api_base=self.alt_llm_api_base,
                                          api_key=self.alt_llm_api_key,
                                          reasoning_effort=self.alt_llm_reasoning_effort,
                                          max_retries=3)
        else:
            alt_llm = None

        if self.alt_reranker_api_base and self.alt_reranker_model:
            alt_reranker = GeneralReranker(model_id=self.alt_reranker_model,
                                           api_base=self.alt_reranker_api_base,
                                           api_key=self.alt_reranker_api_key)
        else:
            alt_reranker = None

        if alt_llm and alt_reranker:
            return alt_llm, alt_reranker
        if alt_llm and not alt_reranker:
            return alt_llm, None

        llm, reranker = await get_default_llms()
        if alt_reranker:
            return llm, alt_reranker
        return llm, reranker

    async def review_documents(self, question: str, next_query: str, query_history: List[QueryHistoryItem], docs: List[SearchResult]) -> Tuple[bool, str | None, List[SearchResult], str | None]:
        """审查本轮候选文档，并返回下一步控制器决策。

        返回值依次表示：证据是否足够、下一轮查询、有用文档、文档摘要。
        这个函数不负责搜索，也不生成最终答案；它只是把 LLM 的结构化
        判断转换成 Python 状态，交给 ``run_streaming`` 驱动下一轮。
        """
        llm, _reranker = await self.get_active_models()

        model_config = get_model_config(llm.model_id)
        # 审查器同时看到原问题、本轮查询和历史摘要，才能判断“还缺什么”，
        # 而不是只根据本轮候选文档机械地说相关或不相关。
        query_history_str = self._format_query_history(query_history)
        prompt = model_config.REVIEW_DOCUMENTS_PROMPT(
            question=question,
            next_query=next_query,
            current_time=datetime.now(timezone.utc),
            query_history_section=query_history_str)
        prompt_tokens = calc_tokens_str(prompt)

        # 给审查模型预留输出空间，并把剩余上下文预算交给文档。
        # 公式的含义是：模型总上下文 = 提示词 + 文档 + 审查输出 + 安全余量。
        answer_max_tokens = self.docs_review_max_tokens
        redundant_tokens = 1024  # for doc header, prompt template, and safety margin overhead
        available_context = self.context_length - \
            prompt_tokens - answer_max_tokens - redundant_tokens
        docs_truncated = await atruncate_docs(docs, available_context)
        context = build_to_context(docs_truncated)
        prompt += context
        self.logger.info("Truncate documents for review",
                         model_context_length=self.context_length,
                         prompt_tokens=prompt_tokens,
                         answer_max_tokens=answer_max_tokens,
                         available_context=available_context,
                         original_count=len(docs),
                         truncated_count=len(docs_truncated),
                         actual_tokens=calc_tokens_str(prompt),
                         IDs=[d.sid for d in docs_truncated])

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        resp_text = ""
        # 这里使用流式接口主要是为了复用项目的 LLM 客户端；控制逻辑只
        # 关心最终 content，reasoning_content 只打印，不参与状态转换。
        async for chunk in llm.complete_chat_streaming(messages, max_tokens=answer_max_tokens):
            if chunk.choices[0].finish_reason is not None:
                break
            delta = chunk.choices[0].delta
            reasoning_content = hasattr(
                delta, 'reasoning_content') and delta.reasoning_content
            reasoning_content = reasoning_content or (
                hasattr(delta, 'reasoning') and delta.reasoning)
            if reasoning_content:
                print(reasoning_content, end="", flush=True)
            elif hasattr(delta, 'content') and delta.content:
                print(delta.content, end="", flush=True)
                resp_text += delta.content

        # REVIEW_DOCUMENTS_PROMPT 要求模型输出 XML 风格标签。解析标签后，
        # 自然语言回答就被压缩成可执行的四元组。
        resp_text = resp_text.strip().lower() if resp_text else ""
        is_sufficient = extract_tag_val(resp_text, "is-sufficient") == "yes"
        new_query = extract_tag_val(resp_text, "new-query")
        useful_doc_ids_str = extract_tag_val(resp_text, "useful-docs")
        useful_docs_summary = extract_tag_val(resp_text, "useful-docs-summary")

        self.logger.info("Review documents completed",
                         question=question,
                         next_query=next_query,
                         query_history=query_history,
                         is_sufficient=is_sufficient,
                         new_query=new_query,
                         useful_doc_ids=useful_doc_ids_str,
                         useful_docs_summary=useful_docs_summary)

        useful_docs = []
        if useful_doc_ids_str:
            # 文档 ID 只能从本轮候选中选择，避免模型凭空引用不存在的文档。
            useful_doc_ids = [id_.strip() for id_
                              in useful_doc_ids_str.split(",") if id_.strip().isdigit()]
            useful_docs = [doc for doc in docs if doc.sid in useful_doc_ids]

        if not is_sufficient and not new_query:
            # 没有下一轮查询就无法继续推进；这里选择安全退出，避免死循环。
            # 代价是模型格式错误可能造成过早停止，属于需要记录的失败案例。
            is_sufficient = True  # force to yes if no new query

        return is_sufficient, new_query, useful_docs, useful_docs_summary

    async def pre_flight_models(self) -> None:
        llm, reranker = await self.get_active_models()
        if self.pre_flight_llm:
            self.logger.info("Performing pre-flight check for LLM")
            test_messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            async for chunk in llm.complete_chat_streaming(test_messages, max_tokens=1):
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
            self.logger.info("Pre-flight Reranker response received",
                             ranked_doc_ids=[doc.sid for doc in ranked_docs])

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """构造一个可流式消费的 RAG Agent 执行器。

        外层函数只返回生成器；真正的搜索、审查和答案生成发生在调用方
        迭代 ``stream`` 时。核心流程可以抽象为：
        ``search -> rerank -> review -> accumulate -> stop/rewrite``。
        """
        async def stream():
            try:
                # Run pre-flight checks but don't await
                asyncio.create_task(self.pre_flight_models())

                yield inter_resp(f"Searching question: {request.question}\n\n",
                                 silent=False, logger=self.logger)
                llm, reranker = await self.get_active_models()

                # acc_docs 是跨轮次累积的最终证据池；后面的答案模型只看它，
                # 不会重新看到每轮所有未选中的候选文档。
                acc_docs: List[SearchResult] = []
                # 用源文档 ID 去重，防止不同查询重复贡献同一份证据。
                acc_docs_id_set = set()
                # 审查器用 sid 引用本轮文档；这个计数器尝试为后续轮次的 sid
                # 做偏移。它是实现细节，不能直接假设所有历史 sid 都全局唯一。
                acc_doc_base_count = 0
                # 这是给审查器看的轻量控制记忆，不等于最终证据池。
                query_history: List[QueryHistoryItem] = []
                # next_query 会被审查器的 <new-query> 更新，驱动下一轮搜索。
                next_query = request.question
                # 只有搜索失败或审查结果异常时才打开更强的查询变体思考。
                qv_think_enabled = False
                tries = 0
                # 先为最终答案预留输出空间，再用剩余上下文容纳累积证据。
                answer_max_tokens = self.answer_max_tokens + 1024
                context_tokens_limit = self.context_length - answer_max_tokens
                # ---------------------------------------------------------------
                # 每一轮都执行同一组状态转移：搜索、重排、审查、累积证据，
                # 然后决定停止、改写查询，或继续下一轮。
                while True:
                    tries += 1
                    # 1) 搜索：search_w_qv 会为 next_query 生成若干查询变体，
                    #    执行召回并融合结果；它仍然是检索器，不是 Agent 决策器。
                    qvs, docs = await search_w_qv(next_query, num_qvs=self.num_qvs, enable_think=qv_think_enabled, logger=self.logger, search_engine=self.search_engine, preset_llm=llm, chunk_max_words=self.chunk_max_words, chunk_overlap_words=self.chunk_overlap_words)
                    docs = [r for r in docs if isinstance(r, SearchResult)]
                    qvs_str = "; ".join(qvs)
                    yield inter_resp(f"Search completed: {qvs_str}\n\n",
                                     silent=False, logger=self.logger)

                    # 2) 重排：Retriever 负责找候选，Reranker 负责按本轮查询
                    #    重新排序；它不决定是否继续搜索。
                    yield inter_resp(f"Reranking {len(docs)} documents...\n\n",
                                     silent=False, logger=self.logger)
                    docs_reranked = await reranker.rerank(next_query, docs)

                    if not docs_reranked:
                        # 召回为空时没有证据可供审查，只能改写查询后重试。
                        # 这条路径是错误恢复，不是“复杂问题”的正常路由分支。
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevant documents, so far we have {len(acc_docs)} relevant documents, reformulating query...\n\n",
                                         silent=False, logger=self.logger)
                        next_query = await reformulate_query(next_query, preset_llm=llm)
                        yield inter_resp(f"Next search ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # 3) 审查前给本轮文档重新编号。审查器返回的是 sid，跨轮次
                    #    偏移可以避免不同轮次出现相同的候选编号。
                    docs_reranked = update_docs_sids(
                        docs_reranked, base_count=acc_doc_base_count)
                    yield inter_resp("Reviewing documents for relevance and sufficiency...\n\n",
                                     silent=False, logger=self.logger)
                    # 审查器的输出是控制信号：是否停止、下一轮搜什么、哪些文档
                    # 可以进入累积证据池，以及供下一轮参考的摘要。
                    is_enough, _next_query, useful_docs, useful_docs_summary = await self.review_documents(request.question, next_query, query_history, docs_reranked)

                    # 只记录压缩后的“本轮结果”，避免下一轮提示词随原始文档线性膨胀。
                    query_history.append(QueryHistoryItem(
                        query=next_query,
                        doc_count=len(useful_docs),
                        summary=useful_docs_summary if useful_docs_summary else 'No relevant documents found'
                    ))

                    # 只有审查器选出的 useful_docs 才能成为最终答案的证据；
                    # 去重集合按源文档 ID 控制重复，base_count 则按选中数量推进编号。
                    acc_doc_base_count += len(useful_docs)
                    for d in useful_docs:
                        if d.id not in acc_docs_id_set:
                            acc_docs_id_set.add(d.id)
                            acc_docs.append(d)

                    if useful_docs_summary:
                        yield inter_resp(f"Found documents: {useful_docs_summary}\n\n",
                                         silent=False, logger=self.logger)

                    # 4) 语义停止：证据审查器认为当前证据足够，进入最终生成。
                    if is_enough:
                        break

                    if not _next_query:
                        # 审查器说“不够”但没有给查询，无法执行正常路由；
                        # 与空结果一样，打开增强查询变体并做一次恢复性改写。
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevant documents for this query, so far we have {len(acc_docs)} relevant documents\n\n",
                                         silent=False, logger=self.logger)
                        next_query = await reformulate_query(next_query, preset_llm=llm)
                        yield inter_resp(f"Next search with better query variants ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # 5) 正常路由：把审查器给出的新查询交给下一轮 search_w_qv。
                    next_query = _next_query
                    yield inter_resp(f"Need more information, so far we have {len(acc_docs)} relevant documents\n\n",
                                     silent=False, logger=self.logger)

                    # 除语义停止外，还要受轮数和上下文预算限制，防止 Agent 无限
                    # 搜索或把过多证据塞进最终答案提示词。
                    current_tokens = sum(calc_tokens(d) for d in acc_docs)
                    if tries >= self.max_tries:
                        yield inter_resp(f"Reached maximum tries ({self.max_tries}), proceeding to generate final answer\n\n",
                                         silent=False, logger=self.logger)
                        break
                    elif current_tokens >= context_tokens_limit:
                        yield inter_resp(f"Reached context token limit ({current_tokens}/{context_tokens_limit} tokens), proceeding to generate final answer\n\n",
                                         silent=False, logger=self.logger)
                        break
                    else:
                        yield inter_resp(f"Next search({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                # 循环结束后，acc_docs 才是最终生成阶段唯一使用的证据集合。
                # ---------------------------------------------------------------

                # 最终生成前再次截断并重新编号：审查阶段的编号只服务于控制决策，
                # 最终编号服务于答案中的引用和返回给前端的 citations。
                acc_docs = await atruncate_docs(acc_docs, context_tokens_limit)
                acc_docs = update_docs_sids(acc_docs)

                yield inter_resp(f"Starting final answer with {len(acc_docs)} documents\n\n",
                                 silent=False, logger=self.logger)
                messages = build_llm_messages(
                    acc_docs, request.question, True, model_id=llm.model_id)

                prompt_tokens = calc_tokens_str(json.dumps(messages))
                # 根据实际提示词长度动态计算生成预算，并留出安全余量。
                gen_max_tokens = self.context_length - prompt_tokens - 1000
                async for chunk in llm.complete_chat_streaming(messages, max_tokens=gen_max_tokens):
                    if chunk.choices[0].finish_reason is not None:
                        # Stream finished
                        break
                    delta = chunk.choices[0].delta
                    reasoning_content = hasattr(
                        delta, 'reasoning_content') and delta.reasoning_content
                    reasoning_content = reasoning_content or (
                        hasattr(delta, 'reasoning') and delta.reasoning)
                    if reasoning_content:
                        # still intermediate steps
                        yield inter_resp(reasoning_content, silent=True, logger=self.logger)
                    elif hasattr(delta, 'content') and delta.content:
                        # final report
                        yield RunStreamingResponse(
                            final_report=delta.content,
                            is_intermediate=False,
                            complete=False
                        )
                    # otherwise ignore empty deltas

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
                    for r in acc_docs if isinstance(r, SearchResult)
                ]
                # 答案流结束后再发送 complete=True，前端据此知道可以收尾并使用引用。
                yield RunStreamingResponse(
                    citations=citations,
                    is_intermediate=False,
                    complete=True,
                    metadata={
                        "answer_model_id": llm.model_id,
                        "query_variants_model_id": llm.model_id,
                        "documents_reviewer_model_id": llm.model_id,
                    },
                )

            except Exception as e:
                # 即使中途异常，也发送终止事件，避免流式调用方永久等待。
                self.logger.exception("Error in run_streaming")
                yield RunStreamingResponse(
                    final_report=f"Error processing question: {str(e)}",
                    citations=[],
                    is_intermediate=False,
                    complete=True,
                    error=str(e)
                )

        return stream


if __name__ == "__main__":
    import sys
    import asyncio

    async def main():
        """Simple test execution for VanillaAgent."""
        print("Testing VanillaAgent with LLM server...")

        # Initialize VanillaAgent
        rag = VanillaAgent()

        try:
            q = sys.argv[1] if len(sys.argv) > 1 \
                else "What is the capital of France?"
            run_request = RunRequest(question=q)
            start_time = asyncio.get_event_loop().time()

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
                        print(f"\n[FINAL] {response.final_report}")
                    if response.final_report:
                        print(response.final_report, end="", flush=True)
                    if response.citations:
                        print(f"\n\nCitations: {len(response.citations)}")
                    if response.error:
                        print(f"\n\nError: {response.error}")
            end_time = asyncio.get_event_loop().time()
            print(f"\n\nTotal time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
