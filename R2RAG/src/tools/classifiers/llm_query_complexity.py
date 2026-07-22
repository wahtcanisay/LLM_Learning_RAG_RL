"""
R2RAG 的 LLM 查询复杂度分类器。

它只回答一个控制问题：当前 query 能否用一次简单搜索回答？
- 模型回答 no  -> 不是复杂问题 -> is_simple=True -> VanillaRAG；
- 模型回答 yes -> 是复杂问题   -> is_simple=False -> VanillaAgent。

这里的“复杂”指检索/研究流程复杂，不等于句子长、医学术语多或语言难懂。
"""

import time
from typing import List, Optional
from openai.types.chat import ChatCompletionMessageParam
from tools.classifiers.typing import PredictionResult
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.logging_utils import get_logger


class QueryComplexityLLM:
    """通过 OpenAI 兼容接口调用 Qwen3-4B，输出二值查询复杂度标签。"""

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 reasoning_parser: Optional[str] = "qwen3",
                 gpu_memory_utilization: Optional[float] = 0.6,
                 max_model_len: Optional[int] = 20000,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        # 本地模型服务配置：这些参数最终交给 VllmConfig，而不是直接加载模型。
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        # 分类任务需要稳定输出，因此默认 temperature=0.0，减少同一问题路由抖动。
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 延迟初始化：构造路由器时不立刻启动/连接 vLLM；第一次 predict() 才准备客户端。
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.logger = get_logger('QueryComplexityLLM')

    async def _ensure_llms(self):
        """确保分类所需的 OpenAI 兼容客户端只初始化一次。"""
        if not self.llm_client:
            # get_llm_mgr() 管理本地 vLLM 服务的启动或复用；VllmConfig 描述服务参数。
            llm_mgr = get_llm_mgr(VllmConfig(model_id=self.model_id,
                                             reasoning_parser=self.reasoning_parser,
                                             gpu_memory_utilization=self.gpu_memory_utilization,
                                             max_model_len=self.max_model_len,
                                             api_key=self.api_key))
            # 这里 await 的是“服务可用并取得客户端”，不是直接取得分类结果。
            self.llm_client = await llm_mgr.get_openai_client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    async def predict(self, query: str) -> PredictionResult:
        """
        判断 query 是否适合单轮检索，并返回统一的 PredictionResult。

        输出给路由器的关键字段是 is_simple；infer_time 用于成本分析，
        reasoning_content 仅写日志，不进入最终路由标签。
        """
        # 第一次调用可能启动模型服务，之后会复用 self.llm_client。
        await self._ensure_llms()

        if not self.llm_client:
            # 类型上 llm_client 是 Optional；初始化失败时明确中止，避免 None 调用报错难定位。
            raise RuntimeError("Failed to initialize LLM client")

        # Prompt 把复杂度操作化为“能否通过一次 Google 搜索轻易找到答案”。
        # 多部分、含糊、多步骤或需要综合多个视角的问题应回答 yes（复杂）。
        # 注意 yes/no 描述的是“是否复杂”，而 PredictionResult 保存的是“是否简单”。
        system_prompt = """Judge if the user question is a complex question. Note that the answer can only be "yes" or "no".

Given the question below, if you are doing the research, do you think the question is very easy and you can find the answer easily with a single search on Google?

If so, it's not a complex question, respond with "no", otherwise, it's a complex question, respond with "yes".

Generally, for straightforward, single question, it's a simple question. If the question is ambiguous, multifaceted, contains multiple parts, has 2 or more sub-questions or requires multiple steps to answer, it's a complex question.

Give the final answer based on your last reasoning, yes indicates it's a complex question or no indicating it's not a complex question.
"""

        messages: List[ChatCompletionMessageParam] = [
            # system 约束分类标准与输出格式，user 只放原始问题。
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 单独计时，后续可以比较动态路由带来的额外延迟是否值得。
        start_time = time.time()
        # content 是最终文本答案；cpl 保留完整响应，可读取模型的 reasoning_content。
        content, cpl = await self.llm_client.complete_chat(messages)
        infer_time = time.time() - start_time

        # 解析策略对模型多输出少量解释具有容错性：只要最终文本含 yes 就判为复杂。
        # 代价是类似 "not yes" 也会被判复杂，因此生产系统更适合结构化输出或严格解析。
        lowered_answer = content.strip().lower() if content else ''
        is_complex = lowered_answer == 'yes' or 'yes' in lowered_answer
        # 路由器读取 is_simple，所以这里对 is_complex 取反。
        is_simple = not is_complex

        # Qwen3 推理模型可能把思考过程放在 reasoning_content；记录它便于分析误路由。
        # 普通模型没有该字段时返回空字符串，不影响分类结果。
        reasoning_content = cpl.choices[0].message.reasoning_content.strip() \
            if 'reasoning_content' in cpl.choices[0].message else ''
        self.logger.info(
            "predict", query=query, is_complex=is_complex,
            reasoning_content=reasoning_content, infer_time=infer_time)

        return PredictionResult(
            query=query,
            # 当前实现是硬标签，不是真正校准过的概率：简单为 1.0，复杂为 0.0。
            is_simple_prob=1.0 if is_simple else 0.0,
            is_simple=is_simple,
            # confidence 固定为 1.0 也不代表模型真的百分百确定，后续实验应注意这个边界。
            confidence=1.0,
            infer_time=infer_time
        )
