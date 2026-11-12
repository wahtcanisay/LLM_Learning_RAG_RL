"""本地 SearchAgent 的单动作回复入口；真实 3B 服务待接入。"""

from dataclasses import dataclass
from pathlib import Path
import sys

# MedSearch 与 MedAgent 是同级目录；直接从 MedAgent 启动时需加入共同父目录。
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .contracts import (Usage, 
                        SearchTask, 
                        SearchResult,
                        )
from .retrieval import retrieve
from MedSearch.rl.protocol import ACTION_PATTERN, parse_action


@dataclass(frozen=True)
class ActionResponse:
    text: str
    usage: Usage

# TokensPrompt包装token作为vllm的可接受输入, 也就是组织输入格式
from vllm.inputs import TokensPrompt

def request_action(
                   context:str,
                   *,
                   generate,
                   tokenizer,
                   sampling_params,
                   ) -> ActionResponse:
    """调用模型生成单条动作回复，返回文本及 token 用量。

    Args:
        context (str): 当前问题与已累积的动作、检索证据组成的上下文
        generate: vLLM 的生成方法，通常传入 llm.generate
        tokenizer: 分词器
        sampling_params: vLLM 生成配置，控制随机性、输出长度和停止方式

    Output:
        ActionResponse:
            text (str): 第一条输入的第一个候选文本，供 parse_action 解析
            usage (Usage): 输入、输出、总 token 数，以及统计是否已知

    Note:
        当前只返回一个候选；即使配置 n > 1，也只取第一个。
        context 已在 SearchAgent.run() 开始时套好模板。
    """

    # 只对当前续写文本分词，保留已有的模板标记。
    input_ids = tokenizer.encode(
        context,
        add_special_tokens=False, # 不额外添加特殊 token，避免与模板重复。
    )

    request = generate(
        [TokensPrompt(prompt_token_ids=input_ids)], # 单条 token ID 输入，外层列表表示批次。
        sampling_params, # 控制本次生成的采样方式、长度上限与停止条件。
        use_tqdm=False, # 关闭批处理进度条，不影响生成内容。
    )[0] # 取第一条输入对应结果
    output = request.outputs[0] # 取出输入的第一个候选答案
    # 以上两个[0] 作为单条输入特化版

    # 用于计算usage
    input_tokens = len(input_ids)
    output_tokens = len(output.token_ids)

    return ActionResponse(
        text=output.text,
        usage=Usage(input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    True) # known=True：本次输入、输出 token 数均已统计。
    )



class SearchAgent:
    def __init__(self,
                 request_action, 
                 retrieve: retrieve,
                 retrieve_config: dict, 
                 retrieve_timeout_s: int,
                 tokenizer,
                 ):
        """保存 SearchAgent 执行所需的外部调用。
        request_action() -> ActionResponse: Search模型调用函数
        retrieve() -> List[Evidence]: 检索函数, 返回Evidence的列表作为检索结果
        """ 
        self.retrieve = retrieve
        self.request_action = request_action
        self.retrieve_config = retrieve_config
        self.retrieve_timeout_s = retrieve_timeout_s
        self.tokenizer = tokenizer
    def run(self, 
            task: SearchTask,
            max_tries: int = 3) -> SearchResult:

        evidence_list = []
        # 模型续写与内部 SearchResult 共用同一份上下文，可含 <think>。
        context = self.tokenizer.apply_chat_template(
            [{"role":"user", "content":task.query}],
            tokenize=False,
            add_generation_prompt=True,
        )

        for step in range(max_tries):
            response = self.request_action(context)
            cur_action = parse_action(response.text)
            if cur_action is None:
                break
            if cur_action.kind == "search" and step == max_tries - 1:
                break

            # 用与 parse_action 相同的规则定位闭标签，保留前面的生成文本。
            action_match = ACTION_PATTERN.search(response.text)
            assert action_match is not None
            context += response.text[:action_match.end()] # 拼接当前动作
            if cur_action.kind == "answer":
                return SearchResult(task.attempt, cur_action.content, evidence_list, context)
            # 检索动作
            evidence = self.retrieve(
                cur_action.content,
                self.retrieve_config,
                self.retrieve_timeout_s,
                top_k=3,
            )
            evidence_list.extend(evidence)
            # information清洗
            information = "\n".join(item.text for item in evidence) if evidence else "No results."
            context += "<information>" + information.strip() + "</information>" # 检索得到information拼接

        return SearchResult(
            task.attempt,
            candidate_answer="",
            evidence=evidence_list,
            context=context
        )
