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


class SearchAgent:
    def __init__(self,
                 request_action, 
                 retrieve: retrieve,
                 retrieve_config: dict, 
                 retrieve_timeout_s: int,
                 ):
        """保存 SearchAgent 执行所需的外部调用。
        request_action() -> ActionResponse: Search模型调用函数
        retrieve() -> List[Evidence]: 检索函数, 返回Evidence的列表作为检索结果
        """ 
        self.retrieve = retrieve
        self.request_action = request_action
        self.retrieve_config = retrieve_config
        self.retrieve_timeout_s = retrieve_timeout_s
    def run(self, 
            task: SearchTask,
            max_tries: int = 3) -> SearchResult:

        evidence_list = []
        # 模型续写与内部 SearchResult 共用同一份上下文，可含 <think>。
        context = task.query

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
