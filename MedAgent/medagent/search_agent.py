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
from MedSearch.rl.protocol import parse_action


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
        context = task.query
        

        for step in range(max_tries):
            response = self.request_action(context)
            cur_action = parse_action(response.text)    
            if cur_action is not None and cur_action.kind == "search":

                if step == max_tries - 1:
                    # 最后一次禁止搜索
                    break 

                evidence = self.retrieve(
                    cur_action.content,
                    self.retrieve_config,
                    self.retrieve_timeout_s,
                    top_k=3,
                )

                
                evidence_list.extend(evidence)

                information = ("\n".join(item.text for item in evidence)
                               if evidence
                               else "No results.")

                context += "<search>" + cur_action.content.strip() + "</search>" + "\n"
                context += "<information>" + information.strip() + "</information>"

            elif cur_action is not None and cur_action.kind == "answer":

                context += "<answer>" + cur_action.content.strip() + "</answer>"
                return SearchResult(
                    task.attempt,
                    cur_action.content,
                    evidence_list,
                    context
                )
            elif cur_action is None:
                break

        return SearchResult(
            task.attempt,
            candidate_answer="",
            evidence=evidence_list,
            context=context
        )
