from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
)

import asyncio
from .protocol import parse_action
from .retriever_client import retrieve

INVALID_ACTION_PROMPT = (
    "\nMy previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should "
    "put the answer between <answer> and </answer>. Let me try again.\n"
)

FINAL_ANSWER_PROMPT = (
    "The search budget is exhausted. I must now give the final answer "
    "between <answer> and </answer>. I must not search again."
)


def truncate_to_action_boundary(response: str) -> str:
    """返回本轮首个闭合 Search-R1 动作结束处的文本。"""
    endings = ("</search>", "</answer>")
    positions = [(response.find(ending), ending) for ending in endings]
    positions = [(position, ending) for position, ending in positions if position >= 0]
    if not positions:
        return response
    position, ending = min(positions)
    return response[: position + len(ending)]


def truncate_generated_action(token_ids, tokenizer):
    """保留原生成 token 的动作前缀，维持与 old log-prob 的逐 token 对齐。"""
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    endings = ("</search>", "</answer>")
    if not any(ending in response for ending in endings):
        return token_ids, response

    for end in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        if any(ending in prefix for ending in endings):
            return token_ids[:end], truncate_to_action_boundary(prefix)

    raise RuntimeError("Closed action disappeared while decoding token prefixes.")

class MedSearchAgentLoop(AgentLoopBase):
    def __init__(self, 
                 search_url, 
                 top_k, 
                 request_timeout, 
                 max_turns,
                 max_observation_length,
                 max_action_length=500,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs) # 继承base的输入，新增仅以下三部分
        self.search_url = search_url
        self.top_k = top_k
        self.request_timeout = request_timeout
        self.max_turns = max_turns
        self.max_observation_length = max_observation_length
        self.max_action_length = max_action_length
        self.response_length = self.rollout_config.response_length

    async def run(self, 
                  sampling_params: dict[str, Any],  # 采样参数
                  priority: int = 0,                # 多个rollout可能存在的调度优先级
                  **kwargs,
                  ) -> AgentLoopOutput:
        """init中需要初始化关键API和数据流"""

        # 取出raw_prompt进行generate
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        runtime_ids = await self.ct_build_initial_tokens(messages) 

        # 合法搜索次数
        searches = 0 
        # 输出轮次
        assistant_turns = 0
        invalid_actions = 0
        answered = False
        num_preempted = 0
        min_global_steps = []
        max_global_steps = []
        # 数据部分初始化
        response_mask = []
        response_logprobs = None
        action_sampling_params = dict(sampling_params)
        action_sampling_params.pop("max_new_tokens", None)
        action_sampling_params["max_tokens"] = self.max_action_length
        while assistant_turns < self.max_turns:
            generated = await self.server_manager.generate(
                            request_id=uuid4().hex,
                            prompt_ids=runtime_ids,
                            sampling_params=action_sampling_params,
                            priority=priority,
                        )
            num_preempted += generated.num_preempted or 0 # 记录优先rollout次数
            if generated.extra_fields.get("min_global_steps") is not None:
                min_global_steps.append(generated.extra_fields["min_global_steps"])
            if generated.extra_fields.get("max_global_steps") is not None:
                max_global_steps.append(generated.extra_fields["max_global_steps"])

            # 本轮首次拿到log_prob, 创建空列表开始累计, 否则沿用原值
            previous_logporbs = response_logprobs
            if previous_logporbs is None and generated.log_probs is not None:
                previous_logporbs = []

            # Search-R1 在首个闭合动作处将控制权交还环境；其后的文本不属于本轮动作。
            assistant_token_ids, processed_prompt = truncate_generated_action(
                generated.token_ids,
                self.tokenizer,
            )
            assistant_logprobs = generated.log_probs
            if assistant_logprobs is not None:
                assistant_logprobs = assistant_logprobs[: len(assistant_token_ids)]

            # generate后产生token加入连续token序列
            merged, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                        runtime_ids,
                        assistant_token_ids,
                        response_mask,                                        
                        previous_logporbs,        
                        assistant_logprobs=assistant_logprobs,
                    )
            
            # 合并结束后的处理, 开始进action分析
            assistant_turns += 1
            runtime_ids = merged.token_ids 
            cur_action = parse_action(processed_prompt)
            
            
            if cur_action is not None and cur_action.kind == "search":
                # 检索生成
                retrieved_prompt = await asyncio.to_thread(retrieve,
                                                          cur_action.content,
                                                          search_url=self.search_url,
                                                          topk=self.top_k,
                                                          timeout=self.request_timeout,)
                # search数+1, 返回的<information>闭合内容作为token加入到runtime_ids当中,补足
                searches += 1
                obs_token_ids = self.tokenizer.encode(
                    retrieved_prompt,
                    add_special_tokens=False,
                )[: self.max_observation_length]
                runtime_ids += obs_token_ids
                response_mask += ([0] * len(obs_token_ids))
                if response_logprobs is not None:
                    response_logprobs += ([0.0] * len(obs_token_ids))
            
            elif cur_action is not None and cur_action.kind == "answer":
                answered = True
                break

            else:
                # 格式错误属于 rollout 行为，不应抛异常中断整批训练；注入环境提示让模型下一轮重试。
                invalid_actions += 1
                correction_ids = self.tokenizer.encode(
                    INVALID_ACTION_PROMPT,
                    add_special_tokens=False,
                )
                runtime_ids += correction_ids
                response_mask += [0] * len(correction_ids)
                if response_logprobs is not None:
                    response_logprobs += [0.0] * len(correction_ids)
            
        if not answered:
            # max_turns后并未达到搜索结果, 直接进行最后一轮尝试    
            final_answer_ids = self.tokenizer.encode(
                FINAL_ANSWER_PROMPT,
                add_special_tokens=False,
            )
            runtime_ids += final_answer_ids
            response_mask += [0] * len(final_answer_ids)
            if response_logprobs is not None:
                response_logprobs += [0.0] * len(final_answer_ids)
            generated = await self.server_manager.generate(
                                        request_id=uuid4().hex,
                                        prompt_ids=runtime_ids,
                                        sampling_params=action_sampling_params,
                                        priority=priority,
                                    )
            num_preempted += generated.num_preempted or 0
            if generated.extra_fields.get("min_global_steps") is not None:
                min_global_steps.append(generated.extra_fields["min_global_steps"])
            if generated.extra_fields.get("max_global_steps") is not None:
                max_global_steps.append(generated.extra_fields["max_global_steps"])

            assistant_token_ids, processed_prompt = truncate_generated_action(
                generated.token_ids,
                self.tokenizer,
            )
            assistant_logprobs = generated.log_probs
            if assistant_logprobs is not None:
                assistant_logprobs = assistant_logprobs[: len(assistant_token_ids)]

            merged, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                                    runtime_ids,
                                    assistant_token_ids,
                                    response_mask,                                        
                                    response_logprobs,        
                                    assistant_logprobs=assistant_logprobs,
                                )
            assistant_turns += 1
            runtime_ids = merged.token_ids 
            cur_action = parse_action(processed_prompt)
            
            if cur_action is not None and cur_action.kind == "answer":
                answered = True
            
            
        # 合并后的token拆分为prompt和response
        response_ids = runtime_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = (runtime_ids[: len(runtime_ids) - len(response_mask)] 
                            if response_mask 
                            else runtime_ids)
        
        # 送入AgentLoopOutput
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length], 
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length]
                if response_logprobs is not None
                else None
            ), 
            num_turns=1 + assistant_turns + searches + invalid_actions,
            metrics={
                "num_preempted": num_preempted or 0, # 调度器暂停、抢占或者重新安排次数
            },
            extra_fields={
                "turns_stats": assistant_turns, # 轮次统计
                "valid_search_stats": searches, # 合法搜索次数
                "invalid_action_stats": invalid_actions, # 非法动作次数，供后续日志或 reward 决策使用
                "answered": answered,           # 是否给出回答           
                # veRL 用该版本范围诊断 rollout 是否来自当前 actor。
                "min_global_steps": min(min_global_steps) if min_global_steps else None,
                "max_global_steps": max(max_global_steps) if max_global_steps else None,
            },
        )
