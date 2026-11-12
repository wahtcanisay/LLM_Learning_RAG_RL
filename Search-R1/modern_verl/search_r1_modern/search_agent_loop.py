import asyncio
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from .protocol import append_environment_tokens, parse_action, retrieve, truncate_to_first_action


INVALID_ACTION = (
    "\nMy previous action is invalid. If I want to search, I should put the query between "
    "<search> and </search>. If I want to give the final answer, I should put the answer "
    "between <answer> and </answer>. Let me try again.\n"
)


class SearchR1AgentLoop(AgentLoopBase):
    """Run Search-R1's XML search/answer loop through current veRL's AgentLoop API."""

    def __init__(
        self,
        *args,
        search_url: str,
        topk: int = 3,
        max_turns: int = 1,
        max_generation_length: int = 96,
        max_observation_length: int = 192,
        request_timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.search_url = search_url
        self.topk = topk
        self.max_turns = max_turns
        self.max_generation_length = max_generation_length
        self.max_observation_length = max_observation_length
        self.request_timeout = request_timeout
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = await self.ct_build_initial_tokens(list(kwargs["raw_prompt"]))
        runtime_ids = prompt_ids
        response_mask: list[int] = []
        assistant_turns = searches = valid_actions = 0
        generate_seconds = tool_seconds = 0.0
        num_preempted = 0

        while assistant_turns < self.max_turns + 1 and len(response_mask) < self.response_length:
            remaining = self.response_length - len(response_mask)
            turn_params = {**sampling_params, "max_tokens": min(self.max_generation_length, remaining)}
            started = time.perf_counter()
            generated = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=runtime_ids,
                sampling_params=turn_params,
            )
            generate_seconds += time.perf_counter() - started
            num_preempted += generated.num_preempted or 0
            assistant_turns += 1

            generated_text = self.tokenizer.decode(generated.token_ids, skip_special_tokens=True)
            action_text = truncate_to_first_action(generated_text)
            action_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
            merged, response_mask, _ = await self.ct_merge_assistant_token(runtime_ids, action_ids, response_mask)
            runtime_ids = merged.token_ids

            action, content = parse_action(action_text)
            if action is not None:
                valid_actions += 1
            if action == "answer":
                break
            if action == "search" and searches < self.max_turns:
                started = time.perf_counter()
                passages = await asyncio.to_thread(
                    retrieve, content, self.search_url, self.topk, self.request_timeout
                )
                tool_seconds += time.perf_counter() - started
                searches += 1
                observation = f"\n\n<information>{passages.strip()}</information>\n\n"
            elif action is None:
                observation = INVALID_ACTION
            else:
                break

            observation_ids = self.tokenizer.encode(observation, add_special_tokens=False)[: self.max_observation_length]
            runtime_ids, response_mask = append_environment_tokens(
                runtime_ids, response_mask, observation_ids, self.response_length
            )

        response_ids = runtime_ids[-len(response_mask) :] if response_mask else []
        final_prompt_ids = runtime_ids[: len(runtime_ids) - len(response_mask)] if response_mask else runtime_ids
        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=assistant_turns + searches + 1,
            metrics={
                "generate_sequences": generate_seconds,
                "tool_calls": tool_seconds,
                "num_preempted": num_preempted,
            },
            extra_fields={
                "turns_stats": assistant_turns,
                "valid_action_stats": valid_actions,
                "valid_search_stats": searches,
            },
        )
