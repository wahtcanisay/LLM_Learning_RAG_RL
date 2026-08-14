# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search-R1 的训练装配入口（学习重要性 P0）。

Hydra 读取 ``config/ppo_trainer.yaml`` 并应用 shell 中的点号参数覆盖；本文件随后：

1. 初始化 Ray；
2. 选择 FSDP/Megatron worker 类与 GPU 资源映射；
3. 创建规则式 ``RewardManager``；
4. 把依赖注入 ``RayPPOTrainer``，再调用 ``init_workers()`` 和 ``fit()``。

真正的 rollout/GRPO 循环在 ``ray_trainer.py``。入口与 trainer 分开，是因为同一个
trainer 还可被其他 main 复用。
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    """按数据来源返回对应的规则打分函数。

    输入：
        ``data_source``（``str``）：当前样本的数据集标识，来自
        ``data_item.non_tensor_batch['data_source']``，例如 ``'nq'``。

    输出：
        可调用对象；当前支持的数据集统一返回
        ``qa_em.compute_score_em(solution_str, ground_truth, ...)``。未知数据集抛出
        ``NotImplementedError``，不会静默套用错误的评分规则。

    调用方式：
        由本文件 ``RewardManager.__call__()`` 逐样本调用。调用方随后把完整的
        ``prompt + response`` 文本和 gold answer 交给返回的打分函数。
    """
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """把一批完整轨迹转换为 token-level reward tensor。

    Search-R1 使用序列级 outcome reward，但 veRL 的 PPO/GRPO 接口接收
    ``[batch, response_len]`` token reward，因此只在最后一个有效 response token
    写入分数，其他位置保持 0。
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # 打印rollout条数方便人工观察
        self.format_score = format_score # 格式分数奖励，默认0，不奖励正确格式

    def __call__(self, data: DataProto):
        """把一批完整 rollout 转成 veRL 使用的 token-level reward。

        输入：
            ``data``（``DataProto``）：由 rollout 结果与原始数据合并后的 batch。
            关键张量/字段为：
            
            这些为generation.py的rollout结果

            - ``data.batch['prompts']``：``torch.Tensor[int]``，shape
              ``[batch_size, prompt_len]``，左侧 padding；
            - ``data.batch['responses']``：``torch.Tensor[int]``，shape
              ``[batch_size, response_len]``，右侧 padding；
            - ``data.batch['attention_mask']``：``torch.Tensor[int/bool]``，shape
              ``[batch_size, prompt_len + response_len]``；

            后续的non_tensor_batch为原始 dataset的 metadata，与rollout结果拼接送入reward环节

            - ``data.non_tensor_batch['data_source']``：每条样本的 ``str`` 数据源；
            - ``data.non_tensor_batch['reward_model']``：字典，其中
              ``ground_truth['target']`` 是 ``str`` 或 ``list[str]`` gold alias。

        输出：
            ``torch.Tensor[float32]``，shape ``[batch_size, response_len]``。
            每行只有最后一个有效 response token 写入序列级分数（默认 0 或 1），
            其余位置为 0；若上游已有同形状 ``rm_scores``，则直接返回该张量。

        调用方式：
            ``main_task()`` 创建训练/验证两个 ``RewardManager`` 并注入
            ``RayPPOTrainer``。训练时由 ``RayPPOTrainer.fit()`` 的
            ``self.reward_fn(batch)`` 调用；验证时由 ``_validate()`` 的
            ``self.val_reward_fn(test_batch)`` 调用。返回值写入
            ``batch.batch['token_level_scores']``，再经 KL 处理成为
            ``token_level_rewards``，最后传给 ``compute_advantage()`` 计算 GRPO
            advantage。
        """

        # 若启用了 learned RM，上游可直接提供同形状 rm_scores；官方 GRPO 不启用。
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) #与responses长度相同

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts'] # 左<PAD>+有效prompt

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # prompt 是左填充，取末尾有效 token；response 是右填充，取开头有效 token。
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] # 取有效prompt长度

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # reward parser 看到的是完整 prompt + rollout；这解释了“双 answer 标签”前提。
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            # outcome reward 放在轨迹终点，之后 advantage 会沿 response mask 广播/计算。
            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """Hydra CLI 入口；Ray driver 执行真正的 ``main_task``。"""
    # 初始化ray分布式框架，启动main_task
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    """在 Ray remote task 中组装 tokenizer、workers、reward 和 trainer。"""
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # strategy 决定分布式后端；当前 3B 官方脚本通常走 FSDP。
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # ActorRollout worker 在 hybrid engine 中复用同一模型权重做训练与 vLLM 采样。
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 训练不打印轨迹；验证每个 data_source 打印一条，便于人工核验格式。
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1) # 验证集reward函数

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
