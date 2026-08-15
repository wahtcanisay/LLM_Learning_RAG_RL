# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """把同题多条 rollout 的序列 reward 转成 GRPO token-level advantage。

    输入：
        ``token_level_rewards``（``torch.Tensor[float]``）：shape
        ``[batch_size, response_length]``。Search-R1 的 outcome reward 通常只写在
        每条 response 的最后一个有效 token；其余位置为 0。

        ``eos_mask``（``torch.Tensor[int/bool/float]``）：shape 同上，1 表示有效
        response token，0 表示 padding。开启 state masking 后，上游仍会在 actor loss
        阶段换用 ``loss_mask`` 排除环境注入的 information token。

        ``index``：长度为 ``batch_size`` 的一维可索引容器。函数签名写作
        ``torch.Tensor``，实际由 ``data.non_tensor_batch['uid']`` 传入；同一道题的
        多条 rollout 必须具有相同且可哈希的 uid。

        ``epsilon``（``float``）：标准差分母的数值稳定项，默认 ``1e-6``。

    输出：
        ``tuple[torch.Tensor, torch.Tensor]``，依次为 ``advantages`` 和
        ``returns``，shape 都是 ``[batch_size, response_length]``。当前实现两者
        返回同一个张量：每条 rollout 的组内标准化分数被广播到所有有效 response
        token，padding 位置为 0。计算位于 ``torch.no_grad()``，不建立梯度图。

    调用方式：
        ``ray_trainer.compute_advantage()`` 在 ``adv_estimator == 'grpo'`` 时调用；
        返回值写入 ``batch['advantages']``/``batch['returns']``。随后
        ``DataParallelPPOActor.update_policy()`` 将 ``advantages`` 交给
        ``compute_policy_loss()``。

    关键边界：
        若同题 rollout 全为 1 或全为 0，则 ``reward - group_mean == 0``，这一组的
        advantage 为 0，不提供策略梯度信号；这是零方差组，不等于整个训练必然坍塌。
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    # 压缩 token 维度：每条 rollout 得到一个序列级标量 score。
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        # 用 uid 将同一道题采样出的多条 rollout 放进同一比较组。
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 只比较组内相对好坏，不直接使用 reward 的绝对大小。
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # 一个 rollout 的相对 advantage 广播到所有有效 response token。
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """用 PPO clipped surrogate objective 计算 actor 的策略梯度损失。

    输入：
        ``old_log_prob``（``torch.Tensor[float]``）：rollout 行为策略生成各 token 时
        保存的 log probability，shape ``[batch_size, response_length]``。

        ``log_prob``（``torch.Tensor[float]``）：当前正在更新的 actor 对同一批 token
        重新前向计算的 log probability，shape 同上，参与反向传播。

        ``advantages``（``torch.Tensor[float]``）：GRPO 组内标准化后广播到 token 的
        advantage，shape 同上；正值提高相应 token 概率，负值降低概率。

        ``eos_mask``（``torch.Tensor[int/bool/float]``）：实际参与 loss 平均的 token
        mask，shape 同上。在 Search-R1 的 state masking 路径中，调用方传入的是
        ``loss_mask``，因此环境注入的 information token 不参与优化。

        ``cliprange``（``float``）：把概率比限制在
        ``[1 - cliprange, 1 + cliprange]`` 的 PPO 裁剪半径。

    输出：
        ``tuple[torch.Tensor, torch.Tensor, torch.Tensor]``，三个都是标量张量：

        - ``pg_loss``：对有效 token 做 masked mean 后的 clipped policy loss；
        - ``pg_clipfrac``：被裁剪分支限制的有效 token 比例；
        - ``ppo_kl``：当前 actor 与 rollout old policy 的近似 KL 监控量。它不是
          ``ref_log_prob`` 计算出的 reference-policy KL loss。

    调用方式：
        由 ``verl/workers/actor/dp_actor.py::DataParallelPPOActor.update_policy()``
        对每个 micro-batch 调用。调用方再组合 entropy bonus，以及配置启用时的
        reference-policy KL loss，执行 ``loss.backward()`` 和 optimizer step。
    """
    # log π_new(a|s) - log π_old(a|s)，再取 exp 得到重要性采样概率比。
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # 比较未裁剪与裁剪后的 surrogate loss；取 max 对应最大化目标时取 min。
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 只在 eos_mask/loss_mask 指定的有效模型输出 token 上取平均。
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
