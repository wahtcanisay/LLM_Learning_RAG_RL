"""失败轨迹 OPD 损失；复用 veRL 的 KL 估计与 policy loss。"""

import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import kl_penalty, compute_policy_loss_vanilla
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def failure_opd_loss(
    student_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    opd_mask: torch.Tensor,
    config: ActorConfig,
) -> tuple[torch.Tensor, dict]:
    """计算失败trajectory的opd_loss,

    Args:
        student_log_probs: 当前学生 logprob，保留梯度
        old_log_probs: 更新前学生 logprob，用作固定基准
        teacher_logprobs: 冻结教师的 logprob
        response_mask: 模型生成位置为 1，环境与 padding 为 0
        opd_mask: 失败trajectory的llm生成token位置为1
        config: veRL actor 配置，含 clipping 与全局 batch 信息
        所有张量为对齐的[B, T]tensor

    Returns:
        (loss, metrics)：OPD loss 与日志
        无 OPD 位置时返回可反传的零 loss 和空字典。
    """

    # 没有OPD位置返回0loss

    if not opd_mask.bool().any():
        return student_log_probs.sum() * 0.0, {}

    # 计算per-token的reverse KL,    kl_estimate = 学生 logprob − 教师 logprob
    kl_estimate = kl_penalty(
        logprob=student_log_probs,
        ref_logprob=teacher_logprobs.detach(),
        kl_penalty="k1",
    )

    opd_advantages = torch.where(
        opd_mask.bool(),                    # 失败动作位置保留kl信号, 成功轨迹置为0
        -kl_estimate.detach(),              # 取负, teacher更认可的token作为正advantage 
        torch.zeros_like(kl_estimate),       
    )

    # opd_advantage 变成 backward 的policy_loss

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs.detach(), # 冻结学生模型
        log_prob=student_log_probs,           # 当前学生模型
        advantages=opd_advantages,            # 教师指导信号
        response_mask=response_mask,         # 模型生成位置，排除observation和padding
        loss_agg_mode="seq-mean-token-mean", # seq-mean-token-mean, 代表先对每条trajectory有效token求平均, 再对有效轨迹求平均
        config=config,                       # veRL的配置参数，这里要提供clipping阈值
    )

    return loss, metrics


def grpo_with_failure_opd_loss(
    config: ActorConfig,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
    *,
    opd_coef: float,
) -> tuple[torch.Tensor, dict]:
    """在原生 GRPO loss 上追加失败 OPD loss。

    Args:
        config: actor 的config配置
        model_output: student的前向结果，包含保留梯度的 log_probs。
        data: 完整 microbatch TensorDict，包含原生 PPO 所需字段；
            开启 OPD 时还须包含 response 对齐的 teacher_logprobs、opd_mask。
        dp_group: 数据并行进程组，原样传给 ppo_loss。
        opd_coef: OPD 权重；为 0 时只返回原生 loss，不读取教师字段。

    Returns:
        (total_loss, metrics)：可反传的合并损失与日志。
        total_loss = task_loss + opd_coef * opd_loss。
        保留原日志，OPD 日志使用 opd/ 前缀，避免覆盖 actor/ 指标。
        不改写原 data 的 advantages 或 response_mask。
    """
    # 正常计算GRPO_loss部分
    task_loss, metrics = ppo_loss(
        config=config,
        model_output=model_output,
        data=data,
        dp_group=dp_group,
    )

    if opd_coef == 0:
        return task_loss, metrics

    student_log_probs = no_padding_2_padding(  # student模型完整上下文输出，取出reponse对应的logprob, 补齐到[B, T]
        model_output["log_probs"],
        data,
    )

    opd_data = data.select(
        "old_log_probs",
        "teacher_logprobs",
        "response_mask",
        "opd_mask",
    ).to_padded_tensor(0.0) # 从data中取出四个字段, 不同长度轨迹补齐, 末尾填0

    # 四张量送入student所在device
    device = student_log_probs.device

    old_log_probs = opd_data["old_log_probs"].to(device)
    teacher_logprobs = opd_data["teacher_logprobs"].to(device)
    response_mask = opd_data["response_mask"].to(device)
    opd_mask = opd_data["opd_mask"].to(device)

    opd_loss, opd_metrics = failure_opd_loss(
        student_log_probs=student_log_probs,
        old_log_probs=old_log_probs,
        teacher_logprobs=teacher_logprobs,
        response_mask=response_mask,
        opd_mask=opd_mask,
        config=config,
    )

    total_loss = task_loss + opd_coef * opd_loss
    # opd日志与actor日志合并
    for key, value in opd_metrics.items():
        opd_key = (
            "opd/" + key[len("actor/"):]
            if key.startswith("actor/")
            else "opd/" + key
        )   
        metrics[opd_key] = value

    return total_loss, metrics