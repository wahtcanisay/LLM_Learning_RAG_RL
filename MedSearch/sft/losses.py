"""离线 Top-k 蒸馏损失"""

from torch import Tensor


def forward_topk_loss(
    student_logits: Tensor,
    target_positions: list[Tensor],
    teacher_topk_ids: list[Tensor],
    teacher_topk_logprobs: list[Tensor],
) -> Tensor:
    """计算有效动作位置的教师 Top-32 蒸馏损失

    Args:
        student_logits: 学生输出，Tensor[B, L, V]。
        target_positions: 每条轨迹的目标 token 位置，list[Tensor[N_i]]。
        teacher_topk_ids: 每条轨迹各目标位置的 32 个教师候选 ID，list[Tensor[N_i, 32]]。
        teacher_topk_logprobs: 上述候选 ID 对应的教师 log 概率，list[Tensor[N_i, 32]]。

    Returns:
        所有目标位置的平均蒸馏 loss，标量 Tensor。
    """
    # 初始化总loss, token总数
    loss_sum = 0.0 
    token_count = 0

    for i, positions in enumerate(target_positions):
        # 位置张量移入学生所在设备
        positions = positions.to(student_logits.device)

        # logits: [N_i, 151665], N_i 为当前轨迹的目标位置数
        logits = student_logits[i, positions - 1, : 151665].float()

        # 写入logprob
        student_logprob = logits.log_softmax(dim=-1)

        # 取第i条teacher轨迹的候选id, 移入学生设备
        teacher_ids = teacher_topk_ids[i].to(student_logits.device)

        # 用teacher_ids,  student词表中取值
        selected_logprobs = student_logprob.gather(dim=-1, index=teacher_ids) # [N_i, 32]

        # 取出teacher的logprob, 送入students所在设备
        teacher_logprobs = teacher_topk_logprobs[i].to(
            device=student_logits.device,
            dtype=logits.dtype
        ).detach()

        # 计算32候选损失项
        loss_terms = teacher_logprobs.exp() * (teacher_logprobs - selected_logprobs)

        # 每个学习位置的32项相加, 截为非负
        token_loss = loss_terms.sum(-1).clamp_min(0) # shape:[Ni]

        loss_sum += token_loss.sum()

        token_count += token_loss.numel()

    return loss_sum / token_count
        
