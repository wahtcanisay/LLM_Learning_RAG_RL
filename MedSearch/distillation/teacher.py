"""教师模型评分接口。"""
import torch
from transformers import AutoModelForCausalLM

def score_tokens(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mode: str = "sampled",
    k: int = 32,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """计算教师对目标位置的 sampled 或 Top-k logprob。

    维度：B 为批次大小，P 为上下文长度，T 为目标长度；
    V 为师生共同的实际 token 词表大小，V_model 为教师输出层大小。

    Args:
        model: 已加载的教师因果语言模型，通常由
            AutoModelForCausalLM.from_pretrained(...) 创建；本函数只调用它评分。
        prefix_ids: [B, P] 的 torch.long 上下文 token ID。
        target_ids: [B, T] 的 torch.long 待评分 token ID。
        mode: ``"sampled"`` 返回实际目标 token 的分数；
            ``"topk"`` 返回每个目标位置最可能的 k 个候选及分数。
        k: Top-k 路径的候选数量；sampled 路径不使用。

    Returns:
        sampled：与 target_ids 对齐的 [B, T] float32 logprob 张量。
        topk：字典，topk_ids 为 [B, T, k] long，
            topk_logprobs 为 [B, T, k] float32。
        Top-k 分数在共同完整词表上归一化，不在选出的 k 项上重新归一化。
    """
    # [B, P] + [B, T-1] -> [B, P+T-1]；最后一个目标无需再作为输入。
    input_ids = torch.cat((prefix_ids, target_ids[:, :-1]), dim=1)
    # [B, P+T-1]；当前输入不含 padding，所有位置均可见。
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # logits：[B, P+T-1, V_model]，尚未归一化的词表分数。
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    start = prefix_ids.shape[1] - 1  # 标量；上下文最后一个位置预测第一个目标 token。
    # [B, T, V]；当前 V=151665，排除模型输出层中 tokenizer 未映射的行。
    target_logits = logits[:, start: start + target_ids.shape[1], :151665]
    # [B, T, V] float32；在共同词表维度上计算 logprob。
    log_probs = torch.log_softmax(target_logits.float(), dim=-1)

    if mode == "sampled":

        sample_log_probs = log_probs.gather(
            dim=-1, 
            index=target_ids.unsqueeze(-1), # [B, T] -> [B, T, 1]
        ).squeeze(-1) # gather：[B, T, 1]；squeeze 后：[B, T]

        return sample_log_probs

    elif mode == "topk":

        # 从 V 个候选取前 k 个；indices 和 values 均为 [B, T, k]。
        topk_result = torch.topk(log_probs, k=k, dim=-1)
        return {
            "topk_ids" : topk_result.indices,  # long；候选 token ID。
            "topk_logprobs" : topk_result.values  # float32；原分布中的 logprob。
        }
    




def load_teacher(
        local_path: str,
        device: str | torch.device,
) -> torch.nn.Module:
    """从本地 checkpoint 加载冻结的 BF16 教师模型。

    Args:
        local_path: 教师 checkpoint 的本地目录。
        device: 模型加载后的目标设备。

    Returns:
        处于 eval 模式且参数不参与梯度计算的教师模型。
    """
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        local_files_only=True, # 只检查本地文件不负责下载
    )

    model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model
