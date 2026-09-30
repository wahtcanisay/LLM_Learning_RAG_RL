"""教师模型评分接口。"""
import torch
from transformers import AutoModelForCausalLM

def score_tokens(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mode: str = "sampled",
    k: int = 32,
) -> torch.Tensor:
    """计算教师模型对目标 token 的逐 token logprob。

    Args:
        model: 已加载的教师因果语言模型，通常由
            AutoModelForCausalLM.from_pretrained(...) 创建；本函数只调用它评分。
        prefix_ids: [B, P] 的 torch.long 上下文 token ID。
        target_ids: [B, T] 的 torch.long 待评分 token ID。
        mode: 评分方式；当前任务只实现 ``"sampled"``。
        k: Top-k 路径的候选数量；sampled 路径不使用。

    Returns:
        [B, T] 的 logprob 张量，与target_ids逐位置对应
    """
    input_ids = torch.cat((prefix_ids, target_ids[:, :-1]), dim=1)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # 前向传播拿到logits分布
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    start = prefix_ids.shape[1] - 1
    target_logits = logits[:, start: start + target_ids.shape[1], :151665] # 151665代表tokenizer的词表大小
    log_probs = torch.log_softmax(target_logits.float(), dim=-1) # 取log_p, 这里回到FP32

    sample_log_probs = log_probs.gather(
        dim=-1, 
        index=target_ids.unsqueeze(-1), # [B, T] -> [B, T, 1]
    ).squeeze(-1) # [B, T, 1] -> [B, T]

    return sample_log_probs


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
