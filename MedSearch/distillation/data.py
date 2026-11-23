"""把一条蒸馏轨迹整理成模型训练样本。"""


def encode_distillation_trace(record: dict) -> dict[str, list[int]]:
    """将原始轨迹编码为冷启动训练样本。

    Args:
        record: dict
        {
            "prompt_ids": ..., # 问题以及上文tokenid
            "response_ids":... , # 模型动作与环境tokenid
            "response_mask":... , # 0代表环境返回掩码
        }
    Returns:
        {
            "input_ids": ...,       # prompt_ids 与 response_ids 拼接
            "attention_mask": ...,  # 有效 token 位置为 1
            "labels": ...,          # 模型生成位置保留 token ID，上文以及环境生成位置为 -100
        }
    """
    prompt_ids, response_ids = record["prompt_ids"], record["response_ids"]
    response_mask = record["response_mask"]
    
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + [
        tokenid if mask == 1 else -100
        for tokenid, mask in zip(response_ids, response_mask)
    ]
    attetion_mask = [1] * len(input_ids)

    return {"input_ids": input_ids,
            "attention_mask": attetion_mask,
            "labels": labels,
            }
