"""多轮 rollout 的 padding/mask 小工具（学习重要性 P1）。

Transformers tokenizer 常把 prompt 左填充以便从序列末尾继续生成，而累计 response
通常右填充。Search-R1 每轮拼接 prompt/action/information 后需要在两种布局间转换。
"""

import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    """只保存张量整理所需的 pad id 与长度上限。"""
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    """不含模型计算，只维护 batch 内 token 的次序、mask 和 position id。"""
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """按 batch 中最长有效序列裁掉整列 padding，并保持各字段形状一致。"""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """稳定移动 pad 到左/右侧，并返回相同重排索引。

        这里不是按 token 值排序：mask 只有 0/1，稳定排序只把 pad 与非 pad 分组。
        """
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id # 得到输入tensor形状的mask列表
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True) # 转化为int排序
        return tensor.gather(1, sorted_indices), sorted_indices # 按下标重新排列原来tensor

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """非 pad token 为 1；检索 information 也是可被模型注意的有效 token。"""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """有效 token 从 0 递增，pad 位置保持 0，兼容左填充输入。"""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """先沿序列维拼接，再把各段内部的 pad 统一移动到目标侧。"""
        concatenated = torch.cat(tensors, dim=1) #拼接三种向量
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left) # PAD整理进入一侧之中
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        把只含 active 样本的生成结果散射回原 batch 大小。

        已 answer 的轨迹在后续轮不会参与生成，因此其本轮位置全部填 pad/空串；
        这能让其余张量和题目元数据仍保持相同 batch 下标。
        """
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str
