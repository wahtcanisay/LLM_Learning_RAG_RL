"""Read and batch frozen teacher trajectories for search distillation"""

from pathlib import Path
import json
import torch

from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

from ..distillation.data import encode_distillation_trace

IGNORE_INDEX = -100
STAGES = ("search_distill",)


def load_samples(path: str | Path) -> list[dict]:
    """Load frozen teacher trajectories without changing their order"""
    path = Path(path)

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()

    raise ValueError(f"distillation data must be JSONL or Parquet: {path}")


class SFTDataset(Dataset):
    """Expose one frozen teacher trajectory as a causal training sample"""

    def __init__(self, samples: list[dict], tokenizer, stage: str, max_length: int):
        if stage not in STAGES:
            raise ValueError(f"unknown SFT stage: {stage}")
        self.samples = samples
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """取出单条轨迹, 保留教师缓存

        sample:
            prompt_ids: 问题与上文 token ID
            response_ids: 模型生成与检索返回的 token ID
            response_mask: 模型生成为 1, 检索返回为 0
        Returns:
            input_ids: prompt_ids 与 response_ids 拼接
            attention_mask: 有效 token 为 1
            labels: 模型生成位置保留 ID, 其余为 -100
            教师缓存字段
        """
        sample = self.samples[index]
        encoded = encode_distillation_trace(sample)

        if sample.get("teacher_topk_ids"):
            encoded["target_positions"] = sample["target_positions"]  # [N], 目标在 input_ids 中的位置
            encoded["teacher_topk_ids"] = sample["teacher_topk_ids"]  # [N, 32], 教师候选 token ID
            encoded["teacher_topk_logprobs"] = sample["teacher_topk_logprobs"]  # [N, 32], 对应的 log 概率

        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"encoded trace has {len(encoded['input_ids'])} tokens, "
                f"exceeding max_length={self.max_length}"
            )
        
        return encoded



def build_data_collator(tokenizer):
    """返回组 batch 的函数: 普通字段补齐, 教师缓存保留各自长度"""

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    def collate(features):
        """输入单条样本列表; 返回普通字段 [B, L] 张量和教师缓存张量列表"""
        sample_dict = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels":feature["labels"],
            }
            for feature in features
        ]

        batch = base_collator(sample_dict)

        if "teacher_topk_ids" in features[0]:
            # 每条缓存各转一个张量, 不补齐; 顺序与普通字段一致
            batch["target_positions"] = [
                torch.tensor(feature["target_positions"], dtype=torch.long)
                for feature in features
            ]

            batch["teacher_topk_ids"] = [
                torch.tensor(feature["teacher_topk_ids"], dtype=torch.long)
                for feature in features
            ]

            batch["teacher_topk_logprobs"] = [
                torch.tensor(feature["teacher_topk_logprobs"], dtype=torch.float32)
                for feature in features
            ]

        return batch

    return collate


