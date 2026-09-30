"""Read and batch frozen teacher trajectories for search distillation."""

from pathlib import Path
import json

from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

from ..distillation.data import encode_distillation_trace

IGNORE_INDEX = -100
STAGES = ("search_distill",)


def load_samples(path: str | Path) -> list[dict]:
    """Load frozen teacher trajectories without changing their order."""
    path = Path(path)

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    raise ValueError(f"distillation data must be JSONL: {path}")


class SFTDataset(Dataset):
    """Expose one frozen teacher trajectory as a causal training sample."""

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
        """Preserve original IDs and exclude environment tokens from labels."""
        sample = self.samples[index]
        encoded = encode_distillation_trace(sample)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"encoded trace has {len(encoded['input_ids'])} tokens, "
                f"exceeding max_length={self.max_length}"
            )
        return encoded



def build_data_collator(tokenizer) -> DataCollatorForSeq2Seq:
    """Reuse the Transformers collator already validated by MedicalGPT."""
    # 函数功能具体是在dataset按照index取出，返回了单条，未padding的dict，
    # 经过这个DataCollatorForSeq2Seq，组成长度一致的tensor
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )
