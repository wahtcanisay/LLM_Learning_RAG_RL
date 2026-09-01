"""Minimal Stage-A/Stage-B dataset wiring for MedSearch SFT."""

from pathlib import Path
import json

from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

from .preprocess import IGNORE_INDEX, encode_stage_a, encode_stage_b


STAGES = ("stage_a", "stage_b")


def load_samples(path: str | Path) -> list[dict]:
    """Load frozen SFT samples without changing their schema or order.

    Contract:
    - Read Stage-A ``.jsonl`` as one JSON object per non-empty line.
    - Read Stage-B ``.json`` as one top-level JSON list.
    - Return the original sample dictionaries in file order.
    - Reject unsupported suffixes instead of guessing their format.

    Student task: implement this function with Python's standard ``json`` module.
    """
    # 数据清洗代码，针对jsonl和json有不同的加载
    path = Path(path)

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            samples = json.load(file)

        if not isinstance(samples, list):
            raise ValueError("JSON SFT data must contain a top-level list")

        return samples

    raise ValueError(f"unsupported SFT data format: {path.suffix}")


class SFTDataset(Dataset):
    """Encode one frozen raw sample when PyTorch requests an item."""

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
        """Return one encoded sample using the encoder selected by ``stage``.

        Contract:
        - ``stage_a`` calls ``encode_stage_a``.
        - ``stage_b`` calls ``encode_stage_b``.
        - Pass ``sample``, ``tokenizer`` and ``max_length`` through unchanged.
        - Do not pad here; padding belongs to the batch collator.

        Student task: implement only this stage dispatch.
        """
        sample = self.samples[index]

        if self.stage == "stage_a":
            return encode_stage_a(sample, self.tokenizer, self.max_length)

        else:
            return encode_stage_b(sample, self.tokenizer, self.max_length)



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
