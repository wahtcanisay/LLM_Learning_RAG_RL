"""Teacher-owned RED checks for train-time Stage-A dev loss wiring."""

import json
from pathlib import Path
import sys

import pytest
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from MedSearch.sft import train
from test_sft_model import LORA_CONFIG, tiny_qwen


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models/Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    return tokenizer


def write_stage_a(path: Path, sample_id: str) -> None:
    sample = {
        "id": sample_id,
        "source": "fixture",
        "mixture_group": "single",
        "messages": [
            {"role": "user", "content": f"question {sample_id}"},
            {"role": "assistant", "content": f"answer {sample_id}"},
        ],
    }
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def make_config(tmp_path: Path) -> dict:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "dev.jsonl"
    write_stage_a(train_path, "train-only")
    write_stage_a(eval_path, "dev-only")
    return {
        **LORA_CONFIG,
        "stage": "stage_a",
        "data_path": str(train_path),
        "eval_data_path": str(eval_path),
        "model_path": "models/Qwen2.5-3B-Instruct",
        "dtype": "float32",
        "use_qlora": False,
        "resume_from_checkpoint": None,
        "max_length": 128,
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 0,
        "training": {
            "output_dir": str(tmp_path / "output"),
            "learning_rate": 2e-4,
            "num_train_epochs": 1,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "seed": 7,
            "report_to": "none",
            "dataloader_pin_memory": False,
        },
    }


def test_build_dataloader_selects_frozen_dev_without_shuffle(tmp_path, tokenizer):
    config = make_config(tmp_path)

    train_loader = train.build_dataloader(config, tokenizer, split="train")
    eval_loader = train.build_dataloader(config, tokenizer, split="eval")

    assert train_loader.dataset.samples[0]["id"] == "train-only"
    assert eval_loader.dataset.samples[0]["id"] == "dev-only"
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(eval_loader.sampler, SequentialSampler)


def test_build_dataloader_rejects_unknown_split(tmp_path, tokenizer):
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match="split"):
        train.build_dataloader(config, tokenizer, split="test")


def test_stage_b_reuses_stage_a_dev_with_stage_a_encoder(tmp_path, tokenizer):
    config = make_config(tmp_path)
    stage_b_path = tmp_path / "stage_b.json"
    stage_b_path.write_text(
        json.dumps(
            [
                {
                    "id": "stage-b-train",
                    "question": "Which option is correct?",
                    "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                    "trajectory": [
                        {"type": "answer", "thinking": "reason", "answer": "A"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    config.update(
        stage="stage_b",
        data_path=str(stage_b_path),
        eval_stage="stage_a",
    )

    train_loader = train.build_dataloader(config, tokenizer, split="train")
    eval_loader = train.build_dataloader(config, tokenizer, split="eval")

    assert train_loader.dataset.stage == "stage_b"
    assert eval_loader.dataset.stage == "stage_a"
    assert train_loader.dataset[0]["labels"]
    assert eval_loader.dataset[0]["labels"]


def test_training_arguments_reuse_native_eval_fields(tmp_path):
    config = make_config(tmp_path)
    config["training"].update(
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
    )

    args = train.build_training_arguments(config, project_root=tmp_path)

    assert args.eval_strategy == "steps"
    assert args.eval_steps == 250
    assert args.per_device_eval_batch_size == 1
    assert args.prediction_loss_only is True


def test_build_trainer_receives_eval_dataset(tmp_path, tokenizer):
    config = make_config(tmp_path)
    train_loader = train.build_dataloader(config, tokenizer, split="train")
    eval_loader = train.build_dataloader(config, tokenizer, split="eval")
    model = train.attach_lora(tiny_qwen(), LORA_CONFIG)

    trainer = train.build_trainer(
        config,
        model,
        tokenizer,
        train_loader,
        eval_dataloader=eval_loader,
        project_root=tmp_path,
    )

    assert trainer.train_dataset is train_loader.dataset
    assert trainer.eval_dataset is eval_loader.dataset
    assert trainer.data_collator is train_loader.collate_fn
