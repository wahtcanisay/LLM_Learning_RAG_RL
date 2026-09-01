"""Teacher-owned one-update checks; real 3B training is never run by pytest."""

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, set_seed

from test_sft_trainer import cpu_config, train
from test_sft_model import LORA_CONFIG, tiny_qwen


@pytest.mark.parametrize("checkpoint", [None, "checkpoint-250"])
def test_run_training_forwards_only_real_resume_requests(checkpoint):
    output = SimpleNamespace(
        global_step=251 if checkpoint else 1,
        metrics={"train_loss": 1.0, "train_runtime": 2.0},
    )
    trainer = SimpleNamespace(train=Mock(return_value=output))

    result = train.run_training(trainer, resume_from_checkpoint=checkpoint)

    if checkpoint is None:
        trainer.train.assert_called_once_with()
    else:
        trainer.train.assert_called_once_with(resume_from_checkpoint=checkpoint)
    assert result == {**output.metrics, "global_step": output.global_step}


def test_smoke_config_isolated_and_original_unchanged(tmp_path):
    assert hasattr(train, "prepare_train_smoke"), "missing teacher smoke guard"
    config = cpu_config()
    before = deepcopy(config)
    first = train.prepare_train_smoke(config, tmp_path)
    second = train.prepare_train_smoke(config, tmp_path)
    assert config == before
    assert first["training"]["output_dir"] != second["training"]["output_dir"]
    assert Path(first["training"]["output_dir"]).is_dir()
    assert Path(first["training"]["output_dir"]).parent == tmp_path / "MedSearch/outputs/smoke"
    assert first["training"]["max_steps"] == 1
    assert first["training"]["logging_nan_inf_filter"] is False
    assert first["training"]["skip_memory_metrics"] is False
    assert first["smoke_sample_count"] == 8


@pytest.mark.parametrize("key,value", [("stage", "stage_b"), ("max_steps", -1), ("max_steps", 2)])
def test_smoke_rejects_other_stages_or_unbounded_training(tmp_path, key, value):
    assert hasattr(train, "prepare_train_smoke"), "missing teacher smoke guard"
    config = cpu_config()
    if key == "stage":
        config[key] = value
    else:
        config["training"][key] = value
    with pytest.raises(ValueError):
        train.prepare_train_smoke(config, tmp_path)
    assert not (tmp_path / "MedSearch/outputs").exists()


def test_run_training_updates_lora_but_not_frozen_base(tmp_path, monkeypatch):
    assert hasattr(train, "run_training"), "missing student training interface"
    config = cpu_config()
    config["training"].update(output_dir=str(tmp_path), disable_tqdm=True)
    set_seed(config["training"]["seed"])
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 3}
    vocab.update({f"t{i}": i for i in [2, *range(4, 64)]})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    model = tiny_qwen()
    model.config.pad_token_id = model.generation_config.pad_token_id = 0
    model.config.eos_token_id = model.generation_config.eos_token_id = 3
    model = train.attach_lora(model, LORA_CONFIG)
    samples = [
        {"input_ids": [4 + i, 12, 13, 3], "attention_mask": [1, 1, 1, 1],
         "labels": [-100, -100, 13, 3]}
        for i in range(8)
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=train.build_data_collator(tokenizer))
    trainer = train.build_trainer(config, model, tokenizer, loader)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    # Observe the real call/return boundary without substituting fake training.
    actual_train = trainer.train
    outputs = []

    def record_train(*args, **kwargs):
        output = actual_train(*args, **kwargs)
        outputs.append((output, dict(output.metrics)))
        return output

    monkeypatch.setattr(trainer, "train", record_train)
    result = train.run_training(trainer)
    assert isinstance(result, dict)
    assert result["global_step"] == trainer.state.global_step == 1
    assert math.isfinite(result["train_loss"])
    assert result["train_runtime"] > 0
    gradients = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    assert gradients and all(math.isfinite(value) for value in gradients)
    assert any(value > 0 for value in gradients)
    changed = []
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        if not torch.equal(before[name], parameter.detach()):
            changed.append(name)
        if "lora_" not in name:
            assert not parameter.requires_grad
            assert torch.equal(before[name], parameter.detach())
    assert changed and all("lora_" in name for name in changed)
    assert trainer.optimizer is not None
    assert not list(tmp_path.glob("checkpoint-*"))
    # Callbacks can remove total_flos from metrics after log_history captures it;
    # compare with TrainOutput.metrics, never infer a return schema from logging.
    assert len(outputs) == 1
    output, original_metrics = outputs[0]
    assert result == {**original_metrics, "global_step": output.global_step}
    assert result is not output.metrics
    assert output.metrics == original_metrics


def test_checkpoint_resume_restores_trainer_state_and_continues(tmp_path):
    config = cpu_config()
    config["training"].update(
        output_dir=str(tmp_path),
        max_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        disable_tqdm=True,
    )
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 3}
    vocab.update({f"t{i}": i for i in [2, *range(4, 64)]})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    samples = [
        {"input_ids": [4 + i, 12, 13, 3], "attention_mask": [1, 1, 1, 1],
         "labels": [-100, -100, 13, 3]}
        for i in range(8)
    ]
    loader = DataLoader(
        samples, batch_size=2, collate_fn=train.build_data_collator(tokenizer)
    )

    set_seed(config["training"]["seed"])
    first_model = train.attach_lora(tiny_qwen(), LORA_CONFIG)
    first = train.build_trainer(config, first_model, tokenizer, loader)
    assert train.run_training(first)["global_step"] == 1
    checkpoint = tmp_path / "checkpoint-1"
    assert checkpoint.is_dir()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "trainer_state.json").is_file()

    config["training"]["max_steps"] = 2
    set_seed(config["training"]["seed"])
    resumed_model = train.attach_lora(tiny_qwen(), LORA_CONFIG)
    resumed = train.build_trainer(config, resumed_model, tokenizer, loader)
    result = train.run_training(resumed, resume_from_checkpoint=str(checkpoint))

    assert result["global_step"] == resumed.state.global_step == 2
    assert resumed.optimizer is not None and resumed.optimizer.state
    assert resumed.lr_scheduler is not None
    assert resumed.lr_scheduler.last_epoch == 2
    assert (tmp_path / "checkpoint-2").is_dir()
    logged_steps = {entry.get("step") for entry in resumed.state.log_history}
    assert {1, 2}.issubset(logged_steps)
