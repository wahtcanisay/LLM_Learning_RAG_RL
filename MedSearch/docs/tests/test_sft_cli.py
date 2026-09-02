"""Teacher checks for the config-only formal SFT entry point."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_sft_trainer import train, cpu_config
from test_sft_model import tiny_qwen


@pytest.mark.parametrize(
    "legacy_flag",
    ["--qlora", "--model-smoke", "--trainer-smoke", "--train-smoke", "--save-adapter"],
)
def test_formal_cli_rejects_legacy_training_switches(legacy_flag):
    with pytest.raises(SystemExit) as error:
        train.main(["--config", "unused.yaml", legacy_flag])
    assert error.value.code == 2


@pytest.mark.parametrize("use_qlora", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_main_runs_configured_training_pipeline(
    monkeypatch, tmp_path, use_qlora, checkpointing
):
    config = cpu_config()
    config["dtype"] = "bfloat16"
    config["use_qlora"] = use_qlora
    config["resume_from_checkpoint"] = None
    config["training"]["output_dir"] = str(tmp_path)
    config["training"]["gradient_checkpointing"] = checkpointing
    monkeypatch.setattr(train, "load_config", lambda path: config)
    order = []
    tokenizer = SimpleNamespace(padding_side="left")
    train_dataloader = [
        {"input_ids": torch.ones(1, 1, dtype=torch.long),
         "attention_mask": torch.ones(1, 1, dtype=torch.long),
         "labels": torch.ones(1, 1, dtype=torch.long)}
    ]
    expected_eval_dataloader = [
        {"input_ids": torch.full((1, 1), 2, dtype=torch.long),
         "attention_mask": torch.ones(1, 1, dtype=torch.long),
         "labels": torch.full((1, 1), 2, dtype=torch.long)}
    ]
    base = tiny_qwen().to(dtype=torch.bfloat16)
    peft_model = object()
    trainer = SimpleNamespace(model=peft_model)

    monkeypatch.setattr(
        train.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        train,
        "build_dataloader",
        lambda *args, split="train", **kwargs: (
            train_dataloader if split == "train" else expected_eval_dataloader
        ),
    )
    monkeypatch.setattr(train, "describe_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(train, "describe_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(train, "set_seed", lambda seed: order.append("seed"))

    def build_quantization(compute_dtype):
        assert compute_dtype is torch.bfloat16
        order.append("quantization")
        return "quantization-config"

    monkeypatch.setattr(train, "build_quantization_config", build_quantization)

    def load_tiny(passed_config, *, quantization_config=None, dtype):
        assert passed_config is config
        assert dtype is torch.bfloat16
        if use_qlora:
            assert quantization_config == "quantization-config"
        else:
            assert quantization_config is None
        order.append("load")
        return base

    def prepare(model, use_gradient_checkpointing):
        assert model is base
        assert use_gradient_checkpointing is checkpointing
        order.append("prepare")
        return model

    def attach(model, passed_config):
        assert model is base
        order.append("attach")
        return peft_model

    def build(
        passed_config,
        model,
        passed_tokenizer,
        dataloader,
        *,
        eval_dataloader=None,
    ):
        assert (passed_config, model, passed_tokenizer, dataloader) == (
            config, peft_model, tokenizer, train_dataloader
        )
        assert eval_dataloader is expected_eval_dataloader
        order.append("trainer")
        return trainer

    def run(passed_trainer, *, resume_from_checkpoint=None):
        assert passed_trainer is trainer
        assert resume_from_checkpoint is None
        order.append("train")
        return {"global_step": 1, "train_loss": 1.0}

    def save(model, passed_tokenizer, output_dir):
        assert (model, passed_tokenizer) == (peft_model, tokenizer)
        assert Path(output_dir) == tmp_path / "adapter"
        order.append("save")

    monkeypatch.setattr(train, "load_base_model", load_tiny)
    monkeypatch.setattr(train, "prepare_qlora_model", prepare)
    monkeypatch.setattr(train, "attach_lora", attach)
    monkeypatch.setattr(train, "build_trainer", build)
    monkeypatch.setattr(train, "run_training", run)
    monkeypatch.setattr(train, "save_adapter", save)

    train.main(["--config", "unused.yaml"])

    middle = ["quantization", "load", "prepare"] if use_qlora else ["load"]
    assert order == ["seed", *middle, "attach", "trainer", "train", "save"]
