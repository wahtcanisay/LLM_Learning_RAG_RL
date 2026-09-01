"""Teacher-owned acceptance checks; never train the real 3B model here."""

import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from MedSearch.sft import train
from test_sft_model import LORA_CONFIG, assert_only_lora_is_trainable, tiny_qwen


def cpu_config():
    config = train.load_config(PROJECT_ROOT / "MedSearch/configs/stage_a.yaml")
    config.update(dtype="float32", num_workers=0)
    config["training"] = {
        "output_dir": "outputs/test",
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "max_steps": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "seed": 20260831,
        "dataloader_pin_memory": False,  # This suite intentionally disables CUDA.
    }
    return config


def test_training_arguments_preserve_config_and_loss_boundaries(tmp_path):
    assert hasattr(train, "build_training_arguments"), "missing teacher argument builder"
    config = cpu_config()
    args = train.build_training_arguments(config, project_root=tmp_path)

    assert isinstance(args, TrainingArguments)
    assert Path(args.output_dir) == tmp_path / "outputs/test"
    assert config["training"]["output_dir"] == "outputs/test"
    assert args.per_device_train_batch_size == config["batch_size"]
    assert args.dataloader_num_workers == 0
    assert args.gradient_accumulation_steps == 4
    assert args.learning_rate == 2e-4
    assert args.max_steps == 1
    assert args.seed == args.data_seed == 20260831
    assert args.gradient_checkpointing
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert not args.bf16 and not args.fp16
    assert args.remove_unused_columns is False
    assert args.label_names == ["labels"]
    assert args.report_to == []
    assert args.eval_strategy == "no" and args.save_strategy == "no"

    config["training"]["output_dir"] = str(tmp_path / "absolute")
    assert Path(train.build_training_arguments(config, tmp_path).output_dir) == tmp_path / "absolute"


def test_training_arguments_allow_native_wandb_reporting(tmp_path):
    config = cpu_config()
    config["training"].update(
        report_to="wandb",
        run_name="medsearch-stage-a-qlora-seed20260831",
    )

    args = train.build_training_arguments(config, project_root=tmp_path)

    assert args.report_to == ["wandb"]
    assert args.run_name == "medsearch-stage-a-qlora-seed20260831"


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_checked_in_training_configs_are_valid_on_cpu(stage, tmp_path):
    config = train.load_config(PROJECT_ROOT / f"MedSearch/configs/{stage}.yaml")
    config["dtype"] = "float32"  # Validate wiring, not GPU/BF16 feasibility.
    args = train.build_training_arguments(config, tmp_path)
    assert Path(args.output_dir) == tmp_path / "MedSearch/outputs" / stage
    assert args.max_steps == (-1 if stage == "stage_a" else 1)
    assert args.optim == "adamw_torch"
    if stage == "stage_a":
        assert args.save_strategy == "steps"
        assert args.save_steps == 250
        assert args.save_total_limit == 2


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
@pytest.mark.parametrize("checkpointing", [True, False])
def test_trainer_reuses_dataset_and_collator_without_starting_training(tmp_path, checkpointing, stage):
    assert hasattr(train, "build_trainer"), "missing student Trainer interface"
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models/Qwen2.5-3B-Instruct", local_files_only=True
    )
    tokenizer.padding_side = "right"
    config = cpu_config()
    stage_config = train.load_config(PROJECT_ROOT / f"MedSearch/configs/{stage}.yaml")
    for key in ("stage", "data_path", "max_length"):
        config[key] = stage_config[key]
    config["training"]["gradient_checkpointing"] = checkpointing
    loader = train.build_dataloader(config, tokenizer)
    # Limit the in-memory fixture, not the frozen data file. Check every batch
    # element against the original collator output, independent of shuffle order.
    loader.dataset.samples = loader.dataset.samples[:2]
    expected = loader.collate_fn([loader.dataset[0], loader.dataset[1]])
    model = train.attach_lora(tiny_qwen(), LORA_CONFIG)

    trainer = train.build_trainer(config, model, tokenizer, loader, project_root=tmp_path)

    assert isinstance(trainer, Trainer)
    assert trainer.model is model
    assert trainer.train_dataset is loader.dataset
    assert trainer.data_collator is loader.collate_fn
    assert trainer.processing_class is tokenizer
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    assert trainer.state.global_step == 0
    assert trainer.eval_dataset is None
    # Transformers 5.10.4 Trainer.__init__ writes args.use_cache to model.config,
    # independently of gradient checkpointing. Its training default is False.
    assert model.config.use_cache is trainer.args.use_cache is False
    assert_only_lora_is_trainable(model)
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"] != -100).any()
    keys = ("input_ids", "attention_mask", "labels")
    assert sorted(zip(*(batch[key].tolist() for key in keys))) == sorted(
        zip(*(expected[key].tolist() for key in keys))
    )
