"""Single MedSearch SFT entry point: assembly checks and bounded training smoke."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile

import yaml
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

from .data import SFTDataset, build_data_collator, load_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CONFIG_KEYS = (
    "stage",
    "data_path",
    "model_path",
    "max_length",
    "batch_size",
    "shuffle",
    "num_workers",
    "use_qlora",
    "resume_from_checkpoint",
)


def load_config(path: str | Path) -> dict:
    """Read and minimally validate one Stage-A or Stage-B YAML config."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("SFT config must be a YAML mapping")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"SFT config missing keys: {', '.join(missing)}")
    if config["stage"] not in ("stage_a", "stage_b"):
        raise ValueError(f"unknown SFT stage: {config['stage']}")
    return config


def build_dataloader(
    config: dict,
    tokenizer,
    project_root: Path = PROJECT_ROOT,
) -> DataLoader:
    """Connect one YAML config to the existing MedSearch data pipeline.

    Contract:
    - Resolve ``data_path`` relative to ``project_root``; absolute paths also work.
    - Reuse ``load_samples`` and ``SFTDataset`` without changing sample schemas.
    - Reuse ``build_data_collator`` for dynamic padding.
    - Pass batch size, shuffle and worker count directly from ``config``.
    - Return the PyTorch ``DataLoader``; do not fetch a batch in this function.

    Student task: implement only this wiring. Do not load a model or create a
    Trainer here.
    """
    path = Path(config["data_path"])

    if not path.is_absolute():
        path = project_root / path

    samples = load_samples(path)
    dataset = SFTDataset(samples, tokenizer, config["stage"], config["max_length"])
    collator_fn = build_data_collator(tokenizer)

    return DataLoader(dataset,
                      batch_size=config["batch_size"],
                      shuffle=config["shuffle"],
                      num_workers=config["num_workers"],
                      collate_fn=collator_fn)


def build_quantization_config(
    compute_dtype: torch.dtype,
) -> BitsAndBytesConfig:
    """Build the QLoRA quantization config; do not load or modify a model.

    Contract:
    - Input is an already resolved torch dtype (bfloat16, float16 or float32),
      not a YAML mapping or dtype string. The caller owns the QLoRA branch.
    - Return a native BitsAndBytesConfig for 4-bit NF4 with double
      quantization. Its compute dtype must be the supplied compute_dtype;
      do not hard-code BF16 or enable 8-bit loading at the same time.
    - Return the config object itself, not a dict, model or quantized weights.
    - Do not load a model/tokenizer, prepare k-bit training, attach LoRA,
      allocate CUDA tensors or construct a Trainer here.

    Reference: MedicalGPT/training/supervised_finetuning.py, QLoRA configuration
    branch around lines 740-765.
    """

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, # 4bit加载量化权重
                                             bnb_4bit_compute_dtype=compute_dtype, # 量化层计算浮点精度
                                             bnb_4bit_use_double_quant=True, # 双重量化进一步减少开销
                                             bnb_4bit_quant_type="nf4") # nf4量化格式
    return quantization_config



def load_base_model(
    config: dict,
    project_root: Path = PROJECT_ROOT,
    *,
    quantization_config: BitsAndBytesConfig | None = None,
    dtype : torch.dtype
):
    """Load a local base model, optionally with the caller's QLoRA config.

    Contract for the student's new quantized branch:
    - Keep existing model_path resolution and local-only loading. The caller
      supplies the already resolved dtype used by both loading branches.
    - None preserves the original non-quantized behavior; no device_map needed.
    - Otherwise receive the already-built 4-bit config object, not a boolean;
      pass that exact object as quantization_config to from_pretrained.
    - This first quantized path targets one visible GPU: device_map={"": 0}.
      Raise RuntimeError mentioning CUDA before loading if CUDA is unavailable.
    - Keep dtype, low_cpu_mem_usage=True and local_files_only=True. Return the
      loaded base model, not an adapter, Trainer, tuple or dictionary.
    - Do not call model.to(), use device_map="auto", offload to CPU/disk, prepare
      k-bit training, attach LoRA or run training here. Those are later steps.

    Student task: replace only the quantized-branch placeholder below. Existing
    callers still use the non-quantized default; formal main remains yours.
    """
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = project_root / model_path


    if quantization_config is not None:
        # Student task: CUDA availability check and native quantized loading.
        if torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    device_map={"":0}
                )
        else:
            raise RuntimeError("CUDA is required for 4-bit loading")

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def prepare_qlora_model(model, use_gradient_checkpointing: bool):
    """Prepare an already loaded 4-bit base, before attaching any LoRA adapter.

    Contract:
    - The caller supplies a bare 4-bit base from load_base_model, not a path,
      config dict, Trainer or PeftModel. Non-quantized callers skip this helper.
    - Reuse PEFT prepare_model_for_kbit_training once; pass the supplied
      use_gradient_checkpointing boolean, including False.
    - Use gradient_checkpointing_kwargs={"use_reentrant": False}, matching      our existing build_training_arguments. Let PEFT handle freezing, dtype
      preparation and checkpointing; do not implement those operations by hand.
    - Return the prepared model itself. PEFT modifies this base in place;
      this is not a copy, an adapter, a dictionary or a tuple.
    - Do not reload/quantize/move the model, attach LoRA, construct a Trainer,
      train or save here. Leave use_cache handling to the existing Trainer path.

    Reference: MedicalGPT/training/supervised_finetuning.py:920-923 (preparation
    before new LoRA). Our checkpointing kwargs match our current HF setup.
    Student task: implement only the native preparation call and return.
    Formal main remains yours; this helper is not wired into any caller yet.
    """
    # Qlora省显存，还有一个部分是要裁撤中间的activation来进行的，算是以时间换空间的操作
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing, # 对中间的activation进行部分清除，backward需要的情况下重新算
                                            gradient_checkpointing_kwargs={"use_reentrant": False}, # 选择checkpoint实现方式
                                            )
    return model

def attach_lora(model, config: dict, project_root: Path = PROJECT_ROOT,):
    """Create Stage-A LoRA or resume the Stage-A adapter for Stage-B.

    Contract:
    - If ``adapter_path`` is null, build ``LoraConfig`` from the four LoRA
      fields and call ``get_peft_model``.
    - If ``adapter_path`` is set, resolve it relative to ``project_root`` and
      load it with ``PeftModel.from_pretrained(..., is_trainable=True)``.
    - Raise ``FileNotFoundError`` before PEFT loading when that path is absent.
    - Return the PEFT model in both branches; do not move it to a device.

    Student task: implement only this adapter dispatch. Do not create an
    optimizer or Trainer here.
    """
    adapter_path = config["adapter_path"]
    if adapter_path is None:
        lora_config = LoraConfig(r=config["lora_rank"],
                                 task_type=TaskType.CAUSAL_LM,
                                 target_modules=config["lora_target_modules"],
                                 inference_mode=False,
                                 lora_alpha=config["lora_alpha"],
                                 lora_dropout=config["lora_dropout"])
        peft_model = get_peft_model(model, lora_config)
    else:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = project_root / adapter_path

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter path does not exist: {adapter_path}"
            )

        peft_model = PeftModel.from_pretrained(model,
                                               str(adapter_path),
                                               is_trainable=True)

    return peft_model







def build_training_arguments(
    config: dict,
    project_root: Path = PROJECT_ROOT,
) -> TrainingArguments:
    """Teacher wiring: reuse HF arguments, preserving our precomputed labels.

    ``training`` contains native TrainingArguments fields. Batch size and worker
    count remain shared with the data smoke. No evaluation or saving until those
    paths have their own acceptance checks; max_steps=1 is only a smoke default.
    """
    training = dict(config["training"])
    training.setdefault("save_strategy", "no")
    training.setdefault("report_to", "none")
    output_dir = Path(training.pop("output_dir"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if config["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"unsupported model dtype: {config['dtype']}")
    return TrainingArguments(
        output_dir=str(output_dir),
        **training,
        per_device_train_batch_size=config["batch_size"],
        dataloader_num_workers=config["num_workers"],
        data_seed=training["seed"],
        bf16=config["dtype"] == "bfloat16",
        fp16=config["dtype"] == "float16",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy="no",
    )


def build_trainer(
    config: dict,
    model,
    tokenizer,
    dataloader: DataLoader,
    project_root: Path = PROJECT_ROOT,
) -> Trainer:
    """Assemble a Trainer around the already verified LoRA and data components.

    Contract:
    - Reuse ``build_training_arguments(config, project_root)``.
    - Disable ``model.config.use_cache`` when gradient checkpointing is enabled;
      otherwise this wrapper need not set it. Transformers 5.10.4 Trainer itself
      writes ``args.use_cache`` (default False) to the model during construction.
      Let Trainer enable checkpointing at training time using the configured
      non-reentrant implementation.
    - Reuse the supplied model, tokenizer, dataloader.dataset and
      dataloader.collate_fn. HF calls the tokenizer argument ``processing_class``.
    - Return a plain HF Trainer; it builds its own shuffled training DataLoader.
      Config ``shuffle`` controls only our separate data smoke DataLoader.
    - Keep our encoded labels intact. Do not re-tokenize, shift labels or replace
      the collator with a language-modeling collator that regenerates labels.
    - Do not call train/evaluate/save, create an optimizer, load another model,
      unfreeze the base weights or implement a Trainer subclass.

    Student task: implement only this assembly, not the training loop.
    """

    train_arg = build_training_arguments(config, project_root)
    if train_arg.gradient_checkpointing:
        model.config.use_cache = False
    trainer = Trainer(model=model,
                      args=train_arg,
                      processing_class=tokenizer, # 模型配套使用的数据处理器，文本模型通常传tokenizer
                      data_collator=dataloader.collate_fn,
                      train_dataset=dataloader.dataset) # 其实这个dataload可以不用吧，

    return trainer


def prepare_train_smoke(config: dict, project_root: Path = PROJECT_ROOT) -> dict:
    """Teacher guard: one Stage-A update, fresh output, no mutation of the YAML."""
    if config["stage"] != "stage_a" or config["training"]["max_steps"] != 1:
        raise ValueError("--train-smoke requires stage_a and max_steps=1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--train-smoke currently supports one process only")
    batch_size = config["batch_size"]
    accumulation = config["training"]["gradient_accumulation_steps"]
    if batch_size <= 0 or accumulation <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    smoke = deepcopy(config)
    smoke["smoke_sample_count"] = batch_size * accumulation
    output_root = project_root / "MedSearch/outputs/smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    smoke["training"].update(
        output_dir=tempfile.mkdtemp(prefix="stage_a-", dir=output_root),
        logging_steps=1,
        logging_nan_inf_filter=False,
        skip_memory_metrics=False,
    )
    return smoke


def run_training(
    trainer: Trainer,
    *,
    resume_from_checkpoint: str | Path | None = None,
) -> dict:
    """Execute the configured training run and return its real scalar metrics.

    Contract:
    - The caller has already configured the model, data, seed and step limit.
    - Delegate training to the supplied HF Trainer exactly once.
    - When ``resume_from_checkpoint`` is None, call ``trainer.train()`` without
      a resume keyword. Otherwise forward that exact path through
      ``trainer.train(resume_from_checkpoint=...)``.
    - Return a new dictionary containing the TrainOutput metrics and an explicit
      ``global_step`` read from that output. Do not return Trainer or TrainOutput.
    - Do not fabricate loss/steps, catch training failures, re-seed/rebuild the
      model, or manually implement backward/optimizer/scheduler operations.
    - Do not evaluate or save an adapter/checkpoint here. HF Trainer owns periodic
      checkpoints; the caller owns final adapter saving and run metadata.

    Student task: replace the fresh-only call below with the two-branch resume
    dispatch. Keep the existing result conversion unchanged.
    """
    if resume_from_checkpoint is None:
        train_result = trainer.train() # .train()返回的TrianOutput有三个属性，更新的step，汇总的训练loss，metrics的指标字典
    else:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    metrics = train_result.metrics
    global_step = train_result.global_step
    result_dict = {**metrics, "global_step": global_step}
    return result_dict



def save_adapter(model: PeftModel, tokenizer, output_dir: str | Path) -> None:
    """Save the current LoRA adapter and its matching tokenizer.

    Contract:
    - Inputs: an already attached, single-device PeftModel, its tokenizer, and
      a fresh empty directory created by the caller. No training happens here.
    - Reuse PEFT/HF save_pretrained APIs. Save adapter_config.json and
      adapter_model.safetensors (safe_serialization=True), plus the tokenizer's
      native files, including its chat template and special-token settings.
    - Our vocabulary/embeddings are unchanged and frozen: explicitly use
      save_embedding_layers=False. Do not merge LoRA or save the full base model.
    - Return None, not paths, metrics, Trainer, or a save_pretrained return value.
    - Do not change weights/tokenizer, catch save failures, create an optimizer,
      or save a full training checkpoint. Adapter reload is not optimizer/step
      resume; run_training and the caller's metric files remain unchanged.

    Reference: MedicalGPT/training/supervised_finetuning.py::save_model.
    Student task: implement only native adapter/tokenizer saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_embedding_layers=False) # 显式声明代表不保存完整的embedding权重矩阵
    tokenizer.save_pretrained(output_dir)



def describe_model(model) -> dict:
    """Verify the frozen-base boundary and summarize real parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    non_lora_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if trainable == 0:
        raise ValueError("LoRA model has no trainable parameters")
    if non_lora_trainable:
        raise ValueError(f"non-LoRA parameters are trainable: {non_lora_trainable[:5]}")
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": trainable / total * 100,
    }


def describe_batch(config: dict, dataloader: DataLoader, batch: dict) -> dict:
    """Validate the smoke batch and return a machine-readable summary."""
    shapes = [batch[key].shape for key in ("input_ids", "attention_mask", "labels")]
    if not shapes[0] == shapes[1] == shapes[2]:
        raise ValueError(f"SFT batch tensors have different shapes: {shapes}")
    if shapes[0][1] > config["max_length"]:
        raise ValueError(
            f"SFT batch length {shapes[0][1]} exceeds max_length={config['max_length']}"
        )
    return {
        "stage": config["stage"],
        "sample_count": len(dataloader.dataset),
        "batch_shape": list(shapes[0]),
        "max_length": config["max_length"],
    }


def main(argv: list[str] | None = None) -> None:
    """Run one configuration-driven MedSearch SFT job.

    Student task: complete the formal model → Trainer → train → final adapter
    sequence below. Smoke sampling, one-step assertions and probe output belong
    in ``MedSearch/docs/tests``, not in this production entry point.
    """
    parser = argparse.ArgumentParser(description="Run MedSearch SFT")
    parser.add_argument("--config", required=True, help="Stage-A or Stage-B YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    # Trainer seeds too late for freshly initialized LoRA weights.
    set_seed(config["training"]["seed"])
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    dataloader = build_dataloader(config, tokenizer)

    # Student task: resolve dtype, build optional QLoRA configuration, then call
    # load_base_model → prepare_qlora_model (QLoRA only) → attach_lora →
    # build_trainer → run_training(resume_from_checkpoint=...) → save_adapter.
    dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
    dtype = dtypes[config["dtype"]]

    if config["use_qlora"]:
        quantization_config = build_quantization_config(dtype)
        model = load_base_model(config, quantization_config=quantization_config, dtype=dtype)
        model = prepare_qlora_model(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])
    else:
        model = load_base_model(config, quantization_config=None, dtype=dtype)

    model = attach_lora(model, config)

    trainer = build_trainer(config, model, tokenizer, dataloader)
    results = run_training(trainer, resume_from_checkpoint=config["resume_from_checkpoint"])

    output_dir = Path(config["training"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    save_adapter(trainer.model, tokenizer, output_dir / "adapter")
if __name__ == "__main__":
    main()
