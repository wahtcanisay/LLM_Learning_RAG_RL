"""Cold-start teacher-trajectory training entry point."""

import argparse
import json
import os
from pathlib import Path
import yaml
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

from .data import SFTDataset, build_data_collator, load_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2] # 指向第三个父目录
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
    "eval_data_path",
    "dtype",
    "adapter_path",
) # pipeline要求的Yaml配置所必须字段


def load_config(path: str | Path) -> dict:
    """
    读取yaml作为config配置
    """
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("SFT config must be a YAML mapping")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"SFT config missing keys: {', '.join(missing)}")
    if config["stage"] != "search_distill":
        raise ValueError(f"unknown SFT stage: {config['stage']}")
    return config


def build_dataloader(
    config: dict,
    tokenizer,
    project_root: Path = PROJECT_ROOT,
    *,
    split: str = "train",
) -> DataLoader:
    """
    构造dataloader，虽然这一部分可能非必须，后续使用的仍然是Dataloader.dataset
    """
    if split == "train":
        path = Path(config["data_path"])
        shuffle = config["shuffle"]
    elif split == "eval":
        path = Path(config["eval_data_path"])
        shuffle = False
    else:
        raise ValueError(f"unknown split: {split}")
    if not path.is_absolute():
        path = project_root / path

    samples = load_samples(path)

    dataset_stage = (
        config["stage"]
        if split == "train"
        else config.get("eval_stage", config["stage"])
    )

    dataset = SFTDataset(samples, tokenizer, dataset_stage, config["max_length"])
    collator_fn = build_data_collator(tokenizer) # 返回DataCollatorForSeq2Seq()

    return DataLoader(dataset,
                      batch_size=config["batch_size"],
                      shuffle=shuffle,
                      num_workers=config["num_workers"],
                      collate_fn=collator_fn)


def build_quantization_config(
    compute_dtype: torch.dtype,
) -> BitsAndBytesConfig:
    """
    qlora所需要的量化加载配置，BitsAndBytesConfig，在加载basemodel时候使用
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
    """
    加载base_model的接口
    """
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = project_root / model_path

    # qlora基模加载配置
    if quantization_config is not None:
        if torch.cuda.is_available():
            # qlora刚需cuda，不存在则报错
            return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    device_map={"":0} # 指定设备
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
    """准备好4-bit的basemodel，即将加入Lora块之前

    加入prepare_model_for_kbit_training的训练配置
    """
    # Qlora省显存，还有一个部分是要裁撤中间的activation来进行的，算是以时间换空间的操作
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing, # 对中间的activation进行部分清除，backward需要的情况下重新算
                                            gradient_checkpointing_kwargs={"use_reentrant": False}, # 选择checkpoint实现方式
                                            )
    return model

def attach_lora(model, config: dict, project_root: Path = PROJECT_ROOT,):
    """加入Lora.

    查找路径是否存在adapter，没有则创建Lora_config，然后加载到base_model上
    """
    adapter_path = config["adapter_path"]
    if adapter_path is None:
        lora_config = LoraConfig(r=config["lora_rank"],
                                 task_type=TaskType.CAUSAL_LM,
                                 target_modules=config["lora_target_modules"],
                                 inference_mode=False,
                                 lora_alpha=config["lora_alpha"], # 缩放系数alpha，
                                                                  # aplha/rank决定了Lora矩阵并入原权重的放大系数
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
        # 从path中加载Lora权重，并将Lora权重设为可训练
        peft_model = PeftModel.from_pretrained(model,
                                               str(adapter_path),
                                               is_trainable=True)

    return peft_model







def build_training_arguments(
    config: dict,
    project_root: Path = PROJECT_ROOT,
) -> TrainingArguments:
    """
    构造训练配置，后续在trainer中加入此config
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
    )


def build_trainer(
    config: dict,
    model,
    tokenizer,
    dataloader: DataLoader,
    project_root: Path = PROJECT_ROOT,
    *,
    eval_dataloader: DataLoader | None = None,
) -> Trainer:
    """组装trainer作为pipeline

    传入dataloader提供train_dataset/eval_dataset
    """

    train_arg = build_training_arguments(config, project_root)
    if train_arg.gradient_checkpointing:
        model.config.use_cache = False
    trainer = Trainer(model=model,
                      args=train_arg,
                      processing_class=tokenizer, # 模型配套使用的数据处理器，文本模型通常传tokenizer
                      data_collator=dataloader.collate_fn,
                      eval_dataset=eval_dataloader.dataset if eval_dataloader is not None else None,
                      train_dataset=dataloader.dataset, ) # 其实这个dataloader可以不用吧，直接传dataset

    return trainer


def run_training(
    trainer: Trainer,
    *,
    resume_from_checkpoint: str | Path | None = None,
) -> dict:
    """执行.train(),返回metrics指标
    这里直接将metrics解包与globalstep一起打包送入一个dict
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
    """保存Lora和tokenizer

    PEFTModel运行保存逻辑默认保存adapter相关参数
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_embedding_layers=False) # 显式声明代表不保存完整的embedding权重矩阵
    tokenizer.save_pretrained(output_dir)


def main(argv: list[str] | None = None) -> None:
    """
    整体训练流程
    """

    parser = argparse.ArgumentParser(description="Run MedSearch SFT")
    parser.add_argument("--config", required=True, help="Search distillation YAML config")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    # Trainer seeds too late for freshly initialized LoRA weights.
    set_seed(config["training"]["seed"])
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    # 加载tokenizer以及训练/测试数据
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    train_dataloader = build_dataloader(config, tokenizer, split="train")
    eval_dataloader = build_dataloader(config, tokenizer, split="eval")

    # dtype用于后续加载量化base_model
    dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
    dtype = dtypes[config["dtype"]]

    # 量化加载/原生
    if config["use_qlora"]:
        quantization_config = build_quantization_config(dtype)
        model = load_base_model(config, quantization_config=quantization_config, dtype=dtype)
        model = prepare_qlora_model(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])
    else:
        model = load_base_model(config, quantization_config=None, dtype=dtype)

    # 加入Lora
    model = attach_lora(model, config)

    # 配置trainer并运行返回results
    trainer = build_trainer(config, model, tokenizer, dataloader=train_dataloader, eval_dataloader=eval_dataloader)
    results = run_training(trainer, resume_from_checkpoint=config["resume_from_checkpoint"])

    # 保存results
    output_dir = Path(config["training"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    # 保存lora权重
    save_adapter(trainer.model, tokenizer, output_dir / "adapter")

if __name__ == "__main__":
    main()
