import sys
from pathlib import Path

import pytest
from peft import PeftModel
from transformers import AutoModelForCausalLM, Qwen2Config


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MEDSEARCH_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MedSearch.sft.train import attach_lora


LORA_CONFIG = {
    "adapter_path": None,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}


def tiny_qwen():
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    config._name_or_path = str(PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct")
    return AutoModelForCausalLM.from_config(config)


def assert_only_lora_is_trainable(model):
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_attach_lora_creates_a_new_trainable_adapter():
    model = attach_lora(tiny_qwen(), LORA_CONFIG, project_root=PROJECT_ROOT)

    assert isinstance(model, PeftModel)
    assert_only_lora_is_trainable(model)


def test_attach_lora_resumes_an_existing_adapter_as_trainable(tmp_path):
    adapter_path = tmp_path / "stage_a_adapter"
    attach_lora(tiny_qwen(), LORA_CONFIG, project_root=PROJECT_ROOT).save_pretrained(
        adapter_path, save_embedding_layers=False
    )
    resume_config = {**LORA_CONFIG, "adapter_path": str(adapter_path)}

    model = attach_lora(tiny_qwen(), resume_config, project_root=PROJECT_ROOT)

    assert isinstance(model, PeftModel)
    assert_only_lora_is_trainable(model)


def test_attach_lora_rejects_a_missing_adapter_path(tmp_path):
    config = {**LORA_CONFIG, "adapter_path": str(tmp_path / "missing")}

    with pytest.raises(FileNotFoundError, match="LoRA adapter path does not exist"):
        attach_lora(tiny_qwen(), config, project_root=PROJECT_ROOT)
