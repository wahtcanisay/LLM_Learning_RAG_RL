"""Teacher-owned configuration checks; no model loading or CUDA execution."""

import pytest
import torch
from transformers import BitsAndBytesConfig

from test_sft_trainer import train


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_qlora_config_uses_nf4_double_quant_and_requested_compute_dtype(compute_dtype):
    assert hasattr(train, "build_quantization_config"), "missing student quantization interface"
    config = train.build_quantization_config(compute_dtype)
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.load_in_8bit is False
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == compute_dtype
