"""Teacher-only CPU preparation checks; not actual 4-bit GPU acceptance."""

from copy import deepcopy
from unittest.mock import patch

import peft
import pytest
import torch

from test_sft_trainer import train
from test_sft_model import tiny_qwen


@pytest.mark.parametrize("checkpointing", [False, True])
def test_prepare_qlora_model_reuses_peft_and_preserves_base(checkpointing):
    assert hasattr(train, "prepare_qlora_model"), "Student preparation interface is missing"
    model = tiny_qwen().to(dtype=torch.bfloat16)
    # Exercise PEFT's k-bit checkpointing branch on a tiny CPU model. The flag
    # is a test double: its weights are NOT quantized, and this proves no GPU
    # kernel/backward or Params4bit preservation. Those require later acceptance.
    model.is_loaded_in_4bit = True
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config_before = deepcopy(model.config.to_dict())

    # Spy delegates to the real PEFT implementation: verify both native API
    # reuse and the resulting model, not just a fabricated mock return value.
    with patch.object(
        train, "prepare_model_for_kbit_training", wraps=peft.prepare_model_for_kbit_training
    ) as prepare:
        result = train.prepare_qlora_model(model, use_gradient_checkpointing=checkpointing)

    assert prepare.call_count == 1
    args, kwargs = prepare.call_args
    assert (args[0] if args else kwargs["model"]) is model
    actual_gc = args[1] if len(args) > 1 else kwargs["use_gradient_checkpointing"]
    assert actual_gc is checkpointing
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert result is model
    assert model.is_gradient_checkpointing is checkpointing
    assert model.config.to_dict() == config_before
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert not any("lora_" in name for name, _ in model.named_parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name].float())
