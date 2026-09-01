"""Teacher-owned adapter round-trip checks; no 3B model or GPU is loaded."""

from copy import deepcopy

import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer, set_seed

from test_sft_trainer import train
from test_sft_model import LORA_CONFIG, PROJECT_ROOT, assert_only_lora_is_trainable, tiny_qwen
import run_sft_3b_smoke as probe


def test_gpu_probe_chat_template_returns_token_ids():
    assert hasattr(probe, "probe_token_ids"), "missing explicit probe token-ID conversion"
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models/Qwen2.5-3B-Instruct", local_files_only=True,
    )
    messages = [{"role": "user", "content": "What does a thermometer measure?"}]
    ids = probe.probe_token_ids(tokenizer, messages)
    assert isinstance(ids, list) and ids and all(isinstance(value, int) for value in ids)
    assert torch.tensor([ids]).ndim == 2
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert ids == tokenizer.encode(rendered, add_special_tokens=False)


def test_probe_logits_match_with_and_without_amp_forward_wrapper():
    assert hasattr(probe, "probe_last_logits"), "missing consistent probe inference precision"
    set_seed(20260831)
    model = train.attach_lora(tiny_qwen(), LORA_CONFIG).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0, std=0.05)
    wrapped = deepcopy(model)
    # Mirrors Accelerate's persistent autocast decoration, on CPU for regression.
    wrapped.forward = torch.autocast("cpu", dtype=torch.bfloat16)(wrapped.forward)
    inputs = {"input_ids": torch.tensor([[4, 12, 13, 3]]),
              "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    expected = probe.probe_last_logits(wrapped, inputs)
    actual = probe.probe_last_logits(model, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_save_adapter_round_trip(tmp_path):
    assert hasattr(train, "save_adapter"), "missing student save_adapter interface"
    set_seed(20260831)
    base = tiny_qwen()
    identical_base = deepcopy(base)  # Reload must use identical frozen weights.
    model = train.attach_lora(base, LORA_CONFIG)
    # Synthetic nonzero adapter: serialization test, NOT a training result.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0, std=0.05)
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models/Qwen2.5-3B-Instruct", local_files_only=True,
    )
    tokenizer.padding_side = "right"
    state_before = {
        k: v.clone()
        for k, v in get_peft_model_state_dict(model, save_embedding_layers=False).items()
    }
    model.eval()  # Disable LoRA dropout for deterministic comparison.
    # Token IDs stay within the tiny model's vocabulary; real chat tokens do not.
    inputs = {"input_ids": torch.tensor([[4, 12, 13, 3]]),
              "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with torch.no_grad():
        expected = model(**inputs, use_cache=False).logits
        with model.disable_adapter():
            base_logits = model(**inputs, use_cache=False).logits
    assert not torch.allclose(expected, base_logits, atol=1e-6, rtol=1e-6)

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()  # Teacher/caller owns creation of the fresh empty folder.
    assert train.save_adapter(model, tokenizer, output_dir) is None
    assert (output_dir / "adapter_config.json").is_file()
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "tokenizer_config.json").is_file()
    assert not (output_dir / "config.json").exists()
    assert not list(output_dir.glob("model*.safetensors"))
    assert not list(output_dir.glob("*.bin"))
    assert not (output_dir / "optimizer.pt").exists()
    saved = load_file(output_dir / "adapter_model.safetensors")
    assert saved.keys() == state_before.keys()
    assert all("lora_" in key for key in saved)
    state_after = get_peft_model_state_dict(model, save_embedding_layers=False)
    for key, expected_tensor in state_before.items():
        assert torch.equal(saved[key], expected_tensor)
        assert torch.equal(state_after[key], expected_tensor)

    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reloaded_tokenizer.special_tokens_map == tokenizer.special_tokens_map
    assert reloaded_tokenizer.chat_template == tokenizer.chat_template
    assert reloaded_tokenizer.padding_side == tokenizer.padding_side
    messages = [{"role": "user", "content": "What is the question?"}]
    assert reloaded_tokenizer.apply_chat_template(messages) == tokenizer.apply_chat_template(messages)
    reloaded = train.attach_lora(identical_base, {**LORA_CONFIG, "adapter_path": str(output_dir)})
    assert_only_lora_is_trainable(reloaded)  # Reuses the Stage-B loading branch.
    reloaded.eval()
    for key, tensor in get_peft_model_state_dict(reloaded, save_embedding_layers=False).items():
        assert torch.equal(tensor, state_before[key])
    with torch.no_grad():
        actual = reloaded(**inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
