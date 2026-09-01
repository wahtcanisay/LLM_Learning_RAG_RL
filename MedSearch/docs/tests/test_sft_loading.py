"""Teacher checks for the loading boundary, not GPU quantization acceptance."""

from copy import deepcopy
import inspect

import pytest
import torch

from test_sft_trainer import train
from test_sft_model import tiny_qwen


def test_nonquantized_loading_preserves_local_model(tmp_path):
    original = tiny_qwen()
    path = tmp_path / "tiny"
    original.save_pretrained(path)
    config = {"model_path": "tiny", "dtype": "float32"}
    before = deepcopy(config)
    loaded = train.load_base_model(config, project_root=tmp_path, dtype=torch.float32)
    assert config == before
    assert not getattr(loaded, "is_loaded_in_4bit", False)
    assert next(loaded.parameters()).device.type == "cpu"
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, original.state_dict()[name])


def test_quantized_loading_passes_config_and_single_gpu_placement(tmp_path, monkeypatch):
    assert "quantization_config" in inspect.signature(train.load_base_model).parameters
    config = {"model_path": "local-base", "dtype": "bfloat16"}
    before = deepcopy(config)
    quantization = train.build_quantization_config(torch.bfloat16)
    calls = []
    sentinel = object()

    def record_loading(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    # Intercept the expensive external loading boundary only. This checks our
    # wiring, not whether the backend actually quantizes weights on CUDA.
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", record_loading)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = train.load_base_model(
        config,
        project_root=tmp_path,
        quantization_config=quantization,
        dtype=torch.bfloat16,
    )
    assert result is sentinel
    assert config == before
    assert len(calls) == 1
    path, kwargs = calls[0]
    assert str(path) == str(tmp_path / "local-base")
    assert kwargs["quantization_config"] is quantization
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["device_map"] == {"": 0}
    assert kwargs["local_files_only"] is True
    assert kwargs["low_cpu_mem_usage"] is True


def test_quantized_loading_rejects_no_cuda_before_loading(tmp_path, monkeypatch):
    assert "quantization_config" in inspect.signature(train.load_base_model).parameters
    quantization = train.build_quantization_config(torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_load(*args, **kwargs):
        pytest.fail("No-CUDA check must occur before model loading")

    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", unexpected_load)
    with pytest.raises(RuntimeError, match="CUDA"):
        train.load_base_model(
            {"model_path": "missing", "dtype": "bfloat16"},
            project_root=tmp_path,
            quantization_config=quantization,
            dtype=torch.bfloat16,
        )
