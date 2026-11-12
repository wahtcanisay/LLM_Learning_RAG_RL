import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from MedSearch.distillation.teacher import load_teacher


def test_load_teacher_from_local_checkpoint_is_frozen_bf16_and_usable(tmp_path):
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    Qwen2ForCausalLM(config).save_pretrained(tmp_path)

    teacher = load_teacher(str(tmp_path), device="cpu")

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert {parameter.dtype for parameter in teacher.parameters()} == {torch.bfloat16}
    with torch.inference_mode():
        logits = teacher(torch.tensor([[10, 20]], dtype=torch.long)).logits
    assert logits.shape == (1, 2, 64)
    assert torch.isfinite(logits).all()
