import copy

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from MedSearch.sft.distillation_trainer import DistillationTrainer
from MedSearch.sft.losses import forward_topk_loss
from MedSearch.sft.train import build_trainer


@pytest.mark.parametrize("with_cache", [False, True])
def test_compute_loss_and_native_trainer_update(tmp_path, with_cache):
    torch.manual_seed(13)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=16, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    ))
    batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 20, 21]]),
    }
    if with_cache:
        teacher = torch.cat((torch.full((32,), 2.), torch.zeros(32))).log_softmax(-1)[:32]
        batch.update(target_positions=[torch.tensor([2, 3])],
                     teacher_topk_ids=[torch.arange(32).repeat(2, 1)],
                     teacher_topk_logprobs=[teacher.repeat(2, 1).requires_grad_()])
    original = copy.deepcopy(batch)
    config = {
        "batch_size": 1, "num_workers": 0, "dtype": "float32",
        "training": {
            "output_dir": str(tmp_path), "use_cpu": True, "max_steps": 1,
            "gradient_accumulation_steps": 2, "learning_rate": 1e-3,
            "disable_tqdm": True, "seed": 13, "logging_steps": 1,
        },
    }
    dataloader = DataLoader([batch, batch], batch_size=1, collate_fn=lambda rows: rows[0])
    trainer = build_trainer(config, model, tokenizer=None, dataloader=dataloader)
    assert isinstance(trainer, DistillationTrainer)
    assert trainer.train_dataset is dataloader.dataset
    assert trainer.data_collator is dataloader.collate_fn
    assert trainer.model_accepts_loss_kwargs is False
    calls = []
    handle = model.register_forward_pre_hook(lambda *unused: calls.append(1))
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True, num_items_in_batch=999)
    assert len(calls) == 1
    expected = outputs.loss
    if with_cache:
        expected = expected + 0.1 * forward_topk_loss(
            outputs.logits, batch["target_positions"], batch["teacher_topk_ids"],
            batch["teacher_topk_logprobs"])
    assert torch.allclose(loss, expected)
    scalar = trainer.compute_loss(model, batch)
    assert scalar.ndim == 0 and torch.allclose(scalar, loss)
    assert len(calls) == 2
    for key in original:
        if isinstance(original[key], list):
            assert all(torch.equal(a, b) for a, b in zip(original[key], batch[key]))
        else:
            assert torch.equal(original[key], batch[key])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.count_nonzero() > 0 for grad in grads)
    if with_cache:
        assert batch["teacher_topk_logprobs"][0].grad is None
    model.zero_grad(set_to_none=True)
    before = model.lm_head.weight.detach().clone()
    calls.clear()
    result = trainer.train()
    assert result.global_step == 1
    assert len(calls) == 2  # Two microbatches, one model forward per microbatch.
    assert torch.isfinite(torch.tensor(result.training_loss))
    loss_logs = [item for item in trainer.state.log_history if "loss" in item]
    assert loss_logs and "ce_loss" in loss_logs[-1] and "top32_loss" in loss_logs[-1]
    logged = loss_logs[-1]
    assert logged["loss"] == pytest.approx(logged["ce_loss"] + .1 * logged["top32_loss"], abs=1e-4)
    assert not torch.equal(before, model.lm_head.weight.detach())
    handle.remove()
