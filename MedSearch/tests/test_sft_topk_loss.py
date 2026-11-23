import pytest
import torch

from MedSearch.sft.losses import forward_topk_loss


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_topk_loss_alignment_shared_vocab_token_mean_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    vocab = 151665
    logits = torch.zeros(2, 6, 151936, device=device)
    # Unmapped output rows must not affect normalization.
    logits[..., vocab:] = 100
    logits[0, 1, :32] = -1
    logits[0, 3, :32] = 1
    logits[1, 0, :32] = -2
    logits.requires_grad_()
    positions = [torch.tensor([2, 4]), torch.tensor([1])]
    ids = [torch.arange(32).repeat(2, 1), torch.arange(32).reshape(1, 32)]
    teacher_full = torch.zeros(vocab, dtype=torch.float64)
    teacher_full[:32] = 8
    teacher_logp = teacher_full.log_softmax(-1)[:32]
    teacher = [teacher_logp.float().repeat(2, 1).requires_grad_(),
               teacher_logp.float().reshape(1, 32).requires_grad_()]

    loss = forward_topk_loss(logits, positions, ids, teacher)
    reference_tokens = []
    for i, pos in ((0, 1), (0, 3), (1, 0)):
        logq = logits[i, pos, :vocab].detach().cpu().double().log_softmax(-1)[:32]
        reference_tokens.append((teacher_logp.exp() * (teacher_logp - logq)).sum().clamp_min(0))
    expected = torch.stack(reference_tokens).mean()
    assert loss.ndim == 0 and loss.dtype == torch.float32
    assert torch.allclose(loss.detach().cpu().double(), expected, atol=2e-6)
    sequence_mean = (torch.stack(reference_tokens[:2]).mean() + reference_tokens[2]) / 2
    assert not torch.allclose(expected, sequence_mean)
    assert teacher_logp.exp().sum() < 1  # Retain original mass, do not renormalize Top-32.

    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert all(item.grad is None for item in teacher)
    active = torch.zeros(2, 6, dtype=torch.bool, device=device)
    active[0, [1, 3]] = True
    active[1, 0] = True
    assert logits.grad[~active].count_nonzero() == 0
    assert logits.grad[..., vocab:].count_nonzero() == 0
    assert (logits.grad[active, :32] < 0).all()


def test_topk_loss_clamps_each_position_before_averaging():
    vocab = 151665
    logits = torch.zeros(1, 3, vocab, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, :32] = 8
        logits[0, 1, :32] = -8
    positions = [torch.tensor([1, 2])]
    ids = [torch.arange(32).repeat(2, 1)]
    teacher = [torch.full((2, 32), -torch.log(torch.tensor(float(vocab))).item())]
    logq = logits[0, :2].detach().log_softmax(-1)[:, :32]
    raw = (teacher[0].exp() * (teacher[0] - logq)).sum(-1)
    assert raw[0] < 0 and raw[1] > 0
    loss = forward_topk_loss(logits, positions, ids, teacher)
    assert torch.allclose(loss, raw.clamp_min(0).mean())
    loss.backward()
    assert logits.grad[0, 0].count_nonzero() == 0
    assert logits.grad[0, 1].count_nonzero() > 0
