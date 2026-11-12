import math
from types import SimpleNamespace

import torch

from MedSearch.distillation.teacher import score_tokens


def test_sampled_scores_use_previous_context_and_mapped_vocab_only():
    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            assert input_ids.tolist() == [[10, 20], [30, 40]]
            assert attention_mask.tolist() == [[1, 1], [1, 1]]
            logits = torch.full((2, 2, 151666), -1000.0)
            logits[0, 0, 20] = 0.0
            logits[0, 0, 21] = 0.0
            logits[0, 1, 20] = 0.0
            logits[0, 1, 21] = 2.0
            logits[1, 0, 40] = 3.0
            logits[1, 0, 41] = 0.0
            logits[1, 1, 40] = 0.0
            logits[1, 1, 41] = 0.0
            logits[:, :, 151665] = 100.0  # 输出层额外行，不能参与归一化
            return SimpleNamespace(logits=logits)

    prefix_ids = torch.tensor([[10], [30]], dtype=torch.long)
    target_ids = torch.tensor([[20, 21], [40, 41]], dtype=torch.long)
    scores = score_tokens(FixedModel(), prefix_ids, target_ids, mode="sampled")

    expected = torch.tensor([
        [-math.log(2), 2 - math.log(math.exp(2) + 1)],
        [3 - math.log(math.exp(3) + 1), -math.log(2)],
    ])
    torch.testing.assert_close(scores, expected)


def test_top32_keeps_full_vocab_probabilities_and_causal_alignment():
    class FixedTopkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, input_ids, attention_mask):
            assert input_ids.tolist() == [[10, 11, 20], [30, 31, 40]]
            assert attention_mask.eq(1).all()
            logits = torch.full((2, 3, 151666), -1000.)
            candidates = torch.arange(64).float() / 100
            for batch in range(2):
                offset = batch * 100
                logits[batch, 1, offset:offset + 64] = candidates
                logits[batch, 2, offset:offset + 64] = candidates.flip(0)
            logits[:, :, 151665] = 1000.  # Unmapped output row must be excluded.
            return SimpleNamespace(logits=logits * self.scale)

    model = FixedTopkModel()
    result = score_tokens(model, torch.tensor([[10, 11], [30, 31]]),
                          torch.tensor([[20, 21], [40, 41]]), mode="topk")
    assert set(result) == {"topk_ids", "topk_logprobs"}
    ids, scores = result["topk_ids"], result["topk_logprobs"]
    assert ids.shape == scores.shape == (2, 2, 32)
    assert ids.dtype == torch.long and scores.dtype == torch.float32
    expected_ids = torch.stack([torch.stack([torch.arange(63, 31, -1), torch.arange(32)]) + offset
                                for offset in (0, 100)])
    torch.testing.assert_close(ids, expected_ids)
    full_logits = torch.arange(64).float() / 100
    expected_scores = (full_logits.flip(0)[:32] - torch.logsumexp(full_logits, dim=0))
    torch.testing.assert_close(scores, expected_scores.expand(2, 2, 32))
    assert (scores.exp().sum(-1) < 1).all()  # No renormalization within Top-32.
    assert ids.max().item() < 151665
    assert not scores.requires_grad and model.scale.grad is None
