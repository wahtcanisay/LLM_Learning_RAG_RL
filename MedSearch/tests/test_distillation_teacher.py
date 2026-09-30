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
