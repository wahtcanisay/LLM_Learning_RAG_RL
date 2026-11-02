from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from MedSearch.tools.cache_teacher_scores import SCHEMA, score_record, validate_cache


def test_cache_keeps_action_positions_and_roundtrips_parquet(tmp_path):
    record = {"trajectory_id": "trace-1", "question_id": "q1", "references": ["gold"],
              "prompt_ids": [10, 11], "response_ids": [20, 21, 90, 91, 30],
              "response_mask": [1, 1, 0, 0, 1]}
    calls = []
    def scorer(model, prefix, target, mode, k):
        calls.append((prefix.tolist(), target.tolist(), mode, k))
        return {"topk_ids": torch.arange(32).expand(1, target.shape[1], 32),
                "topk_logprobs": torch.full((1, target.shape[1], 32), -5.)}
    with patch("MedSearch.tools.cache_teacher_scores.score_tokens", side_effect=scorer):
        row = score_record(torch.nn.Linear(1, 1), record)
    assert calls == [([[10, 11]], [[20, 21]], "topk", 32),
                     ([[10, 11, 20, 21, 90, 91]], [[30]], "topk", 32)]
    assert row["target_positions"] == [2, 3, 6]
    assert "references" not in row
    path = tmp_path / "cache.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), path)
    assert validate_cache(path) == {"rows": 1, "scored_tokens": 3}
    assert pq.read_table(path).to_pylist() == [row]
