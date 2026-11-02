"""用真实 veRL 批次检查教师阶段的调度与分数传递，不加载大模型。"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transfer_queue import KVBatchMeta
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from MedSearch.rl import distillation_hook as hook


def make_phase_batch(accuracies, events):
    rows = [
        {"uid": "group-0", "prompts": torch.tensor([10]),
         "responses": torch.tensor([20, 90, 30]), "response_mask": torch.tensor([1, 0, 1]),
         "old_log_probs": torch.full((3,), -0.5), "advantages": torch.ones(3),
         "extra_fields": {"reward_extra_info": {"acc": accuracies[0]}, "references": ["gold"]}},
        {"uid": "group-0", "prompts": torch.tensor([11, 12]),
         "responses": torch.tensor([21, 31]), "response_mask": torch.tensor([1, 1]),
         "old_log_probs": torch.full((2,), -0.5), "advantages": -torch.ones(2),
         "extra_fields": {"reward_extra_info": {"acc": accuracies[1]}, "references": ["gold"]}},
        {"uid": "padding", "prompts": torch.tensor([10]),
         "responses": torch.tensor([0]), "response_mask": torch.tensor([0]),
         "old_log_probs": torch.zeros(1), "advantages": torch.zeros(1), "extra_fields": {}},
    ]
    data = list_of_dict_to_tensordict(rows)
    meta = KVBatchMeta(partition_id="phase-test", keys=["trace-0", "trace-1", "padding"],
                       tags=[{}, {}, {"is_padding": True}])

    def move(device, **kwargs):
        assert kwargs == {"model": False, "optimizer": True, "grad": False}
        events.append(device)

    runtime = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(fsdp_config=SimpleNamespace(param_offload=True)))),
        actor_rollout_wg=SimpleNamespace(to=move),
    )
    return runtime, meta, data, rows


@pytest.mark.parametrize("accuracies", [(1, 1), (1, 0), (0, 0)],
                         ids=["all_correct", "mixed", "all_failed"])
def test_phase_orders_resource_switch_and_returns_teacher_scores(accuracies):
    events = []
    runtime, meta, data, rows = make_phase_batch(accuracies, events)
    teacher = torch.nn.Linear(1, 1)

    def score_fn(failures):
        assert events == ["cpu"]
        assert [item["original_index"] for item in failures] == [
            i for i, acc in enumerate(accuracies) if acc == 0]
        events.append("score")
        return hook.score_failed_batch(teacher, failures)

    keys, tags = list(meta.keys), [dict(tag) for tag in meta.tags]
    with patch.object(hook, "score_tokens", side_effect=lambda model, prefix, target, mode: -target.float() / 10):
        result = hook.run_failure_teacher_phase(runtime, meta, data, score_fn)

    assert result is data
    assert events == (["cpu", "score", "device"] if 0 in accuracies else [])
    assert meta.keys == keys and meta.tags == tags and len(data) == len(rows)
    for i, row in enumerate(rows):
        for field, expected in row.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(data[i][field], expected)
            else:
                assert data[i][field] == expected
        expected_mask = row["response_mask"].float() if i < 2 and accuracies[i] == 0 else torch.zeros_like(row["response_mask"], dtype=torch.float32)
        torch.testing.assert_close(data[i]["opd_mask"], expected_mask)
        torch.testing.assert_close(data[i]["teacher_logprobs"], -row["responses"].float() / 10 * expected_mask)


def test_phase_restores_student_and_propagates_scoring_error():
    events = []
    runtime, meta, data, rows = make_phase_batch((1, 0), events)

    def score_fn(failures):
        events.append("score")
        raise RuntimeError("teacher scoring failed")

    with patch.object(hook, "attach_teacher_scores", wraps=hook.attach_teacher_scores) as attach:
        with pytest.raises(RuntimeError, match="teacher scoring failed"):
            hook.run_failure_teacher_phase(runtime, meta, data, score_fn)
    assert events == ["cpu", "score", "device"]
    attach.assert_not_called()
    assert set(data.keys()) == set(rows[0])
