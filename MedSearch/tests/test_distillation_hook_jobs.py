from MedSearch.rl.distillation_hook import build_teacher_score_jobs
from unittest.mock import patch

import pytest
import torch

from MedSearch.rl.distillation_hook import score_failed_trajectory
from MedSearch.rl.distillation_hook import score_failed_batch
from MedSearch.rl.distillation_hook import attach_teacher_scores


def test_teacher_jobs_keep_observations_in_prefix_only():
    jobs = build_teacher_score_jobs(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 90, 91, 30],
        response_mask=[1, 1, 0, 0, 1],
    )
    assert jobs == [
        {"prefix_ids": [10, 11], "target_ids": [20, 21], "response_start": 0},
        {"prefix_ids": [10, 11, 20, 21, 90, 91], "target_ids": [30], "response_start": 4},
    ]


def test_teacher_jobs_ignore_observation_only_trace():
    assert build_teacher_score_jobs([10], [90, 91], [0, 0]) == []


def test_failed_trajectory_scores_actions_and_scatters_to_response():
    teacher = torch.nn.Linear(1, 1)
    calls = []

    def fake_score(model, prefix, target, mode):
        calls.append((prefix.tolist(), target.tolist(), mode))
        assert model is teacher
        assert prefix.dtype == target.dtype == torch.long
        return -target.float() / 10

    with patch("MedSearch.rl.distillation_hook.score_tokens", side_effect=fake_score):
        result = score_failed_trajectory(teacher, [10, 11], [20, 21, 90, 91, 30], [1, 1, 0, 0, 1], True)

    assert calls == [([[10, 11]], [[20, 21]], "sampled"),
                     ([[10, 11, 20, 21, 90, 91]], [[30]], "sampled")]
    torch.testing.assert_close(result["teacher_logprobs"], torch.tensor([-2., -2.1, 0., 0., -3.]))
    torch.testing.assert_close(result["opd_mask"], torch.tensor([1., 1., 0., 0., 1.]))
    for value in result.values():
        assert value.device.type == "cpu" and value.dtype == torch.float32
        assert not value.requires_grad


@pytest.mark.parametrize("response_ids,response_mask,failed", [
    ([20, 21], [1, 1], False),
    ([90, 91], [0, 0], True),
])
def test_unscored_trajectory_returns_same_dictionary_without_teacher(response_ids, response_mask, failed):
    with patch("MedSearch.rl.distillation_hook.score_tokens") as scorer:
        result = score_failed_trajectory(None, [10], response_ids, response_mask, failed)
    scorer.assert_not_called()
    assert set(result) == {"teacher_logprobs", "opd_mask"}
    for value in result.values():
        torch.testing.assert_close(value, torch.zeros(len(response_ids)))


def test_failed_action_requires_teacher():
    with pytest.raises(ValueError):
        score_failed_trajectory(None, [10], [20], [1], True)


@pytest.mark.parametrize("accuracies", [(0, 0, 1, 0), (1, 1, 1, 1)])
def test_failure_batch_with_real_verl_tensordict_preserves_complete_batch(accuracies):
    from transfer_queue import KVBatchMeta
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict
    from MedSearch.rl.distillation_hook import build_failure_batch

    rows = []
    for i, acc in enumerate(accuracies):
        rows.append({
            "uid": f"group-{i // 2}",
            "prompts": torch.tensor([10] * (i + 1)),
            "responses": torch.tensor([20, 90, 30] + [31] * i),
            "response_mask": torch.tensor([1, 0, 1] + [1] * i),
            "extra_fields": {"reward_extra_info": {"acc": acc}, "references": ["gold"]},
        })
    # Synthetic padding has no reward result and must be skipped before reading it.
    rows.append({"uid": "padding", "prompts": torch.tensor([10]),
                 "responses": torch.tensor([0]), "response_mask": torch.tensor([0]),
                 "extra_fields": {}})
    data = list_of_dict_to_tensordict(rows)
    meta = KVBatchMeta(partition_id="failure-batch-test", keys=[f"trace-{i}" for i in range(5)],
                       tags=[{} for _ in range(4)] + [{"is_padding": True}])
    keys_before = list(meta.keys)
    tags_before = [dict(tag) for tag in meta.tags]

    result = build_failure_batch(meta, data)

    assert result == [{
        "original_index": i, "group_id": rows[i]["uid"], "trajectory_id": f"trace-{i}",
        "prompt_ids": rows[i]["prompts"].tolist(),
        "response_ids": rows[i]["responses"].tolist(),
        "response_mask": rows[i]["response_mask"].tolist(),
    } for i, acc in enumerate(accuracies) if acc == 0]
    assert meta.keys == keys_before and meta.tags == tags_before
    assert len(data) == len(rows)
    for i, expected in enumerate(rows):
        assert data[i]["uid"] == expected["uid"]
        assert data[i]["extra_fields"] == expected["extra_fields"]
        for field in ("prompts", "responses", "response_mask"):
            torch.testing.assert_close(data[i][field], expected[field])


def test_failed_batch_empty_does_not_score():
    with patch("MedSearch.rl.distillation_hook.score_failed_trajectory") as scorer:
        assert score_failed_batch(None, []) == []
    scorer.assert_not_called()


def test_failed_batch_accepts_documented_keyword():
    assert score_failed_batch(teacher_model=None, failure_batch=[]) == []


def test_failed_batch_keeps_identity_and_variable_length_scores():
    from copy import deepcopy

    failures = [
        {"original_index": 3, "trajectory_id": "trace-3", "group_id": "g",
         "prompt_ids": [10], "response_ids": [20, 90, 30], "response_mask": [1, 0, 1]},
        {"original_index": 0, "trajectory_id": "trace-0", "group_id": "g",
         "prompt_ids": [11, 12], "response_ids": [21], "response_mask": [1]},
    ]
    original = deepcopy(failures)
    teacher = torch.nn.Linear(1, 1)
    with patch("MedSearch.rl.distillation_hook.score_tokens", side_effect=lambda model, prefix, target, mode: -target.float() / 10):
        results = score_failed_batch(teacher, failures)
    assert failures == original
    assert [(r["original_index"], r["trajectory_id"]) for r in results] == [(3, "trace-3"), (0, "trace-0")]
    assert all(set(r) == {"original_index", "trajectory_id", "teacher_logprobs", "opd_mask"} for r in results)
    for result, logprobs, mask in zip(results, [[-2., 0., -3.], [-2.1]], [[1., 0., 1.], [1.]]):
        torch.testing.assert_close(result["teacher_logprobs"], torch.tensor(logprobs))
        torch.testing.assert_close(result["opd_mask"], torch.tensor(mask))
        assert all(result[k].device.type == "cpu" and result[k].dtype == torch.float32
                   for k in ("teacher_logprobs", "opd_mask"))


def test_failed_batch_propagates_scoring_error():
    failures = [{"original_index": i, "trajectory_id": f"trace-{i}",
                 "prompt_ids": [10], "response_ids": [20], "response_mask": [1]} for i in range(3)]
    first_result = {"teacher_logprobs": torch.tensor([-2.]), "opd_mask": torch.ones(1)}
    with patch("MedSearch.rl.distillation_hook.score_failed_trajectory",
               side_effect=[first_result, RuntimeError("teacher scoring failed")]) as scorer:
        with pytest.raises(RuntimeError, match="teacher scoring failed"):
            score_failed_batch(None, failures)
    assert scorer.call_count == 2
    assert all(call.kwargs["failed"] is True for call in scorer.call_args_list)


def make_attach_batch(lengths):
    from transfer_queue import KVBatchMeta
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    rows = [{"responses": torch.arange(length),
             "response_mask": torch.ones(length) if i != 3 else torch.zeros(length),
             "old_log_probs": torch.full((length,), -0.5),
             "advantages": torch.full((length,), float(i)),
             "uid": f"group-{i // 2}"} for i, length in enumerate(lengths)]
    return (KVBatchMeta(partition_id="attach-test", keys=[f"trace-{i}" for i in range(4)],
                        tags=[{}, {}, {}, {"is_padding": True}]),
            list_of_dict_to_tensordict(rows), rows)


@pytest.mark.parametrize("lengths", [(2, 3, 1, 4), (3, 3, 3, 3)])
@pytest.mark.parametrize("has_scores", [True, False])
def test_attach_scores_real_tensordict_alignment_and_preservation(lengths, has_scores):
    meta, data, original = make_attach_batch(lengths)
    original_keys, original_tags = list(meta.keys), [dict(tag) for tag in meta.tags]
    scores = [{"original_index": i, "trajectory_id": f"trace-{i}",
               "teacher_logprobs": torch.full((lengths[i],), -float(i + 1)),
               "opd_mask": torch.ones(lengths[i])} for i in (2, 0)] if has_scores else []

    assert attach_teacher_scores(meta, data, scores) is data
    assert meta.keys == original_keys and meta.tags == original_tags
    assert set(data.keys()) == set(original[0]) | {"teacher_logprobs", "opd_mask"}
    for i, row in enumerate(original):
        for field, value in row.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(data[i][field], value)
            else:
                assert data[i][field] == value
        scored = has_scores and i in (0, 2)
        torch.testing.assert_close(data[i]["teacher_logprobs"],
                                   torch.full((lengths[i],), -float(i + 1) if scored else 0.))
        torch.testing.assert_close(data[i]["opd_mask"], torch.full((lengths[i],), float(scored)))
        assert data[i]["teacher_logprobs"].device.type == "cpu"
        assert data[i]["opd_mask"].dtype == torch.float32
    # Verify new fields follow the same row when the full batch is sliced.
    from verl.utils.tensordict_utils import index_select_tensor_dict
    subset = index_select_tensor_dict(data, [2, 0])
    for new_index, old_index in enumerate((2, 0)):
        for field in ("teacher_logprobs", "opd_mask", "responses"):
            torch.testing.assert_close(subset[new_index][field], data[old_index][field])


@pytest.mark.parametrize("invalid", ["negative_index", "large_index", "identity", "logprob_shape", "mask_shape"])
def test_attach_rejects_bad_scores_before_mutating_batch(invalid):
    meta, data, original = make_attach_batch((2, 3, 1, 4))
    valid = {"original_index": 0, "trajectory_id": "trace-0",
             "teacher_logprobs": torch.ones(2), "opd_mask": torch.ones(2)}
    bad = {"original_index": 1, "trajectory_id": "trace-1",
           "teacher_logprobs": torch.ones(3), "opd_mask": torch.ones(3)}
    if invalid in ("negative_index", "large_index"):
        bad["original_index"] = -1 if invalid == "negative_index" else 4
    elif invalid == "identity":
        bad["trajectory_id"] = "wrong-trajectory"
    else:
        field = "teacher_logprobs" if invalid == "logprob_shape" else "opd_mask"
        bad[field] = torch.ones(1, 3)
    with pytest.raises(ValueError):
        attach_teacher_scores(meta, data, [valid, bad])
    assert set(data.keys()) == set(original[0])
    for i, row in enumerate(original):
        for field, value in row.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(data[i][field], value)
            else:
                assert data[i][field] == value
