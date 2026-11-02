"""Exercise the combined loss with real veRL response alignment."""

import torch
from verl.utils.tensordict_utils import list_of_dict_to_tensordict, assign_non_tensor_data
from verl.workers.utils.losses import ppo_loss

from MedSearch.distillation.losses import grpo_with_failure_opd_loss
from MedSearch.tests.test_failure_opd_loss import actor_config


def make_batch(with_teacher=True, all_success=False):
    full_log_probs = (-2. + torch.arange(12).float() / 100).requires_grad_()
    response_probs = [full_log_probs[1:4].detach().clone(), full_log_probs[7:11].detach().clone()]
    rows = []
    for i, (prompt_length, response_length) in enumerate(((2, 3), (3, 4))):
        mask = torch.tensor([1., 0., 1.] if i == 0 else [1., 1., 0., 1.])
        row = {"prompts": torch.arange(prompt_length), "responses": torch.arange(response_length),
               "response_mask": mask, "old_log_probs": response_probs[i],
               "advantages": torch.full((response_length,), 1. if i == 0 else -1.)}
        if with_teacher:
            row["teacher_logprobs"] = response_probs[i] + 1. if i == 0 else torch.zeros(response_length)
            row["opd_mask"] = mask.clone() if i == 0 and not all_success else torch.zeros(response_length)
        rows.append(row)
    data = list_of_dict_to_tensordict(rows)
    for name, value in {"dp_size": 1, "batch_num_tokens": 5, "global_batch_size": 2}.items():
        assign_non_tensor_data(data, name, value)
    return full_log_probs, data, rows


def assert_original_fields_unchanged(data, rows):
    for i, row in enumerate(rows):
        for field, original in row.items():
            torch.testing.assert_close(data[i][field], original)


def test_zero_coefficient_matches_native_without_teacher_fields():
    student, data, rows = make_batch(with_teacher=False)
    expected, expected_metrics = ppo_loss(actor_config(), {"log_probs": student}, data)
    result, metrics = grpo_with_failure_opd_loss(actor_config(), {"log_probs": student}, data, opd_coef=0.)
    torch.testing.assert_close(result, expected)
    expected_grad = torch.autograd.grad(expected, student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(result, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert set(metrics) == set(expected_metrics)
    assert "teacher_logprobs" not in data and "opd_mask" not in data
    assert_original_fields_unchanged(data, rows)


def test_nonzero_coefficient_has_hand_computed_response_aligned_gradient():
    student, data, rows = make_batch()
    baseline, baseline_metrics = ppo_loss(actor_config(), {"log_probs": student}, data)
    coef = 0.1
    result, metrics = grpo_with_failure_opd_loss(actor_config(), {"log_probs": student}, data, opd_coef=coef)
    # Teacher is +1 on failed tokens: failed sequence contributes -1 / 2 sequences.
    torch.testing.assert_close(result, baseline - coef / 2)
    baseline_grad = torch.autograd.grad(baseline, student, retain_graph=True)[0]
    result_grad = torch.autograd.grad(result, student)[0]
    expected_delta = torch.zeros_like(student)
    # Each of two failed action tokens receives 1/(2 tokens * 2 sequences).
    expected_delta[[1, 3]] = -coef / 4
    torch.testing.assert_close(result_grad - baseline_grad, expected_delta)
    assert set(baseline_metrics).issubset(metrics)
    assert "opd/pg_clipfrac" in metrics and "opd/ppo_kl" in metrics
    for key in baseline_metrics:
        torch.testing.assert_close(torch.as_tensor(metrics[key].values), torch.as_tensor(baseline_metrics[key].values))
    assert_original_fields_unchanged(data, rows)


def test_all_success_nonzero_coefficient_matches_native():
    student, data, rows = make_batch(all_success=True)
    baseline, _ = ppo_loss(actor_config(), {"log_probs": student}, data)
    result, metrics = grpo_with_failure_opd_loss(actor_config(), {"log_probs": student}, data, opd_coef=0.1)
    torch.testing.assert_close(result, baseline)
    baseline_grad = torch.autograd.grad(baseline, student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(result, student)[0]
    torch.testing.assert_close(actual_grad, baseline_grad)
    assert not any(key.startswith("opd/") for key in metrics)
    assert_original_fields_unchanged(data, rows)

