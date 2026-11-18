"""Check OPD masking, gradient direction and full-batch normalization."""

import pytest
import torch
from verl.workers.config import ActorConfig

from MedSearch.distillation.losses import failure_opd_loss


def actor_config(global_batch_size=None):
    config = ActorConfig(strategy="fsdp", rollout_n=2, ppo_micro_batch_size_per_gpu=1,
                         clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
    if global_batch_size is not None:
        config.global_batch_info["global_batch_size"] = global_batch_size
    return config


@pytest.mark.parametrize("has_actions", [True, False])
def test_no_opd_returns_differentiable_zero(has_actions):
    student = torch.full((2, 3), -2., requires_grad=True)
    old = student.detach().clone().requires_grad_()
    teacher = torch.full((2, 3), -1., requires_grad=True)
    response = torch.ones_like(student) if has_actions else torch.zeros_like(student)
    loss, metrics = failure_opd_loss(student, old, teacher, response, torch.zeros_like(student), actor_config())
    assert loss.item() == 0 and torch.isfinite(loss) and metrics == {}
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    assert old.grad is None and teacher.grad is None


def test_mixed_batch_keeps_success_in_denominator_and_inputs_unchanged():
    student = torch.tensor([[-2., -2., -2.], [-2., -2., -2.], [-2., -2., -2.]], requires_grad=True)
    old = student.detach().clone().requires_grad_()
    teacher = torch.tensor([[-1., -3., -1.], [0., 0., 0.], [0., 0., 0.]], requires_grad=True)
    response = torch.tensor([[1., 0., 1.], [1., 1., 1.], [0., 0., 0.]])
    opd = torch.tensor([[1., 0., 1.], [0., 0., 0.], [0., 0., 0.]])
    inputs = [student, old, teacher, response, opd]
    originals = [x.detach().clone() for x in inputs]
    config = actor_config()
    loss, _ = failure_opd_loss(*inputs, config)
    # Failed sequence loss is -1; success contributes zero; padding is excluded.
    torch.testing.assert_close(loss, torch.tensor(-0.5))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[-0.25, 0., -0.25], [0., 0., 0.], [0., 0., 0.]]))
    assert old.grad is None and teacher.grad is None
    for actual, original in zip(inputs, originals):
        torch.testing.assert_close(actual.detach(), original)
    assert config.global_batch_info == {}


def test_all_failed_has_correct_gradient_direction():
    student = torch.tensor([[-2., -2.]], requires_grad=True)
    old = student.detach().clone().requires_grad_()
    teacher = torch.tensor([[-1., -3.]], requires_grad=True)
    mask = torch.ones_like(student)
    loss, _ = failure_opd_loss(student, old, teacher, mask, mask, actor_config())
    loss.backward()
    # Negative gradient raises student logprob under gradient descent; positive lowers it.
    torch.testing.assert_close(student.grad, torch.tensor([[-0.5, 0.5]]))
    assert old.grad is None and teacher.grad is None


def test_global_denominator_is_preserved_across_microbatches():
    student = torch.tensor([[-2., -2.], [-2., -2.]], requires_grad=True)
    teacher = torch.tensor([[-1., -1.], [0., 0.]])
    old = student.detach().clone()
    response = torch.ones_like(student)
    opd = torch.tensor([[1., 1.], [0., 0.]])
    config = actor_config(global_batch_size=2)
    whole, _ = failure_opd_loss(student, old, teacher, response, opd, config)
    parts = [failure_opd_loss(student[i:i+1], old[i:i+1], teacher[i:i+1],
                              response[i:i+1], opd[i:i+1], config)[0] for i in range(2)]
    torch.testing.assert_close(sum(parts), whole)
    full_grad = torch.autograd.grad(whole, student, retain_graph=True)[0]
    split_grad = torch.autograd.grad(sum(parts), student)[0]
    torch.testing.assert_close(split_grad, full_grad)


def test_native_clipping_stops_excessive_positive_update():
    student = torch.tensor([[-2.]], requires_grad=True)
    old = torch.tensor([[-3.]], requires_grad=True)
    teacher = torch.tensor([[-1.]], requires_grad=True)
    mask = torch.ones_like(student)
    loss, _ = failure_opd_loss(student, old, teacher, mask, mask, actor_config())
    torch.testing.assert_close(loss, torch.tensor(-1.2))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    assert old.grad is None and teacher.grad is None
