"""Analytical checks of the paper objective, independent of the trainer."""

import math
from types import SimpleNamespace

import pytest
import torch

from va_opd.objective import (
    chunked_reverse_kl,
    grouped_token_weights,
    response_weights_from_means,
    reverse_kl_rows,
    sampled_log_probs,
    token_weights_for_token_mean,
    visual_advantage,
    weighted_prompt_mean,
)
from verl.trainer.distillation import DistillationSpec, build_distillation_config, chunked_distillation


def test_visual_advantage_is_positive_masked_and_stop_gradient():
    original = torch.tensor([[-1.0, -3.0, -2.0, float("nan")]], requires_grad=True)
    pixelated = torch.tensor([[-2.0, -2.0, -2.0, float("nan")]], requires_grad=True)
    result = visual_advantage(original, pixelated, torch.tensor([[1, 1, 1, 0]]))
    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert not result.requires_grad


def test_teacher_va_uses_untempered_distribution():
    original = torch.tensor([[2.0, 0.0]])
    pixelated = torch.tensor([[0.0, 0.0]])
    label = torch.tensor([0])
    expected = math.log(2 / (1 + math.exp(-2)))
    actual = visual_advantage(sampled_log_probs(original, label), sampled_log_probs(pixelated, label))
    assert actual.item() == pytest.approx(expected, abs=1e-7)
    tempered = sampled_log_probs(original / 0.7, label) - sampled_log_probs(pixelated / 0.7, label)
    assert not torch.allclose(actual, tempered)


def test_paper_figure_population_normalization():
    means = torch.tensor([0.045, 0.030, 0.020, 0.010], requires_grad=True)
    weights = response_weights_from_means(means, ["prompt"] * 4)
    # Values printed in the supplied method figure, rounding to three digits.
    torch.testing.assert_close(weights, torch.tensor([0.656, 0.206, 0.095, 0.044]), atol=8e-4, rtol=0)
    assert weights.sum().item() == pytest.approx(1)
    assert not weights.requires_grad


def test_equal_scores_and_singletons_have_uniform_response_weights():
    result = response_weights_from_means(torch.tensor([0.0, 0.0, 9.0]), ["a", "a", "b"])
    torch.testing.assert_close(result, torch.tensor([0.5, 0.5, 1.0]))


def test_group_permutation_and_numeric_tensor_ids():
    means = torch.tensor([0.5, 1.0, 0.1, 0.7, 0.2])
    ids = torch.tensor([7, 9, 7, 9, 7])
    permutation = torch.tensor([4, 1, 0, 3, 2])
    expected = response_weights_from_means(means, ids)
    actual = response_weights_from_means(means[permutation], ids[permutation])
    torch.testing.assert_close(actual, expected[permutation])


def test_grouped_kl_gives_half_mass_to_high_tokens_regardless_of_length():
    scores = torch.tensor([[9.0, 1, 1, 1, 1, 99], [9.0, 1, 1, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    weights, _ = grouped_token_weights(scores, mask, ["short", "long"])
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125, 0]))
    # ceil(.2 * 6) = 2, so the first two tied-rank positions receive .25.
    torch.testing.assert_close(weights[1], torch.tensor([0.25, 0.25, 0.125, 0.125, 0.125, 0.125]))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))


def test_one_token_receives_all_rollout_mass():
    weights, _ = grouped_token_weights(torch.tensor([[2.0], [3.0]]), torch.ones(2, 1), ["a", "a"])
    expected = response_weights_from_means(torch.tensor([2.0, 3.0]), ["a", "a"])
    torch.testing.assert_close(weights[:, 0], expected)
    assert weights.sum().item() == pytest.approx(1)


def test_mask_ignores_nan_padding_and_handles_noncontiguous_tokens():
    mask = torch.tensor([[0, 1, 0, 1, 1]])
    weights, _ = grouped_token_weights(torch.tensor([[float("nan"), 1, float("nan"), 3, 2]]), mask, ["a"])
    torch.testing.assert_close(weights, torch.tensor([[0.0, 0.25, 0.0, 0.5, 0.25]]))


def test_empty_rollout_rejected_without_losing_prompt_mass():
    with pytest.raises(ValueError, match="at least one"):
        grouped_token_weights(torch.zeros(2, 3), torch.tensor([[1, 0, 0], [0, 0, 0]]), ["a", "a"])


def test_seeded_ties_are_reproducible():
    scores = torch.zeros(1, 20)

    def compute(seed):
        return grouped_token_weights(
            scores, torch.ones_like(scores), [0], generator=torch.Generator().manual_seed(seed)
        )[0]

    first = compute(23)
    torch.testing.assert_close(first, compute(23))
    assert not torch.equal(first, compute(24))
    assert (first == 0.5 / 4).sum().item() == 4
    assert first.sum().item() == pytest.approx(1)


def test_global_response_weight_override_supports_distributed_shards():
    means = torch.tensor([1.0, 2.0, 3.0, 4.0])
    global_weights = response_weights_from_means(means, ["a"] * 4)
    local_scores = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    local, _ = grouped_token_weights(
        local_scores, torch.ones_like(local_scores), response_weights=global_weights[[0, 2]]
    )
    torch.testing.assert_close(local.sum(-1), global_weights[[0, 2]])
    assert local.sum().item() < 1


def test_prompt_average_matches_grouped_loss_and_token_mean_adapter():
    scores = torch.tensor([[5.0, 1, 1, 1, 1], [1.0, 5, 1, 1, 1], [2.0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    group_ids = ["a", "a", "b"]
    weights, _ = grouped_token_weights(scores, mask, group_ids)
    losses = torch.tensor([[10.0, 2, 2, 2, 2], [2.0, 20, 2, 2, 2], [3.0, 0, 0, 0, 0]], requires_grad=True)
    # Prompt a: .5*(.5*10+.5*2) + .5*(.5*20+.5*2)=8.5.
    # Prompt b: a singleton response token, KL=3. Average=5.75.
    direct = weighted_prompt_mean(losses, weights, group_ids)
    assert direct.item() == pytest.approx(5.75)
    adapted = token_weights_for_token_mean(weights, mask, group_ids)
    torch.testing.assert_close(direct, (losses * adapted).sum() / mask.sum())
    direct.backward()
    torch.testing.assert_close(losses.grad, weights / 2)


def test_full_reverse_kl_analytic_direction_and_teacher_frozen():
    student = torch.tensor([[math.log(0.75), math.log(0.25)]], requires_grad=True)
    teacher = torch.tensor([[math.log(0.5), math.log(0.5)]], requires_grad=True)
    loss = reverse_kl_rows(student, teacher)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    loss.sum().backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert student.grad[0, 0] > 0 and student.grad[0, 1] < 0


def test_shared_support_with_differently_padded_lm_heads():
    student = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0, -4.0, 20.0]], requires_grad=True)
    loss = reverse_kl_rows(student, teacher, vocab_size=2)
    torch.testing.assert_close(loss, torch.zeros(1))
    loss.sum().backward()
    assert student.grad[0, 2].item() == 0
    assert teacher.grad is None
    with pytest.raises(ValueError, match="explicit shared"):
        reverse_kl_rows(student, teacher)


def test_chunked_kl_backward_matches_independent_double_precision_reference():
    generator = torch.Generator().manual_seed(41)
    student = torch.randn(7, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(7, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    reference_student = student.detach().clone().requires_grad_()
    p = reference_student.softmax(-1)
    q = teacher.detach().softmax(-1)
    expected = (p * (p.log() - q.log())).sum(-1)
    expected.sum().backward()
    actual = chunked_reverse_kl(student, teacher, chunk_size=2)
    actual.sum().backward()
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(student.grad, reference_student.grad, atol=2e-7, rtol=2e-6)
    assert teacher.grad is None


def test_runtime_adapter_preserves_full_kl_and_metric_contract():
    student = torch.tensor([[math.log(0.75), math.log(0.25), -10.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0, -1.0, -2.0]], requires_grad=True)
    loss, sampled, stats = chunked_distillation(student, torch.tensor([0]), teacher, DistillationSpec(2), chunk_size=1)
    assert loss.item() == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5), abs=1e-7)
    torch.testing.assert_close(sampled, student.log_softmax(-1)[:, 0])
    assert stats.shape == (1, 3) and not stats.requires_grad
    assert stats[0, 1] > 0 and stats[0, 2] > 0
    loss.sum().backward()
    assert teacher.grad is None
    assert student.grad[0, 2] == 0


@pytest.mark.parametrize(
    "change",
    [{"divergence": "forward_kl"}, {"support": "student_top_k"}, {"target": "visual_gain"}, {"temperature": 0.7}],
)
def test_runtime_rejects_other_objectives(change):
    with pytest.raises(ValueError, match="require"):
        DistillationSpec.from_config({"vocab_size": 4, **change})


def test_distillation_config_requires_direct_original_teacher_target():
    algorithm = SimpleNamespace(
        distill_loss_coef=1.0,
        policy_loss_coef=0.0,
        distill_divergence="reverse_kl",
        distill_support="full",
        distill_temperature=1.0,
        distill_temperature_scope="all",
        distill_target="teacher",
        distill_chunk_size=128,
        distill_weighting="va_opd",
    )
    config = build_distillation_config(algorithm, 100)
    assert config["weighting"] == "va_opd" and config["teacher_view"] == "original" and config["is_clip"] is None
    algorithm.policy_loss_coef = 1.0
    with pytest.raises(ValueError, match="policy_loss_coef=0"):
        build_distillation_config(algorithm, 100)


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": float("nan")}, {"eps": 0}])
def test_bad_response_normalization_is_rejected(kwargs):
    with pytest.raises(ValueError):
        response_weights_from_means(torch.ones(1), [0], **kwargs)
