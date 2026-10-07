# Copyright 2026 the VA-OPD authors. Licensed under Apache-2.0.
"""The VA-OPD objective, independent of the distributed training runtime.

All weights are stop-gradient quantities. The counterfactual image supplies only
visual advantage; reverse KL always uses the teacher on the original image.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _support_size(student: torch.Tensor, teacher: torch.Tensor, vocab_size: int | None) -> int:
    if student.ndim < 2 or teacher.ndim < 2 or student.shape[:-1] != teacher.shape[:-1]:
        raise ValueError("Student and teacher must cover the same token rows.")
    if vocab_size is None:
        if student.shape[-1] != teacher.shape[-1]:
            raise ValueError("Unequal LM-head widths require an explicit shared vocab_size.")
        vocab_size = student.shape[-1]
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or not 0 < vocab_size <= min(student.shape[-1], teacher.shape[-1])
    ):
        raise ValueError("vocab_size must fit both LM heads and be positive.")
    return vocab_size


def sampled_log_probs(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int | None = None) -> torch.Tensor:
    """Score sampled tokens at T=1, optionally excluding padded LM-head rows.

    The rollout sampling temperature does not change the teacher distribution
    used for VA. Call this on both teacher views with the same vocabulary.
    """
    if logits.ndim < 2 or labels.shape != logits.shape[:-1]:
        raise ValueError("labels must identify one token for every logit row.")
    support = logits.shape[-1] if vocab_size is None else vocab_size
    if isinstance(support, bool) or not isinstance(support, int) or not 0 < support <= logits.shape[-1]:
        raise ValueError("Invalid vocabulary support.")
    if labels.dtype not in (torch.int32, torch.int64) or bool(((labels < 0) | (labels >= support)).any()):
        raise ValueError("Sampled tokens must be integer IDs inside the vocabulary support.")
    values = logits[..., :support].float()
    return values.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1) - torch.logsumexp(values, dim=-1)


def visual_advantage(
    original_log_probs: torch.Tensor, pixelated_log_probs: torch.Tensor, response_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Equation (2): positive teacher log-probability difference, detached."""
    if original_log_probs.shape != pixelated_log_probs.shape:
        raise ValueError("Teacher views must score exactly the same sampled tokens.")
    scores = (original_log_probs.detach().float() - pixelated_log_probs.detach().float()).clamp_min(0)
    if response_mask is not None:
        if response_mask.shape != scores.shape:
            raise ValueError("response_mask and teacher scores must have identical shapes.")
        scores = scores.masked_fill(~response_mask.bool(), 0)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Visual advantage contains non-finite valid-token scores.")
    return scores


def _groups(group_ids: Sequence[Any] | torch.Tensor, count: int) -> dict[Any, list[int]]:
    keys = group_ids.detach().cpu().tolist() if isinstance(group_ids, torch.Tensor) else list(group_ids)
    if len(keys) != count:
        raise ValueError("Every response must have one prompt group ID.")
    groups: dict[Any, list[int]] = {}
    for row, key in enumerate(keys):
        try:
            groups.setdefault(key, []).append(row)
        except TypeError as exc:
            raise ValueError("Prompt group IDs must be hashable scalars.") from exc
    if not groups:
        raise ValueError("At least one prompt group is required.")
    return groups


def response_weights_from_means(
    means: torch.Tensor, group_ids: Sequence[Any] | torch.Tensor, temperature: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """Equations (3–4): population z-score, then softmax within each prompt.

    For distributed use, gather response means and prompt IDs first, compute
    these weights over complete sibling groups, and select the local rows.
    Equal-VA siblings receive uniform weights, including a singleton group.
    """
    if means.ndim != 1 or not bool(torch.isfinite(means).all()):
        raise ValueError("Response means must be a finite one-dimensional tensor.")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("temperature and eps must be finite and positive.")
    groups = _groups(group_ids, means.numel())
    values = means.detach().double()
    weights = torch.empty_like(values)
    for rows in groups.values():
        group = values[rows]
        normalized = (group - group.mean()) / (group.std(correction=0) + eps)
        weights[rows] = torch.softmax(normalized / temperature, dim=0)
    return weights.float()


def grouped_token_weights(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any] | torch.Tensor | None = None,
    temperature: float = 1.0,
    high_fraction: float = 0.2,
    high_weight: float = 0.5,
    generator: torch.Generator | None = None,
    eps: float = 1e-6,
    response_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equations (5–6), represented as one coefficient per valid token.

    The high group contains ceil(p_v * T) tokens. Each group receives its
    prescribed mass regardless of its size. If the low group is empty (e.g.
    T=1), all response mass goes to the high group. Empty responses are invalid.
    Ties use token order by default; supplying a seeded generator chooses a
    random secondary key, without changing VA ranks. The paper does not specify
    a tie rule or rounding convention; these are explicit implementation choices.

    With response_weights supplied, this also supports a local distributed shard
    whose other siblings are on another rank. Without it, group_ids must cover
    complete sibling groups. Weights and diagnostics are detached.
    """
    if scores.ndim != 2 or response_mask.shape != scores.shape:
        raise ValueError("scores and response_mask must have shape (responses, tokens).")
    if not math.isfinite(high_fraction) or not 0 < high_fraction <= 1:
        raise ValueError("high_fraction must be in (0, 1].")
    if not math.isfinite(high_weight) or not 0 <= high_weight <= 1:
        raise ValueError("high_weight must be in [0, 1].")
    mask = response_mask.bool()
    values = scores.detach().float().masked_fill(~mask, 0)
    lengths = mask.sum(-1)
    if bool((lengths == 0).any()) or lengths.numel() == 0:
        raise ValueError("Every rollout must contain at least one response token.")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Valid-token VA scores must be finite and nonnegative.")
    if response_weights is None:
        if group_ids is None:
            raise ValueError("group_ids is required without precomputed response weights.")
        response_weights = response_weights_from_means(values.sum(-1) / lengths, group_ids, temperature, eps)
    elif response_weights.shape != lengths.shape:
        raise ValueError("response_weights must have one entry per response.")
    response_weights = response_weights.detach().to(device=values.device, dtype=torch.float32)
    if not bool(torch.isfinite(response_weights).all()) or bool((response_weights < 0).any()):
        raise ValueError("Response weights must be finite and nonnegative.")

    weights = torch.zeros_like(values)
    high_mass = values.new_zeros(())
    high_zero = 0
    high_count = 0
    for row in range(values.shape[0]):
        positions = torch.nonzero(mask[row], as_tuple=True)[0]
        row_scores = values[row, positions]
        order = torch.arange(positions.numel(), device=values.device)
        if generator is not None:
            secondary = torch.rand(positions.numel(), generator=generator, device=generator.device).to(values.device)
            order = torch.argsort(secondary, stable=True)
        order = order[torch.argsort(row_scores[order], descending=True, stable=True)]
        n_high = max(1, math.ceil(high_fraction * positions.numel()))
        high, low = positions[order[:n_high]], positions[order[n_high:]]
        mass = response_weights[row]
        if low.numel():
            weights[row, high] = mass * high_weight / high.numel()
            weights[row, low] = mass * (1 - high_weight) / low.numel()
        else:
            weights[row, high] = mass / high.numel()
        high_mass += values[row, high].sum()
        high_zero += int((values[row, high] == 0).sum())
        high_count += high.numel()
    total_mass = values.sum()
    metrics = {
        "distill/high_group_score_share": float(high_mass / total_mass) if total_mass > 0 else 0.0,
        "distill/high_group_zero_score_frac": high_zero / high_count,
        "distill/response_weight_max": float(response_weights.max()),
        "distill/response_weight_min": float(response_weights.min()),
    }
    return weights, metrics


def reverse_kl_rows(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, vocab_size: int | None = None
) -> torch.Tensor:
    """Full-distribution KL(student || original-image teacher), at T=1.

    An explicit shared vocabulary permits different padded LM-head widths. Both
    distributions are normalized over that entire tokenizer support, not top-k.
    Only the student distribution carries gradients.
    """
    size = _support_size(student_logits, teacher_logits, vocab_size)
    log_p = F.log_softmax(student_logits[..., :size].float(), dim=-1)
    log_q = F.log_softmax(teacher_logits.detach()[..., :size].float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def chunked_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    vocab_size: int | None = None,
    chunk_size: int = 256,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Bound fp32 loss intermediates to chunk_size rows, recomputing backward."""
    size = _support_size(student_logits, teacher_logits, vocab_size)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    shape = student_logits.shape[:-1]
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape(-1, teacher_logits.shape[-1])
    if student.shape[0] == 0:
        return student_logits.sum(-1).float()
    parts = []
    for start in range(0, student.shape[0], chunk_size):
        inputs = student[start : start + chunk_size], teacher[start : start + chunk_size]

        def loss_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return reverse_kl_rows(left, right, size)

        if checkpoint_chunks and torch.is_grad_enabled() and student.requires_grad:
            parts.append(checkpoint(loss_rows, *inputs, use_reentrant=False))
        else:
            parts.append(loss_rows(*inputs))
    return torch.cat(parts).reshape(shape)


def token_weights_for_token_mean(
    weights: torch.Tensor, response_mask: torch.Tensor, group_ids: Sequence[Any] | torch.Tensor
) -> torch.Tensor:
    """Rescale coefficients for a token-mean reducer: c_t * N_tokens / N_prompts."""
    if weights.shape != response_mask.shape:
        raise ValueError("weights and response_mask must have identical shapes.")
    groups = _groups(group_ids, weights.shape[0])
    return weights.detach() * response_mask.bool().sum().to(weights.dtype) / len(groups)


def weighted_prompt_mean(
    per_token_kl: torch.Tensor, weights: torch.Tensor, group_ids: Sequence[Any] | torch.Tensor
) -> torch.Tensor:
    """Average the summed Eq. (6) objectives over complete prompt groups."""
    if per_token_kl.shape != weights.shape or per_token_kl.ndim != 2:
        raise ValueError("Losses and coefficients must have shape (responses, tokens).")
    return (per_token_kl * weights.detach()).sum() / len(_groups(group_ids, weights.shape[0]))
