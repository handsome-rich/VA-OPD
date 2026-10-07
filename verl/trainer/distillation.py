# Copyright 2026 the VA-OPD authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Runtime adapter for the paper's Standard OPD and VA-OPD objectives.

The mathematical implementation lives in :mod:`va_opd.objective`. This adapter
keeps the trainer's batch and metric interfaces, while explicitly rejecting
unrelated target constructions, truncated vocabularies, and policy gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from va_opd.objective import (
    grouped_token_weights,
    reverse_kl_rows,
    sampled_log_probs,
)
from va_opd.objective import (
    token_weights_for_token_mean as token_weights_for_token_mean,
)

DIVERGENCES = ("reverse_kl",)
SUPPORTS = ("full",)
TARGETS = ("teacher",)
TEMPERATURE_SCOPES = ("all",)


@dataclass(frozen=True)
class DistillationSpec:
    """The shared tokenizer support and fixed original-image reverse-KL target."""

    vocab_size: int
    divergence: str = "reverse_kl"
    support: str = "full"
    temperature: float = 1.0
    temperature_scope: str = "all"
    target: str = "teacher"
    # Accepted config fields for compatibility, with no role in this objective.
    jsd_beta: float = 0.5
    top_k: int = 100

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistillationSpec":
        spec = cls(**{key: value for key, value in config.items() if key in cls.__dataclass_fields__})
        validate_distillation_spec(spec)
        return spec


def validate_distillation_spec(spec: DistillationSpec) -> None:
    if isinstance(spec.vocab_size, bool) or not isinstance(spec.vocab_size, int) or spec.vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer.")
    expected = {
        "divergence": "reverse_kl",
        "support": "full",
        "target": "teacher",
        "temperature": 1.0,
        "temperature_scope": "all",
    }
    for field, value in expected.items():
        if getattr(spec, field) != value:
            raise ValueError(f"VA-OPD and Standard OPD require {field}={value!r}; got {getattr(spec, field)!r}.")


def stat_names(spec: DistillationSpec) -> tuple[str, ...]:
    validate_distillation_spec(spec)
    return "entropy", "student_oov_mass", "teacher_oov_mass"


def _rows(
    student: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = reverse_kl_rows(student, teacher, vocab_size)
    # This compatibility log-probability matches the actor's whole-head score.
    # The KL distributions themselves use the explicit tokenizer support.
    log_probs = sampled_log_probs(student, labels)
    with torch.no_grad():
        s = student.detach().float()
        t = teacher.detach().float()
        s_lse = torch.logsumexp(s, dim=-1)
        s_log_probs = s - s_lse.unsqueeze(-1)
        entropy = -(s_log_probs.exp() * s_log_probs).sum(-1)
        s_padding = -torch.expm1(torch.logsumexp(s[:, :vocab_size], dim=-1) - s_lse)
        t_padding = -torch.expm1(torch.logsumexp(t[:, :vocab_size], dim=-1) - torch.logsumexp(t, dim=-1))
        stats = torch.stack((entropy, s_padding.clamp_min(0), t_padding.clamp_min(0)), dim=-1)
    return loss, log_probs, stats


def chunked_distillation(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    spec: DistillationSpec,
    chunk_size: int = 256,
    teacher_contrast_logits: torch.Tensor | None = None,
    student_contrast_logits: torch.Tensor | None = None,
    row_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all-vocabulary reverse KL in fp32 checkpointed row chunks.

    Padded LM heads may have different widths; shared tokenizer IDs must fit
    both. Only response rows are passed here, so no prompt/padding loss is added.
    The teacher is detached even if a caller accidentally enables its gradients.
    """
    validate_distillation_spec(spec)
    if any(value is not None for value in (teacher_contrast_logits, student_contrast_logits, row_gate)):
        raise ValueError("The counterfactual teacher supplies VA only, never an alternate KL target.")
    if student_logits.ndim != 2 or teacher_logits.ndim != 2 or student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError("Student and teacher logits must cover the same response rows.")
    if labels.shape != student_logits.shape[:1] or bool(((labels < 0) | (labels >= spec.vocab_size)).any()):
        raise ValueError("One sampled tokenizer ID is required per response row.")
    if spec.vocab_size > min(student_logits.shape[-1], teacher_logits.shape[-1]):
        raise ValueError("The shared tokenizer support exceeds an LM-head width.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if student_logits.shape[0] == 0:
        empty = student_logits.sum(-1).float()
        return empty, empty, student_logits.new_empty((0, len(stat_names(spec))), dtype=torch.float32)
    teacher_logits = teacher_logits.detach()
    outputs = []
    for start in range(0, student_logits.shape[0], chunk_size):
        end = start + chunk_size
        inputs = student_logits[start:end], labels[start:end], teacher_logits[start:end], spec.vocab_size
        if torch.is_grad_enabled() and student_logits.requires_grad:
            outputs.append(checkpoint(_rows, *inputs, use_reentrant=False))
        else:
            outputs.append(_rows(*inputs))
    return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs))


def compute_grouped_token_weights(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    softmax_temperature: float,
    high_fraction: float,
    high_weight: float,
    generator: torch.Generator | None,
    eps: float = 1e-6,
    response_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Trainer-facing argument names for :func:`grouped_token_weights`."""
    return grouped_token_weights(
        scores, response_mask, index, softmax_temperature, high_fraction, high_weight, generator, eps, response_weights
    )


def build_distillation_config(
    algorithm: Any, vocab_size: int, end_token_ids: tuple[int, ...] = (), black_pixel_values: list[float] | None = None
) -> dict[str, Any] | None:
    if algorithm.distill_loss_coef == 0:
        return None
    if algorithm.distill_loss_coef < 0 or algorithm.policy_loss_coef != 0:
        raise ValueError("On-policy distillation uses a positive KL coefficient and policy_loss_coef=0.")
    if (
        getattr(algorithm, "teacher_view", "original") != "original"
        or getattr(algorithm, "distill_contrast_view", "none") != "none"
    ):
        raise ValueError("The KL target must use the original-image teacher view.")
    if getattr(algorithm, "distill_is_clip", None) is not None:
        raise ValueError("The paper objective does not apply importance-sampling weights.")
    weighting = getattr(algorithm, "distill_weighting", "none")
    if weighting not in ("none", "va_opd"):
        raise ValueError("distill_weighting must be 'none' or 'va_opd'.")
    config = {
        "vocab_size": int(vocab_size),
        "divergence": algorithm.distill_divergence,
        "support": algorithm.distill_support,
        "temperature": float(algorithm.distill_temperature),
        "temperature_scope": algorithm.distill_temperature_scope,
        "target": algorithm.distill_target,
        "loss_coef": float(algorithm.distill_loss_coef),
        "policy_loss_coef": 0.0,
        "chunk_size": int(algorithm.distill_chunk_size),
        "is_clip": None,
        "teacher_view": "original",
        "contrast_view": "none",
        "views": [],
        "weighting": weighting,
    }
    DistillationSpec.from_config(config)
    if config["chunk_size"] < 1:
        raise ValueError("distill_chunk_size must be positive.")
    return config


def view_keys(name: str) -> dict[str, str]:
    return {
        field: f"distill_view_{name}_{field}"
        for field in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "multi_modal_data",
            "multi_modal_inputs",
            "multi_modal_cache_id",
        )
    }


def resolve_end_token_ids(model_path: str, tokenizer: Any) -> tuple[int, ...]:
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(model_path).eos_token_id
    except OSError:
        eos = None
    if eos is None:
        eos = tokenizer.eos_token_id
    values = [eos] if isinstance(eos, int) else list(eos or [])
    return tuple(sorted(set(int(value) for value in values)))


def black_pixel_values(processor: Any) -> list[float]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not getattr(image_processor, "do_normalize", False):
        raise ValueError("The image processor must normalize pixel values.")
    return [-float(mean) / float(std) for mean, std in zip(image_processor.image_mean, image_processor.image_std)]


def masked_stat_means(stats: torch.Tensor, response_mask: torch.Tensor, names: tuple[str, ...]) -> dict[str, float]:
    mask = response_mask.bool()
    if not bool(mask.any()):
        return {}
    return {name: float(value) for name, value in zip(names, stats.detach()[mask].float().mean(0))}


class DistillationMetricSums:
    """Token-weighted metric totals, invariant to microbatch partitioning."""

    def __init__(self, names: tuple[str, ...], with_is_weights: bool):
        self.columns = [column for column, name in enumerate(names) if name != "entropy"]
        self.names = [f"distill/{names[column]}" for column in self.columns]
        self.with_is_weights = with_is_weights
        self.sums: torch.Tensor | None = None

    def add(
        self,
        loss: torch.Tensor,
        loss_weight: torch.Tensor,
        stats: torch.Tensor,
        is_weights: torch.Tensor | None,
        response_mask: torch.Tensor,
    ) -> None:
        mask = response_mask.bool()
        totals = torch.cat(
            (
                (loss.detach().float() * loss_weight).reshape(1),
                stats.detach()[mask][:, self.columns].float().sum(0),
                (is_weights.detach()[mask].float().sum() if is_weights is not None else loss.new_zeros(())).reshape(1),
                mask.sum().float().reshape(1),
            )
        ).to(loss.device)
        self.sums = totals if self.sums is None else self.sums + totals

    def reduce(self, loss_denominator: torch.Tensor) -> dict[str, float]:
        if self.sums is None:
            raise RuntimeError("No distillation metrics were accumulated.")
        sums = self.sums.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        tokens = sums[-1].clamp_min(1)
        metrics = {"distill/loss": float(sums[0] / loss_denominator.float().clamp_min(1))}
        metrics.update({name: float(value / tokens) for name, value in zip(self.names, sums[1:-2])})
        if self.with_is_weights:
            metrics["distill/is_weight_mean"] = float(sums[-2] / tokens)
        return metrics


def importance_weights(log_probs: torch.Tensor, old_log_probs: torch.Tensor, clip: float | None) -> None:
    if clip is not None:
        raise ValueError("The paper uses direct reverse KL without importance-sampling clipping.")
    return None
