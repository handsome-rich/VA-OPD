# Copyright 2026 VA-OPD authors. Licensed under Apache-2.0.
"""Prompt-normalized VA weights across data-parallel shards.

Only one scalar mean, token count and prompt ID per response crosses ranks.
Teacher logits stay on their own GPU and are reused by the KL objective.
"""

import torch
import torch.distributed as dist

from .objective import grouped_token_weights, response_weights_from_means


@torch.no_grad()
def distributed_token_weights(
    scores, response_mask, group_ids, expected_rollouts=4, temperature=1.0, high_fraction=0.2, high_weight=0.5
):
    lengths = response_mask.bool().sum(-1)
    means = scores.masked_fill(~response_mask.bool(), 0).sum(-1) / lengths.clamp_min(1)
    ids = group_ids.to(device=scores.device, dtype=torch.long)
    if ids.shape != means.shape:
        raise ValueError("One numeric prompt ID is required per response.")
    active = dist.is_available() and dist.is_initialized()
    world, rank = (dist.get_world_size(), dist.get_rank()) if active else (1, 0)
    counts = [means.numel()]
    if world > 1:
        local_count = torch.tensor([means.numel()], device=scores.device, dtype=torch.long)
        gathered_counts = [torch.empty_like(local_count) for _ in range(world)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(count.item()) for count in gathered_counts]

    def gather(values):
        if world == 1:
            return values
        padded = values.new_zeros(max(counts))
        padded[: values.numel()] = values
        outputs = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(outputs, padded)
        return torch.cat([part[:count] for part, count in zip(outputs, counts)])

    all_means, all_ids, all_lengths = gather(means), gather(ids), gather(lengths)
    if (all_lengths == 0).any():
        raise ValueError("Empty responses cannot participate in VA normalization.")
    unique, sibling_counts = torch.unique(all_ids, return_counts=True)
    if (sibling_counts != expected_rollouts).any():
        raise ValueError(f"Every prompt must have exactly {expected_rollouts} on-policy responses across all ranks.")
    all_weights = response_weights_from_means(all_means, all_ids, temperature=temperature)
    offset = sum(counts[:rank])
    local_weights = all_weights[offset : offset + counts[rank]]
    weights, metrics = grouped_token_weights(
        scores,
        response_mask,
        response_weights=local_weights,
        high_fraction=high_fraction,
        high_weight=high_weight,
    )
    # Actor backward first divides by local tokens and then multiplies by
    # local_tokens * world / global_tokens. FSDP averages gradients by world.
    # This factor therefore yields sum_prompt(loss_prompt) / num_prompts.
    weights *= all_lengths.sum().float() / unique.numel()
    metrics["distill/visual_advantage"] = float(all_means.mean())
    return weights, metrics
