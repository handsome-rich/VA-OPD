from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from va_opd.distributed import distributed_token_weights
from va_opd.objective import grouped_token_weights


def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        scores = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 100
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
            ]
        )
        ids = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        sl = slice(rank * 4, (rank + 1) * 4)
        weights, _ = distributed_token_weights(scores[sl], mask[sl], ids[sl])
        parameter = torch.tensor(1.0, requires_grad=True)
        kl = torch.arange(40, dtype=torch.float32).reshape(8, 5) + 1
        # The actor scales by world before FSDP averages the gradients.
        loss = (parameter * kl[sl] * weights * mask[sl]).sum() * 2 / mask.sum()
        loss.backward()
        torch.save({"weights": weights, "gradient": parameter.grad}, Path(output) / f"{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_cross_rank_groups_and_ddp_loss_scale(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable in this PyTorch build")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    parts = [torch.load(tmp_path / f"{rank}.pt", weights_only=True) for rank in range(2)]
    scores = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 100
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ]
    )
    ids = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    reference, _ = grouped_token_weights(scores, mask, ids)
    torch.testing.assert_close(torch.cat([part["weights"] for part in parts]), reference * mask.sum() / 2)
    expected = (reference * (torch.arange(40).reshape(8, 5) + 1)).sum() / 2
    torch.testing.assert_close(torch.stack([part["gradient"] for part in parts]).mean(), expected)
