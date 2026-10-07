"""Execute the actual actor integration with tiny logits instead of GPU models."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from va_opd.distributed import distributed_token_weights
from va_opd.objective import sampled_log_probs, visual_advantage
from verl.trainer.distillation import DistillationSpec, chunked_distillation


def actor_method():
    # Importing the full Ray/vLLM actor needs a CUDA installation. Compile its actual
    # method unchanged; the fake model supplies exactly the same response-row interface.
    file = Path(__file__).parents[1] / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(file.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    method = next(
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch_distill"
    )
    namespace = {
        "torch": torch,
        "Any": object,
        "DistillationSpec": DistillationSpec,
        "chunked_distillation": chunked_distillation,
        "distributed_token_weights": distributed_token_weights,
        "sampled_log_probs": sampled_log_probs,
        "visual_advantage": visual_advantage,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[method.name]


class FakeActor:
    def __init__(self, mask):
        self.config = SimpleNamespace(padding_free=True)
        self.actor_module = object()
        self.teacher_module = torch.nn.Identity()
        self.calls = []
        self.mask = mask.bool()
        rows = int(mask.sum())
        gen = torch.Generator().manual_seed(125)
        self.student = torch.randn(rows, 5, generator=gen, requires_grad=True)
        self.original = torch.randn(rows, 5, generator=gen, requires_grad=True)
        self.pixelated = torch.randn(rows, 5, generator=gen, requires_grad=True)

    def _get_actor_model_config(self):
        return SimpleNamespace(model_type="qwen3_vl")

    def _prepare_response_rows(self, batch, mask, length):
        return {"labels": batch["responses"][mask.bool()], "view": "original"}

    def _prepare_view_rows(self, batch, name, rows, length):
        return {**rows, "view": name}

    def _packed_forward(self, module, views):
        assert len(views) == 1
        view = views[0]["view"]
        self.calls.append(("student" if module is self.actor_module else "teacher", view))
        if module is self.actor_module:
            return [self.student]
        return [self.original if view == "original" else self.pixelated]

    def _rows_to_response(self, values, rows, length):
        result = values.new_zeros((*self.mask.shape, *values.shape[1:]))
        result[self.mask] = values
        return result


@pytest.mark.parametrize("method,teacher_passes", [("va_opd", 2), ("none", 1)])
def test_actor_reuses_teacher_and_has_only_student_gradient(method, teacher_passes):
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    actor = FakeActor(mask)
    inputs = {
        "responses": torch.zeros_like(mask),
        "response_mask": mask,
        "distill_group_ids": torch.zeros(4, dtype=torch.long),
    }
    cfg = {
        "vocab_size": 5,
        "chunk_size": 3,
        "weighting": method,
        "rollouts_per_prompt": 4,
        "va_temperature": 1.0,
        "va_high_fraction": 0.2,
        "va_high_weight": 0.5,
    }
    _, kl, _ = actor_method()(actor, inputs, cfg)
    assert actor.calls.count(("teacher", "original")) == 1
    assert sum(who == "teacher" for who, _ in actor.calls) == teacher_passes
    if method == "va_opd":
        weights = inputs["distill_token_weights"]
        assert torch.all(weights[~mask.bool()] == 0)
        assert float(weights.sum()) == pytest.approx(float(mask.sum()))
        loss = (kl * weights).sum() / mask.sum()
    else:
        loss = ((kl * mask).sum(-1) / mask.sum(-1)).mean()
    loss.backward()
    assert actor.student.grad is not None and torch.isfinite(actor.student.grad).all()
    assert actor.original.grad is None and actor.pixelated.grad is None
