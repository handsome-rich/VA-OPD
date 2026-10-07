import json
from pathlib import Path

import pytest

from va_opd.prepare_data import normalize_rows, validation_indices, write_dataset
from va_opd.train import load_config, validate_data_paths

CONFIG = Path(__file__).parents[1] / "configs/va_opd.yaml"


def test_exact_paper_defaults_and_derived_microbatch():
    c = load_config(CONFIG)
    assert c.data.rollout_batch_size == c.worker.actor.global_batch_size == 16
    assert c.worker.rollout.n == 4
    assert c.worker.rollout.temperature == 0.7 and c.worker.rollout.top_p == 0.95
    assert c.data.max_response_length == 4096
    assert tuple(c.worker.actor.optim.betas) == (0.9, 0.95)
    assert c.worker.actor.optim.weight_decay == 0.1
    assert c.worker.actor.optim.lr_scheduler_type == "cosine"
    assert c.worker.actor.optim.lr_warmup_ratio == 0.05
    assert c.worker.actor.micro_batch_size_per_device_for_update == 8
    assert c.algorithm.policy_loss_coef == 0 and c.worker.rollout.ban_ids_beyond_tokenizer
    c = load_config(CONFIG, ["trainer.n_gpus_per_node=4"])
    assert c.worker.actor.micro_batch_size_per_device_for_update == 16


def test_standard_opd_uses_same_rollout_and_equal_response_reduction():
    c = load_config(CONFIG, method="opd")
    assert c.algorithm.distill_weighting == "none"
    assert c.worker.actor.loss_avg_mode == "seq"
    assert c.worker.rollout.temperature == 0.7 and c.worker.rollout.top_p == 0.95


@pytest.mark.parametrize(
    "override",
    [
        "worker.actor.micro_batch_size_per_device_for_update=4",
        "worker.actor.dynamic_batching=true",
        "algorithm.policy_loss_coef=1",
        "algorithm.online_filtering=true",
        "data.format_prompt=README.md",
        "worker.actor.ppo_epochs=2",
        "worker.teacher.source=ema",
        "trainer.save_freq=10",
    ],
)
def test_configuration_rejects_nonpaper_objectives_or_invalid_update(override):
    with pytest.raises((ValueError, AssertionError)):
        load_config(CONFIG, [override])


def test_disjoint_data_manifest_and_embedded_images(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    train = list(
        normalize_rows(
            [{"problem": "Train?", "answer": "1", "images": [Image.new("RGB", (20, 20))]}], "geometry3k", "train"
        )
    )
    val = list(
        normalize_rows(
            [{"problem": f"Question {i}?", "answer": "2", "images": [Image.new("RGB", (20, 20))]} for i in range(300)],
            "geometry3k",
            "validation",
        )
    )
    manifest = write_dataset(tmp_path, train, val)
    assert len(manifest["validation_ids"]) == 200
    assert not set(manifest["train_ids"]) & set(manifest["validation_ids"])
    assert len(pq.read_table(tmp_path / "validation200.parquet")) == 200
    c = load_config(
        CONFIG, [f"data.train_files={tmp_path}/train.parquet", f"data.val_files={tmp_path}/validation200.parquet"]
    )
    validate_data_paths(c)
    assert pq.read_table(tmp_path / "train.parquet").to_pylist()[0]["images"][0]["bytes"]
    file = tmp_path / "validation200.parquet"
    original = file.read_bytes()
    file.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        validate_data_paths(c)
    file.write_bytes(original)
    manifest["validation_ids"][0] = manifest["train_ids"][0]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        validate_data_paths(c)


def test_validation_sampling_reproducible_and_does_not_use_test():
    assert validation_indices(300) == validation_indices(300)
    assert validation_indices(300, seed=1) != validation_indices(300, seed=2)
    with pytest.raises(ValueError):
        validation_indices(199)
