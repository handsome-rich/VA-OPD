# Copyright 2026 VA-OPD authors. Licensed under the Apache License, Version 2.0.
"""Validate the paper recipe, then launch the bundled Ray/FSDP/vLLM trainer."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=(), method="va_opd"):
    from verl.trainer.config import PPOConfig

    source = OmegaConf.load(path)
    source = OmegaConf.merge(source, OmegaConf.from_dotlist(list(overrides)))
    world = int(source.trainer.n_gpus_per_node) * int(source.trainer.nnodes)
    local = int(source.data.rollout_batch_size) * int(source.worker.rollout.n)
    if world < 1 or local % world:
        raise ValueError("Responses per step must divide evenly across the GPU ranks.")
    actor = source.worker.actor
    expected_micro = local // world
    configured = actor.get("micro_batch_size_per_device_for_update", expected_micro)
    if configured != expected_micro:
        raise ValueError(f"No gradient accumulation: update microbatch must be {expected_micro}, got {configured}.")
    actor.micro_batch_size_per_device_for_update = expected_micro
    if method == "opd":
        source.algorithm.distill_weighting = "none"
        source.algorithm.visual_sensitivity_reference = "current"
        source.algorithm.corrupt_image = None
        source.algorithm.corrupt_image_kwargs = {}
        # Eq. 1 averages each response's token KL before averaging responses.
        actor.loss_avg_mode = "seq"
    elif method != "va_opd":
        raise ValueError("method must be va_opd or opd")
    for field in ("train_files", "val_files"):
        file = Path(source.data[field]).expanduser()
        source.data[field] = str(file if file.is_absolute() else ROOT / file)
    reward = source.worker.reward.reward_function
    if reward:
        file, func = reward.rsplit(":", 1)
        file = Path(file)
        source.worker.reward.reward_function = f"{file if file.is_absolute() else ROOT / file}:{func}"
    merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), source)
    config = OmegaConf.to_object(merged)
    config.deep_post_init()
    validate_recipe(config)
    return config


def validate_recipe(config):
    a, w, d = config.algorithm, config.worker, config.data
    if a.policy_loss_coef != 0 or a.distill_loss_coef != 1 or not a.disable_kl or a.use_kl_loss:
        raise ValueError("VA-OPD and Standard OPD use only the direct distillation loss.")
    if a.distill_divergence != "reverse_kl" or a.distill_support != "full" or a.distill_target != "teacher":
        raise ValueError("The objective requires full-vocabulary KL(student || original teacher).")
    if a.distill_temperature != 1 or a.distill_is_clip is not None:
        raise ValueError("Distillation uses raw distributions at T=1 and no importance clipping.")
    if w.teacher.source != "model":
        raise ValueError("The paper uses a frozen pretrained teacher, not an EMA teacher.")
    if w.actor.ulysses_size != 1 or w.actor.dynamic_batching:
        raise ValueError("The released synchronous VA update requires Ulysses size=1 and dynamic_batching=false.")
    if w.actor.ppo_epochs != 1 or w.actor.global_batch_size != d.rollout_batch_size:
        raise ValueError("Each rollout batch receives exactly one optimizer update.")
    if a.online_filtering:
        raise ValueError("The paper retains truncated responses and does not reward-filter rollouts.")
    if d.format_prompt or d.system_prompt or d.override_chat_template:
        raise ValueError("Use the released model chat template and original dataset questions.")
    if w.actor.model.plain_think_tokens != "false":
        raise ValueError("Keep the released tokenizer unchanged (plain_think_tokens=false).")
    if config.trainer.val_freq != config.trainer.save_freq or config.trainer.save_freq <= 0:
        raise ValueError("Validation and save intervals must match so each selected checkpoint was evaluated.")
    if config.trainer.save_limit != -1:
        raise ValueError("Keep every evaluated checkpoint until validation selection is complete.")


def validate_data_paths(config):
    import hashlib

    import pyarrow.parquet as pq

    def check_file(file, manifest, ids_key):
        file = Path(file)
        expected = manifest.get("files", {}).get(file.name)
        if expected is None:
            raise ValueError(f"{file.name} is not recorded in its preparation manifest.")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {file}; prepare the dataset again.")
        ids = pq.read_table(file, columns=["sample_id"])["sample_id"].to_pylist()
        recorded = manifest.get(ids_key, [])
        if len(ids) != len(set(ids)) or ids != recorded:
            raise ValueError(f"Actual sample IDs in {file} differ from its manifest or contain duplicates.")
        return ids

    for label, file in (("training", config.data.train_files), ("validation", config.data.val_files)):
        if not Path(file).is_file():
            raise FileNotFoundError(f"Missing {label} data: {file}. Run python -m va_opd.prepare_data first.")
    manifest_path = Path(config.data.val_files).parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Validation needs a manifest.json recording the disjoint Geometry3K split.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset") != "geometry3k" or len(manifest.get("validation_ids", [])) != 200:
        raise ValueError("Checkpoint selection requires exactly 200 held-out Geometry3K problems.")
    if set(manifest["validation_ids"]) & set(manifest.get("train_ids", [])):
        raise ValueError("Training and validation IDs overlap.")
    validation_ids = check_file(config.data.val_files, manifest, "validation_ids")
    train_manifest_path = Path(config.data.train_files).parent / "manifest.json"
    if not train_manifest_path.is_file():
        raise ValueError("Training needs its own preparation manifest.json.")
    train_manifest = json.loads(train_manifest_path.read_text())
    train_ids = check_file(config.data.train_files, train_manifest, "train_ids")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Actual training and validation IDs overlap.")
    if set(train_manifest.get("train_content_sha256", [])) & set(manifest.get("validation_content_sha256", [])):
        raise ValueError("Actual training and validation content overlap; prepare a disjoint validation selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/va_opd.yaml")
    parser.add_argument("--method", choices=("va_opd", "opd"), default="va_opd")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate settings without loading GPU dependencies or data."
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set, args.method)
    print(json.dumps(config.to_dict(), indent=2))
    if args.dry_run:
        print("Configuration valid. No model loading or GPU work performed.")
        return
    validate_data_paths(config)
    # Importing the GPU stack is deliberately deferred until after validation.
    import ray

    from verl.trainer.main import Runner

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
    )
    ray.get(Runner.remote().run.remote(config))


if __name__ == "__main__":
    main()
