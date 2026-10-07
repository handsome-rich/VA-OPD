# Implementation and reproduction notes

The training recipe follows the method equations and Appendix A of [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924).

## Objective

The student generates four responses per prompt. The frozen teacher evaluates those same token IDs and prefixes once with the original image and once with its pixelated counterpart. Both distributions use probability temperature **1**, independent of the **0.7** trajectory sampling temperature.

`va_opd/objective.py` implements positive sampled-token log-ratio VA, population-standard-deviation group normalization with epsilon `1e-6`, sibling softmax at temperature `1`, and separate token-group means of full-vocabulary **KL(student || original-image teacher)**. VA, sorting and coefficients are detached. The pixelated distribution supplies VA only. Rewards are used to monitor accuracy and select a checkpoint; there is no reward term, policy gradient, reference-model KL, importance weighting or entropy regularizer in the objective.

Each prompt contributes equally. Siblings may move to different ranks during load balancing: `va_opd/distributed.py` gathers only response means, lengths and numeric prompt IDs, computes softmax over complete sibling groups, then returns the local coefficients. It verifies exactly four responses per prompt. The rescaling by global response-token count / prompt count compensates for the backend's token-mean and FSDP gradient-average reductions. A two-process Gloo test checks this against the serial objective with unequal lengths and siblings split across ranks.

`verl/workers/actor/dp_actor.py` reuses the original teacher logits for both VA and reverse KL. Standard OPD needs one teacher view; VA-OPD adds the pixelated view. The driver does not perform an earlier teacher-scoring pass. The teacher is frozen and explicitly placed in evaluation mode. Only response rows enter the KL, which is calculated in fp32 chunks with activation recomputation. Response logits are stored in bf16, and the divergence uses bounded fp32 chunks.

The released entrypoint supports synchronous FSDP data parallelism with Ulysses size 1. Every rank forwards all its local rollouts before its one backward/update. Dynamic batching, gradient accumulation and multiple update epochs are rejected because they could normalize incomplete sibling groups. The update microbatch is derived from `prompts × rollouts / GPU ranks`: eight responses on each of eight GPUs.

## Settings checked against Appendix A

| Setting | Released paper recipe |
|---|---|
| Student | Qwen3-VL-2B-Instruct |
| Frozen teacher | Qwen3-VL-8B-Instruct; 4B and 32B overrides |
| Prompts × responses per optimizer step | 16 × 4 |
| Epochs | 5 |
| AdamW | LR `1e-6`, betas `(0.9, 0.95)`, weight decay `0.1` |
| Schedule | Cosine decay; 5% linear warmup |
| Student sampling | Temperature `0.7`, top-p `0.95` |
| Maximum response | 4096 tokens; truncated responses are retained |
| Attention / compute | FlashAttention 2 / bf16 |
| Paper hardware | One node, 8 × A100 80 GB |
| Pixelation | Bilinear downsample each side to 10%, nearest-neighbor restore |
| VA weights | τ `1`, epsilon `1e-6`, high fraction `0.2`, high-group mass `0.5` |
| Validation sampling | Temperature `1`, eight responses per question |

Standard OPD uses the same optimization and rollout settings. Its reduction follows Eq. 1: mean token KL within each response, then mean over responses. At equal VA the sibling weights become uniform, but default token groups still each receive half the response mass; equal VA alone does not turn VA-OPD into Standard OPD.

## Preprocessing and runtime settings

The preprocessing, token ranking and checkpoint settings are configured as follows:

- **Validation IDs:** keep the 2101 official Geometry3K training examples and choose 200 of the official validation examples with seed 42. Both training datasets use this Geometry3K selection. `manifest.json` records the exact IDs, source fingerprints, content hashes and parquet checksums. The training entrypoint verifies the actual files, 200 unique validation IDs and train/validation disjointness, including ViRL39K content hashes.
- **Prompt/image budget:** prompt cap 8192; image pixel bounds 262144–4194304. Overlong prompts raise an error instead of cutting image placeholders or rewriting the question. Increase `data.max_prompt_length` together with `worker.rollout.max_num_batched_tokens` if necessary. The 4096 response limit is a paper setting.
- **Model/tokenizer:** the vision tower remains trainable. The released tokenizer and chat template are kept unchanged. Training questions are not given an added system message or CoT format prompt. Student/teacher token mappings and processors must be compatible.
- **Token support:** normalize KL and VA over all tokenizer IDs, excluding LM-head padding rows. Sampling bans those padding IDs. This is full-vocabulary KL, with no top-k approximation.
- **Masks and ranking:** the first generated EOS is included, tokens after it and padding are excluded; a response truncated without EOS keeps all generated tokens. High-group size is `ceil(0.2 × valid_length)`. Ties retain token order. A one-token response gets all its response mass; empty responses are rejected.
- **Monitoring and checkpoints:** seed 42, gradient clipping 1, validation/save every 50 updates, all checkpoints retained. Full optimizer/scheduler/dataloader state is saved for resume. Selection uses accuracy on the held-out set, with the earliest checkpoint winning ties. The final checkpoint is also validated. Final test benchmarks never select a checkpoint.
- **Evaluation top-p/length:** record generation top-p and response length in the benchmark run manifest; see [evaluation.md](evaluation.md).

The training entrypoint provides Standard OPD and VA-OPD recipes. The README includes the paper's GRPO, PAPO, CoT-SFT and off-policy KD comparisons.

## Runtime and checks

The bundled runtime uses the verl / EasyR1 lineage with the upstream license notices preserved. The public `va_opd` package contains the rewritten method, validated configuration entrypoint, data preparation and evaluation aggregation. Use a fresh environment so an independently installed `verl` package does not shadow the bundled one.

The GPU extra pins vLLM `0.17.1` (which requires PyTorch `2.10.0`) and Transformers `>=4.57,<5`. Install a compatible CUDA PyTorch environment, then the extra, then FlashAttention with `--no-build-isolation`.

```bash
pip install -e '.[data,dev]'
pytest -q
python -m va_opd.train --config configs/va_opd.yaml --dry-run
python -m va_opd.train --config configs/va_opd.yaml --method opd --dry-run
bash -n scripts/train.sh
```

CPU tests cover analytical KL values and gradients, the paper's population-normalization example, masks, group weights, image geometry, the actual actor integration with tiny model logits, two-rank reduction, recipe checks, split/checksum validation, answer parsing and official-score aggregation.

For configuration overrides, use `--set KEY=VALUE` (repeatable), for example:

```bash
bash scripts/train.sh --dry-run
TEACHER_PATH=Qwen/Qwen3-VL-32B-Instruct bash scripts/train.sh
bash scripts/train.sh --set trainer.max_steps=5
```

Checkpoints appear under `checkpoints/VA-OPD/<experiment>/global_step_<N>/actor/`. `checkpoint_tracker.json` records the selected step and most recent step. Export the selected actor's sharded weights with `python scripts/model_merger.py --local_dir <actor-directory>` before external benchmark inference. Inspect `python scripts/model_merger.py --help` for the export arguments.
