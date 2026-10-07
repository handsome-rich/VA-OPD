# Evaluation and checkpoint selection

The paper uses the original benchmark prompts, without a rewritten system
prompt or an injected chain-of-thought instruction. The student samples eight
independent responses per example at temperature **1.0**. **avg@8** averages
per-response correctness; it does not count an example as correct whenever any
response is correct, and does not take a majority vote. Evaluation `top_p` is
not specified in the supplied paper; record your choice in the run manifest.

| Benchmark | Scoring in the paper | Reported score |
|---|---|---|
| WeMath | Official rule evaluator | (Strict + Loose) / 2 |
| MathVista, **testmini** | Official GPT judge | Accuracy |
| MathVerse, **testmini** | Official rule evaluator | Accuracy |
| HallusionBench | Official GPT judge | (aAcc + fAcc + qAcc) / 3 |
| AI2D | Official rule evaluator | Accuracy |
| MMMU | Official GPT judge | Accuracy |
| MMStar | Official GPT judge | Accuracy |
| OCRBench | Official rule evaluator, complete 1000-example set | Final Score / 10 |

LLM-based scoring uses **`gpt-4o-2024-08-06`**, temperature **0**, and each
benchmark's official judge prompt verbatim. Supply the official evaluator and
its released prompts when running a benchmark. This repository provides a
protocol manifest, answer extraction, score aggregation and checkpoint
selection. It does **not** include a complete download, generation or GPT
judging pipeline, and does not replace official scorers with a local heuristic.
The manifest is [configs/evaluation.yaml](../configs/evaluation.yaml).

## Answer extraction

`va_opd.evaluation.first_committed_answer` selects the first balanced
`\boxed{...}` or explicit `Answer:`, `Final answer:`, `The answer is`, or Chinese
answer marker in output order. A later correction does not replace an earlier
commitment. Nested LaTeX braces are supported. An answer marker immediately
followed by a boxed answer returns the box content. If no recognizable
commitment exists, the **unchanged raw response** is returned for the official
evaluator. Semantic equivalence, option matching and correctness remain the
official evaluator's responsibility.

This is a documented deterministic extraction convention. The paper specifies
first-committed extraction, but does not provide its original parser source.
Use the same extraction function and official scorer for every compared
method. Free-form answer-marker payloads are read to the end of their line;
the function does not guess where arbitrary explanatory prose ends.

```python
from va_opd.evaluation import first_committed_answer

answer = first_committed_answer(r"Initially, \boxed{A}. Later, \boxed{B}.")
assert answer == "A"
```

## Aggregate official benchmark results

For each benchmark, put sample 0 from every example in one official scorer
run, sample 1 in a second run, and so on through sample 7. Every run must contain
the **same complete example set** and use the same prompts, parser and scorer.
Keep a generation manifest with ordered example IDs and random seeds to check
this correspondence: the aggregator validates counts, but cannot verify the
identity or independence of responses from summary metrics alone.

Export an object with `schema_version: 1`, the protocol below, and all eight
benchmark objects. A benchmark object has `split`, `metric_unit`, and eight
`runs`. Each run contains `sample_index` (0--7), `num_examples`, and `metrics`.
Accuracy metrics must be **percentages on 0--100**, not fractions on 0--1;
OCRBench uses `metric_unit: "points_1000"` and its unnormalized 0--1000 Final
Score. Lower-case metric names below are export schema names, rather than a
claim about an official evaluator's output field names.

```json
{
  "schema_version": 1,
  "protocol": {
    "samples_per_prompt": 8,
    "student_temperature": 1.0,
    "aggregation": "avg@8",
    "prompt_policy": "released",
    "answer_policy": "first_committed",
    "judge_model": "gpt-4o-2024-08-06",
    "judge_temperature": 0.0,
    "judge_prompt_policy": "official_verbatim"
  },
  "benchmarks": {
    "mathvista": {
      "split": "testmini",
      "metric_unit": "percent",
      "runs": [
        {"sample_index": 0, "num_examples": 1000, "metrics": {"accuracy": 66.4}}
      ]
    }
  }
}
```

The JSON above illustrates **one run's schema only** and intentionally fails
full aggregation until runs 1--7 and the seven remaining benchmarks are
provided. Metric keys for each benchmark are:

| Export benchmark key | Export metric keys |
|---|---|
| `wemath` | `strict`, `loose` |
| `mathvista`, `mathverse`, `ai2d`, `mmmu`, `mmstar` | `accuracy` |
| `hallusionbench` | `aAcc`, `fAcc`, `qAcc` |
| `ocrbench` | `final_score` |

```bash
python -m va_opd.evaluation \
  --results results/official-eight-samples.json \
  --output results/summary.json
```

The summary contains the eight 0--100 scores, component metrics, math average
(WeMath, MathVista, MathVerse) and visual average (HallusionBench, AI2D, MMMU,
MMStar, OCRBench). Component scores are averaged across eight runs before the
reported benchmark formula is applied. This equals mean per-response
correctness for fixed-set accuracy metrics. WeMath and HallusionBench retain
their official structured metrics and formulas; they must be scored by the
official evaluator, rather than reduced to a guessed binary substitute.

The CLI rejects missing benchmarks, duplicate sample indices, nonfinite or
out-of-range values, unsupported metric units, unequal example counts,
incorrect testmini splits, and protocol metadata that differs from the paper.
An exporter that labels fractions as percentages must be corrected at its
source: the numerical range alone cannot distinguish a valid low percentage
from a mislabeled fraction.

## Choose the checkpoint before test benchmarking

Select checkpoints using **200 held-out Geometry3K problems disjoint from
training**. The supplied appendix does not release the exact held-out IDs or
split seed. Save both ID lists and the selected split seed with your run; a new
split should not be described as the paper's original split. Never choose a
checkpoint by MathVista, MathVerse or any other test benchmark score.

Export a validation file in this form, with full ID lists:

```json
{
  "dataset": "geometry3k",
  "split": "validation",
  "training_ids": ["train-0", "train-1"],
  "validation_ids": ["heldout-0", "heldout-1"],
  "checkpoints": [
    {"checkpoint": "checkpoints/global_step_50", "global_step": 50,
     "num_examples": 200, "score": 62.0},
    {"checkpoint": "checkpoints/global_step_100", "global_step": 100,
     "num_examples": 200, "score": 63.0}
  ]
}
```

The shortened example requires **all 200 validation IDs** before selection.
Scores are validation percentages on 0--100. Exact ties choose the earliest
`global_step`; this tie rule is a release convention because the paper does
not specify one. Paths must identify distinct checkpoints.

```bash
python -m va_opd.evaluation \
  --select-checkpoint results/geometry3k-validation.json \
  --output results/selected-checkpoint.json
```

The selector validates the benchmark source, 200 validation IDs, their
disjointness from training, complete validation counts and finite score bounds.
As with aggregation, metadata checks do not prove the provenance of an
externally supplied score: preserve generated answers and evaluator outputs
for audit. Run the eight test benchmarks only after fixing the checkpoint.
