"""Paper-aligned extraction, official-score aggregation and checkpoint selection.

This module consumes scores produced by the official benchmark evaluators. It
does not substitute a local heuristic for a benchmark's rules or judge prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SAMPLES_PER_PROMPT = 8
JUDGE_MODEL = "gpt-4o-2024-08-06"
MATH_BENCHMARKS = ("wemath", "mathvista", "mathverse")
VISUAL_BENCHMARKS = ("hallusionbench", "ai2d", "mmmu", "mmstar", "ocrbench")
BENCHMARKS = MATH_BENCHMARKS + VISUAL_BENCHMARKS
METRICS = {
    "wemath": ("strict", "loose"),
    "hallusionbench": ("aAcc", "fAcc", "qAcc"),
    "ocrbench": ("final_score",),
    **{name: ("accuracy",) for name in ("mathvista", "mathverse", "ai2d", "mmmu", "mmstar")},
}
_BOX = re.compile(r"\\boxed\s*\{")
_ANSWER = re.compile(
    r"\b(?:final\s+answer\s*:|answer\s*:|the\s+answer\s+is\b)"
    r"|(?:最终\s*)?答案\s*(?:[：:]|是|为)",
    re.IGNORECASE,
)


def _balanced_box(text: str, match: re.Match[str]) -> str | None:
    """Read a boxed answer without truncating nested LaTeX braces."""
    start = match.end()
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        # A brace escaped by an odd number of backslashes is a literal brace.
        slashes = 0
        previous = index - 1
        while previous >= 0 and text[previous] == "\\":
            slashes += 1
            previous -= 1
        if slashes % 2:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = text[start:index].strip()
                return value or None
    return None


def _marker_answer(text: str, match: re.Match[str]) -> str | None:
    remainder = text[match.end() :].lstrip(" \t")
    # Do not silently read a later answer when the first marker has no payload.
    if not remainder or remainder.startswith(("\n", "\r")):
        return None
    line = remainder.splitlines()[0].strip()
    box_match = _BOX.search(line)
    if box_match and not line[: box_match.start()].strip(" $*:"):
        return _balanced_box(line, box_match)
    # Remove only presentation wrappers, leaving semantic interpretation to
    # the official benchmark evaluator.
    value = line.strip().rstrip("。")
    if value.endswith("."):
        value = value[:-1].rstrip()
    value = value.strip("*$ ")
    return value or None


def first_committed_answer(text: str) -> str:
    """Extract the earliest explicit answer, preserving raw text as fallback.

    Recognized commitments are a balanced ``\\boxed{...}`` or an explicit
    answer marker. Their order in the output determines precedence. This is a
    deterministic parsing convention, not semantic judging or option guessing.
    """
    if not isinstance(text, str):
        raise TypeError("Model output must be a string.")
    candidates: list[tuple[int, str]] = []
    for match in _BOX.finditer(text):
        answer = _balanced_box(text, match)
        if answer is not None:
            candidates.append((match.start(), answer))
    for match in _ANSWER.finditer(text):
        answer = _marker_answer(text, match)
        if answer is not None:
            candidates.append((match.start(), answer))
    return min(candidates, key=lambda item: item[0])[1] if candidates else text


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}.")
    return value


def _number(value: Any, field: str, maximum: float = 100.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= maximum:
        raise ValueError(f"{field} must be finite and in [0, {maximum:g}].")
    return value


def _validate_protocol(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _integer(payload.get("schema_version"), "schema_version", 1) != 1:
        raise ValueError("Only official-results schema_version 1 is supported.")
    protocol = _mapping(payload.get("protocol"), "protocol")
    required = {
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "student_temperature": 1.0,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": 0.0,
        "judge_prompt_policy": "official_verbatim",
        "prompt_policy": "released",
        "answer_policy": "first_committed",
        "aggregation": "avg@8",
    }
    for key, expected in required.items():
        value = protocol.get(key)
        if isinstance(expected, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise ValueError(f"protocol.{key} must be {expected!r}.")
        elif value != expected:
            raise ValueError(f"protocol.{key} must be {expected!r}.")
    return protocol


def aggregate_official_scores(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Average eight independent official benchmark scores, all on 0--100.

    Each run must score the same complete benchmark split. Percentages are
    explicit; a 0--1 fraction mislabeled as a percentage cannot be detected
    from its value alone, so exporters must provide the correct metric_unit.
    """
    payload = _mapping(payload, "results")
    protocol = _validate_protocol(payload)
    benchmarks = _mapping(payload.get("benchmarks"), "benchmarks")
    if set(benchmarks) != set(BENCHMARKS):
        missing = sorted(set(BENCHMARKS) - set(benchmarks))
        unexpected = sorted(set(benchmarks) - set(BENCHMARKS))
        raise ValueError(f"Expected all eight benchmarks; missing={missing}, unexpected={unexpected}.")
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for name in BENCHMARKS:
        benchmark = _mapping(benchmarks[name], f"benchmarks.{name}")
        split = benchmark.get("split")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"{name}.split must identify the official evaluated split.")
        if name in {"mathvista", "mathverse"} and split != "testmini":
            raise ValueError(f"{name} must use the testmini split.")
        unit = "points_1000" if name == "ocrbench" else "percent"
        if benchmark.get("metric_unit") != unit:
            raise ValueError(f"{name}.metric_unit must be {unit!r}.")
        runs = benchmark.get("runs")
        if not isinstance(runs, list) or len(runs) != SAMPLES_PER_PROMPT:
            raise ValueError(f"{name} requires exactly eight independently sampled runs.")
        indices: set[int] = set()
        run_counts: set[int] = set()
        values = {metric: [] for metric in METRICS[name]}
        for run_number, raw_run in enumerate(runs):
            run = _mapping(raw_run, f"{name}.runs[{run_number}]")
            index = _integer(run.get("sample_index"), f"{name}.sample_index")
            if index >= SAMPLES_PER_PROMPT or index in indices:
                raise ValueError(f"{name}.sample_index must cover 0 through 7 exactly once.")
            indices.add(index)
            count = _integer(run.get("num_examples"), f"{name}.num_examples", 1)
            run_counts.add(count)
            metrics = _mapping(run.get("metrics"), f"{name}.metrics")
            if set(metrics) != set(METRICS[name]):
                raise ValueError(f"{name}.metrics must contain exactly {METRICS[name]!r}.")
            for metric, per_run in values.items():
                per_run.append(_number(metrics[metric], f"{name}.{metric}", 1000.0 if name == "ocrbench" else 100.0))
        if len(run_counts) != 1:
            raise ValueError(f"{name} runs must contain the same number of examples.")
        if name == "ocrbench" and run_counts != {1000}:
            raise ValueError("OCRBench must evaluate its official 1000-example set.")
        counts[name] = run_counts.pop()
        components[name] = {metric: math.fsum(per_run) / SAMPLES_PER_PROMPT for metric, per_run in values.items()}
        if name == "ocrbench":
            scores[name] = components[name]["final_score"] / 10.0
        else:
            scores[name] = math.fsum(components[name].values()) / len(components[name])
    return {
        "schema_version": 1,
        "protocol": dict(protocol),
        "score_unit": "percent",
        "scores": scores,
        "math_average": math.fsum(scores[name] for name in MATH_BENCHMARKS) / len(MATH_BENCHMARKS),
        "visual_average": math.fsum(scores[name] for name in VISUAL_BENCHMARKS) / len(VISUAL_BENCHMARKS),
        "official_components": components,
        "num_examples": counts,
    }


def _ids(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list of example IDs.")
    identifiers: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)) or not str(item).strip():
            raise ValueError(f"{field} must contain only nonempty strings or integer IDs.")
        identifiers.append(str(item))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field} contains duplicate example IDs.")
    return set(identifiers)


def select_validation_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Choose only on a disjoint 200-example Geometry3K validation set.

    The validation score is a percentage. Equal scores choose the earliest
    global step. Callers must export genuine scores and stable dataset IDs;
    metadata validation cannot establish the provenance of an external score.
    """
    payload = _mapping(payload, "checkpoint selection")
    if payload.get("dataset") != "geometry3k" or payload.get("split") != "validation":
        raise ValueError("Checkpoint selection requires Geometry3K validation, never test benchmarks.")
    training_ids = _ids(payload.get("training_ids"), "training_ids")
    validation_ids = _ids(payload.get("validation_ids"), "validation_ids")
    if len(validation_ids) != 200:
        raise ValueError("Checkpoint selection requires exactly 200 held-out Geometry3K examples.")
    if training_ids & validation_ids:
        raise ValueError("Geometry3K training and validation IDs must be disjoint.")
    candidates = payload.get("checkpoints")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("checkpoints must be a nonempty list.")
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_candidate in candidates:
        candidate = _mapping(raw_candidate, "checkpoint")
        path = candidate.get("checkpoint")
        if not isinstance(path, str) or not path.strip() or path in seen_paths:
            raise ValueError("Each checkpoint must have a unique nonempty checkpoint path.")
        seen_paths.add(path)
        step = _integer(candidate.get("global_step"), "checkpoint.global_step")
        score = _number(candidate.get("score"), "checkpoint.score")
        if _integer(candidate.get("num_examples"), "checkpoint.num_examples", 1) != 200:
            raise ValueError("Every checkpoint must be scored on all 200 validation examples.")
        validated.append({"checkpoint": path, "global_step": step, "score": score, "num_examples": 200})
    best = min(validated, key=lambda candidate: (-candidate["score"], candidate["global_step"]))
    return {"dataset": "geometry3k", "split": "validation", **best}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="Official benchmark result JSON to aggregate.")
    source.add_argument("--select-checkpoint", type=Path, help="Geometry3K validation result JSON to select from.")
    parser.add_argument("--output", type=Path, help="Write JSON to this path; otherwise print to stdout.")
    args = parser.parse_args(argv)
    try:
        with (args.results or args.select_checkpoint).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        result = aggregate_official_scores(payload) if args.results else select_validation_checkpoint(payload)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
