import json

import pytest

from va_opd.evaluation import (
    BENCHMARKS,
    METRICS,
    aggregate_official_scores,
    first_committed_answer,
    main,
    select_validation_checkpoint,
)


def official_results():
    payload = {
        "schema_version": 1,
        "protocol": {
            "samples_per_prompt": 8,
            "student_temperature": 1.0,
            "aggregation": "avg@8",
            "prompt_policy": "released",
            "answer_policy": "first_committed",
            "judge_model": "gpt-4o-2024-08-06",
            "judge_temperature": 0.0,
            "judge_prompt_policy": "official_verbatim",
        },
        "benchmarks": {},
    }
    for benchmark in BENCHMARKS:
        payload["benchmarks"][benchmark] = {
            "split": "testmini" if benchmark in {"mathvista", "mathverse"} else "official",
            "metric_unit": "points_1000" if benchmark == "ocrbench" else "percent",
            "runs": [
                {
                    "sample_index": sample,
                    "num_examples": 1000 if benchmark == "ocrbench" else 20,
                    "metrics": {metric: 850 if benchmark == "ocrbench" else 50 for metric in METRICS[benchmark]},
                }
                for sample in range(8)
            ],
        }
    return payload


def validation_results():
    return {
        "dataset": "geometry3k",
        "split": "validation",
        "training_ids": [str(index) for index in range(200, 220)],
        "validation_ids": [str(index) for index in range(200)],
        "checkpoints": [
            {"checkpoint": "step100", "global_step": 100, "num_examples": 200, "score": 60.0},
            {"checkpoint": "step50", "global_step": 50, "num_examples": 200, "score": 60.0},
            {"checkpoint": "step10", "global_step": 10, "num_examples": 200, "score": 59.0},
        ],
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"First \boxed{A}; corrected \boxed{B}", "A"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\{x\}}", r"\{x\}"),
        ("Answer: 42\nFinal answer: 43", "42"),
        ("Final answer: **A**.\nAnswer: B", "A"),
        (r"The answer is $\boxed{42}$", "42"),
        ("答案：A\n最终答案：B", "A"),
        ("Reasoning without any explicit commitment.", "Reasoning without any explicit commitment."),
        (r"\boxed{unfinished", r"\boxed{unfinished"),
        ("Answer:\nmore reasoning", "Answer:\nmore reasoning"),
        ("", ""),
    ],
)
def test_first_committed_answer(text, expected):
    assert first_committed_answer(text) == expected


def test_marker_and_box_use_output_order():
    assert first_committed_answer("Answer: 7\n" + r"\boxed{8}") == "7"
    assert first_committed_answer(r"\boxed{8}" + "\nAnswer: 7") == "8"


def test_output_type_validation():
    with pytest.raises(TypeError):
        first_committed_answer(None)


def test_official_formulas_and_avg8():
    payload = official_results()
    for run in payload["benchmarks"]["wemath"]["runs"]:
        run["metrics"] = {"strict": 20.0, "loose": 80.0}
    for run in payload["benchmarks"]["hallusionbench"]["runs"]:
        run["metrics"] = {"aAcc": 30.0, "fAcc": 60.0, "qAcc": 90.0}
    for run in payload["benchmarks"]["mathvista"]["runs"]:
        run["metrics"]["accuracy"] = 100.0 if run["sample_index"] == 0 else 0.0
    result = aggregate_official_scores(payload)
    assert result["scores"]["wemath"] == 50.0
    assert result["scores"]["hallusionbench"] == 60.0
    assert result["scores"]["mathvista"] == 12.5  # Not pass@8 or a majority vote.
    assert result["scores"]["ocrbench"] == 85.0
    assert result["math_average"] == pytest.approx(37.5)
    assert result["visual_average"] == pytest.approx(59.0)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -1, 100.001, True, "50"])
def test_invalid_accuracy_metrics_rejected(bad_score):
    payload = official_results()
    payload["benchmarks"]["ai2d"]["runs"][0]["metrics"]["accuracy"] = bad_score
    with pytest.raises(ValueError):
        aggregate_official_scores(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["benchmarks"].pop("ocrbench"),
        lambda p: p["benchmarks"]["mathvista"].update(split="test"),
        lambda p: p["benchmarks"]["ai2d"].update(metric_unit="fraction"),
        lambda p: p["benchmarks"]["ai2d"]["runs"].pop(),
        lambda p: p["benchmarks"]["ai2d"]["runs"][1].update(sample_index=0),
        lambda p: p["benchmarks"]["ai2d"]["runs"][1].update(num_examples=19),
        lambda p: p["benchmarks"]["ocrbench"]["runs"][0].update(num_examples=999),
        lambda p: p["benchmarks"]["ocrbench"]["runs"][0]["metrics"].update(final_score=1001),
        lambda p: p["protocol"].update(judge_model="gpt-4o"),
        lambda p: p["protocol"].update(student_temperature=0.7),
        lambda p: p["protocol"].update(aggregation="pass@8"),
        lambda p: p["protocol"].update(judge_temperature=True),
    ],
)
def test_protocol_mismatches_rejected(mutation):
    payload = official_results()
    mutation(payload)
    with pytest.raises(ValueError):
        aggregate_official_scores(payload)


def test_checkpoint_selection_ties_choose_earliest_step():
    selected = select_validation_checkpoint(validation_results())
    assert selected["checkpoint"] == "step50"
    assert selected["dataset"] == "geometry3k"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(dataset="mathvista"),
        lambda p: p.update(split="test"),
        lambda p: p["validation_ids"].pop(),
        lambda p: p["training_ids"].append("0"),
        lambda p: p["validation_ids"].append("0"),
        lambda p: p["checkpoints"][0].update(score=float("nan")),
        lambda p: p["checkpoints"][0].update(num_examples=199),
        lambda p: p["checkpoints"][0].update(global_step=True),
    ],
)
def test_checkpoint_leakage_and_invalid_scores_rejected(mutation):
    payload = validation_results()
    mutation(payload)
    with pytest.raises(ValueError):
        select_validation_checkpoint(payload)


def test_cli_round_trip_and_selection(tmp_path):
    source = tmp_path / "official.json"
    output = tmp_path / "nested" / "summary.json"
    source.write_text(json.dumps(official_results()), encoding="utf-8")
    main(["--results", str(source), "--output", str(output)])
    assert json.loads(output.read_text())["scores"]["ocrbench"] == 85.0
    source.write_text(json.dumps(validation_results()), encoding="utf-8")
    main(["--select-checkpoint", str(source), "--output", str(output)])
    assert json.loads(output.read_text())["global_step"] == 50


def test_cli_bad_json_fails_closed(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--results", str(source)])
    assert error.value.code == 2
