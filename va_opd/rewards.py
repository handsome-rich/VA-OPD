# Copyright 2026 VA-OPD authors. Licensed under Apache-2.0.
"""Geometry3K accuracy for monitoring and held-out checkpoint selection only."""

from va_opd.evaluation import first_committed_answer

REWARD_NAME = "geometry_accuracy"
REWARD_TYPE = "batch"


def compute_score(reward_inputs):
    from mathruler.grader import grade_answer

    results = []
    for row in reward_inputs:
        answer = first_committed_answer(row["response"])
        correct = float(grade_answer(answer, str(row["ground_truth"])))
        results.append({"overall": correct, "accuracy": correct})
    return results
