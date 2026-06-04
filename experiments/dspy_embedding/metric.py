# experiments/dspy_embedding/metric.py
from __future__ import annotations

import dspy


def retrieval_recall_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    expected = set(example.risk_ids)

    try:
        retrieved = set(prediction.risk_ids)
    except (AttributeError, TypeError):
        retrieved = set()

    hits = expected & retrieved
    recall = len(hits) / len(expected) if expected else 1.0

    if pred_name is None:
        return recall

    missing = sorted(expected - retrieved)
    feedback = (
        f"Recall={recall:.3f} ({len(hits)}/{len(expected)}) "
        f"retrieved={len(retrieved)}"
    )
    if missing:
        feedback += f" | Missing: {', '.join(missing[:10])}"
        if len(missing) > 10:
            feedback += f" (+{len(missing) - 10} more)"

    return dspy.Prediction(score=recall, feedback=feedback)
