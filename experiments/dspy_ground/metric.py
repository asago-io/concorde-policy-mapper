from __future__ import annotations

import dspy

DECISION_WEIGHT = 0.8
QUOTE_WEIGHT = 0.2


def _token_f1(predicted_text: str, expected_text: str) -> float:
    pred_tokens = set(predicted_text.lower().split())
    exp_tokens = set(expected_text.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(exp_tokens)
    return 2 * p * r / (p + r)


def ground_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    expected = {v["risk_id"]: v["grounded"] for v in example.expected_verdicts}
    expected_quotes = {
        v["risk_id"]: v.get("expected_quotes", [])
        for v in example.expected_verdicts
    }

    try:
        predicted_verdicts = prediction.verdicts
        if not predicted_verdicts:
            predicted = {}
            predicted_quotes = {}
        else:
            predicted = {}
            predicted_quotes = {}
            for v in predicted_verdicts:
                if hasattr(v, "risk_id"):
                    predicted[v.risk_id] = v.grounded
                    predicted_quotes[v.risk_id] = list(v.quotes) if hasattr(v, "quotes") else []
                elif isinstance(v, dict):
                    predicted[v["risk_id"]] = v["grounded"]
                    predicted_quotes[v["risk_id"]] = v.get("quotes", [])
    except (AttributeError, TypeError):
        predicted = {}
        predicted_quotes = {}

    tp = sum(1 for rid in expected if expected[rid] and predicted.get(rid, False))
    fp = sum(1 for rid in predicted if predicted[rid] and not expected.get(rid, False))
    fn = sum(1 for rid in expected if expected[rid] and not predicted.get(rid, False))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    decision_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    quote_scores = []
    for rid in expected:
        if expected[rid] and predicted.get(rid, False):
            exp_q = expected_quotes.get(rid, [])
            pred_q = predicted_quotes.get(rid, [])
            if exp_q and pred_q:
                exp_text = " ".join(exp_q)
                pred_text = " ".join(pred_q)
                quote_scores.append(_token_f1(pred_text, exp_text))

    quote_f1 = sum(quote_scores) / len(quote_scores) if quote_scores else 0.0

    combined = DECISION_WEIGHT * decision_f1 + QUOTE_WEIGHT * quote_f1

    if pred_name is None:
        return combined

    n_expected_pos = sum(1 for v in expected.values() if v)
    n_predicted_pos = sum(1 for v in predicted.values() if v)

    feedback = (
        f"combined={combined:.3f} "
        f"decision_F1={decision_f1:.3f} P={precision:.3f} R={recall:.3f} "
        f"(TP={tp} FP={fp} FN={fn}) "
        f"quote_F1={quote_f1:.3f} (n={len(quote_scores)}) "
        f"expected_pos={n_expected_pos} predicted_pos={n_predicted_pos} "
        f"total_candidates={len(expected)}"
    )

    return dspy.Prediction(score=combined, feedback=feedback)
