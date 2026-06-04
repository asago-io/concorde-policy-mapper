"""Tests for embedding retrieval recall metric."""
import dspy

from experiments.dspy_embedding.metric import retrieval_recall_metric


def test_perfect_recall():
    example = dspy.Example(risk_ids=["a", "b", "c"])
    pred = dspy.Prediction(risk_ids=["a", "b", "c", "d"])
    score = retrieval_recall_metric(example, pred)
    assert score == 1.0


def test_partial_recall():
    example = dspy.Example(risk_ids=["a", "b", "c", "d"])
    pred = dspy.Prediction(risk_ids=["a", "c"])
    score = retrieval_recall_metric(example, pred)
    assert score == 0.5


def test_zero_recall():
    example = dspy.Example(risk_ids=["a", "b"])
    pred = dspy.Prediction(risk_ids=["x", "y"])
    score = retrieval_recall_metric(example, pred)
    assert score == 0.0


def test_empty_expected():
    example = dspy.Example(risk_ids=[])
    pred = dspy.Prediction(risk_ids=["a"])
    score = retrieval_recall_metric(example, pred)
    assert score == 1.0


def test_feedback_with_pred_name():
    example = dspy.Example(risk_ids=["a", "b", "c"])
    pred = dspy.Prediction(risk_ids=["a"])
    result = retrieval_recall_metric(example, pred, pred_name="test")
    assert hasattr(result, "score")
    assert abs(result.score - 1 / 3) < 0.01
    assert "Missing" in result.feedback
    assert "b" in result.feedback
    assert "c" in result.feedback
