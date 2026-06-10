"""Tests for embedding optimization dataset loader."""

from pathlib import Path

import pytest

_GT_DIR = Path(__file__).resolve().parent.parent / "evals" / "ground_truth"
_POLICY_DIR = Path(__file__).resolve().parent.parent / "policy_examples"


@pytest.mark.skipif(
    not (_GT_DIR / "sap.yaml").exists() or not any(_POLICY_DIR.glob("sap.*")),
    reason="Ground truth or policy file for 'sap' not available",
)
@pytest.mark.slow
def test_build_example_sap():
    from experiments.dspy_embedding.dataset import _build_example

    example = _build_example("sap", chunk_max_tokens=512)
    assert example is not None
    assert len(example.chunk_texts) > 0
    assert len(example.risk_ids) > 0
    assert example.policy_name == "sap"
    # All risk IDs should be risk-level, not category-level
    for rid in example.risk_ids:
        assert not rid.startswith("nist-ai-rmf")
        assert not rid.startswith("owasp-llm-2.0")


def test_build_example_nonexistent():
    from experiments.dspy_embedding.dataset import _build_example

    example = _build_example("nonexistent-policy-xyz")
    assert example is None
