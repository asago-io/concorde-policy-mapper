import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from concorde_policy_mapper.extract.models import ExtractionResult
from concorde_policy_mapper.extract.pipeline import run_extraction


def _make_risk(id, name, description, concern=""):
    return SimpleNamespace(
        id=id,
        name=name,
        description=description,
        concern=concern,
        risk_type="",
        isDefinedByTaxonomy="test-taxonomy",
        isPartOf="",
    )


MOCK_RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs favoring certain groups"),
    _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data by AI"),
]


def test_run_extraction_returns_extraction_result(mock_config, tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("AI systems must avoid bias and protect personal data. Training data integrity is critical.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.version == "0.3"
    assert result.source_documents == [str(doc)]
    assert result.retrieval_stats.total_chunks >= 1


def test_run_extraction_empty_document(mock_config, tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("")

    mock_client = MagicMock()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.risks == []


def test_run_extraction_no_risks(mock_config, tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("Some policy text.")

    mock_client = MagicMock()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=[],
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.risks == []
    assert result.retrieval_stats.total_chunks == 0


def test_run_extraction_multiple_documents(mock_config, tmp_path):
    docs = []
    for i in range(3):
        doc = tmp_path / f"doc{i}.txt"
        doc.write_text(f"Policy document number {i} about AI governance.")
        docs.append(doc)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=docs,
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert len(result.source_documents) == 3
    assert result.retrieval_stats.total_chunks >= 3


def test_run_extraction_metadata(mock_config, tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("AI risk document.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert result.metadata["model"] == "test-model"
    assert result.metadata["top_n_accept"] == 10
    assert result.metadata["top_n_judge"] == 10


def test_run_extraction_populates_chunks(mock_config, tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("AI systems must avoid bias and protect personal data. Training data integrity is critical.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert len(result.chunks) >= 1
    chunk = result.chunks[0]
    assert chunk.index == 0
    assert chunk.source == str(doc)
    assert len(chunk.text_preview) > 0
    assert len(chunk.text_preview) <= 200
    assert chunk.candidates_retrieved >= 0


def test_run_extraction_populates_llm_calls(mock_config, tmp_path):
    """LLM calls list is present (may be empty if no borderline candidates)."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert isinstance(result.llm_calls, list)


def test_run_extraction_no_judge_no_grounding(mock_config, tmp_path):
    """With no_judge+no_grounding, no LLM calls are made and all candidates become RiskMatch with empty evidence."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = MagicMock()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
        no_judge=True,
        no_grounding=True,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_judge"] is True
    assert result.metadata["no_grounding"] is True
    mock_client.chat.completions.create.assert_not_called()
    assert len(result.llm_calls) == 0
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"
    assert result.retrieval_stats.grounding_filtered == 0


def test_run_extraction_no_grounding_with_judge(mock_config, tmp_path):
    """With no_grounding only, judge runs but grounding is skipped."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
        no_grounding=True,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_grounding"] is True
    assert result.metadata["no_judge"] is False
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"
    assert result.retrieval_stats.grounding_filtered == 0


def test_run_extraction_no_judge_no_grounding_no_cross_encoder(mock_config, tmp_path):
    """With no_judge+no_grounding and no cross-encoder, pure BM25+semantic output."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = MagicMock()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
        no_judge=True,
        no_grounding=True,
        use_cross_encoder=False,
        rrf_min_score=0.001,
    )

    assert isinstance(result, ExtractionResult)
    mock_client.chat.completions.create.assert_not_called()
    for risk in result.risks:
        assert risk.accepted_by in ("rrf", "auto_promoted")


def test_run_extraction_no_cross_encoder(mock_config, tmp_path):
    """With use_cross_encoder=False, no judge calls are made and metadata reflects the mode."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = []

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
        use_cross_encoder=False,
        rrf_min_score=0.01,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["use_cross_encoder"] is False
    assert result.metadata["rrf_min_score"] == 0.01
    assert result.metadata["cross_encoder_model"] is None
    judge_calls = [c for c in result.llm_calls if c.stage == "judge"]
    assert len(judge_calls) == 0
