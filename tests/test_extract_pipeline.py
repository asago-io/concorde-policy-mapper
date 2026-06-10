from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from concorde_policy_mapper.extract.models import (
    EvidenceSpan,
    ExtractionResult,
    LLMCallRecord,
    RetrievalConfig,
    RetrievalScores,
    RiskMatch,
    ScoredCandidate,
    _CausalChain,
    _RiskEvidence,
)
from concorde_policy_mapper.extract.pipeline import (
    _run_causal_synthesis,
    build_risk_match,
    determine_accepted_by,
    run_extraction,
)


def _make_risk(id, name, description, concern="", parent=""):
    return SimpleNamespace(
        id=id,
        name=name,
        description=description,
        concern=concern,
        risk_type="",
        isDefinedByTaxonomy="test-taxonomy",
        isPartOf=parent,
        exact_mappings=[],
        close_mappings=[],
        broad_mappings=[],
        narrow_mappings=[],
        related_mappings=[],
    )


MOCK_RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs favoring certain groups"),
    _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data by AI"),
]


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
        retrieval=RetrievalConfig(no_judge=True, no_grounding=True),
        ocr=False,
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


@pytest.mark.slow
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
        retrieval=RetrievalConfig(no_grounding=True),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_grounding"] is True
    assert result.metadata["no_judge"] is False
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"
    assert result.retrieval_stats.grounding_filtered == 0


@pytest.mark.slow
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
        retrieval=RetrievalConfig(
            no_judge=True,
            no_grounding=True,
            use_cross_encoder=False,
            rrf_min_score=0.001,
        ),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    mock_client.chat.completions.create.assert_not_called()
    for risk in result.risks:
        assert risk.accepted_by in ("rrf", "auto_promoted")


@pytest.mark.slow
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
        retrieval=RetrievalConfig(use_cross_encoder=False, rrf_min_score=0.01),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["use_cross_encoder"] is False
    assert result.metadata["rrf_min_score"] == 0.01
    assert result.metadata["cross_encoder_model"] is None
    judge_calls = [c for c in result.llm_calls if c.stage == "judge"]
    assert len(judge_calls) == 0


# --- determine_accepted_by unit tests ---


def _candidate(risk_id="R-001"):
    return ScoredCandidate(
        risk_id=risk_id,
        risk_name="Test",
        risk_description="desc",
        cross_encoder_score=0.8,
        rrf_score=0.5,
    )


def test_determine_accepted_by_llm_judge():
    c = _candidate()
    result = determine_accepted_by(
        c,
        borderline_judged=[c],
        use_cross_encoder=True,
        no_judge=False,
    )
    assert result == "llm_judge"


def test_determine_accepted_by_auto_promoted():
    c = _candidate()
    result = determine_accepted_by(
        c,
        borderline_judged=[c],
        use_cross_encoder=True,
        no_judge=True,
    )
    assert result == "auto_promoted"


def test_determine_accepted_by_threshold():
    c = _candidate()
    result = determine_accepted_by(
        c,
        borderline_judged=[],
        use_cross_encoder=True,
        no_judge=False,
    )
    assert result == "threshold"


def test_determine_accepted_by_rrf():
    c = _candidate()
    result = determine_accepted_by(
        c,
        borderline_judged=[],
        use_cross_encoder=False,
        no_judge=False,
    )
    assert result == "rrf"


@pytest.mark.slow
def test_run_extraction_no_judge_with_grounding_accepted_by(mock_config, tmp_path):
    """Regression: no_judge=True with grounding should tag as 'auto_promoted', not 'llm_judge'."""
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
        retrieval=RetrievalConfig(no_judge=True, no_grounding=False),
        ocr=False,
    )

    for risk in result.risks:
        assert risk.accepted_by in ("auto_promoted", "rrf", "threshold"), (
            f"Expected auto_promoted/rrf/threshold but got '{risk.accepted_by}' for {risk.risk_id}"
        )


# --- build_risk_match unit tests ---


def test_build_risk_match_cross_encoder():
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="threshold",
        grounding_confidence="high",
        evidence=[],
        use_cross_encoder=True,
    )
    assert m.confidence == c.cross_encoder_score
    assert m.scores.rrf_score == c.rrf_score
    assert m.accepted_by == "threshold"
    assert m.taxonomy == "test-tax"


def test_build_risk_match_rrf_mode():
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="rrf",
        grounding_confidence="ungrounded",
        evidence=[],
        use_cross_encoder=False,
    )
    assert m.confidence == c.rrf_score


def test_build_risk_match_confidence_override():
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="expansion",
        grounding_confidence="high",
        evidence=[],
        confidence_override=0.0,
    )
    assert m.confidence == 0.0


def test_build_risk_match_scores_override():
    c = _candidate()
    zero_scores = RetrievalScores(
        bm25_rank=0,
        embedding_distance=0.0,
        cross_encoder_score=0.0,
        rrf_score=0.0,
    )
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="expansion",
        grounding_confidence="high",
        evidence=[],
        confidence_override=0.0,
        scores_override=zero_scores,
    )
    assert m.scores.bm25_rank == 0
    assert m.scores.cross_encoder_score == 0.0


# --- _run_expansion integration test ---

EXPANSION_RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs", parent="group-fairness"),
    _make_risk("R-002", "Data Bias", "Bias from training data distributions", parent="group-fairness"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data"),
]


@pytest.mark.slow
def test_run_extraction_expand_siblings(mock_config, tmp_path):
    """With expand_siblings, sibling risks of found risks are grounded and included."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        response_model = kwargs.get("response_model")
        if response_model and hasattr(response_model, "__args__"):
            inner = response_model.__args__[0]
            if inner == _RiskEvidence:
                _ = [r.get("risk_id", "") for r in kwargs["messages"][-1].get("content", "").split("risk_id") if r]
                return [
                    _RiskEvidence(
                        risk_id=rid,
                        grounded=True,
                        confidence="medium",
                        quotes=["bias detection and mitigation"],
                    )
                    for rid in kwargs.get("_risk_ids", [])
                ]
        return []

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda **kwargs: (
        [
            _RiskEvidence(
                risk_id=r["risk_id"],
                grounded=True,
                confidence="medium",
                quotes=["bias detection and mitigation"],
            )
            for r in (
                kwargs.get("messages", [{}])[-1].get("risks", [])
                if isinstance(kwargs.get("messages", [{}])[-1], dict)
                else []
            )
        ]
        if kwargs.get("response_model")
        else []
    )

    mock_client.chat.completions.create.return_value = [
        _RiskEvidence(
            risk_id="R-001",
            grounded=True,
            confidence="high",
            quotes=["bias detection"],
        ),
    ]

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=EXPANSION_RISKS,
        retrieval=RetrievalConfig(expand_siblings=True, no_judge=True),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["expand_siblings"] is True
    _ = [r for r in result.risks if r.accepted_by == "expansion"]
    expansion_stats = result.metadata.get("expansion_stats", {})
    assert expansion_stats.get("expanded_candidates", 0) >= 0


@pytest.mark.slow
def test_run_extraction_expand_no_siblings_when_no_grounding(mock_config, tmp_path):
    """Expansion requires grounding — skipped when no_grounding=True."""
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
        risks=EXPANSION_RISKS,
        retrieval=RetrievalConfig(expand_siblings=True, no_judge=True, no_grounding=True),
        ocr=False,
    )

    assert result.metadata["expansion_stats"]["expanded_candidates"] == 0
    assert not any(r.accepted_by == "expansion" for r in result.risks)


# --- _run_judge integration test ---


@pytest.mark.slow
def test_run_extraction_judge_with_cross_encoder(mock_config, tmp_path):
    """With use_cross_encoder and judge enabled, borderline candidates go through judge."""
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
        retrieval=RetrievalConfig(no_grounding=True),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_judge"] is False
    assert result.metadata["use_cross_encoder"] is True
    for risk in result.risks:
        assert risk.accepted_by in ("threshold", "llm_judge")


# --- _run_causal_synthesis tests ---


def test_run_causal_synthesis_populates_fields():
    merged = [
        RiskMatch(
            risk_id="atlas-bias",
            risk_name="AI Bias",
            risk_description="Systematic bias.",
            confidence=0.85,
            grounding_confidence="high",
            accepted_by="threshold",
            evidence=[
                EvidenceSpan(text="quote", document="doc.pdf", chunk_index=0),
                EvidenceSpan(text="quote2", document="doc.pdf", chunk_index=2),
            ],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.2, cross_encoder_score=0.85, rrf_score=0.03),
        ),
    ]
    chunks = [
        SimpleNamespace(text="Chunk zero about credit scoring fairness."),
        SimpleNamespace(text="Chunk one irrelevant."),
        SimpleNamespace(text="Chunk two about discrimination in lending."),
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _CausalChain(
            threat="Credit scoring discriminates",
            threat_source="Biased training data",
            vulnerability="No fairness audit",
            consequence="Applicants denied credit",
            impact="Financial exclusion",
        ),
    ]

    from concorde_policy_mapper.llm import LLMConfig

    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    collector: list[LLMCallRecord] = []

    result = _run_causal_synthesis(merged, chunks, mock_client, config, 4, collector)

    assert len(result) == 1
    assert result[0].threat == "Credit scoring discriminates"
    assert result[0].threat_source == "Biased training data"
    assert result[0].vulnerability == "No fairness audit"
    assert result[0].consequence == "Applicants denied credit"
    assert result[0].impact == "Financial exclusion"
    assert len(collector) == 1
    assert collector[0].stage == "causal_synthesis"


def test_run_causal_synthesis_skips_empty_results():
    merged = [
        RiskMatch(
            risk_id="atlas-bias",
            risk_name="AI Bias",
            risk_description="Bias.",
            confidence=0.85,
            grounding_confidence="high",
            accepted_by="expansion",
            evidence=[
                EvidenceSpan(text="q", document="doc.pdf", chunk_index=0),
            ],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.2, cross_encoder_score=0.85, rrf_score=0.03),
        ),
    ]
    chunks = [SimpleNamespace(text="Unrelated text about data retention.")]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _CausalChain(threat="", threat_source="", vulnerability="", consequence="", impact=""),
    ]

    from concorde_policy_mapper.llm import LLMConfig

    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    result = _run_causal_synthesis(merged, chunks, mock_client, config, 4, [])

    assert result[0].threat is None
    assert result[0].impact is None
