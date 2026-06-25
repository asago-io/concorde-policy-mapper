from unittest.mock import MagicMock

from asago_policy_mapper.extract.models import LLMCallRecord, ScoredCandidate, _JudgeVerdict, _JudgeVerdicts
from asago_policy_mapper.extract.parse import Chunk
from asago_policy_mapper.extract.retrieve import (
    build_chunk_contexts,
    build_padded_text,
    classify_by_rank,
    classify_by_threshold,
    classify_candidates,
    judge_borderline,
)


def test_build_padded_text_middle_chunk():
    chunks = [
        Chunk(text="First chunk. It has two sentences.", source="a.pdf", index=0),
        Chunk(text="Middle chunk with content.", source="a.pdf", index=1),
        Chunk(text="Last chunk. Final sentence here.", source="a.pdf", index=2),
    ]
    padded = build_padded_text(chunks, chunk_index=1, context_sentences=2)
    assert "First chunk." in padded or "It has two sentences." in padded
    assert "Middle chunk with content." in padded
    assert "Last chunk." in padded or "Final sentence here." in padded


def test_build_padded_text_first_chunk():
    chunks = [
        Chunk(text="First chunk only.", source="a.pdf", index=0),
        Chunk(text="Second chunk text.", source="a.pdf", index=1),
    ]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert "First chunk only." in padded
    assert "Second chunk text." in padded


def test_build_padded_text_last_chunk():
    chunks = [
        Chunk(text="First chunk. With detail.", source="a.pdf", index=0),
        Chunk(text="Last chunk alone.", source="a.pdf", index=1),
    ]
    padded = build_padded_text(chunks, chunk_index=1, context_sentences=2)
    assert "Last chunk alone." in padded
    assert "First chunk." in padded or "With detail." in padded


def test_build_padded_text_single_chunk():
    chunks = [Chunk(text="Only chunk.", source="a.pdf", index=0)]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert padded == "Only chunk."


def test_build_padded_text_no_cross_document():
    chunks = [
        Chunk(text="Doc A last chunk.", source="a.pdf", index=0),
        Chunk(text="Doc B first chunk.", source="b.pdf", index=0),
    ]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert "Doc B first chunk." not in padded


def test_classify_candidates_rank_based():
    """Rank-based mode: top N accepted, next M borderline, rest discarded."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-004", risk_name="D", risk_description="d", cross_encoder_score=0.30),
        ScoredCandidate(risk_id="R-005", risk_name="E", risk_description="e", cross_encoder_score=0.10),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=2,
        top_n_judge=2,
    )
    assert [c.risk_id for c in accepted] == ["R-001", "R-002"]
    assert [c.risk_id for c in borderline] == ["R-003", "R-004"]
    assert [c.risk_id for c in discarded] == ["R-005"]


def test_classify_candidates_rank_based_floor():
    """Min score floor rejects low-scoring candidates regardless of rank."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.10),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=2,
        top_n_judge=2,
        min_score_floor=0.3,
    )
    assert [c.risk_id for c in accepted] == ["R-001", "R-002"]
    assert len(borderline) == 0
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_candidates_legacy_threshold():
    """Legacy threshold mode when threshold_high is set."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.15),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.3,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_candidates_bm25_rescue():
    """Candidates with strong BM25 rank are rescued to borderline even with low cross-encoder score."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.001, bm25_rank=5),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.001, bm25_rank=15),
        ScoredCandidate(risk_id="R-004", risk_name="D", risk_description="d", cross_encoder_score=0.001, bm25_rank=0),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.15,
        bm25_rescue_rank=10,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert set(c.risk_id for c in discarded) == {"R-003", "R-004"}


def test_classify_candidates_bm25_rescue_disabled():
    """With bm25_rescue_rank=0, no rescue happens."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.001, bm25_rank=5),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.15,
        bm25_rescue_rank=0,
    )
    assert len(accepted) == 0
    assert len(borderline) == 0
    assert len(discarded) == 1


def test_classify_candidates_bm25_rescue_rank_based():
    """BM25 rescue works in rank-based mode too."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.001, bm25_rank=5),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=1,
        top_n_judge=0,
        bm25_rescue_rank=10,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert len(discarded) == 0


def test_judge_borderline_accepts_relevant():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        items=[MagicMock(risk_id="R-001", relevant=True, justification="Text discusses bias.")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Model Bias",
            risk_description="Systematic errors favoring certain groups.",
            cross_encoder_score=0.5,
        ),
    ]
    accepted = judge_borderline(
        candidates,
        chunk_text="The AI system may produce biased outcomes.",
        client=mock_client,
        model="test-model",
    )
    assert len(accepted) == 1
    assert accepted[0].risk_id == "R-001"


def test_judge_borderline_rejects_irrelevant():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        items=[MagicMock(risk_id="R-005", relevant=False, justification="No mention of jobs.")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-005",
            risk_name="Workforce Displacement",
            risk_description="Automation leading to unemployment.",
            cross_encoder_score=0.4,
        ),
    ]
    accepted = judge_borderline(
        candidates,
        chunk_text="The AI system must be transparent.",
        client=mock_client,
        model="test-model",
    )
    assert len(accepted) == 0


def test_judge_borderline_empty_candidates():
    mock_client = MagicMock()
    accepted = judge_borderline([], chunk_text="Text.", client=mock_client, model="m")
    assert accepted == []
    mock_client.chat.completions.create.assert_not_called()


def test_judge_borderline_captures_call():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _JudgeVerdicts(
        items=[_JudgeVerdict(risk_id="R-001", relevant=True, justification="Relevant")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Bias",
            risk_description="Model bias risk",
            cross_encoder_score=0.5,
        ),
        ScoredCandidate(
            risk_id="R-002",
            risk_name="Privacy",
            risk_description="Privacy risk",
            cross_encoder_score=0.4,
        ),
    ]

    collector: list[LLMCallRecord] = []
    result = judge_borderline(
        candidates,
        "Some chunk text about AI bias.",
        mock_client,
        "test-model",
        call_collector=collector,
        chunk_index=3,
    )

    assert len(result) == 1
    assert result[0].risk_id == "R-001"
    assert len(collector) == 1
    assert collector[0].stage == "judge"
    assert collector[0].chunk_index == 3
    assert "R-001" in collector[0].risk_ids
    assert "R-002" in collector[0].risk_ids
    assert collector[0].result_summary == "1/2 accepted"
    assert collector[0].duration_ms >= 0


def test_judge_borderline_no_collector():
    """Existing behavior: no collector arg, no error."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _JudgeVerdicts(items=[])
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Bias",
            risk_description="Model bias risk",
            cross_encoder_score=0.5,
        ),
    ]

    result = judge_borderline(candidates, "Text", mock_client, "test-model")
    assert result == []


def test_retrieve_chunk_no_cross_encoder():
    """With use_cross_encoder=False, all candidates go to accepted, no borderline."""
    from asago_policy_mapper.extract.retrieve import retrieve_chunk

    mock_index = MagicMock()
    mock_index.hybrid_search.return_value = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", rrf_score=0.02),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", rrf_score=0.015),
    ]
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]
    cr = retrieve_chunk(chunks, 0, mock_index, use_cross_encoder=False, rrf_min_score=0.01)
    assert len(cr.accepted) == 2
    assert len(cr.borderline) == 0
    assert len(cr.borderline_judged) == 0
    assert cr.stats["auto_accepted"] == 2
    assert cr.stats["borderline"] == 0
    mock_index.hybrid_search.assert_called_once()
    call_kwargs = mock_index.hybrid_search.call_args
    assert call_kwargs[1]["rrf_min_score"] == 0.01


# --- classify_by_rank / classify_by_threshold direct tests ---


def test_classify_by_rank_accepts_top_n():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.50),
    ]
    accepted, borderline, discarded = classify_by_rank(
        candidates,
        top_n_accept=1,
        top_n_judge=1,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_by_rank_floor_discards():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.80),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.30),
    ]
    accepted, borderline, discarded = classify_by_rank(
        candidates,
        top_n_accept=5,
        top_n_judge=5,
        min_score_floor=0.50,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert len(borderline) == 0
    assert [c.risk_id for c in discarded] == ["R-002"]


def test_classify_by_threshold_accepts_above_high():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.05),
    ]
    accepted, borderline, discarded = classify_by_threshold(
        candidates,
        threshold_high=0.80,
        threshold_low=0.20,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_by_threshold_bm25_rescue():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.05, bm25_rank=2),
    ]
    accepted, borderline, discarded = classify_by_threshold(
        candidates,
        threshold_high=0.80,
        threshold_low=0.20,
        bm25_rescue_rank=5,
    )
    assert len(accepted) == 0
    assert [c.risk_id for c in borderline] == ["R-001"]
    assert len(discarded) == 0


# --- build_chunk_contexts tests ---


def test_build_chunk_contexts_expands_both_neighbors():
    chunks = [
        Chunk(text="previous chunk text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next chunk text", source="a.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "previous chunk text" in contexts[1]
    assert "core chunk text" in contexts[1]
    assert "next chunk text" in contexts[1]


def test_build_chunk_contexts_no_budget():
    chunks = [
        Chunk(text="previous text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next text", source="a.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=3)
    assert contexts[1] == "core chunk text"


def test_build_chunk_contexts_different_source_skipped():
    chunks = [
        Chunk(text="prev from other doc", source="b.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next from other doc", source="c.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert contexts[1] == "core chunk text"


def test_build_chunk_contexts_first_chunk():
    chunks = [
        Chunk(text="core chunk", source="a.pdf", index=0),
        Chunk(text="next chunk", source="a.pdf", index=1),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "core chunk" in contexts[0]
    assert "next chunk" in contexts[0]


def test_build_chunk_contexts_last_chunk():
    chunks = [
        Chunk(text="prev chunk", source="a.pdf", index=0),
        Chunk(text="core chunk", source="a.pdf", index=1),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "prev chunk" in contexts[1]
    assert "core chunk" in contexts[1]


def test_build_chunk_contexts_single_chunk():
    chunks = [Chunk(text="only chunk here", source="a.pdf", index=0)]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert contexts[0] == "only chunk here"


def test_build_chunk_contexts_section_headings():
    """Section headings are inserted at section transitions."""
    chunks = [
        Chunk(text="intro text", source="a.pdf", index=0, section="Introduction"),
        Chunk(text="method text", source="a.pdf", index=1, section="Methods"),
        Chunk(text="more methods", source="a.pdf", index=2, section="Methods"),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    # Context for chunk 1 should have heading when transitioning from Introduction
    assert "intro text" in contexts[1]
    assert "## Methods" in contexts[1]
    # Context for chunk 2 should not repeat section heading (same section as chunk 1)
    assert "method text" in contexts[2]
    assert contexts[2].count("## Methods") <= 1


def test_build_chunk_contexts_symmetric_expansion():
    """Expansion alternates between before and after."""
    chunks = [
        Chunk(text="chunk zero with some words", source="a.pdf", index=0),
        Chunk(text="chunk one with some words", source="a.pdf", index=1),
        Chunk(text="core chunk", source="a.pdf", index=2),
        Chunk(text="chunk three with some words", source="a.pdf", index=3),
        Chunk(text="chunk four with some words", source="a.pdf", index=4),
    ]
    # Budget for 3 neighbors (core=2 words, each neighbor=5 words, budget=17 words)
    contexts = build_chunk_contexts(chunks, max_context_tokens=17)
    ctx = contexts[2]
    assert "core chunk" in ctx
    assert "chunk one with some words" in ctx
    assert "chunk three with some words" in ctx
