from unittest.mock import MagicMock

from concorde_policy_mapper.extract.models import LLMCallRecord, ScoredCandidate, _JudgeVerdict
from concorde_policy_mapper.extract.parse import Chunk
from concorde_policy_mapper.extract.retrieve import (
    _pad_with_budget,
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
    mock_client.chat.completions.create.return_value = [
        MagicMock(risk_id="R-001", relevant=True, justification="Text discusses bias."),
    ]
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
    mock_client.chat.completions.create.return_value = [
        MagicMock(risk_id="R-005", relevant=False, justification="No mention of jobs."),
    ]
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
    mock_client.chat.completions.create.return_value = [
        _JudgeVerdict(risk_id="R-001", relevant=True, justification="Relevant"),
    ]
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
    mock_client.chat.completions.create.return_value = []
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
    from concorde_policy_mapper.extract.retrieve import retrieve_chunk

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


# --- _pad_with_budget tests ---


def test_pad_with_budget_includes_full_prev_and_next():
    """Budget large enough for both neighbors: output includes all three chunks."""
    chunks = [
        Chunk(text="previous chunk text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next chunk text", source="a.pdf", index=2),
    ]
    # core=3 words, prev=3, next=3 → need max_tokens >= 9
    result = _pad_with_budget(chunks, chunk_index=1, max_tokens=100)
    assert "previous chunk text" in result
    assert "core chunk text" in result
    assert "next chunk text" in result


def test_pad_with_budget_no_budget():
    """max_tokens equals core chunk word count: no room for neighbors."""
    chunks = [
        Chunk(text="previous text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next text", source="a.pdf", index=2),
    ]
    # core is 3 words, budget = 3 - 3 = 0
    result = _pad_with_budget(chunks, chunk_index=1, max_tokens=3)
    assert result == "core chunk text"


def test_pad_with_budget_prev_exceeds_half_budget():
    """Prev chunk too large for half budget: takes sentences in reverse order."""
    chunks = [
        Chunk(text="First sentence of prev. Second sentence of prev. Third sentence of prev.", source="a.pdf", index=0),
        Chunk(text="core", source="a.pdf", index=1),
        Chunk(text="next", source="a.pdf", index=2),
    ]
    # core=1 word, budget=6-1=5, half=2
    # prev has 3 sentences (4, 4, 4 words each) — all exceed half budget (2)
    # next=1 word, fits remaining budget
    result = _pad_with_budget(chunks, chunk_index=1, max_tokens=6)
    assert "First sentence of prev." not in result
    assert "core" in result
    assert "next" in result


def test_pad_with_budget_next_exceeds_budget():
    """Next chunk too large for remaining budget: takes sentences in forward order."""
    chunks = [
        Chunk(text="core", source="a.pdf", index=0),
        Chunk(text="Short next. A much longer sentence that will not fit in the budget.", source="a.pdf", index=1),
    ]
    # core=1 word, budget=4-1=3
    # No prev (index 0). Next sentences: "Short next."=2 words (fits), long=11 words (won't fit)
    result = _pad_with_budget(chunks, chunk_index=0, max_tokens=4)
    assert "core" in result
    assert "Short next." in result
    assert "A much longer sentence" not in result


def test_pad_with_budget_different_source_skipped():
    """Prev and next from different source: only core chunk returned."""
    chunks = [
        Chunk(text="prev from other doc", source="b.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next from other doc", source="c.pdf", index=2),
    ]
    result = _pad_with_budget(chunks, chunk_index=1, max_tokens=100)
    assert result == "core chunk text"


def test_pad_with_budget_first_chunk():
    """chunk_index=0: no prev available, only core + next."""
    chunks = [
        Chunk(text="core chunk", source="a.pdf", index=0),
        Chunk(text="next chunk", source="a.pdf", index=1),
    ]
    result = _pad_with_budget(chunks, chunk_index=0, max_tokens=100)
    assert "core chunk" in result
    assert "next chunk" in result


def test_pad_with_budget_last_chunk():
    """Last chunk: no next available, only prev + core."""
    chunks = [
        Chunk(text="prev chunk", source="a.pdf", index=0),
        Chunk(text="core chunk", source="a.pdf", index=1),
    ]
    result = _pad_with_budget(chunks, chunk_index=1, max_tokens=100)
    assert "prev chunk" in result
    assert "core chunk" in result


def test_pad_with_budget_single_chunk():
    """Only one chunk in list: returns just core text."""
    chunks = [Chunk(text="only chunk here", source="a.pdf", index=0)]
    result = _pad_with_budget(chunks, chunk_index=0, max_tokens=100)
    assert result == "only chunk here"
