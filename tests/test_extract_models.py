from concorde_policy_mapper.extract.models import (
    ChunkSummary,
    EvidenceSpan,
    ExtractionResult,
    LLMCallRecord,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
    ScoredCandidate,
    _CausalChain,
)


def test_evidence_span_defaults():
    span = EvidenceSpan(
        text="AI systems must be monitored.",
        document="policy.pdf",
        chunk_index=0,
        sentence_index=2,
        cross_encoder_score=0.85,
    )
    assert span.page is None
    assert span.section is None


def test_evidence_span_with_provenance():
    span = EvidenceSpan(
        text="AI systems must be monitored.",
        document="policy.pdf",
        page=3,
        section="Section 4: Oversight",
        chunk_index=1,
        sentence_index=0,
        cross_encoder_score=0.92,
    )
    assert span.page == 3
    assert span.section == "Section 4: Oversight"


def test_scored_candidate():
    c = ScoredCandidate(
        risk_id="atlas-R-0042",
        risk_name="Discriminatory Outcomes",
        risk_description="AI system produces biased results.",
    )
    assert c.bm25_rank == 0
    assert c.embedding_distance == 0.0
    assert c.cross_encoder_score == 0.0
    assert c.rrf_score == 0.0


def test_risk_match_serialization():
    match = RiskMatch(
        risk_id="atlas-R-0042",
        risk_name="Discriminatory Outcomes",
        risk_description="AI system produces biased results.",
        confidence=0.85,
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text="The system may produce different outcomes.",
                document="policy.pdf",
                chunk_index=0,
                sentence_index=1,
                cross_encoder_score=0.85,
            )
        ],
        scores=RetrievalScores(
            bm25_rank=3,
            embedding_distance=0.25,
            cross_encoder_score=0.85,
            rrf_score=0.034,
        ),
    )
    d = match.model_dump()
    assert d["risk_id"] == "atlas-R-0042"
    assert len(d["evidence"]) == 1
    assert d["scores"]["bm25_rank"] == 3


def test_chunk_summary_defaults():
    cs = ChunkSummary(
        index=0,
        source="policy.pdf",
        text_preview="AI systems must be monitored...",
        candidates_retrieved=50,
        auto_accepted=3,
        borderline=7,
        discarded=40,
    )
    assert cs.page is None
    assert cs.section is None
    assert cs.accepted_risk_ids == []


def test_chunk_summary_with_all_fields():
    cs = ChunkSummary(
        index=2,
        source="policy.pdf",
        page=5,
        section="Section 3",
        text_preview="Some policy text...",
        candidates_retrieved=30,
        auto_accepted=2,
        borderline=5,
        discarded=23,
        accepted_risk_ids=["R-001", "R-002"],
    )
    assert cs.page == 5
    assert cs.accepted_risk_ids == ["R-001", "R-002"]


def test_llm_call_record():
    record = LLMCallRecord(
        call_id="judge-001",
        stage="judge",
        chunk_index=3,
        risk_ids=["R-001", "R-002"],
        messages=[{"role": "user", "content": "Judge these risks"}],
        response={"verdicts": []},
        duration_ms=150.5,
        result_summary="0/2 accepted",
    )
    d = record.model_dump()
    assert d["call_id"] == "judge-001"
    assert d["stage"] == "judge"
    assert len(d["risk_ids"]) == 2


def test_llm_call_record_defaults():
    record = LLMCallRecord(
        call_id="ground-001",
        stage="grounding",
        chunk_index=0,
        risk_ids=["R-001"],
        messages=[],
        response="",
        duration_ms=0,
        result_summary="",
    )
    assert record.stage == "grounding"


def test_extraction_result_defaults():
    result = ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=5,
            total_candidates_retrieved=120,
            auto_accepted=8,
            llm_judged=3,
            grounding_filtered=1,
        ),
    )
    assert result.version == "0.3"
    assert result.token_usage == {}
    assert result.metadata == {}
    assert result.retrieval_stats.timing_ms == {}
    assert result.chunks == []
    assert result.llm_calls == []
    assert result.eval is None


def test_extraction_result_v03_defaults():
    result = ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=5,
            total_candidates_retrieved=120,
            auto_accepted=8,
            llm_judged=3,
            grounding_filtered=1,
        ),
    )
    assert result.version == "0.3"
    assert result.chunks == []
    assert result.llm_calls == []
    assert result.eval is None


def test_extraction_result_v03_with_chunks_and_calls():
    result = ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=1,
            total_candidates_retrieved=10,
            auto_accepted=1,
            llm_judged=0,
            grounding_filtered=0,
        ),
        chunks=[
            ChunkSummary(
                index=0,
                source="policy.pdf",
                text_preview="Some text...",
                candidates_retrieved=10,
                auto_accepted=1,
                borderline=2,
                discarded=7,
                accepted_risk_ids=["R-001"],
            )
        ],
        llm_calls=[
            LLMCallRecord(
                call_id="ground-001",
                stage="grounding",
                chunk_index=0,
                risk_ids=["R-001"],
                messages=[{"role": "user", "content": "Ground this"}],
                response={"grounded": True},
                duration_ms=200,
                result_summary="1/1 grounded",
            )
        ],
        eval={"precision": 0.9, "recall": 0.8, "f1": 0.85, "pass": True},
    )
    assert len(result.chunks) == 1
    assert len(result.llm_calls) == 1
    assert result.eval["f1"] == 0.85


def test_extraction_result_v03_serialization_roundtrip():
    result = ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=0,
            total_candidates_retrieved=0,
            auto_accepted=0,
            llm_judged=0,
            grounding_filtered=0,
        ),
        chunks=[],
        llm_calls=[],
    )
    d = result.model_dump()
    restored = ExtractionResult(**d)
    assert restored.version == "0.3"
    assert restored.chunks == []
    assert restored.llm_calls == []
    assert restored.eval is None


def test_causal_chain_model():
    chain = _CausalChain(
        threat="Automated profiling leads to discriminatory lending decisions",
        threat_source="AI-driven credit scoring systems",
        vulnerability="No human review of automated decisions",
        consequence="Qualified applicants denied credit based on protected characteristics",
        impact="Systemic financial exclusion of marginalized communities",
    )
    assert chain.threat.startswith("Automated")
    assert chain.impact.startswith("Systemic")
    data = chain.model_dump()
    assert set(data.keys()) == {"threat", "threat_source", "vulnerability", "consequence", "impact"}


def test_llm_call_record_causal_synthesis_stage():
    record = LLMCallRecord(
        call_id="causal-001",
        stage="causal_synthesis",
        chunk_index=3,
        risk_ids=["atlas-bias"],
        messages=[{"role": "user", "content": "test"}],
        response={"threat": "test"},
        duration_ms=100.0,
        result_summary="synthesized",
    )
    assert record.stage == "causal_synthesis"
