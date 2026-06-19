from asago_policy_mapper.extract.merge import merge_matches
from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    RetrievalScores,
    RiskMatch,
)


def _make_match(risk_id, confidence, doc="a.pdf", chunk_index=0, sentence_index=0):
    return RiskMatch(
        risk_id=risk_id,
        risk_name=f"Risk {risk_id}",
        risk_description=f"Description of {risk_id}.",
        confidence=confidence,
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text=f"Evidence for {risk_id} in {doc} chunk {chunk_index}.",
                document=doc,
                chunk_index=chunk_index,
                sentence_index=sentence_index,
                cross_encoder_score=confidence,
            )
        ],
        scores=RetrievalScores(
            bm25_rank=1,
            embedding_distance=0.2,
            cross_encoder_score=confidence,
            rrf_score=0.03,
        ),
    )


def test_merge_same_risk_across_chunks():
    matches = [
        _make_match("R-001", 0.8, chunk_index=0),
        _make_match("R-001", 0.9, chunk_index=2),
        _make_match("R-002", 0.7, chunk_index=1),
    ]
    merged = merge_matches(matches, max_evidence=3)
    assert len(merged) == 2

    r001 = next(m for m in merged if m.risk_id == "R-001")
    assert r001.confidence == 0.9
    assert len(r001.evidence) == 2
    assert r001.evidence[0].cross_encoder_score >= r001.evidence[1].cross_encoder_score


def test_merge_same_risk_across_documents():
    matches = [
        _make_match("R-001", 0.8, doc="a.pdf"),
        _make_match("R-001", 0.75, doc="b.pdf"),
    ]
    merged = merge_matches(matches, max_evidence=3)
    assert len(merged) == 1
    r = merged[0]
    assert r.confidence == 0.8
    docs = {e.document for e in r.evidence}
    assert docs == {"a.pdf", "b.pdf"}


def test_merge_caps_evidence_count():
    matches = [_make_match("R-001", 0.5 + i * 0.1, chunk_index=i) for i in range(5)]
    merged = merge_matches(matches, max_evidence=3)
    assert len(merged[0].evidence) == 3


def test_merge_deduplicates_identical_text():
    m1 = _make_match("R-001", 0.8, chunk_index=0)
    m2 = _make_match("R-001", 0.85, chunk_index=0)
    merged = merge_matches([m1, m2], max_evidence=3)
    assert len(merged[0].evidence) == 1
    assert merged[0].confidence == 0.85


def test_merge_empty_input():
    assert merge_matches([], max_evidence=3) == []
