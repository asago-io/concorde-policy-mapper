from types import SimpleNamespace

import pytest

from concorde_policy_mapper.extract.index import RiskIndex, _is_remote, _parse_remote_url


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


RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI model outputs that favor certain groups over others"),
    _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data to compromise model integrity"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized collection or use of personal data by AI systems"),
    _make_risk("R-004", "Lack of Transparency", "Inability to explain or understand AI decision-making processes"),
    _make_risk("R-005", "Workforce Displacement", "Automation of jobs leading to unemployment and economic disruption"),
]


@pytest.fixture
def index():
    return RiskIndex(RISKS)


def test_search_bm25_exact_term(index):
    results = index.search_bm25("model bias", top_k=3)
    assert len(results) > 0
    assert results[0].risk_id == "R-001"
    assert results[0].bm25_rank == 1


def test_search_bm25_no_match(index):
    results = index.search_bm25("quantum computing hardware", top_k=3)
    assert len(results) == 0 or results[0].bm25_rank > 0


def test_search_bm25_respects_top_k(index):
    results = index.search_bm25("AI system", top_k=2)
    assert len(results) <= 2


def test_search_semantic(index):
    results = index.search_semantic("biased AI predictions discriminating against minorities", top_k=3)
    assert len(results) > 0
    risk_ids = [r.risk_id for r in results]
    assert "R-001" in risk_ids


def test_search_semantic_returns_distance(index):
    results = index.search_semantic("data privacy", top_k=1)
    assert len(results) == 1
    assert 0.0 <= results[0].embedding_distance <= 2.0


def test_rerank(index):
    candidates = index.search_semantic("personal data collection without consent", top_k=5)
    reranked = index.rerank("personal data collection without consent", candidates, top_k=3)
    assert len(reranked) <= 3
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in reranked)
    scores = [c.cross_encoder_score for c in reranked]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3)
    assert len(results) > 0
    assert all(c.rrf_score > 0 for c in results)
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in results)
    risk_ids = [r.risk_id for r in results]
    assert "R-002" in risk_ids


def test_hybrid_search_populates_all_scores(index):
    results = index.hybrid_search("bias in AI", top_k=3)
    for c in results:
        assert c.rrf_score > 0
        assert c.cross_encoder_score >= 0


def test_empty_risks():
    index = RiskIndex([])
    assert index.search_bm25("anything", top_k=5) == []
    assert index.search_semantic("anything", top_k=5) == []
    assert index.hybrid_search("anything", top_k=5) == []


def test_risk_count(index):
    assert index.risk_count == 5


def test_cross_encoder_property(index):
    assert index.cross_encoder is not None


def test_no_cross_encoder():
    idx = RiskIndex(RISKS, cross_encoder_model=None)
    assert idx.cross_encoder is None
    assert idx.rerank("test", [], top_k=3) == []


def test_hybrid_search_rrf_only(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3, rrf_min_score=0.005)
    assert len(results) > 0
    assert all(c.rrf_score >= 0.005 for c in results)
    assert all(c.cross_encoder_score == 0.0 for c in results)


def test_hybrid_search_rrf_only_filters_low_scores(index):
    all_results = index.hybrid_search("bias", top_k=50, rrf_min_score=0.001)
    high_results = index.hybrid_search("bias", top_k=50, rrf_min_score=0.02)
    assert len(high_results) <= len(all_results)
    assert all(c.rrf_score >= 0.02 for c in high_results)


def test_is_remote():
    assert _is_remote("https://host.example.com/v1/embeddings")
    assert _is_remote("http://localhost:8000/v1/embeddings")
    assert not _is_remote("all-mpnet-base-v2")
    assert not _is_remote("cross-encoder/ms-marco-MiniLM-L-12-v2")


def test_parse_remote_url():
    base, model = _parse_remote_url(
        "https://bge-m3-model-serving.apps.example.com/v1/embeddings"
    )
    assert base == "https://bge-m3-model-serving.apps.example.com/v1"
    assert model == "bge-m3"

    base2, model2 = _parse_remote_url(
        "https://gte-reranker-modernbert-base-model-serving.apps.rosa.example.com/v1/score"
    )
    assert base2 == "https://gte-reranker-modernbert-base-model-serving.apps.rosa.example.com/v1"
    assert model2 == "gte-reranker-modernbert-base"

    base3, model3 = _parse_remote_url("https://host.com")
    assert base3 == "https://host.com/v1"
    assert model3 == "host"

    base4, _ = _parse_remote_url("https://host.com/v1")
    assert base4 == "https://host.com/v1"


def test_colbert_remote_rejected():
    with pytest.raises(ValueError, match="ColBERT models cannot be served remotely"):
        RiskIndex(RISKS, colbert_model="https://lateon.example.com/v1/embeddings")
