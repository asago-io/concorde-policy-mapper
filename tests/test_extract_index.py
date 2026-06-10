from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from concorde_policy_mapper.extract.index import (
    RiskIndex,
    _is_remote,
    _make_score_normalizer,
    _parse_remote_url,
    _RemoteBiEncoder,
    _RemoteCrossEncoder,
    _rrf_fuse,
)
from concorde_policy_mapper.extract.models import ScoredCandidate


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


@pytest.mark.slow
def test_search_bm25_exact_term(index):
    results = index.search_bm25("model bias", top_k=3)
    assert len(results) > 0
    assert results[0].risk_id == "R-001"
    assert results[0].bm25_rank == 1


@pytest.mark.slow
def test_search_bm25_no_match(index):
    results = index.search_bm25("quantum computing hardware", top_k=3)
    assert len(results) == 0 or results[0].bm25_rank > 0


@pytest.mark.slow
def test_search_bm25_respects_top_k(index):
    results = index.search_bm25("AI system", top_k=2)
    assert len(results) <= 2


@pytest.mark.slow
def test_search_semantic(index):
    results = index.search_semantic("biased AI predictions discriminating against minorities", top_k=3)
    assert len(results) > 0
    risk_ids = [r.risk_id for r in results]
    assert "R-001" in risk_ids


@pytest.mark.slow
def test_search_semantic_returns_distance(index):
    results = index.search_semantic("data privacy", top_k=1)
    assert len(results) == 1
    assert 0.0 <= results[0].embedding_distance <= 2.0


@pytest.mark.slow
def test_rerank(index):
    candidates = index.search_semantic("personal data collection without consent", top_k=5)
    reranked = index.rerank("personal data collection without consent", candidates, top_k=3)
    assert len(reranked) <= 3
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in reranked)
    scores = [c.cross_encoder_score for c in reranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_hybrid_search(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3)
    assert len(results) > 0
    assert all(c.rrf_score > 0 for c in results)
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in results)
    risk_ids = [r.risk_id for r in results]
    assert "R-002" in risk_ids


@pytest.mark.slow
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


@pytest.mark.slow
def test_risk_count(index):
    assert index.risk_count == 5


@pytest.mark.slow
def test_cross_encoder_property(index):
    assert index.cross_encoder is not None


@pytest.mark.slow
def test_no_cross_encoder():
    idx = RiskIndex(RISKS, cross_encoder_model=None)
    assert idx.cross_encoder is None
    assert idx.rerank("test", [], top_k=3) == []


@pytest.mark.slow
def test_hybrid_search_rrf_only(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3, rrf_min_score=0.005)
    assert len(results) > 0
    assert all(c.rrf_score >= 0.005 for c in results)
    assert all(c.cross_encoder_score == 0.0 for c in results)


@pytest.mark.slow
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
    base, model = _parse_remote_url("https://bge-m3-model-serving.apps.example.com/v1/embeddings")
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


# --- _rrf_fuse tests ---


def _sc(risk_id, ce_score=0.0, embed_dist=0.0, bm25_rank=0):
    return ScoredCandidate(
        risk_id=risk_id,
        risk_name=risk_id,
        risk_description="",
        cross_encoder_score=ce_score,
        embedding_distance=embed_dist,
        bm25_rank=bm25_rank,
    )


def test_rrf_fuse_basic():
    list_a = [_sc("R-001"), _sc("R-002")]
    list_b = [_sc("R-002", embed_dist=0.3), _sc("R-003", embed_dist=0.5)]
    rrf_scores, candidate_data, bm25_ranks = _rrf_fuse(list_a, list_b, rrf_k=60)
    assert "R-001" in rrf_scores
    assert "R-002" in rrf_scores
    assert "R-003" in rrf_scores
    assert rrf_scores["R-002"] > rrf_scores["R-001"]
    assert bm25_ranks["R-001"] == 1
    assert bm25_ranks["R-002"] == 2
    assert "R-003" not in bm25_ranks
    assert candidate_data["R-001"].risk_id == "R-001"


def test_rrf_fuse_first_occurrence_wins():
    list_a = [_sc("R-001", ce_score=0.9)]
    list_b = [_sc("R-001", ce_score=0.1)]
    _, candidate_data, _ = _rrf_fuse(list_a, list_b)
    assert candidate_data["R-001"].cross_encoder_score == 0.9


# --- _make_score_normalizer tests ---


def test_score_normalizer_clip():
    norm = _make_score_normalizer(is_nli=False, apply_sigmoid=False)
    raw = np.array([0.5, 1.5, -0.3])
    result = norm(raw)
    np.testing.assert_array_almost_equal(result, [0.5, 1.0, 0.0])


def test_score_normalizer_sigmoid():
    norm = _make_score_normalizer(is_nli=False, apply_sigmoid=True)
    raw = np.array([0.0])
    result = norm(raw)
    assert abs(result[0] - 0.5) < 1e-6


def test_score_normalizer_nli_2d():
    norm = _make_score_normalizer(is_nli=True, apply_sigmoid=False)
    raw = np.array([[1.0, 2.0, 3.0]])
    result = norm(raw)
    assert result.shape == (1,)
    assert result[0] > 0.5


# --- _RemoteBiEncoder tests ---


@patch("openai.OpenAI")
def test_remote_bi_encoder_encode(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3]),
            SimpleNamespace(index=1, embedding=[0.4, 0.5, 0.6]),
        ]
    )

    encoder = _RemoteBiEncoder("https://bge-m3-model-serving.apps.example.com/v1/embeddings")
    result = encoder.encode(["hello", "world"], normalize=True)

    assert result.shape == (2, 3)
    # Verify rows are L2-normalized (unit vectors)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
    mock_client.embeddings.create.assert_called_once()


@patch("openai.OpenAI")
def test_remote_bi_encoder_encode_batches(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # First batch: 2 texts
    batch1_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        ]
    )
    # Second batch: 1 text
    batch2_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 1.0]),
        ]
    )
    mock_client.embeddings.create.side_effect = [batch1_response, batch2_response]

    encoder = _RemoteBiEncoder(
        "https://bge-m3-model-serving.apps.example.com/v1/embeddings",
        batch_size=2,
    )
    result = encoder.encode(["a", "b", "c"], normalize=True)

    assert result.shape == (3, 2)
    assert mock_client.embeddings.create.call_count == 2
    # Verify first call got batch of 2, second call got batch of 1
    calls = mock_client.embeddings.create.call_args_list
    assert calls[0].kwargs["input"] == ["a", "b"]
    assert calls[1].kwargs["input"] == ["c"]


# --- _RemoteCrossEncoder tests ---


@patch("httpx.Client")
def test_remote_cross_encoder_predict(mock_httpx_client_cls):
    mock_client = MagicMock()
    mock_httpx_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "score": 0.42},
            {"index": 0, "score": 0.85},
        ]
    }
    mock_client.post.return_value = mock_response

    encoder = _RemoteCrossEncoder("https://gte-reranker-model-serving.apps.example.com/v1/score")
    pairs = [("risk desc A", "chunk text"), ("risk desc B", "chunk text")]
    result = encoder.predict(pairs)

    assert result.shape == (2,)
    # Results should be sorted by index: index 0 -> 0.85, index 1 -> 0.42
    np.testing.assert_allclose(result, [0.85, 0.42])
    mock_client.post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


@patch("httpx.Client")
def test_remote_cross_encoder_predict_empty(mock_httpx_client_cls):
    mock_client = MagicMock()
    mock_httpx_client_cls.return_value = mock_client

    encoder = _RemoteCrossEncoder("https://gte-reranker-model-serving.apps.example.com/v1/score")
    result = encoder.predict([])

    assert result.shape == (0,)
    mock_client.post.assert_not_called()
