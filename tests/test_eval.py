import json

import pytest

from concorde_policy_mapper.evals.eval import (
    _derive_categories,
    _infer_taxonomy,
    _load_risk_to_category_map,
    _sanitise_risk_id,
    evaluate_extraction,
)
from concorde_policy_mapper.extract.models import (
    ExtractionResult,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
)


@pytest.fixture
def tmp_ground_truth(tmp_path):
    gt = tmp_path / "test-policy.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n  - atlas-transparency\n  - atlas-accountability\n")
    return gt


@pytest.fixture
def tmp_extraction(tmp_path):
    data = {
        "version": "0.2",
        "risks": [
            {
                "risk_id": "atlas-bias",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
            {
                "risk_id": "atlas-privacy",
                "risk_name": "Privacy",
                "risk_description": "",
                "confidence": 0.8,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 2, "embedding_distance": 0.2, "cross_encoder_score": 0.8, "rrf_score": 0.4},
            },
            {
                "risk_id": "atlas-hallucination",
                "risk_name": "Hallucination",
                "risk_description": "",
                "confidence": 0.7,
                "grounding_confidence": "medium",
                "accepted_by": "llm_judge",
                "evidence": [],
                "scores": {"bm25_rank": 3, "embedding_distance": 0.3, "cross_encoder_score": 0.7, "rrf_score": 0.3},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 10,
            "total_candidates_retrieved": 50,
            "auto_accepted": 2,
            "llm_judged": 1,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))
    return ext


def test_evaluate_extraction_metrics(tmp_ground_truth, tmp_extraction):
    result = evaluate_extraction(tmp_ground_truth, tmp_extraction, policy_name="test-policy")

    assert result["policy"] == "test-policy"
    assert result["total_expected"] == 4
    assert result["total_extracted"] == 3
    assert result["matched"] == 2
    assert set(result["matched_ids"]) == {"atlas-bias", "atlas-privacy"}
    assert set(result["missing"]) == {"atlas-transparency", "atlas-accountability"}
    assert result["spurious"] == ["atlas-hallucination"]
    assert result["precision"] == pytest.approx(2 / 3, abs=0.001)
    assert result["recall"] == pytest.approx(2 / 4, abs=0.001)


def test_evaluate_extraction_pass_fail(tmp_ground_truth, tmp_extraction):
    result = evaluate_extraction(tmp_ground_truth, tmp_extraction)
    # recall=0.5 < 0.80, so should fail
    assert result["pass"] is False


def test_evaluate_extraction_perfect_match(tmp_path):
    gt = tmp_path / "perfect.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n")
    data = {
        "version": "0.2",
        "risks": [
            {
                "risk_id": "atlas-bias",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
            {
                "risk_id": "atlas-privacy",
                "risk_name": "Privacy",
                "risk_description": "",
                "confidence": 0.8,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 2, "embedding_distance": 0.2, "cross_encoder_score": 0.8, "rrf_score": 0.4},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 10,
            "total_candidates_retrieved": 50,
            "auto_accepted": 2,
            "llm_judged": 0,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["pass"] is True
    assert result["missing"] == []
    assert result["spurious"] == []


def test_evaluate_extraction_strips_whitespace(tmp_path):
    gt = tmp_path / "ws.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n")
    data = {
        "version": "0.2",
        "risks": [
            {
                "risk_id": "atlas-bias ",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 10,
            "total_candidates_retrieved": 50,
            "auto_accepted": 1,
            "llm_judged": 0,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext)
    assert result["matched"] == 1
    assert result["recall"] == 1.0


def test_evaluate_extraction_custom_thresholds(tmp_ground_truth, tmp_extraction):
    # recall=0.5, precision=0.667 — passes with relaxed thresholds
    result = evaluate_extraction(
        tmp_ground_truth,
        tmp_extraction,
        min_recall=0.4,
        min_precision=0.5,
    )
    assert result["pass"] is True


def test_sanitise_risk_id():
    assert _sanitise_risk_id("atlas-bias") == "atlas-bias"
    assert _sanitise_risk_id("atlas-bias ") == "atlas-bias"
    assert (
        _sanitise_risk_id("ai-risk-taxonomy-not-labeling-content-as-ai-generated Not labeling content as AI-generated")
        == "ai-risk-taxonomy-not-labeling-content-as-ai-generated"
    )


def test_infer_taxonomy():
    assert _infer_taxonomy("atlas-bias") == "ibm-risk-atlas"
    assert _infer_taxonomy("atlas-hallucination") == "ibm-risk-atlas"
    assert _infer_taxonomy("credo-risk-021") == "credo-ucf"
    assert _infer_taxonomy("mit-ai-risk-subdomain-3.1") == "mit-ai-risk-repository"
    assert _infer_taxonomy("nist-data-privacy") == "nist-ai-rmf"
    assert _infer_taxonomy("ai-risk-taxonomy-profiling") == "ai-risk-taxonomy"
    assert _infer_taxonomy("ail-child-exploitation") == "ailuminate-v1.0"
    assert _infer_taxonomy("granite-guardian-harm") == "ibm-granite-guardian"
    assert _infer_taxonomy("llm01-prompt-injection") == "owasp-llm-2.0"
    assert _infer_taxonomy("llm102025-unbounded-consumption") == "owasp-llm-2.0"
    assert _infer_taxonomy("shieldgemma-dangerous-content") == "shieldgemma-taxonomy"
    assert _infer_taxonomy("unknown-risk-id") == "unknown"


def test_evaluate_extraction_per_taxonomy(tmp_path):
    gt = tmp_path / "multi-tax.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n  - nist-data-privacy\n  - credo-risk-021\n")
    data = {
        "version": "0.3",
        "risks": [
            {
                "risk_id": "atlas-bias",
                "taxonomy": "ibm-risk-atlas",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
            {
                "risk_id": "atlas-privacy",
                "taxonomy": "ibm-risk-atlas",
                "risk_name": "Privacy",
                "risk_description": "",
                "confidence": 0.8,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 2, "embedding_distance": 0.2, "cross_encoder_score": 0.8, "rrf_score": 0.4},
            },
            {
                "risk_id": "atlas-hallucination",
                "taxonomy": "ibm-risk-atlas",
                "risk_name": "Hallucination",
                "risk_description": "",
                "confidence": 0.7,
                "grounding_confidence": "medium",
                "accepted_by": "llm_judge",
                "evidence": [],
                "scores": {"bm25_rank": 3, "embedding_distance": 0.3, "cross_encoder_score": 0.7, "rrf_score": 0.3},
            },
            {
                "risk_id": "nist-data-privacy",
                "taxonomy": "nist-ai-rmf",
                "risk_name": "Data Privacy",
                "risk_description": "",
                "confidence": 0.85,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.85, "rrf_score": 0.5},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 10,
            "total_candidates_retrieved": 50,
            "auto_accepted": 3,
            "llm_judged": 1,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext, policy_name="multi-tax")
    assert "per_taxonomy" in result
    pt = result["per_taxonomy"]

    assert "ibm-risk-atlas" in pt
    atlas = pt["ibm-risk-atlas"]
    assert atlas["expected"] == 2
    assert atlas["matched"] == 2
    assert atlas["extracted"] == 3
    assert atlas["precision"] == pytest.approx(2 / 3, abs=0.001)
    assert atlas["recall"] == 1.0

    assert "nist-ai-rmf" in pt
    nist = pt["nist-ai-rmf"]
    assert nist["expected"] == 1
    assert nist["matched"] == 1
    assert nist["precision"] == 1.0
    assert nist["recall"] == 1.0

    assert "credo-ucf" in pt
    credo = pt["credo-ucf"]
    assert credo["expected"] == 1
    assert credo["matched"] == 0
    assert credo["recall"] == 0.0


def test_evaluate_extraction_per_taxonomy_from_filtered(tmp_path):
    """Taxonomy map also reads from grounding_filtered_candidates."""
    gt = tmp_path / "filtered.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n")
    data = {
        "version": "0.3",
        "risks": [],
        "grounding_filtered_candidates": [
            {
                "risk_id": "atlas-bias",
                "taxonomy": "ibm-risk-atlas",
                "risk_name": "Bias",
                "cross_encoder_score": 0.5,
                "accepted_by": "threshold",
                "chunk_index": 0,
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 1,
            "total_candidates_retrieved": 1,
            "auto_accepted": 1,
            "llm_judged": 0,
            "grounding_filtered": 1,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext)
    pt = result["per_taxonomy"]
    assert "ibm-risk-atlas" in pt
    assert pt["ibm-risk-atlas"]["expected"] == 1
    assert pt["ibm-risk-atlas"]["matched"] == 0


def test_per_taxonomy_in_existing_eval(tmp_ground_truth, tmp_extraction):
    """Existing test fixtures now produce per_taxonomy in result."""
    result = evaluate_extraction(tmp_ground_truth, tmp_extraction)
    assert "per_taxonomy" in result
    assert len(result["per_taxonomy"]) > 0


def test_evaluate_extraction_enriched_gt_format(tmp_path, tmp_extraction):
    """Enriched GT format with risks[].id and evidence is parsed correctly."""
    gt = tmp_path / "enriched.yaml"
    gt.write_text(
        "risks:\n"
        "  - id: atlas-bias\n"
        "    name: Bias\n"
        "    evidence:\n"
        '      - text: "The system must avoid bias."\n'
        '        section: "Fairness"\n'
        "  - id: atlas-privacy\n"
        "    name: Privacy\n"
        "    evidence: []\n"
        "  - id: atlas-transparency\n"
        "    name: Transparency\n"
        "    evidence: []\n"
        "  - id: atlas-accountability\n"
        "    name: Accountability\n"
        "    evidence: []\n"
    )
    result = evaluate_extraction(gt, tmp_extraction, policy_name="enriched")
    assert result["total_expected"] == 4
    assert result["matched"] == 2
    assert set(result["matched_ids"]) == {"atlas-bias", "atlas-privacy"}


# --- Category-level eval tests ---


@pytest.fixture
def tmp_sssom(tmp_path):
    """Create a minimal SSSOM mapping file for testing."""
    sssom = tmp_path / "test.sssom.tsv"
    sssom.write_text(
        "# test mapping\n"
        "subject_id\tsubject_source\tpredicate_id\tobject_id\tobject_source\tmapping_justification\n"
        "atlas-bias\tibm-risk-atlas\tskos:broadMatch\tnist-harmful-bias-or-homogenization\tnist-ai-rmf\ttest\n"
        "atlas-privacy\tibm-risk-atlas\tskos:broadMatch\tnist-data-privacy\tnist-ai-rmf\ttest\n"
        "atlas-privacy\tibm-risk-atlas\tskos:broadMatch\tllm022025-sensitive-information-disclosure\towasp-llm-2.0\ttest\n"
        "atlas-hallucination\tibm-risk-atlas\tskos:exactMatch\tnist-confabulation\tnist-ai-rmf\ttest\n"
        "atlas-transparency\tibm-risk-atlas\tskos:relatedMatch\tnist-value-chain-and-component-integration\tnist-ai-rmf\ttest\n"
        "credo-risk-021\tcredo-ucf\tskos:broadMatch\tnist-information-integrity\tnist-ai-rmf\ttest\n"
    )
    return sssom


def test_load_risk_to_category_map(tmp_sssom):
    mapping = _load_risk_to_category_map(tmp_sssom)
    assert "atlas-bias" in mapping
    assert "nist-harmful-bias-or-homogenization" in mapping["atlas-bias"]["nist-ai-rmf"]
    assert "atlas-privacy" in mapping
    assert "nist-data-privacy" in mapping["atlas-privacy"]["nist-ai-rmf"]
    assert "llm022025-sensitive-information-disclosure" in mapping["atlas-privacy"]["owasp-llm-2.0"]


def test_load_risk_to_category_map_excludes_related(tmp_sssom):
    mapping = _load_risk_to_category_map(tmp_sssom)
    assert "atlas-transparency" not in mapping


def test_derive_categories(tmp_sssom):
    mapping = _load_risk_to_category_map(tmp_sssom)
    risk_ids = {"atlas-bias", "atlas-privacy", "credo-risk-021"}
    cats = _derive_categories(risk_ids, mapping)
    assert "nist-ai-rmf" in cats
    assert cats["nist-ai-rmf"] == {
        "nist-harmful-bias-or-homogenization",
        "nist-data-privacy",
        "nist-information-integrity",
    }
    assert "owasp-llm-2.0" in cats
    assert cats["owasp-llm-2.0"] == {"llm022025-sensitive-information-disclosure"}


def test_category_eval_in_evaluate_extraction(tmp_path, tmp_sssom):
    gt = tmp_path / "cat-eval.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n  - credo-risk-021\n")
    data = {
        "version": "0.3",
        "risks": [
            {
                "risk_id": "atlas-bias",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
            {
                "risk_id": "atlas-hallucination",
                "risk_name": "Hallucination",
                "risk_description": "",
                "confidence": 0.7,
                "grounding_confidence": "medium",
                "accepted_by": "llm_judge",
                "evidence": [],
                "scores": {"bm25_rank": 3, "embedding_distance": 0.3, "cross_encoder_score": 0.7, "rrf_score": 0.3},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 10,
            "total_candidates_retrieved": 50,
            "auto_accepted": 1,
            "llm_judged": 1,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext, sssom_path=tmp_sssom)

    assert "category_eval" in result
    ce = result["category_eval"]

    # NIST: GT derives {harmful-bias, data-privacy, info-integrity}
    #       Extracted derives {harmful-bias, confabulation} (from atlas-bias + atlas-hallucination)
    #       Matched: {harmful-bias}
    assert "nist-ai-rmf" in ce
    nist = ce["nist-ai-rmf"]
    assert "nist-harmful-bias-or-homogenization" in nist["matched"]
    assert "nist-confabulation" in nist["spurious"]
    assert "nist-data-privacy" in nist["missing"]
    assert "nist-information-integrity" in nist["missing"]

    # Risk-level: matched 1/3 (atlas-bias), missed atlas-privacy + credo-risk-021
    assert result["matched"] == 1
    # Category-level NIST: matched 1/3 — more forgiving if we'd found atlas-privacy
    assert nist["recall"] == pytest.approx(1 / 3, abs=0.001)


def test_category_eval_missing_sssom(tmp_path):
    gt = tmp_path / "no-sssom.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n")
    data = {
        "version": "0.3",
        "risks": [
            {
                "risk_id": "atlas-bias",
                "risk_name": "Bias",
                "risk_description": "",
                "confidence": 0.9,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [],
                "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
            },
        ],
        "source_documents": ["policy.md"],
        "retrieval_stats": {
            "total_chunks": 1,
            "total_candidates_retrieved": 1,
            "auto_accepted": 1,
            "llm_judged": 0,
            "grounding_filtered": 0,
        },
    }
    ext = tmp_path / "risk-extraction.json"
    ext.write_text(json.dumps(data))

    result = evaluate_extraction(gt, ext, sssom_path=tmp_path / "nonexistent.tsv")
    assert result["category_eval"] == {}


# --- Schema drift smoke test ---


def test_extraction_result_schema_compatible_with_eval(tmp_path):
    """Catch schema drift between ExtractionResult (Pydantic) and eval (raw JSON).

    Constructs an ExtractionResult via Pydantic, serializes to JSON, and runs
    eval against known ground truth. If ExtractionResult field names drift from
    what eval reads (e.g. risk_id renamed), F1 drops to zero.
    """
    result = ExtractionResult(
        risks=[
            RiskMatch(
                risk_id="atlas-bias",
                risk_name="Bias",
                risk_description="",
                taxonomy="ibm-risk-atlas",
                confidence=0.9,
                grounding_confidence="high",
                accepted_by="threshold",
                evidence=[],
                scores=RetrievalScores(
                    bm25_rank=1,
                    embedding_distance=0.1,
                    cross_encoder_score=0.9,
                    rrf_score=0.5,
                ),
            ),
            RiskMatch(
                risk_id="atlas-privacy",
                risk_name="Privacy",
                risk_description="",
                taxonomy="ibm-risk-atlas",
                confidence=0.8,
                grounding_confidence="high",
                accepted_by="threshold",
                evidence=[],
                scores=RetrievalScores(
                    bm25_rank=2,
                    embedding_distance=0.2,
                    cross_encoder_score=0.8,
                    rrf_score=0.4,
                ),
            ),
        ],
        source_documents=["policy.md"],
        retrieval_stats=RetrievalStats(
            total_chunks=5,
            total_candidates_retrieved=20,
            auto_accepted=2,
            llm_judged=0,
            grounding_filtered=0,
        ),
    )

    ext = tmp_path / "risk-extraction.json"
    ext.write_text(result.model_dump_json())

    gt = tmp_path / "gt.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n")

    metrics = evaluate_extraction(gt, ext)
    assert metrics["f1"] == 1.0, (
        f"Schema drift: ExtractionResult JSON no longer matches what eval reads. "
        f"matched={metrics['matched']}, missing={metrics['missing']}, spurious={metrics['spurious']}"
    )
