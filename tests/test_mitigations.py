import yaml
from pathlib import Path

import pytest

from concorde_policy_mapper.extract.mitigations import (
    build_risk_crossmap,
    enrich_with_mitigations,
    load_mitigation_index,
    build_action_descriptions,
)
from concorde_policy_mapper.extract.models import (
    EvidenceSpan,
    MitigationRef,
    RetrievalScores,
    RiskMatch,
)


@pytest.fixture
def sample_index_path(tmp_path):
    data = {
        "atlas-hallucination": [
            {"id": "owasp-act-09-01", "name": "Use RAG", "source": "owasp-llm-2.0"},
            {"id": "GV-1.2-003", "name": "GV-1.2-003", "source": "nist-ai-rmf"},
        ],
        "atlas-prompt-injection": [
            {"id": "owasp-act-01-01", "name": "Constrain model", "source": "owasp-llm-2.0"},
        ],
    }
    path = tmp_path / "test_index.yaml"
    path.write_text(yaml.dump(data))
    return path


def _make_risk(risk_id):
    return RiskMatch(
        risk_id=risk_id,
        risk_name=f"Risk {risk_id}",
        risk_description=f"Desc {risk_id}",
        confidence=0.85,
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[],
        scores=RetrievalScores(
            bm25_rank=1,
            embedding_distance=0.2,
            cross_encoder_score=0.85,
            rrf_score=0.03,
        ),
    )


def test_load_mitigation_index(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    assert "atlas-hallucination" in index
    assert len(index["atlas-hallucination"]) == 2
    ref = index["atlas-hallucination"][0]
    assert isinstance(ref, MitigationRef)
    assert ref.action_id == "owasp-act-09-01"
    assert ref.action_name == "Use RAG"
    assert ref.source == "owasp-llm-2.0"


def test_load_missing_file_returns_empty(tmp_path):
    index = load_mitigation_index(tmp_path / "nonexistent.yaml")
    assert index == {}


def test_enrich_with_mitigations(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    risks = [_make_risk("atlas-hallucination"), _make_risk("atlas-prompt-injection")]
    enrich_with_mitigations(risks, index)

    assert len(risks[0].mitigations) == 2
    assert risks[0].mitigations[0].action_id == "owasp-act-09-01"
    assert len(risks[1].mitigations) == 1
    assert risks[1].mitigations[0].action_id == "owasp-act-01-01"


def test_enrich_unknown_risk_gets_empty(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    risks = [_make_risk("atlas-unknown-risk")]
    enrich_with_mitigations(risks, index)
    assert risks[0].mitigations == []


def test_risk_match_serializes_mitigations():
    risk = _make_risk("atlas-hallucination")
    risk.mitigations = [
        MitigationRef(action_id="act-1", action_name="Do X", source="nist-ai-rmf"),
    ]
    data = risk.model_dump()
    assert data["mitigations"] == [
        {"action_id": "act-1", "action_name": "Do X", "description": None, "source": "nist-ai-rmf", "category": None, "strategy": None},
    ]


def test_risk_match_default_mitigations_empty():
    risk = _make_risk("atlas-hallucination")
    assert risk.mitigations == []
    data = risk.model_dump()
    assert data["mitigations"] == []


def test_enrich_with_descriptions(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    descs = {"owasp-act-09-01": "Use RAG to ground outputs in factual sources."}
    risks = [_make_risk("atlas-hallucination")]
    enrich_with_mitigations(risks, index, descs)
    assert risks[0].mitigations[0].description == "Use RAG to ground outputs in factual sources."
    assert risks[0].mitigations[1].description is None


def test_enrich_without_descriptions(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    risks = [_make_risk("atlas-hallucination")]
    enrich_with_mitigations(risks, index)
    assert all(m.description is None for m in risks[0].mitigations)


def test_enrich_resolves_non_atlas_via_crossmap(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    crossmap = {"credo-risk-016": {"atlas-hallucination"}}
    risks = [_make_risk("credo-risk-016")]
    enrich_with_mitigations(risks, index, risk_crossmap=crossmap)
    assert len(risks[0].mitigations) == 2
    assert risks[0].mitigations[0].action_id == "owasp-act-09-01"


def test_enrich_no_crossmap_for_unmapped_risk(sample_index_path):
    index = load_mitigation_index(sample_index_path)
    crossmap = {}
    risks = [_make_risk("credo-risk-999")]
    enrich_with_mitigations(risks, index, risk_crossmap=crossmap)
    assert risks[0].mitigations == []


def test_real_index_parses():
    """Smoke test: if the generated index file exists, verify it loads."""
    real_path = Path(__file__).resolve().parents[1] / "data" / "atlas_risk_to_actions.yaml"
    if not real_path.exists():
        pytest.skip("atlas_risk_to_actions.yaml not yet generated")
    index = load_mitigation_index(real_path)
    assert len(index) > 0
    for risk_id, actions in index.items():
        assert risk_id.startswith("atlas-"), f"Unexpected risk ID prefix: {risk_id}"
        for a in actions:
            assert a.action_id
            assert a.source


# ---------------------------------------------------------------------------
# Tests for build_action_descriptions
# ---------------------------------------------------------------------------


def _make_nexus_kg(tmp_path):
    """Create the Nexus knowledge_graph directory structure and return the kg path."""
    kg = tmp_path / "src" / "ai_atlas_nexus" / "data" / "knowledge_graph"
    kg.mkdir(parents=True)
    return kg


def test_build_action_descriptions_from_mock_nexus(tmp_path):
    kg = _make_nexus_kg(tmp_path)

    # MIT controls
    (kg / "mit_ai_risk_repository_data_controls.yaml").write_text(
        yaml.dump({"controls": [
            {"id": "mit-ctrl-001", "description": "Monitor model drift"},
            {"id": "mit-ctrl-002", "description": "Validate training data"},
        ]})
    )
    # NIST actions
    (kg / "nist_ai_rmf_actions_data.yaml").write_text(
        yaml.dump({"actions": [
            {"id": "GV-1.2-003", "description": "Establish AI governance"},
        ]})
    )
    # AIUC-1 rules
    (kg / "aiuc1_data.yaml").write_text(
        yaml.dump({"rules": [
            {"id": "aiuc1-req-a1", "description": "Ensure transparency"},
        ]})
    )

    # Local OWASP data
    data_dir = tmp_path / "local_data"
    data_dir.mkdir()
    (data_dir / "owasp_llm_2.0_actions_data.yaml").write_text(
        yaml.dump({"actions": [
            {"id": "owasp-act-01-01", "description": "Constrain model inputs"},
        ]})
    )

    descs = build_action_descriptions(str(tmp_path), data_dir=data_dir)

    assert descs["mit-ctrl-001"] == "Monitor model drift"
    assert descs["mit-ctrl-002"] == "Validate training data"
    assert descs["GV-1.2-003"] == "Establish AI governance"
    assert descs["aiuc1-req-a1"] == "Ensure transparency"
    assert descs["owasp-act-01-01"] == "Constrain model inputs"
    assert len(descs) == 5


def test_build_action_descriptions_missing_files(tmp_path):
    """Non-existent nexus dir returns empty dict."""
    descs = build_action_descriptions(
        str(tmp_path / "nonexistent"),
        data_dir=tmp_path / "also_nonexistent",
    )
    assert descs == {}


def test_build_action_descriptions_credo_filter(tmp_path):
    """Only entries whose id starts with 'credo-act-' are loaded from credo.yaml."""
    kg = _make_nexus_kg(tmp_path)

    (kg / "credo.yaml").write_text(
        yaml.dump({"actions": [
            {"id": "credo-act-001", "description": "Audit model fairness"},
            {"id": "credo-risk-016", "description": "Should be excluded (risk, not action)"},
            {"id": "credo-ctrl-005", "description": "Should be excluded (ctrl, not action)"},
        ]})
    )

    descs = build_action_descriptions(str(tmp_path))

    assert "credo-act-001" in descs
    assert descs["credo-act-001"] == "Audit model fairness"
    assert "credo-risk-016" not in descs
    assert "credo-ctrl-005" not in descs


def test_build_action_descriptions_skips_entries_without_id_or_description(tmp_path):
    """Entries missing id or description are silently skipped."""
    kg = _make_nexus_kg(tmp_path)

    (kg / "nist_ai_rmf_actions_data.yaml").write_text(
        yaml.dump({"actions": [
            {"id": "GV-1.1-001", "description": "Valid entry"},
            {"id": "", "description": "Empty id"},
            {"id": "GV-1.1-002"},  # missing description
            {"description": "No id key"},
        ]})
    )

    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    descs = build_action_descriptions(str(tmp_path), data_dir=empty_data_dir)

    assert descs == {"GV-1.1-001": "Valid entry"}


# ---------------------------------------------------------------------------
# Tests for build_risk_crossmap
# ---------------------------------------------------------------------------


def test_build_risk_crossmap_basic(tmp_path):
    """Non-atlas id mapped to atlas target produces non_atlas → {atlas} entry."""
    mappings_dir = _make_nexus_kg(tmp_path) / "mappings"
    mappings_dir.mkdir()

    (mappings_dir / "credo-ucf.sssom_from_tsv_data.yaml").write_text(
        yaml.dump({"entries": [
            {
                "id": "credo-risk-016",
                "close_mappings": ["atlas-hallucination"],
                "exact_mappings": ["atlas-prompt-injection"],
            },
        ]})
    )

    crossmap = build_risk_crossmap(str(tmp_path))

    assert "credo-risk-016" in crossmap
    assert crossmap["credo-risk-016"] == {"atlas-hallucination", "atlas-prompt-injection"}


def test_build_risk_crossmap_bidirectional(tmp_path):
    """Atlas in id and atlas in target both resolve to non_atlas → atlas."""
    mappings_dir = _make_nexus_kg(tmp_path) / "mappings"
    mappings_dir.mkdir()

    (mappings_dir / "mit-ai-risk-repository_ibm-risk-atlas_from_tsv_data.yaml").write_text(
        yaml.dump({"entries": [
            # atlas in target (non-atlas id)
            {
                "id": "mit-risk-042",
                "broad_mappings": ["atlas-data-poisoning"],
            },
            # atlas in id (non-atlas target)
            {
                "id": "atlas-model-theft",
                "related_mappings": ["mit-risk-099"],
            },
        ]})
    )

    crossmap = build_risk_crossmap(str(tmp_path))

    assert crossmap["mit-risk-042"] == {"atlas-data-poisoning"}
    assert crossmap["mit-risk-099"] == {"atlas-model-theft"}


def test_build_risk_crossmap_missing_files(tmp_path):
    """Non-existent mappings dir returns empty dict."""
    crossmap = build_risk_crossmap(str(tmp_path / "nonexistent"))
    assert crossmap == {}


def test_build_risk_crossmap_multiple_predicates_merge(tmp_path):
    """Mappings from different predicates for the same id are merged into one set."""
    mappings_dir = _make_nexus_kg(tmp_path) / "mappings"
    mappings_dir.mkdir()

    (mappings_dir / "credo-ucf.sssom_from_tsv_data.yaml").write_text(
        yaml.dump({"entries": [
            {
                "id": "credo-risk-001",
                "close_mappings": ["atlas-hallucination"],
                "broad_mappings": ["atlas-data-poisoning"],
                "exact_mappings": ["atlas-hallucination"],  # duplicate, should deduplicate
            },
        ]})
    )

    crossmap = build_risk_crossmap(str(tmp_path))

    assert crossmap["credo-risk-001"] == {"atlas-hallucination", "atlas-data-poisoning"}
