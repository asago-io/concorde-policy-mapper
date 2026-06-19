"""Tests for variant-selective grounding (ground_variants + pipeline integration)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from asago_policy_mapper.extract.attribute import ground_variants
from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    LLMCallRecord,
    RetrievalScores,
    RiskMatch,
    _RiskEvidence,
)
from asago_policy_mapper.extract.pipeline import (
    _ground_variants_one,
    _run_variant_grounding,
)
from asago_policy_mapper.prompts import render_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARENT_ID = "ai-risk-taxonomy-unauthorized-processing"
PARENT_NAME = "Unauthorized processing"
PARENT_DESC = "Processing of personal data without legal basis or consent."

VARIANTS = [
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---biometric-data-(facial-recognition)",
        "name": "Biometric data (facial recognition)",
        "description": "Unauthorized processing of biometric data for facial recognition.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---health-data",
        "name": "Health data",
        "description": "Unauthorized processing of health records.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---financial-data",
        "name": "Financial data",
        "description": "Unauthorized processing of financial records.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---location-data",
        "name": "Location data",
        "description": "Unauthorized processing of geolocation information.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---criminal-records",
        "name": "Criminal records",
        "description": "Unauthorized processing of criminal record data.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---children's-data",
        "name": "Children's data",
        "description": "Unauthorized processing of minors' personal data.",
        "taxonomy": "ai-risk-taxonomy",
    },
]

VARIANT_MAP = {PARENT_ID: VARIANTS}


def _make_parent_match(chunk_index=0):
    return RiskMatch(
        risk_id=PARENT_ID,
        risk_name=PARENT_NAME,
        risk_description=PARENT_DESC,
        taxonomy="ai-risk-taxonomy",
        confidence=0.85,
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text="Biometric data processing without consent.",
                document="policy.pdf",
                chunk_index=chunk_index,
                cross_encoder_score=0.9,
            ),
        ],
        scores=RetrievalScores(
            bm25_rank=3,
            embedding_distance=0.15,
            cross_encoder_score=0.85,
            rrf_score=0.04,
        ),
    )


def _make_non_parent_match():
    return RiskMatch(
        risk_id="atlas-bias",
        risk_name="AI Bias",
        risk_description="Systematic bias in AI outputs.",
        taxonomy="ibm-risk-atlas",
        confidence=0.90,
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text="AI systems must be fair.",
                document="policy.pdf",
                chunk_index=0,
            ),
        ],
        scores=RetrievalScores(
            bm25_rank=1,
            embedding_distance=0.1,
            cross_encoder_score=0.90,
            rrf_score=0.05,
        ),
    )


def _make_chunks(texts):
    return [SimpleNamespace(text=t, page=i, section=f"Section {i}") for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# ground_variants() unit tests
# ---------------------------------------------------------------------------


class TestGroundVariants:
    def test_selective_grounding(self):
        """LLM grounds 2 of 6 variants; only those 2 are returned."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["facial recognition data was collected without consent"],
            ),
            _RiskEvidence(
                risk_id=VARIANTS[1]["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
                rejection_reason="No health data mentioned.",
            ),
            _RiskEvidence(
                risk_id=VARIANTS[2]["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
                rejection_reason="No financial data mentioned.",
            ),
            _RiskEvidence(
                risk_id=VARIANTS[3]["risk_id"],
                grounded=True,
                confidence="medium",
                quotes=["GPS tracking without user awareness"],
            ),
            _RiskEvidence(
                risk_id=VARIANTS[4]["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
            ),
            _RiskEvidence(
                risk_id=VARIANTS[5]["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
            ),
        ]

        result = ground_variants(
            chunk_text="facial recognition data was collected without consent. GPS tracking without user awareness.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS,
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=2,
            page=5,
            section="Privacy",
        )

        assert len(result) == 2
        assert VARIANTS[0]["risk_id"] in result
        assert VARIANTS[3]["risk_id"] in result

        evidence, confidence = result[VARIANTS[0]["risk_id"]]
        assert confidence == "high"
        assert len(evidence) == 1
        assert evidence[0].text == "facial recognition data was collected without consent"
        assert evidence[0].document == "policy.pdf"
        assert evidence[0].chunk_index == 2
        assert evidence[0].page == 5
        assert evidence[0].section == "Privacy"

        evidence_loc, conf_loc = result[VARIANTS[3]["risk_id"]]
        assert conf_loc == "medium"
        assert evidence_loc[0].text == "GPS tracking without user awareness"

    def test_none_grounded(self):
        """LLM grounds 0 variants; empty result."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=v["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
            )
            for v in VARIANTS
        ]

        result = ground_variants(
            chunk_text="Generic personal data processing without specifics.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS,
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=0,
        )

        assert result == {}

    def test_empty_variants(self):
        """No variants → no LLM call, empty result."""
        mock_client = MagicMock()

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=[],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert result == {}
        mock_client.chat.completions.create.assert_not_called()

    def test_ignores_unknown_variant_ids(self):
        """Verdicts for IDs not in the variant list are ignored."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id="some-unknown-id",
                grounded=True,
                confidence="high",
                quotes=["Some evidence."],
            ),
        ]

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:1],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert result == {}

    def test_skips_empty_quotes(self):
        """Grounded verdict with only empty quotes → not included."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["", "  ", ""],
            ),
        ]

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:1],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert VARIANTS[0]["risk_id"] not in result

    def test_captures_call_record(self, mock_client):
        """Call collector records a variant_grounding stage entry."""
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["Biometric data collected."],
            ),
        ]

        collector: list[LLMCallRecord] = []
        ground_variants(
            chunk_text="Biometric data collected.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:2],
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=3,
            call_collector=collector,
        )

        assert len(collector) == 1
        record = collector[0]
        assert record.stage == "variant_grounding"
        assert record.call_id == "ground-variant-001"
        assert record.chunk_index == 3
        assert record.risk_ids == [VARIANTS[0]["risk_id"], VARIANTS[1]["risk_id"]]
        assert "1/2 variants grounded" in record.result_summary
        assert PARENT_ID in record.result_summary
        assert record.duration_ms >= 0


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestGroundVariantsPrompt:
    def test_renders_with_all_fields(self):
        messages = render_prompt(
            "ground_variants",
            {
                "chunk_text": "Biometric data collected without consent.",
                "parent_id": PARENT_ID,
                "parent_name": PARENT_NAME,
                "parent_description": PARENT_DESC,
                "variants": VARIANTS[:2],
            },
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "variant" in messages[0]["content"].lower()
        assert PARENT_ID in messages[1]["content"]
        assert VARIANTS[0]["risk_id"] in messages[1]["content"]
        assert "Biometric data collected without consent." in messages[1]["content"]


# ---------------------------------------------------------------------------
# Pipeline integration: _ground_variants_one
# ---------------------------------------------------------------------------


class TestGroundVariantsOne:
    def test_returns_grounded_variants(self):
        """_ground_variants_one calls ground_variants and returns results."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["biometric data"],
            ),
        ]

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["biometric data processing without consent."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            VARIANT_MAP,
            mock_client,
            "test-model",
            None,
        )

        assert pm is parent
        assert VARIANTS[0]["risk_id"] in grounded

    def test_no_variants_returns_empty(self):
        """Parent with no variants in the map → empty dict."""
        mock_client = MagicMock()
        parent = _make_parent_match()
        chunks = _make_chunks(["Some text."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            {},
            mock_client,
            "test-model",
            None,
        )

        assert grounded == {}
        mock_client.chat.completions.create.assert_not_called()

    def test_no_evidence_returns_empty(self):
        """Parent with no evidence spans → empty dict."""
        mock_client = MagicMock()
        parent = _make_parent_match()
        parent.evidence = []
        chunks = _make_chunks(["Some text."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            VARIANT_MAP,
            mock_client,
            "test-model",
            None,
        )

        assert grounded == {}
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline integration: _run_variant_grounding
# ---------------------------------------------------------------------------


class TestRunVariantGrounding:
    def _make_config(self):
        return SimpleNamespace(model="test-model")

    def test_partitions_parent_and_non_parent(self):
        """Parent matches get variant grounding; non-parent pass through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["biometric data"],
            ),
        ]

        parent = _make_parent_match(chunk_index=0)
        non_parent = _make_non_parent_match()
        chunks = _make_chunks(["biometric data was processed without consent."])

        result = _run_variant_grounding(
            [parent, non_parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        result_ids = {m.risk_id for m in result}
        assert non_parent.risk_id in result_ids
        assert VARIANTS[0]["risk_id"] in result_ids
        assert PARENT_ID not in result_ids

    def test_variant_inherits_parent_scores(self):
        """Variant matches inherit the parent's retrieval scores."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[1]["risk_id"],
                grounded=True,
                confidence="medium",
                quotes=["health records"],
            ),
        ]

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["health records were accessed."])

        result = _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        assert len(result) == 1
        variant = result[0]
        assert variant.risk_id == VARIANTS[1]["risk_id"]
        assert variant.risk_name == VARIANTS[1]["name"]
        assert variant.taxonomy == "ai-risk-taxonomy"
        assert variant.scores.bm25_rank == parent.scores.bm25_rank
        assert variant.scores.cross_encoder_score == parent.scores.cross_encoder_score
        assert variant.scores.rrf_score == parent.scores.rrf_score
        assert variant.accepted_by == parent.accepted_by
        assert variant.grounding_confidence == "medium"

    def test_no_parents_returns_unchanged(self):
        """When no parent matches exist, returns input unchanged."""
        non_parent = _make_non_parent_match()
        chunks = _make_chunks(["Some text."])

        result = _run_variant_grounding(
            [non_parent],
            chunks,
            VARIANT_MAP,
            MagicMock(),
            self._make_config(),
            None,
        )

        assert len(result) == 1
        assert result[0].risk_id == non_parent.risk_id

    def test_zero_variants_grounded_drops_parent(self):
        """If LLM grounds zero variants, parent is dropped entirely."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=v["risk_id"],
                grounded=False,
                confidence="low",
                quotes=[],
            )
            for v in VARIANTS
        ]

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["Generic data processing discussion."])

        result = _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        assert len(result) == 0

    def test_uses_chunk_contexts_when_provided(self):
        """When chunk_contexts is provided, uses padded text instead of raw chunk."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _RiskEvidence(
                risk_id=VARIANTS[0]["risk_id"],
                grounded=True,
                confidence="high",
                quotes=["biometric data from padded context"],
            ),
        ]

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["short chunk"])
        padded = ["biometric data from padded context with extra surrounding text"]

        _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
            chunk_contexts=padded,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = messages[-1]["content"] if messages else ""
        assert "padded context" in user_msg
