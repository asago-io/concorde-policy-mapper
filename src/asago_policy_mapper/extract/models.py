from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class ScoredCandidate(BaseModel):
    risk_id: str
    risk_name: str
    risk_description: str
    bm25_rank: int = 0
    embedding_distance: float = 0.0
    cross_encoder_score: float = 0.0
    rrf_score: float = 0.0


DPV_RISK_CONTROLS = Literal[
    "AvoidanceControl",
    "MitigationControl",
    "ModificationControl",
    "MonitorControl",
    "InterruptionControl",
    "InterventionControl",
    "ReductionControl",
    "ResolutionControl",
    "InvestigationControl",
    "OverrideControl",
    "TransferControl",
]


class MitigationRef(BaseModel):
    action_id: str
    action_name: str | None = None
    description: str | None = None
    source: str
    category: Literal["technical", "operational", "governance"] | None = None
    risk_control: DPV_RISK_CONTROLS | None = None


class EvidenceSpan(BaseModel):
    text: str
    document: str
    page: int | None = None
    section: str | None = None
    chunk_index: int
    sentence_index: int = 0
    cross_encoder_score: float = 0.0


class RetrievalScores(BaseModel):
    bm25_rank: int
    embedding_distance: float
    cross_encoder_score: float
    rrf_score: float


class RiskMatch(BaseModel):
    risk_id: str
    risk_name: str
    risk_description: str
    taxonomy: str = ""
    confidence: float
    grounding_confidence: str
    accepted_by: str
    evidence: list[EvidenceSpan]
    scores: RetrievalScores
    mitigations: list[MitigationRef] = []
    threat: str | None = None
    threat_source: str | None = None
    vulnerability: str | None = None
    consequence: str | None = None
    impact: str | None = None


class RetrievalStats(BaseModel):
    total_chunks: int
    total_candidates_retrieved: int
    auto_accepted: int
    llm_judged: int
    grounding_filtered: int
    timing_ms: dict[str, float] = {}


class ChunkSummary(BaseModel):
    index: int
    source: str
    page: int | None = None
    section: str | None = None
    text_preview: str
    candidates_retrieved: int
    auto_accepted: int
    borderline: int
    discarded: int
    bm25_rescued: int = 0
    accepted_risk_ids: list[str] = []


class LLMCallRecord(BaseModel):
    call_id: str
    stage: Literal["judge", "grounding", "variant_grounding", "causal_synthesis", "query_gen"]
    chunk_index: int = -1
    risk_ids: list[str]
    messages: list[dict]
    response: dict | str | list[dict]
    duration_ms: float
    result_summary: str


class FilteredCandidate(BaseModel):
    risk_id: str
    risk_name: str
    taxonomy: str = ""
    cross_encoder_score: float
    rrf_score: float = 0.0
    bm25_rank: int = 0
    accepted_by: str
    chunk_index: int


class ExtractionResult(BaseModel):
    version: str = "0.3"
    risks: list[RiskMatch]
    source_documents: list[str]
    token_usage: dict = {}
    retrieval_stats: RetrievalStats
    metadata: dict = {}
    chunks: list[ChunkSummary] = []
    llm_calls: list[LLMCallRecord] = []
    grounding_filtered_candidates: list[FilteredCandidate] = []
    eval: dict | None = None


class _JudgeVerdict(BaseModel):
    risk_id: str
    relevant: bool
    justification: str


class _JudgeVerdicts(BaseModel):
    items: list[_JudgeVerdict]


class _GroundingVerdict(BaseModel):
    risk_id: str
    grounded: bool
    confidence: Literal["high", "medium", "low"]


class _RiskEvidence(BaseModel):
    risk_id: str
    grounded: bool
    confidence: Literal["high", "medium", "low"]
    quotes: list[str]


class _RiskEvidenceList(BaseModel):
    items: list[_RiskEvidence]


class _CausalChain(BaseModel):
    threat: str
    threat_source: str
    vulnerability: str
    consequence: str
    impact: str


class _CausalChains(BaseModel):
    items: list[_CausalChain]


@dataclass
class RetrievalConfig:
    bi_encoder_model: str = "all-mpnet-base-v2"
    query_instruction: str = (
        "Given a text passage from an AI governance policy document, retrieve AI risk"
        " descriptions that are relevant to the concepts, requirements, or concerns"
        " discussed in the passage"
    )
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_type: str = "score"
    colbert_model: str | None = None
    chunk_max_tokens: int = 512
    top_n_accept: int = 10
    top_n_judge: int = 10
    min_score_floor: float = 0.70
    threshold_high: float | None = None
    threshold_low: float | None = None
    bm25_rescue_rank: int = 0
    rrf_min_score: float = 0.01
    use_cross_encoder: bool = True
    no_judge: bool = False
    no_grounding: bool = False
    judge_prompt: str = "judge_risk"
    judge_context_tokens: int = 0
    expand_siblings: bool = True
    grounding_passes: int = 3
    expansion_passes: int = 3
    no_causal_synthesis: bool = False
    grounding_batch_size: int = 0
    query_gen: bool = True

    @property
    def effective_cross_encoder_model(self) -> str | None:
        if not self.use_cross_encoder or self.colbert_model:
            return None
        return self.cross_encoder_model

    @property
    def effective_rrf_min_score(self) -> float:
        return self.rrf_min_score if not self.use_cross_encoder else 0.0

    def to_metadata(self) -> dict:
        return {
            "bi_encoder_model": self.bi_encoder_model,
            "cross_encoder_model": self.effective_cross_encoder_model,
            "cross_encoder_type": self.cross_encoder_type,
            "use_cross_encoder": self.use_cross_encoder,
            "colbert_model": self.colbert_model,
            "chunk_max_tokens": self.chunk_max_tokens,
            "top_n_accept": self.top_n_accept,
            "top_n_judge": self.top_n_judge,
            "min_score_floor": self.min_score_floor,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "bm25_rescue_rank": self.bm25_rescue_rank,
            "rrf_min_score": self.rrf_min_score,
            "judge_prompt": self.judge_prompt,
            "no_judge": self.no_judge,
            "no_grounding": self.no_grounding,
            "judge_context_tokens": self.judge_context_tokens,
            "expand_siblings": self.expand_siblings,
            "grounding_passes": self.grounding_passes,
            "expansion_passes": self.expansion_passes,
            "no_causal_synthesis": self.no_causal_synthesis,
            "query_gen": self.query_gen,
        }
