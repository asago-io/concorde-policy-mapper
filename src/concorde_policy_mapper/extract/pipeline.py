from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from concorde_policy_mapper.extract.attribute import ground_and_extract_evidence
from concorde_policy_mapper.extract.index import RiskIndex
from concorde_policy_mapper.extract.merge import merge_matches
from concorde_policy_mapper.extract.models import (
    ChunkSummary,
    ExtractionResult,
    FilteredCandidate,
    LLMCallRecord,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
)
from concorde_policy_mapper.extract.parse import chunk_documents, parse_document
from concorde_policy_mapper.extract.retrieve import (
    build_padded_text,
    judge_borderline,
    retrieve_chunk,
)
from concorde_policy_mapper.llm import LLMConfig

logger = logging.getLogger(__name__)

_AGENTIC_PHRASES = {
    "agentic",
    "ai agent",
    "ai agents",
    "autonomous agent",
    "autonomous agents",
    "agent framework",
    "tool-calling",
    "tool calling",
    "function calling",
    "agentic ai",
    "agentic system",
    "agentic systems",
    "multi-agent",
    "multiagent",
}


def _document_discusses_agents(texts: list[str]) -> bool:
    combined = " ".join(texts).lower()
    return any(phrase in combined for phrase in _AGENTIC_PHRASES)


def _judge_one(i, cr, chunks, client, model, call_collector, judge_prompt="judge_risk", max_context_tokens=0):
    padded = build_padded_text(chunks, i, max_context_tokens=max_context_tokens)
    judged = judge_borderline(
        cr.borderline, padded, client, model,
        call_collector=call_collector, chunk_index=i,
        prompt_name=judge_prompt,
    )
    return i, judged


def _ground_one(cr, chunks, client, model, call_collector):
    chunk = chunks[cr.chunk_index]
    grounded = ground_and_extract_evidence(
        chunk_text=chunk.text,
        candidates=cr.accepted,
        client=client,
        model=model,
        document=chunk.source,
        chunk_index=cr.chunk_index,
        page=chunk.page,
        section=chunk.section,
        call_collector=call_collector,
    )
    return cr, grounded


def run_extraction(
    documents: list[Path],
    client,
    config: LLMConfig,
    risks: list,
    ocr: bool = False,
    chunk_max_tokens: int = 512,
    top_n_accept: int = 10,
    top_n_judge: int = 10,
    min_score_floor: float = 0.70,
    bi_encoder_model: str = "all-mpnet-base-v2",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    bm25_rescue_rank: int = 0,
    use_cross_encoder: bool = True,
    rrf_min_score: float = 0.01,
    colbert_model: str | None = None,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    no_judge: bool = False,
    no_grounding: bool = False,
    judge_prompt: str = "judge_risk",
    judge_context_tokens: int = 0,
) -> ExtractionResult:
    timing: dict[str, float] = {}
    max_workers = config.max_concurrent

    t0 = time.time()
    parsed = [parse_document(doc, ocr=ocr) for doc in documents]
    parsed = [p for p in parsed if p.content.strip()]
    if not parsed:
        logger.warning("All documents are empty after parsing")
        return _empty_result(documents)
    timing["parse_ms"] = (time.time() - t0) * 1000

    t0 = time.time()
    chunks = chunk_documents(parsed, max_tokens=chunk_max_tokens)
    if not chunks:
        return _empty_result(documents)
    timing["chunk_ms"] = (time.time() - t0) * 1000

    if not _document_discusses_agents([p.content for p in parsed]):
        original_count = len(risks)
        risks = [r for r in risks if getattr(r, "risk_type", None) != "agentic"]
        filtered = original_count - len(risks)
        if filtered:
            logger.info(
                "Filtered %d agentic risks (no agent terminology in documents)",
                filtered,
            )

    t0 = time.time()
    if not risks:
        logger.error("No risks loaded from Nexus")
        return _empty_result(documents)
    index = RiskIndex(
        risks,
        bi_encoder_model=bi_encoder_model,
        cross_encoder_model=cross_encoder_model if use_cross_encoder and not colbert_model else None,
        colbert_model=colbert_model,
    )
    timing["index_ms"] = (time.time() - t0) * 1000

    t0 = time.time()
    chunk_results = []
    for i in range(len(chunks)):
        cr = retrieve_chunk(
            chunks,
            i,
            index,
            top_n_accept=top_n_accept,
            top_n_judge=top_n_judge,
            min_score_floor=min_score_floor,
            bm25_rescue_rank=bm25_rescue_rank,
            use_cross_encoder=use_cross_encoder,
            rrf_min_score=rrf_min_score if not use_cross_encoder else 0.0,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
        )
        chunk_results.append(cr)
    timing["retrieve_ms"] = (time.time() - t0) * 1000

    call_collector: list[LLMCallRecord] = []

    chunk_summaries = [
        ChunkSummary(
            index=cr.chunk_index,
            source=cr.source,
            page=cr.page,
            section=cr.section,
            text_preview=chunks[cr.chunk_index].text[:200],
            candidates_retrieved=cr.stats.get("candidates_retrieved", 0),
            auto_accepted=cr.stats.get("auto_accepted", 0),
            borderline=cr.stats.get("borderline", 0),
            discarded=cr.stats.get("discarded", 0),
            bm25_rescued=cr.stats.get("bm25_rescued", 0),
        )
        for cr in chunk_results
    ]

    t0 = time.time()
    if no_judge:
        for cr in chunk_results:
            if cr.borderline:
                cr.borderline_judged = list(cr.borderline)
                cr.accepted.extend(cr.borderline)
    elif use_cross_encoder:
        judge_tasks = [
            (i, cr) for i, cr in enumerate(chunk_results) if cr.borderline
        ]
        if judge_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_judge_one, i, cr, chunks, client, config.model, call_collector, judge_prompt, judge_context_tokens): i
                    for i, cr in judge_tasks
                }
                for future in as_completed(futures):
                    idx, judged = future.result()
                    chunk_results[idx].borderline_judged = judged
                    chunk_results[idx].accepted.extend(judged)
    timing["judge_ms"] = (time.time() - t0) * 1000

    t0 = time.time()
    all_matches: list[RiskMatch] = []
    all_filtered: list[FilteredCandidate] = []

    if no_grounding:
        for cr in chunk_results:
            for candidate in cr.accepted:
                if candidate in cr.borderline_judged:
                    accepted_by = "auto_promoted" if no_judge else "llm_judge"
                elif use_cross_encoder:
                    accepted_by = "threshold"
                else:
                    accepted_by = "rrf"
                conf_score = candidate.cross_encoder_score if use_cross_encoder else candidate.rrf_score
                all_matches.append(
                    RiskMatch(
                        risk_id=candidate.risk_id,
                        risk_name=candidate.risk_name,
                        risk_description=candidate.risk_description,
                        taxonomy=index.get_taxonomy(candidate.risk_id),
                        confidence=conf_score,
                        grounding_confidence="ungrounded",
                        accepted_by=accepted_by,
                        evidence=[],
                        scores=RetrievalScores(
                            bm25_rank=candidate.bm25_rank,
                            embedding_distance=candidate.embedding_distance,
                            cross_encoder_score=candidate.cross_encoder_score,
                            rrf_score=candidate.rrf_score,
                        ),
                    )
                )
        timing["grounding_ms"] = 0.0
        grounding_filtered = 0
    else:
        total_candidates_for_grounding = 0
        total_grounded = 0

        ground_tasks = [cr for cr in chunk_results if cr.accepted]
        for cr in ground_tasks:
            total_candidates_for_grounding += len(cr.accepted)

        if ground_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_ground_one, cr, chunks, client, config.model, call_collector): cr
                    for cr in ground_tasks
                }
                for future in as_completed(futures):
                    cr, grounded = future.result()
                    total_grounded += len(grounded)
                    grounded_ids = set(grounded.keys())
                    for candidate in cr.accepted:
                        if use_cross_encoder:
                            accepted_by = (
                                "llm_judge"
                                if candidate in cr.borderline_judged
                                else "threshold"
                            )
                        else:
                            accepted_by = "rrf"
                        rid = candidate.risk_id
                        if rid in grounded_ids:
                            evidence, confidence = grounded[rid]
                            conf_score = candidate.cross_encoder_score if use_cross_encoder else candidate.rrf_score
                            all_matches.append(
                                RiskMatch(
                                    risk_id=rid,
                                    risk_name=candidate.risk_name,
                                    risk_description=candidate.risk_description,
                                    taxonomy=index.get_taxonomy(rid),
                                    confidence=conf_score,
                                    grounding_confidence=confidence,
                                    accepted_by=accepted_by,
                                    evidence=evidence,
                                    scores=RetrievalScores(
                                        bm25_rank=candidate.bm25_rank,
                                        embedding_distance=candidate.embedding_distance,
                                        cross_encoder_score=candidate.cross_encoder_score,
                                        rrf_score=candidate.rrf_score,
                                    ),
                                )
                            )
                        else:
                            all_filtered.append(
                                FilteredCandidate(
                                    risk_id=rid,
                                    risk_name=candidate.risk_name,
                                    taxonomy=index.get_taxonomy(rid),
                                    cross_encoder_score=candidate.cross_encoder_score,
                                    rrf_score=candidate.rrf_score,
                                    bm25_rank=candidate.bm25_rank,
                                    accepted_by=accepted_by,
                                    chunk_index=cr.chunk_index,
                                )
                            )
        timing["grounding_ms"] = (time.time() - t0) * 1000
        grounding_filtered = total_candidates_for_grounding - total_grounded

    chunk_risk_ids: dict[int, list[str]] = {}
    if no_grounding:
        for cr in chunk_results:
            for candidate in cr.accepted:
                chunk_risk_ids.setdefault(cr.chunk_index, []).append(candidate.risk_id)
    else:
        for m in all_matches:
            for ev in m.evidence:
                chunk_risk_ids.setdefault(ev.chunk_index, []).append(m.risk_id)
    for cs in chunk_summaries:
        cs.accepted_risk_ids = sorted(set(chunk_risk_ids.get(cs.index, [])))

    t0 = time.time()
    merged = merge_matches(all_matches)
    timing["merge_ms"] = (time.time() - t0) * 1000

    total_stats = RetrievalStats(
        total_chunks=len(chunks),
        total_candidates_retrieved=sum(
            cr.stats.get("candidates_retrieved", 0) for cr in chunk_results
        ),
        auto_accepted=sum(
            cr.stats.get("auto_accepted", 0) for cr in chunk_results
        ),
        llm_judged=sum(len(cr.borderline_judged) for cr in chunk_results),
        grounding_filtered=grounding_filtered,
        timing_ms=timing,
    )

    return ExtractionResult(
        risks=merged,
        source_documents=[str(d) for d in documents],
        retrieval_stats=total_stats,
        metadata={
            "model": config.model,
            "bi_encoder_model": bi_encoder_model,
            "cross_encoder_model": cross_encoder_model if use_cross_encoder else None,
            "use_cross_encoder": use_cross_encoder,
            "colbert_model": colbert_model,
            "chunk_max_tokens": chunk_max_tokens,
            "top_n_accept": top_n_accept,
            "top_n_judge": top_n_judge,
            "min_score_floor": min_score_floor,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "bm25_rescue_rank": bm25_rescue_rank,
            "rrf_min_score": rrf_min_score,
            "judge_prompt": judge_prompt,
            "no_judge": no_judge,
            "no_grounding": no_grounding,
            "judge_context_tokens": judge_context_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        chunks=chunk_summaries,
        llm_calls=call_collector,
        grounding_filtered_candidates=all_filtered,
    )


def _empty_result(documents: list[Path]) -> ExtractionResult:
    return ExtractionResult(
        risks=[],
        source_documents=[str(d) for d in documents],
        retrieval_stats=RetrievalStats(
            total_chunks=0,
            total_candidates_retrieved=0,
            auto_accepted=0,
            llm_judged=0,
            grounding_filtered=0,
        ),
    )
