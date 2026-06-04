from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from concorde_policy_mapper.extract.attribute import ground_and_extract_evidence, ground_risk_group
from concorde_policy_mapper.extract.index import RiskIndex
from concorde_policy_mapper.extract.merge import merge_matches
from concorde_policy_mapper.extract.models import (
    ChunkSummary,
    ExtractionResult,
    FilteredCandidate,
    LLMCallRecord,
    RetrievalConfig,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
    ScoredCandidate,
)
from concorde_policy_mapper.extract.parse import chunk_documents, parse_document
from concorde_policy_mapper.extract.retrieve import (
    build_padded_text,
    judge_borderline,
    retrieve_chunk,
)
from concorde_policy_mapper.llm import LLMConfig

logger = logging.getLogger(__name__)


@contextmanager
def timed(timing, key):
    t0 = time.time()
    try:
        yield
    finally:
        timing[key] = (time.time() - t0) * 1000


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


def _ground_one(cr, chunks, client, model, call_collector, passes=1):
    chunk = chunks[cr.chunk_index]
    merged: dict[str, tuple[list, str]] = {}
    for _ in range(passes):
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
        for rid, val in grounded.items():
            if rid not in merged:
                merged[rid] = val
    return cr, merged


def determine_accepted_by(candidate, *, borderline_judged, use_cross_encoder, no_judge):
    if candidate in borderline_judged:
        return "auto_promoted" if no_judge else "llm_judge"
    return "threshold" if use_cross_encoder else "rrf"


def build_risk_match(
    candidate,
    *,
    taxonomy,
    accepted_by,
    grounding_confidence,
    evidence,
    use_cross_encoder=True,
    confidence_override=None,
    scores_override=None,
):
    confidence = (
        confidence_override
        if confidence_override is not None
        else (candidate.cross_encoder_score if use_cross_encoder else candidate.rrf_score)
    )
    scores = scores_override or RetrievalScores(
        bm25_rank=candidate.bm25_rank,
        embedding_distance=candidate.embedding_distance,
        cross_encoder_score=candidate.cross_encoder_score,
        rrf_score=candidate.rrf_score,
    )
    return RiskMatch(
        risk_id=candidate.risk_id,
        risk_name=candidate.risk_name,
        risk_description=candidate.risk_description,
        taxonomy=taxonomy,
        confidence=confidence,
        grounding_confidence=grounding_confidence,
        accepted_by=accepted_by,
        evidence=list(evidence),
        scores=scores,
    )


def _collect_ungrounded(chunk_results, index, retrieval):
    matches = []
    for cr in chunk_results:
        for candidate in cr.accepted:
            accepted_by = determine_accepted_by(
                candidate, borderline_judged=cr.borderline_judged,
                use_cross_encoder=retrieval.use_cross_encoder, no_judge=retrieval.no_judge,
            )
            matches.append(
                build_risk_match(
                    candidate,
                    taxonomy=index.get_taxonomy(candidate.risk_id),
                    accepted_by=accepted_by,
                    grounding_confidence="ungrounded",
                    evidence=[],
                    use_cross_encoder=retrieval.use_cross_encoder,
                )
            )
    return matches


def _run_grounding(chunk_results, chunks, client, config, retrieval, index, call_collector, grounding_passes=1):
    all_matches = []
    all_filtered = []
    total_candidates = 0
    total_grounded = 0
    max_workers = config.max_concurrent

    ground_tasks = [cr for cr in chunk_results if cr.accepted]
    for cr in ground_tasks:
        total_candidates += len(cr.accepted)

    if ground_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_ground_one, cr, chunks, client, config.model, call_collector, grounding_passes): cr
                for cr in ground_tasks
            }
            for future in as_completed(futures):
                cr, grounded = future.result()
                total_grounded += len(grounded)
                grounded_ids = set(grounded.keys())
                for candidate in cr.accepted:
                    accepted_by = determine_accepted_by(
                        candidate, borderline_judged=cr.borderline_judged,
                        use_cross_encoder=retrieval.use_cross_encoder, no_judge=retrieval.no_judge,
                    )
                    rid = candidate.risk_id
                    if rid in grounded_ids:
                        evidence, confidence = grounded[rid]
                        all_matches.append(
                            build_risk_match(
                                candidate,
                                taxonomy=index.get_taxonomy(rid),
                                accepted_by=accepted_by,
                                grounding_confidence=confidence,
                                evidence=evidence,
                                use_cross_encoder=retrieval.use_cross_encoder,
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

    grounding_filtered = total_candidates - total_grounded
    return all_matches, all_filtered, grounding_filtered


def _run_judge(chunk_results, chunks, client, config, retrieval, call_collector):
    max_workers = config.max_concurrent
    if retrieval.no_judge:
        for cr in chunk_results:
            if cr.borderline:
                cr.borderline_judged = list(cr.borderline)
                cr.accepted.extend(cr.borderline)
    elif retrieval.use_cross_encoder:
        judge_tasks = [
            (i, cr) for i, cr in enumerate(chunk_results) if cr.borderline
        ]
        if judge_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_judge_one, i, cr, chunks, client, config.model, call_collector, retrieval.judge_prompt, retrieval.judge_context_tokens): i
                    for i, cr in judge_tasks
                }
                for future in as_completed(futures):
                    idx, judged = future.result()
                    chunk_results[idx].borderline_judged = judged
                    chunk_results[idx].accepted.extend(judged)


def _build_chunk_risk_map(chunk_results, all_matches, no_grounding):
    chunk_risk_ids: dict[int, list[str]] = {}
    if no_grounding:
        for cr in chunk_results:
            for candidate in cr.accepted:
                chunk_risk_ids.setdefault(cr.chunk_index, []).append(candidate.risk_id)
    else:
        for m in all_matches:
            for ev in m.evidence:
                chunk_risk_ids.setdefault(ev.chunk_index, []).append(m.risk_id)
    return chunk_risk_ids


def _run_expansion(
    risks, merged, chunk_results, chunks, documents,
    index, client, config, max_workers, call_collector,
    expansion_passes: int = 1,
) -> tuple[list[RiskMatch], dict]:
    from concorde_policy_mapper.extract.expand import (
        build_expansion_graph,
        expand_with_siblings,
        group_for_grounding,
    )

    stats = {"expanded_candidates": 0, "expanded_grounded": 0, "expansion_groups": 0}
    expansion_graph = build_expansion_graph(risks)
    risk_lookup = {
        r.id: {"name": r.name or "", "description": r.description or ""}
        for r in risks
    }
    risk_to_parent = {}
    for r in risks:
        parent = getattr(r, "isPartOf", "") or ""
        if parent:
            risk_to_parent[r.id] = parent

    merged_ids = {m.risk_id for m in merged}
    expanded = expand_with_siblings(merged_ids, expansion_graph, risk_lookup)
    stats["expanded_candidates"] = len(expanded)

    if not expanded:
        return merged, stats

    found_risk_chunks: dict[str, set[int]] = {}
    for cr in chunk_results:
        for candidate in cr.accepted:
            cid = candidate.risk_id.split(" ")[0].strip()
            found_risk_chunks.setdefault(cid, set()).add(cr.chunk_index)

    groups = group_for_grounding(
        expanded, found_risk_chunks, risk_to_parent, len(chunks),
    )
    stats["expansion_groups"] = len(groups)

    def _ground_group(group):
        return ground_risk_group(
            chunks=chunks,
            chunk_indices=group.chunk_indices,
            risks=list(group.risk_lookup.values()),
            client=client,
            model=config.model,
            document=str(documents[0]) if documents else "",
            call_collector=call_collector,
        )

    def _ground_group_multi(group, passes):
        merged_results: dict[str, tuple[list[EvidenceSpan], str]] = {}
        for _ in range(passes):
            grounded = _ground_group(group)
            for rid, (evidence, confidence) in grounded.items():
                if rid not in merged_results:
                    merged_results[rid] = (evidence, confidence)
        return merged_results

    expansion_matches: list[RiskMatch] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _ground_group_multi if expansion_passes > 1 else _ground_group,
                g, *([expansion_passes] if expansion_passes > 1 else []),
            ): g
            for g in groups
        }
        for future in as_completed(futures):
            grounded = future.result()
            stats["expanded_grounded"] += len(grounded)
            for rid, (evidence, confidence) in grounded.items():
                info = risk_lookup.get(rid, {})
                stub = ScoredCandidate(
                    risk_id=rid,
                    risk_name=info.get("name", ""),
                    risk_description=info.get("description", ""),
                )
                expansion_matches.append(
                    build_risk_match(
                        stub,
                        taxonomy=index.get_taxonomy(rid),
                        accepted_by="expansion",
                        grounding_confidence=confidence,
                        evidence=evidence,
                        confidence_override=0.0,
                    )
                )

    if expansion_matches:
        merged = merge_matches(merged + expansion_matches)

    return merged, stats


def run_extraction(
    documents: list[Path],
    client,
    config: LLMConfig,
    risks: list,
    retrieval: RetrievalConfig | None = None,
    *,
    ocr: bool = False,
) -> ExtractionResult:
    retrieval = retrieval or RetrievalConfig()
    timing: dict[str, float] = {}
    max_workers = config.max_concurrent

    with timed(timing, "parse_ms"):
        parsed = [parse_document(doc, ocr=ocr) for doc in documents]
        parsed = [p for p in parsed if p.content.strip()]
        if not parsed:
            logger.warning("All documents are empty after parsing")
            return _empty_result(documents)

    with timed(timing, "chunk_ms"):
        chunks = chunk_documents(parsed, max_tokens=retrieval.chunk_max_tokens)
        if not chunks:
            return _empty_result(documents)

    if not _document_discusses_agents([p.content for p in parsed]):
        original_count = len(risks)
        risks = [r for r in risks if getattr(r, "risk_type", None) != "agentic"]
        filtered = original_count - len(risks)
        if filtered:
            logger.info(
                "Filtered %d agentic risks (no agent terminology in documents)",
                filtered,
            )

    with timed(timing, "index_ms"):
        if not risks:
            logger.error("No risks loaded from Nexus")
            return _empty_result(documents)
        index = RiskIndex(
            risks,
            bi_encoder_model=retrieval.bi_encoder_model,
            cross_encoder_model=retrieval.effective_cross_encoder_model,
            colbert_model=retrieval.colbert_model,
            query_instruction=retrieval.query_instruction,
            cross_encoder_type=retrieval.cross_encoder_type,
        )

    with timed(timing, "retrieve_ms"):
        chunk_results = []
        for i in range(len(chunks)):
            cr = retrieve_chunk(
                chunks,
                i,
                index,
                top_n_accept=retrieval.top_n_accept,
                top_n_judge=retrieval.top_n_judge,
                min_score_floor=retrieval.min_score_floor,
                bm25_rescue_rank=retrieval.bm25_rescue_rank,
                use_cross_encoder=retrieval.use_cross_encoder,
                rrf_min_score=retrieval.effective_rrf_min_score,
                threshold_high=retrieval.threshold_high,
                threshold_low=retrieval.threshold_low,
            )
            chunk_results.append(cr)

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

    with timed(timing, "judge_ms"):
        _run_judge(chunk_results, chunks, client, config, retrieval, call_collector)

    if retrieval.no_grounding:
        all_matches = _collect_ungrounded(chunk_results, index, retrieval)
        all_filtered: list[FilteredCandidate] = []
        timing["grounding_ms"] = 0.0
        grounding_filtered = 0
    else:
        with timed(timing, "grounding_ms"):
            all_matches, all_filtered, grounding_filtered = _run_grounding(
                chunk_results, chunks, client, config, retrieval, index, call_collector,
                grounding_passes=retrieval.grounding_passes,
            )

    chunk_risk_ids = _build_chunk_risk_map(chunk_results, all_matches, retrieval.no_grounding)
    for cs in chunk_summaries:
        cs.accepted_risk_ids = sorted(set(chunk_risk_ids.get(cs.index, [])))

    with timed(timing, "merge_ms"):
        merged = merge_matches(all_matches)

    expansion_stats = {"expanded_candidates": 0, "expanded_grounded": 0, "expansion_groups": 0}
    if retrieval.expand_siblings and not retrieval.no_grounding and client is not None:
        with timed(timing, "expansion_ms"):
            merged, expansion_stats = _run_expansion(
                risks, merged, chunk_results, chunks, documents,
                index, client, config, max_workers, call_collector,
                expansion_passes=retrieval.expansion_passes,
            )

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
            **retrieval.to_metadata(),
            "expansion_stats": expansion_stats,
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
