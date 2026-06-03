from __future__ import annotations

import logging
import time

from concorde_policy_mapper.extract.models import EvidenceSpan, LLMCallRecord, ScoredCandidate, _RiskEvidence
from concorde_policy_mapper.prompts import render_prompt

logger = logging.getLogger(__name__)


def ground_and_extract_evidence(
    chunk_text: str,
    candidates: list[ScoredCandidate],
    client,
    model: str,
    document: str,
    chunk_index: int,
    page: int | None = None,
    section: str | None = None,
    *,
    call_collector: list[LLMCallRecord] | None = None,
) -> dict[str, tuple[list[EvidenceSpan], str]]:
    if not candidates:
        return {}

    messages = render_prompt(
        "ground_evidence",
        {
            "chunk_text": chunk_text,
            "risks": [
                {
                    "risk_id": c.risk_id,
                    "risk_name": c.risk_name,
                    "risk_description": c.risk_description,
                }
                for c in candidates
            ],
        },
    )

    t0 = time.time()
    verdicts: list[_RiskEvidence] = client.chat.completions.create(
        model=model,
        response_model=list[_RiskEvidence],
        messages=messages,
        temperature=0.0,
    )
    duration_ms = (time.time() - t0) * 1000

    candidate_ids = {c.risk_id for c in candidates}
    result: dict[str, tuple[list[EvidenceSpan], str]] = {}

    for v in verdicts:
        if not v.grounded or v.risk_id not in candidate_ids:
            continue
        spans = [
            EvidenceSpan(
                text=quote,
                document=document,
                page=page,
                section=section,
                chunk_index=chunk_index,
            )
            for quote in v.quotes
            if quote.strip()
        ]
        if spans:
            result[v.risk_id] = (spans, v.confidence)

    if call_collector is not None:
        call_id = f"ground-{len([c for c in call_collector if c.stage == 'grounding']) + 1:03d}"
        call_collector.append(
            LLMCallRecord(
                call_id=call_id,
                stage="grounding",
                chunk_index=chunk_index,
                risk_ids=[c.risk_id for c in candidates],
                messages=messages,
                response=[v.model_dump() for v in verdicts],
                duration_ms=duration_ms,
                result_summary=f"{len(result)}/{len(candidates)} grounded",
            )
        )

    return result


def ground_risk_group(
    chunks: list,
    chunk_indices: list[int],
    risks: list[dict],
    client,
    model: str,
    document: str,
    *,
    call_collector: list[LLMCallRecord] | None = None,
) -> dict[str, tuple[list[EvidenceSpan], str]]:
    """Ground a group of related risks against multiple document chunks."""
    if not risks or not chunk_indices:
        return {}

    passages = []
    for idx in chunk_indices:
        if idx < len(chunks):
            passages.append({"index": idx, "text": chunks[idx].text})

    if not passages:
        return {}

    messages = render_prompt(
        "ground_group",
        {
            "passages": passages,
            "risks": risks,
        },
    )

    t0 = time.time()
    verdicts: list[_RiskEvidence] = client.chat.completions.create(
        model=model,
        response_model=list[_RiskEvidence],
        messages=messages,
        temperature=0.0,
    )
    duration_ms = (time.time() - t0) * 1000

    risk_ids = {r["risk_id"] for r in risks}
    result: dict[str, tuple[list[EvidenceSpan], str]] = {}

    for v in verdicts:
        if not v.grounded or v.risk_id not in risk_ids:
            continue
        spans = [
            EvidenceSpan(
                text=quote,
                document=document,
                chunk_index=chunk_indices[0] if chunk_indices else 0,
            )
            for quote in v.quotes
            if quote.strip()
        ]
        if spans:
            result[v.risk_id] = (spans, v.confidence)

    if call_collector is not None:
        call_id = f"ground-expand-{len([c for c in call_collector if 'expand' in c.call_id]) + 1:03d}"
        call_collector.append(
            LLMCallRecord(
                call_id=call_id,
                stage="grounding",
                chunk_index=chunk_indices[0] if chunk_indices else -1,
                risk_ids=[r["risk_id"] for r in risks],
                messages=messages,
                response=[v.model_dump() for v in verdicts],
                duration_ms=duration_ms,
                result_summary=f"{len(result)}/{len(risks)} grounded (expansion)",
            )
        )

    return result
