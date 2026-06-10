from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from concorde_policy_mapper.extract.models import LLMCallRecord
from concorde_policy_mapper.extract.parse import Chunk
from concorde_policy_mapper.llm import SlimModel
from concorde_policy_mapper.prompts import render_prompt

logger = logging.getLogger(__name__)


@dataclass
class ChunkGroup:
    chunk_indices: list[int]
    section: str | None


@dataclass
class QueryResult:
    query: str
    chunk_indices: list[int]
    section: str | None


class GeneratedQueries(SlimModel):
    queries: list[str]


def group_chunks(chunks: list[Chunk], max_group_size: int = 5) -> list[ChunkGroup]:
    if not chunks:
        return []

    groups: list[ChunkGroup] = []

    for chunk in chunks:
        if chunk.section is not None:
            if groups and groups[-1].section == chunk.section:
                groups[-1].chunk_indices.append(chunk.index)
            else:
                groups.append(ChunkGroup(chunk_indices=[chunk.index], section=chunk.section))
        else:
            if groups:
                groups[-1].chunk_indices.append(chunk.index)
            else:
                groups.append(ChunkGroup(chunk_indices=[chunk.index], section=None))

    split: list[ChunkGroup] = []
    for g in groups:
        for i in range(0, len(g.chunk_indices), max_group_size):
            split.append(ChunkGroup(
                chunk_indices=g.chunk_indices[i : i + max_group_size],
                section=g.section,
            ))

    return split


def generate_queries(
    chunks: list[Chunk],
    groups: list[ChunkGroup],
    client,
    model: str,
    call_collector: list[LLMCallRecord] | None = None,
    *,
    max_workers: int = 32,
    return_fallbacks: bool = False,
) -> list[QueryResult] | tuple[list[QueryResult], list[int]]:
    results: list[QueryResult] = []
    fallback_indices: list[int] = []

    def _process_group(group: ChunkGroup) -> tuple[ChunkGroup, GeneratedQueries | None]:
        chunk_text = "\n\n".join(chunks[i].text for i in group.chunk_indices)
        messages = render_prompt("generate_queries", {"chunk_text": chunk_text})

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                response_model=GeneratedQueries,
                messages=messages,
            )
        except Exception:
            logger.warning(
                "Query generation failed for group section=%s chunks=%s, falling back",
                group.section,
                group.chunk_indices,
            )
            return group, None
        duration_ms = (time.time() - t0) * 1000

        if call_collector is not None:
            call_collector.append(
                LLMCallRecord(
                    call_id=f"querygen-{group.chunk_indices[0]}",
                    stage="query_gen",
                    chunk_index=group.chunk_indices[0],
                    risk_ids=[],
                    messages=messages,
                    response={"queries": response.queries},
                    duration_ms=duration_ms,
                    result_summary=f"{len(response.queries)} queries generated",
                )
            )

        return group, response

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_group, g): g for g in groups}
        group_results: list[tuple[ChunkGroup, GeneratedQueries | None]] = []
        for future in as_completed(futures):
            group_results.append(future.result())

    # Sort by first chunk index to preserve document order
    group_results.sort(key=lambda gr: gr[0].chunk_indices[0])

    for group, response in group_results:
        if response is None:
            fallback_indices.extend(group.chunk_indices)
            continue
        for query in response.queries:
            if query.strip():
                results.append(QueryResult(
                    query=query.strip(),
                    chunk_indices=list(group.chunk_indices),
                    section=group.section,
                ))

    if return_fallbacks:
        return results, fallback_indices
    return results
