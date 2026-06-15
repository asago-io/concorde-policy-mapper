"""Sibling expansion for risk candidates.

After retrieval + merge produces a document-level risk list, expands to
related risks via parent groups (isPartOf) and cross-taxonomy mappings
(exact/close/broad/narrow/related_mappings). Groups expanded risks with
their relevant chunks for efficient document-level grounding.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpandedRisk:
    risk_id: str
    risk_name: str
    risk_description: str
    source_risk_id: str


@dataclass
class GroundingGroup:
    parent: str
    risk_ids: list[str]
    risk_lookup: dict[str, dict]
    chunk_indices: list[int]


def build_expansion_graph(risks: list) -> dict[str, set[str]]:
    """Build a mapping from each risk_id to its expansion set.

    Expansion set = parent siblings + cross-taxonomy mapping targets.
    Only includes risks present in the active catalog.
    """
    catalog_ids = set()
    parent_to_children: dict[str, set[str]] = defaultdict(set)
    risk_to_parent: dict[str, str] = {}
    risk_graph: dict[str, set[str]] = defaultdict(set)

    for r in risks:
        rid = r.id
        catalog_ids.add(rid)
        parent = getattr(r, "isPartOf", "") or ""
        if parent:
            parent_to_children[parent].add(rid)
            risk_to_parent[rid] = parent

        for mtype in ("exact_mappings", "close_mappings", "broad_mappings", "narrow_mappings", "related_mappings"):
            mappings = getattr(r, mtype, None) or []
            if isinstance(mappings, str):
                mappings = [mappings]
            for target in mappings:
                risk_graph[rid].add(target)
                risk_graph[target].add(rid)

    graph: dict[str, set[str]] = {}
    for rid in catalog_ids:
        expanded = set()
        parent = risk_to_parent.get(rid, "")
        if parent:
            expanded.update(parent_to_children[parent])
        expanded.update(risk_graph.get(rid, set()))
        expanded.discard(rid)
        expanded &= catalog_ids
        graph[rid] = expanded

    logger.info(
        "Expansion graph: %d risks, %d with expansions, avg %.1f expanded per risk",
        len(catalog_ids),
        sum(1 for v in graph.values() if v),
        sum(len(v) for v in graph.values()) / len(graph) if graph else 0,
    )
    return graph


def expand_with_siblings(
    merged_risk_ids: set[str],
    expansion_graph: dict[str, set[str]],
    risk_lookup: dict[str, dict],
) -> list[ExpandedRisk]:
    """Expand merged risk IDs to include siblings not already found.

    Returns ExpandedRisk entries for risks that are in the expansion set
    of any found risk but weren't found by retrieval themselves.
    """
    already_found = {rid.split(" ")[0].strip() for rid in merged_risk_ids}
    expanded_ids: dict[str, str] = {}

    for rid in already_found:
        siblings = expansion_graph.get(rid, set())
        if not siblings and "---" in rid:
            siblings = expansion_graph.get(rid.rsplit("---", 1)[0], set())
        for sibling in siblings:
            if sibling not in already_found and sibling not in expanded_ids:
                expanded_ids[sibling] = rid

    result = []
    for rid, source in expanded_ids.items():
        info = risk_lookup.get(rid)
        if not info:
            continue
        result.append(
            ExpandedRisk(
                risk_id=rid,
                risk_name=info.get("name", ""),
                risk_description=info.get("description", ""),
                source_risk_id=source,
            )
        )

    logger.info(
        "Expanded %d found risks → %d new sibling candidates",
        len(already_found),
        len(result),
    )
    return result


def group_for_grounding(
    expanded: list[ExpandedRisk],
    found_risk_chunks: dict[str, set[int]],
    risk_to_parent: dict[str, str],
    total_chunks: int,
    max_risks_per_group: int = 15,
    max_chunks_per_group: int = 20,
) -> list[GroundingGroup]:
    """Group expanded risks by parent category with their relevant chunks.

    Relevant chunks = union of chunks where any found sibling of the group
    was a retrieval candidate.
    """
    groups: dict[str, list[ExpandedRisk]] = defaultdict(list)
    for er in expanded:
        parent = risk_to_parent.get(er.risk_id, risk_to_parent.get(er.source_risk_id, "ungrouped"))
        groups[parent].append(er)

    result = []
    for parent, risks in groups.items():
        source_rids = {er.source_risk_id for er in risks}
        relevant_chunks = set()
        for source_rid in source_rids:
            relevant_chunks.update(found_risk_chunks.get(source_rid, set()))

        if not relevant_chunks:
            continue

        chunk_list = sorted(relevant_chunks)
        if len(chunk_list) > max_chunks_per_group:
            chunk_list = chunk_list[:max_chunks_per_group]

        risk_list = risks
        if len(risk_list) > max_risks_per_group:
            batches = [risk_list[i : i + max_risks_per_group] for i in range(0, len(risk_list), max_risks_per_group)]
        else:
            batches = [risk_list]

        for batch in batches:
            lookup = {
                er.risk_id: {
                    "risk_id": er.risk_id,
                    "risk_name": er.risk_name,
                    "risk_description": er.risk_description,
                }
                for er in batch
            }
            result.append(
                GroundingGroup(
                    parent=parent,
                    risk_ids=[er.risk_id for er in batch],
                    risk_lookup=lookup,
                    chunk_indices=chunk_list,
                )
            )

    logger.info(
        "Grouped %d expanded risks into %d grounding groups",
        len(expanded),
        len(result),
    )
    return result
