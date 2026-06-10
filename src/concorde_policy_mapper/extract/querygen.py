from __future__ import annotations

import logging
from dataclasses import dataclass

from concorde_policy_mapper.extract.parse import Chunk

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
