from types import SimpleNamespace

from concorde_policy_mapper.extract.expand import (
    ExpandedRisk,
    build_expansion_graph,
    expand_with_siblings,
    group_for_grounding,
)


def _make_risk(
    rid,
    is_part_of="",
    exact_mappings=None,
    close_mappings=None,
    broad_mappings=None,
    narrow_mappings=None,
    related_mappings=None,
):
    return SimpleNamespace(
        id=rid,
        isPartOf=is_part_of,
        exact_mappings=exact_mappings,
        close_mappings=close_mappings,
        broad_mappings=broad_mappings,
        narrow_mappings=narrow_mappings,
        related_mappings=related_mappings,
    )


# ---------------------------------------------------------------------------
# build_expansion_graph
# ---------------------------------------------------------------------------


def test_build_expansion_graph_parent_siblings():
    """Three risks sharing a parent should each expand to the other two."""
    risks = [
        _make_risk("R-001", is_part_of="parent-A"),
        _make_risk("R-002", is_part_of="parent-A"),
        _make_risk("R-003", is_part_of="parent-A"),
    ]
    graph = build_expansion_graph(risks)

    assert graph["R-001"] == {"R-002", "R-003"}
    assert graph["R-002"] == {"R-001", "R-003"}
    assert graph["R-003"] == {"R-001", "R-002"}


def test_build_expansion_graph_cross_mappings():
    """Two risks with exact_mappings to each other produce bidirectional edges."""
    risks = [
        _make_risk("R-001", exact_mappings=["R-002"]),
        _make_risk("R-002"),
    ]
    graph = build_expansion_graph(risks)

    assert "R-002" in graph["R-001"]
    assert "R-001" in graph["R-002"]


def test_build_expansion_graph_string_mapping():
    """A mapping attribute that is a string (not a list) is handled correctly."""
    risks = [
        _make_risk("R-001", close_mappings="R-002"),
        _make_risk("R-002"),
    ]
    graph = build_expansion_graph(risks)

    assert "R-002" in graph["R-001"]
    assert "R-001" in graph["R-002"]


def test_build_expansion_graph_filters_out_of_catalog():
    """Mapping targets not present in the catalog are filtered out."""
    risks = [
        _make_risk("R-001", exact_mappings=["R-999"]),
        _make_risk("R-002"),
    ]
    graph = build_expansion_graph(risks)

    # R-999 is not in the catalog, so it must not appear
    assert "R-999" not in graph["R-001"]
    assert graph["R-001"] == set()


def test_build_expansion_graph_no_expansions():
    """An isolated risk with no parent and no mappings has an empty expansion set."""
    risks = [
        _make_risk("R-001"),
        _make_risk("R-002", is_part_of="parent-A"),
    ]
    graph = build_expansion_graph(risks)

    assert graph["R-001"] == set()


def test_build_expansion_graph_multiple_mapping_types():
    """Edges from different mapping types are all included."""
    risks = [
        _make_risk("R-001", broad_mappings=["R-002"], narrow_mappings=["R-003"]),
        _make_risk("R-002"),
        _make_risk("R-003"),
    ]
    graph = build_expansion_graph(risks)

    assert graph["R-001"] == {"R-002", "R-003"}
    assert "R-001" in graph["R-002"]
    assert "R-001" in graph["R-003"]


def test_build_expansion_graph_none_mappings():
    """None-valued mapping attributes do not cause errors."""
    risks = [
        _make_risk("R-001", exact_mappings=None, related_mappings=None),
        _make_risk("R-002"),
    ]
    graph = build_expansion_graph(risks)

    assert graph["R-001"] == set()


def test_build_expansion_graph_combined_parent_and_mapping():
    """Siblings from parentage and mapping targets are both included."""
    risks = [
        _make_risk("R-001", is_part_of="parent-A", exact_mappings=["R-003"]),
        _make_risk("R-002", is_part_of="parent-A"),
        _make_risk("R-003"),
    ]
    graph = build_expansion_graph(risks)

    assert graph["R-001"] == {"R-002", "R-003"}


# ---------------------------------------------------------------------------
# expand_with_siblings
# ---------------------------------------------------------------------------


def test_expand_with_siblings_basic():
    """Siblings not already found are returned as ExpandedRisk."""
    expansion_graph = {
        "R-001": {"R-002", "R-003"},
    }
    risk_lookup = {
        "R-002": {"name": "Risk Two", "description": "Desc two"},
        "R-003": {"name": "Risk Three", "description": "Desc three"},
    }
    result = expand_with_siblings({"R-001"}, expansion_graph, risk_lookup)

    expanded_ids = {er.risk_id for er in result}
    assert expanded_ids == {"R-002", "R-003"}
    for er in result:
        assert er.source_risk_id == "R-001"
        assert er.risk_name == risk_lookup[er.risk_id]["name"]
        assert er.risk_description == risk_lookup[er.risk_id]["description"]


def test_expand_with_siblings_skips_already_found():
    """Risks already in merged_risk_ids are not returned."""
    expansion_graph = {
        "R-001": {"R-002", "R-003"},
    }
    risk_lookup = {
        "R-002": {"name": "Risk Two", "description": "Desc two"},
        "R-003": {"name": "Risk Three", "description": "Desc three"},
    }
    result = expand_with_siblings({"R-001", "R-002"}, expansion_graph, risk_lookup)

    expanded_ids = {er.risk_id for er in result}
    assert "R-002" not in expanded_ids
    assert "R-003" in expanded_ids


def test_expand_with_siblings_skips_missing_lookup():
    """Siblings not present in risk_lookup are silently skipped."""
    expansion_graph = {
        "R-001": {"R-002", "R-003"},
    }
    risk_lookup = {
        "R-002": {"name": "Risk Two", "description": "Desc two"},
        # R-003 intentionally absent
    }
    result = expand_with_siblings({"R-001"}, expansion_graph, risk_lookup)

    expanded_ids = {er.risk_id for er in result}
    assert expanded_ids == {"R-002"}


def test_expand_with_siblings_strips_trailing_name():
    """Risk IDs with trailing names ('R-001 bias') are stripped to 'R-001'."""
    expansion_graph = {
        "R-001": {"R-002"},
    }
    risk_lookup = {
        "R-002": {"name": "Risk Two", "description": "Desc two"},
    }
    # The merged set contains an ID with a trailing name part
    result = expand_with_siblings({"R-001 bias risk"}, expansion_graph, risk_lookup)

    expanded_ids = {er.risk_id for er in result}
    assert "R-002" in expanded_ids


def test_expand_with_siblings_empty_graph():
    """No expansions when the graph has no edges for found risks."""
    expansion_graph = {
        "R-001": set(),
    }
    risk_lookup = {
        "R-002": {"name": "Risk Two", "description": "Desc two"},
    }
    result = expand_with_siblings({"R-001"}, expansion_graph, risk_lookup)
    assert result == []


def test_expand_with_siblings_no_duplicates():
    """Each sibling appears only once even if reachable from multiple found risks."""
    expansion_graph = {
        "R-001": {"R-003"},
        "R-002": {"R-003"},
    }
    risk_lookup = {
        "R-003": {"name": "Risk Three", "description": "Desc three"},
    }
    result = expand_with_siblings({"R-001", "R-002"}, expansion_graph, risk_lookup)

    assert len(result) == 1
    assert result[0].risk_id == "R-003"


# ---------------------------------------------------------------------------
# group_for_grounding
# ---------------------------------------------------------------------------


def test_group_for_grounding_basic():
    """Groups by parent and includes correct chunk indices."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
        ExpandedRisk("R-003", "Risk Three", "Desc three", source_risk_id="R-001"),
    ]
    found_risk_chunks = {"R-001": {0, 2, 5}}
    risk_to_parent = {"R-001": "parent-A", "R-002": "parent-A", "R-003": "parent-A"}

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    assert len(groups) == 1
    g = groups[0]
    assert g.parent == "parent-A"
    assert set(g.risk_ids) == {"R-002", "R-003"}
    assert g.chunk_indices == [0, 2, 5]
    assert "R-002" in g.risk_lookup
    assert "R-003" in g.risk_lookup


def test_group_for_grounding_no_chunks_skipped():
    """Groups whose source risks have no chunks are excluded."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
    ]
    found_risk_chunks = {}  # R-001 has no chunks
    risk_to_parent = {"R-002": "parent-A"}

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    assert len(groups) == 0


def test_group_for_grounding_truncates_chunks():
    """Chunk list is truncated to max_chunks_per_group."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
    ]
    found_risk_chunks = {"R-001": set(range(50))}
    risk_to_parent = {"R-002": "parent-A"}

    groups = group_for_grounding(
        expanded,
        found_risk_chunks,
        risk_to_parent,
        total_chunks=100,
        max_chunks_per_group=5,
    )

    assert len(groups) == 1
    assert len(groups[0].chunk_indices) == 5
    # Chunks should be sorted and truncated from the front
    assert groups[0].chunk_indices == [0, 1, 2, 3, 4]


def test_group_for_grounding_splits_large_groups():
    """Groups larger than max_risks_per_group are split into batches."""
    expanded = [
        ExpandedRisk(f"R-{i:03d}", f"Risk {i}", f"Desc {i}", source_risk_id="R-000")
        for i in range(1, 26)  # 25 expanded risks
    ]
    found_risk_chunks = {"R-000": {0, 1}}
    risk_to_parent = {f"R-{i:03d}": "parent-A" for i in range(26)}

    groups = group_for_grounding(
        expanded,
        found_risk_chunks,
        risk_to_parent,
        total_chunks=10,
        max_risks_per_group=10,
    )

    # 25 risks / 10 per group = 3 batches
    assert len(groups) == 3
    all_risk_ids = []
    for g in groups:
        assert len(g.risk_ids) <= 10
        assert g.parent == "parent-A"
        assert g.chunk_indices == [0, 1]
        all_risk_ids.extend(g.risk_ids)
    # All 25 risks are present across batches
    assert len(all_risk_ids) == 25


def test_group_for_grounding_falls_back_to_source_parent():
    """When expanded risk has no parent, falls back to source_risk_id's parent."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
    ]
    found_risk_chunks = {"R-001": {3, 7}}
    # R-002 has no parent entry, but R-001 does
    risk_to_parent = {"R-001": "parent-B"}

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    assert len(groups) == 1
    assert groups[0].parent == "parent-B"


def test_group_for_grounding_ungrouped_fallback():
    """When neither expanded risk nor source has a parent, falls back to 'ungrouped'."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
    ]
    found_risk_chunks = {"R-001": {0}}
    risk_to_parent = {}  # neither R-002 nor R-001 has a parent

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    assert len(groups) == 1
    assert groups[0].parent == "ungrouped"


def test_group_for_grounding_multiple_parents():
    """Expanded risks from different parents are placed in separate groups."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
        ExpandedRisk("R-004", "Risk Four", "Desc four", source_risk_id="R-003"),
    ]
    found_risk_chunks = {"R-001": {0}, "R-003": {5}}
    risk_to_parent = {
        "R-002": "parent-A",
        "R-004": "parent-B",
    }

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    assert len(groups) == 2
    parents = {g.parent for g in groups}
    assert parents == {"parent-A", "parent-B"}


def test_group_for_grounding_lookup_contains_correct_fields():
    """Each group's risk_lookup contains risk_id, risk_name, risk_description."""
    expanded = [
        ExpandedRisk("R-002", "Risk Two", "Desc two", source_risk_id="R-001"),
    ]
    found_risk_chunks = {"R-001": {0}}
    risk_to_parent = {"R-002": "parent-A"}

    groups = group_for_grounding(expanded, found_risk_chunks, risk_to_parent, total_chunks=10)

    lookup = groups[0].risk_lookup["R-002"]
    assert lookup == {
        "risk_id": "R-002",
        "risk_name": "Risk Two",
        "risk_description": "Desc two",
    }
