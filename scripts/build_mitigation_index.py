#!/usr/bin/env python3
"""Build a unified atlas-risk-id → actions mitigation index.

Reads 5 local data files with direct action → atlas-* risk mappings,
assigns categories from data/mitigation_categories.yaml, and writes
a single YAML index file.

All transitive cross-framework mappings have been pre-resolved into
the per-source data files. No Nexus dependency at build time.

Usage:
    python scripts/build_mitigation_index.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Category assignment ───────────────────────────────────────────────────


def _load_categories(data_dir: Path) -> dict:
    path = data_dir / "mitigation_categories.yaml"
    if not path.exists():
        print(f"  WARNING: {path} not found, categories will be omitted")
        return {}
    return _load_yaml(path)


def _build_mit_action_groups(data_dir: Path) -> dict[str, str]:
    """Parse MIT YAML comment headers to map action IDs → top-level group number."""
    path = data_dir / "mit_ai_risk_mitigation_to_atlas_data.yaml"
    if not path.exists():
        return {}
    group_re = re.compile(r"#\s*[-=]+\s*Group\s+(\d+)")
    action_re = re.compile(r"^\s*-\s*id:\s*(\S+)")

    action_groups: dict[str, str] = {}
    current_group: str | None = None
    with open(path) as f:
        for line in f:
            gm = group_re.search(line)
            if gm:
                current_group = gm.group(1)
                continue
            am = action_re.match(line)
            if am and current_group:
                action_groups[am.group(1)] = current_group
    return action_groups


def _resolve_rule(value) -> dict:
    """Normalize a rule value to {category: ..., risk_control: ...} dict."""
    if isinstance(value, dict):
        return value
    return {"category": value}


def _assign_labels(
    action_id: str,
    source: str,
    categories: dict,
    mit_groups: dict[str, str],
) -> dict | None:
    """Return {category: ..., risk_control: ...} for an action, or None."""
    explicit = categories.get("actions", {})
    if action_id in explicit:
        return _resolve_rule(explicit[action_id])

    if source == "mit-ai-risk-repository":
        group = mit_groups.get(action_id)
        if group:
            val = categories.get("mit_groups", {}).get(group)
            if val:
                return _resolve_rule(val)

    if source == "nist-ai-rmf":
        prefix = action_id.split("-")[0]
        val = categories.get("nist_prefixes", {}).get(prefix)
        if val:
            return _resolve_rule(val)

    if source == "aiuc1":
        m = re.match(r"aiuc1-req-([a-f])", action_id)
        if m:
            val = categories.get("aiuc1_prefixes", {}).get(m.group(1))
            if val:
                return _resolve_rule(val)

    return None


# ── Source collectors ─────────────────────────────────────────────────────


_SOURCES = [
    ("mit_ai_risk_mitigation_to_atlas_data.yaml", "mit-ai-risk-repository"),
    ("owasp_llm_2.0_actions_data.yaml", "owasp-llm-2.0"),
    ("nist_ai_rmf_actions_to_atlas_data.yaml", "nist-ai-rmf"),
    ("credo_ucf_actions_to_atlas_data.yaml", "credo-ucf"),
    ("aiuc1_actions_to_atlas_data.yaml", "aiuc1"),
]


def _collect_source(
    data_dir: Path,
    filename: str,
    source: str,
    index: dict[str, list[dict]],
) -> int:
    path = data_dir / filename
    if not path.exists():
        print(f"  SKIP {source}: {path} not found")
        return 0
    raw = _load_yaml(path)
    count = 0
    for action in raw.get("actions", []):
        atlas_risks = [r for r in action.get("hasRelatedRisk", []) if r.startswith("atlas-")]
        for risk_id in atlas_risks:
            entry = {"id": action["id"], "source": source}
            if action.get("name"):
                entry["name"] = action["name"]
            index[risk_id].append(entry)
            count += 1
    return count


# ── Dedup and output ──────────────────────────────────────────────────────


def _dedup_actions(index: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Remove duplicate action entries per risk (same id + source)."""
    deduped: dict[str, list[dict]] = {}
    for risk_id, actions in sorted(index.items()):
        seen: set[tuple[str, str]] = set()
        unique: list[dict] = []
        for a in actions:
            key = (a["id"], a["source"])
            if key not in seen:
                seen.add(key)
                unique.append(a)
        deduped[risk_id] = unique
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Build mitigation index")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/atlas_risk_to_actions.yaml)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    output_path = Path(args.output) if args.output else data_dir / "atlas_risk_to_actions.yaml"

    index: dict[str, list[dict]] = defaultdict(list)

    print("Collecting mitigations from 5 sources (all direct mappings):")
    source_stats = {}
    for filename, source in _SOURCES:
        links = _collect_source(data_dir, filename, source, index)
        label = source.upper().replace("-", " ").split()[0]
        source_stats[source] = links
        print(f"  {label:6s}: {links} action-risk links")

    index = _dedup_actions(index)

    categories = _load_categories(data_dir)
    mit_groups = _build_mit_action_groups(data_dir) if categories else {}

    labeled = 0
    unlabeled = 0
    for actions in index.values():
        for action in actions:
            labels = _assign_labels(
                action["id"], action["source"], categories, mit_groups,
            )
            if labels:
                if "category" in labels:
                    action["category"] = labels["category"]
                if "risk_control" in labels:
                    action["risk_control"] = labels["risk_control"]
                labeled += 1
            else:
                unlabeled += 1

    total_actions = sum(len(v) for v in index.values())
    print(f"\nIndex: {len(index)} Atlas risks → {total_actions} total action entries")
    print(f"Labels: {labeled} assigned, {unlabeled} unassigned")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_line = ", ".join(
        f"{src.split('-')[0].upper()} ({n})" for src, n in source_stats.items()
    )
    header = (
        "# Auto-generated by scripts/build_mitigation_index.py\n"
        "# Maps Atlas risk IDs to recommended mitigation actions across 5 frameworks.\n"
        f"# Sources: {stats_line}\n"
        f"# Total: {len(index)} risks, {total_actions} action entries\n"
        f"# Labels: {labeled} assigned, {unlabeled} unassigned\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(
            dict(index),
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
