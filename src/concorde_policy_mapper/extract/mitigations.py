from __future__ import annotations

import logging
from pathlib import Path

import yaml

from concorde_policy_mapper.extract.models import MitigationRef, RiskMatch

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "atlas_risk_to_actions.yaml"
)


def load_mitigation_index(
    path: Path | None = None,
) -> dict[str, list[MitigationRef]]:
    path = path or _DEFAULT_INDEX_PATH
    if not path.exists():
        logger.warning("Mitigation index not found at %s", path)
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return {}

    index: dict[str, list[MitigationRef]] = {}
    for risk_id, actions in raw.items():
        index[risk_id] = [
            MitigationRef(
                action_id=a["id"],
                action_name=a.get("name"),
                source=a["source"],
                category=a.get("category"),
                strategy=a.get("strategy"),
            )
            for a in actions
        ]
    return index


def build_action_descriptions(
    nexus_base_dir: str,
    data_dir: Path | None = None,
) -> dict[str, str]:
    """Build action_id → description lookup from Nexus YAML files and local data."""
    kg = Path(nexus_base_dir) / "src" / "ai_atlas_nexus" / "data" / "knowledge_graph"
    data_dir = data_dir or Path(__file__).resolve().parents[3] / "data"
    descs: dict[str, str] = {}

    sources = [
        (kg / "mit_ai_risk_repository_data_controls.yaml", "controls"),
        (kg / "nist_ai_rmf_actions_data.yaml", "actions"),
        (kg / "aiuc1_data.yaml", "rules"),
        (data_dir / "owasp_llm_2.0_actions_data.yaml", "actions"),
    ]
    for path, key in sources:
        if not path.exists():
            continue
        with open(path) as f:
            raw = yaml.safe_load(f)
        for entry in raw.get(key, []):
            eid = entry.get("id", "")
            desc = entry.get("description")
            if eid and desc:
                descs[eid] = desc.strip()

    credo_path = kg / "credo.yaml"
    if credo_path.exists():
        with open(credo_path) as f:
            raw = yaml.safe_load(f)
        for entry in raw.get("actions", []):
            eid = entry.get("id", "")
            if eid.startswith("credo-act-"):
                desc = entry.get("description")
                if desc:
                    descs[eid] = desc.strip()

    return descs


def build_risk_crossmap(nexus_base_dir: str) -> dict[str, set[str]]:
    """Build non-atlas-risk-id → set[atlas-risk-id] from Nexus mapping files."""
    kg = Path(nexus_base_dir) / "src" / "ai_atlas_nexus" / "data" / "knowledge_graph" / "mappings"
    crossmap: dict[str, set[str]] = {}

    mapping_files = [
        "credo-ucf.sssom_from_tsv_data.yaml",
        "mit-ai-risk-repository_ibm-risk-atlas_from_tsv_data.yaml",
    ]
    predicates = ("close_mappings", "related_mappings", "broad_mappings", "exact_mappings")

    for filename in mapping_files:
        path = kg / filename
        if not path.exists():
            continue
        with open(path) as f:
            raw = yaml.safe_load(f)
        for entry in raw.get("entries", []):
            eid = entry["id"]
            for pred in predicates:
                for target in entry.get(pred, []):
                    if eid.startswith("atlas-") or target.startswith("atlas-"):
                        non_atlas = eid if not eid.startswith("atlas-") else target
                        atlas = target if target.startswith("atlas-") else eid
                        if non_atlas != atlas:
                            crossmap.setdefault(non_atlas, set()).add(atlas)

    return crossmap


def enrich_with_mitigations(
    risks: list[RiskMatch],
    index: dict[str, list[MitigationRef]],
    descriptions: dict[str, str] | None = None,
    risk_crossmap: dict[str, set[str]] | None = None,
) -> None:
    for risk in risks:
        mitigations = index.get(risk.risk_id, [])

        if not mitigations and risk_crossmap:
            atlas_ids = risk_crossmap.get(risk.risk_id, set())
            seen: set[str] = set()
            for atlas_id in sorted(atlas_ids):
                for m in index.get(atlas_id, []):
                    if m.action_id not in seen:
                        seen.add(m.action_id)
                        mitigations.append(m)

        if descriptions:
            for m in mitigations:
                m.description = descriptions.get(m.action_id)
        risk.mitigations = mitigations
