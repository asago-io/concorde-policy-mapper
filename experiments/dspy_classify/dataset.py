"""Build DSPy training/eval examples for NIST risk classification.

For each policy:
1. Load extracted risks from a baseline run (non-NIST risks only)
2. Load GT to determine which NIST risks should apply
3. Build dspy.Example with (extracted_risks, target_categories, expected_verdicts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dspy
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_GT_DIR = _ROOT / "evals" / "ground_truth"

TRAIN_POLICIES = [
    "sap", "cisco-supplier", "firstsource", "guy-nhs", "rdash-nhs",
    "dhs-gov", "eu-com", "ovic", "camden-borough-work", "llvm",
]
EVAL_POLICIES = [
    "ars", "leicestershire_police", "lse-legreg", "aus-gov", "lenovo",
    "prosus", "new-york-state", "lse-marking", "ebay", "vps",
]

CLASSIFY_TAXONOMY = "nist-ai-rmf"


def _load_enriched_gt(policy: str) -> list[dict]:
    gt_path = _GT_DIR / f"{policy}.yaml"
    if not gt_path.exists():
        return []
    data = yaml.safe_load(gt_path.read_text())
    if "risks" in data:
        return data["risks"]
    return []


def _format_extracted_risks(risks: list[dict]) -> str:
    lines = []
    for r in risks:
        quote = r.get("evidence_quote", "")
        if quote:
            lines.append(f"- **{r['risk_name']}** ({r['risk_id']}): \"{quote}\"")
        else:
            lines.append(f"- **{r['risk_name']}** ({r['risk_id']})")
    return "\n".join(lines)


def _format_target_categories(targets: list[dict]) -> str:
    lines = []
    for t in targets:
        lines.append(f"- **{t['name']}** ({t['id']}): {t['description']}")
    return "\n".join(lines)


def _build_example(
    policy: str,
    run_dir: Path,
    nist_risks: list[dict],
) -> dspy.Example | None:
    gt_risks = _load_enriched_gt(policy)
    if not gt_risks:
        return None

    gt_nist_ids = {
        r["id"] for r in gt_risks
        if r["id"].startswith("nist-")
    }

    result_path = run_dir / policy / "risk-extraction.json"
    if not result_path.exists():
        return None
    data = json.loads(result_path.read_text())

    extracted = []
    for r in data.get("risks", []):
        if r.get("taxonomy", "") == CLASSIFY_TAXONOMY:
            continue
        quote = ""
        evidence = r.get("evidence", [])
        if evidence:
            quote = evidence[0].get("text", "")[:200]
        extracted.append({
            "risk_id": r["risk_id"],
            "risk_name": r.get("risk_name", ""),
            "evidence_quote": quote,
        })

    if not extracted:
        return None

    expected_verdicts = []
    for nr in nist_risks:
        expected_verdicts.append({
            "risk_id": nr["id"],
            "applies": nr["id"] in gt_nist_ids,
        })

    return dspy.Example(
        extracted_risks=_format_extracted_risks(extracted),
        target_categories=_format_target_categories(
            [{"id": nr["id"], "name": nr["name"], "description": nr["description"]} for nr in nist_risks]
        ),
        expected_verdicts=expected_verdicts,
    ).with_inputs("extracted_risks", "target_categories")


def load_dataset(
    run_dir: Path,
    nexus_base_dir: str,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    from ai_atlas_nexus import AIAtlasNexus

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    nist_risks = [
        {"id": r.id, "name": r.name or "", "description": r.description or ""}
        for r in nexus.get_all_risks()
        if getattr(r, "isDefinedByTaxonomy", "") == CLASSIFY_TAXONOMY
    ]
    logger.info("Loaded %d NIST risks", len(nist_risks))

    train = []
    for policy in TRAIN_POLICIES:
        ex = _build_example(policy, run_dir, nist_risks)
        if ex:
            train.append(ex)
            n_pos = sum(1 for v in ex.expected_verdicts if v["applies"])
            logger.info("  train %s: %d NIST expected", policy, n_pos)

    val = []
    for policy in EVAL_POLICIES:
        ex = _build_example(policy, run_dir, nist_risks)
        if ex:
            val.append(ex)
            n_pos = sum(1 for v in ex.expected_verdicts if v["applies"])
            logger.info("  eval  %s: %d NIST expected", policy, n_pos)

    logger.info("Dataset: %d train, %d eval examples", len(train), len(val))
    return train, val
