"""Build DSPy training/eval examples for the risk judge from enriched GT.

For each policy document:
1. Parse + chunk (same as pipeline)
2. Match GT evidence to chunks to find positive risks per chunk
3. Mine hard negatives using the cross-encoder: score all non-GT risks
   against each chunk and take the highest-scoring false matches
4. Build dspy.Example with (chunk_text, candidate_risks, expected_verdicts)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import dspy
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_GT_DIR = _ROOT / "evals" / "ground_truth"
_POLICY_DIR = _ROOT / "policy_examples"

TRAIN_POLICIES = [
    "sap", "cisco-supplier", "firstsource", "guy-nhs", "rdash-nhs",
    "dhs-gov", "eu-com", "ovic", "camden-borough-work", "llvm",
    "amadeus", "fs-isac", "gray",
]
EVAL_POLICIES = [
    "ars", "leicestershire_police", "lse-legreg", "aus-gov", "lenovo",
    "prosus", "new-york-state", "lse-marking", "ebay", "vps",
    "npcc", "penn", "st-johns", "icrc",
]

_EXCLUDED_TAXONOMIES = {
    "nist-ai-rmf", "owasp-llm-2.0", "ailuminate-v1.0", "owasp-asi",
    "shieldgemma-taxonomy", "mit-ai-risk-repository-causal", "ibm-granite-guardian",
}

MAX_NEGATIVES_PER_CHUNK = 5
MAX_CANDIDATES_PER_EXAMPLE = 12


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _evidence_in_chunk(evidence_text: str, chunk_text: str) -> bool:
    ev_norm = _normalize(evidence_text)
    chunk_norm = _normalize(chunk_text)
    if ev_norm in chunk_norm:
        return True
    ev_words = set(ev_norm.split())
    chunk_words = set(chunk_norm.split())
    if not ev_words:
        return False
    overlap = len(ev_words & chunk_words) / len(ev_words)
    return overlap >= 0.6


def _find_policy_file(policy: str) -> Path | None:
    for ext in [".md", ".pdf", ".docx", ".html", ".txt"]:
        p = _POLICY_DIR / f"{policy}{ext}"
        if p.exists():
            return p
    return None


def _load_enriched_gt(policy: str) -> list[dict]:
    gt_path = _GT_DIR / f"{policy}.yaml"
    if not gt_path.exists():
        return []
    data = yaml.safe_load(gt_path.read_text())
    if "risks" in data:
        return data["risks"]
    return []


def _mine_hard_negatives(
    chunk_text: str,
    gt_ids: set[str],
    all_risks: list[dict],
    cross_encoder,
    top_k: int = MAX_NEGATIVES_PER_CHUNK,
) -> list[dict]:
    non_gt = [r for r in all_risks if r["risk_id"] not in gt_ids]
    if not non_gt:
        return []

    pairs = [
        (f"{r['risk_name']}: {r['risk_description']}", chunk_text)
        for r in non_gt
    ]
    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

    scored = sorted(zip(non_gt, sigmoid_scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        desc = c.get("risk_description", "")
        if desc:
            lines.append(f"- {c['risk_id']}: {c.get('risk_name', '')} — {desc[:200]}")
        else:
            lines.append(f"- {c['risk_id']}: {c.get('risk_name', '')}")
    return "\n".join(lines)


def _build_examples_for_policy(
    policy: str,
    all_risks: list[dict],
    risk_lookup: dict[str, dict],
    cross_encoder,
) -> list[dspy.Example]:
    gt_risks = _load_enriched_gt(policy)
    if not gt_risks:
        return []

    gt_ids = {r["id"] for r in gt_risks}
    gt_evidence = {r["id"]: r.get("evidence", []) for r in gt_risks}
    gt_names = {r["id"]: r.get("name", "") for r in gt_risks}

    policy_file = _find_policy_file(policy)
    if not policy_file:
        logger.warning("No policy file for %s", policy)
        return []

    from concorde_policy_mapper.extract.parse import parse_document, chunk_documents
    try:
        parsed = parse_document(policy_file)
    except Exception as e:
        logger.warning("Parse error for %s: %s", policy, e)
        return []
    chunks = chunk_documents([parsed], max_tokens=512)
    if not chunks:
        return []

    examples = []
    for chunk in chunks:
        positives = []
        for rid in gt_ids:
            evidence_list = gt_evidence.get(rid, [])
            if not evidence_list:
                continue
            if any(_evidence_in_chunk(ev.get("text", ""), chunk.text) for ev in evidence_list):
                info = risk_lookup.get(rid, {})
                positives.append({
                    "risk_id": rid,
                    "risk_name": gt_names.get(rid, info.get("name", "")),
                    "risk_description": info.get("description", ""),
                })

        if not positives:
            continue

        n_neg = min(MAX_NEGATIVES_PER_CHUNK, MAX_CANDIDATES_PER_EXAMPLE - len(positives))
        hard_negatives = _mine_hard_negatives(
            chunk.text, gt_ids, all_risks, cross_encoder, top_k=n_neg,
        )

        candidates = list(positives) + hard_negatives
        random.shuffle(candidates)

        if len(candidates) > MAX_CANDIDATES_PER_EXAMPLE:
            positive_ids = {c["risk_id"] for c in positives}
            pos_cands = [c for c in candidates if c["risk_id"] in positive_ids]
            neg_cands = [c for c in candidates if c["risk_id"] not in positive_ids]
            candidates = pos_cands[:MAX_CANDIDATES_PER_EXAMPLE]
            remaining = MAX_CANDIDATES_PER_EXAMPLE - len(candidates)
            candidates.extend(neg_cands[:remaining])

        expected_verdicts = []
        positive_ids = {c["risk_id"] for c in positives}
        for c in candidates:
            expected_verdicts.append({
                "risk_id": c["risk_id"],
                "relevant": c["risk_id"] in positive_ids,
            })

        example = dspy.Example(
            chunk_text=chunk.text,
            candidate_risks=_format_candidates(candidates),
            expected_verdicts=expected_verdicts,
        ).with_inputs("chunk_text", "candidate_risks")

        examples.append(example)

    return examples


def load_dataset(
    nexus_base_dir: str,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    random.seed(seed)

    from ai_atlas_nexus import AIAtlasNexus
    from sentence_transformers import CrossEncoder

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    risk_lookup = {}
    all_risks = []
    for r in nexus.get_all_risks():
        taxonomy = getattr(r, "isDefinedByTaxonomy", "") or ""
        if taxonomy in _EXCLUDED_TAXONOMIES:
            continue
        rid = r.id
        name = r.name or ""
        desc = r.description or ""
        risk_lookup[rid] = {"name": name, "description": desc}
        all_risks.append({
            "risk_id": rid,
            "risk_name": name,
            "risk_description": desc,
        })

    logger.info("Loaded %d risk-level risks from Nexus (excluded %d category taxonomies)",
                len(all_risks), len(_EXCLUDED_TAXONOMIES))
    logger.info("Loading cross-encoder for hard negative mining...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    logger.info("Building train examples from %d policies...", len(TRAIN_POLICIES))
    train = []
    for policy in TRAIN_POLICIES:
        examples = _build_examples_for_policy(
            policy, all_risks, risk_lookup, cross_encoder,
        )
        train.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))

    logger.info("Building eval examples from %d policies...", len(EVAL_POLICIES))
    val = []
    for policy in EVAL_POLICIES:
        examples = _build_examples_for_policy(
            policy, all_risks, risk_lookup, cross_encoder,
        )
        val.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))

    logger.info("Dataset: %d train, %d eval examples", len(train), len(val))
    return train, val
