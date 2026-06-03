"""Build DSPy training/eval examples for the risk judge from pipeline runs.

Uses actual pipeline candidates from a battery run (with --no-grounding) as
the source of positives and negatives. Each judge call in the run becomes a
training example with the real chunk text, candidate risks, and GT-based verdicts.

Two modes:
  1. Pipeline-mined (default): load judge calls from a --run-dir battery run.
     Candidates are the actual borderline risks the judge saw in production.
  2. Legacy cross-encoder mining: mine hard negatives per chunk using a local
     cross-encoder (previous approach, kept for comparison).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import dspy
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


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        desc = c.get("risk_description", "")
        if desc:
            lines.append(f"- {c['risk_id']}: {c.get('risk_name', '')} — {desc[:200]}")
        else:
            lines.append(f"- {c['risk_id']}: {c.get('risk_name', '')}")
    return "\n".join(lines)


def _find_policy_file(policy: str) -> Path | None:
    for ext in [".md", ".pdf", ".docx", ".html", ".txt"]:
        p = _POLICY_DIR / f"{policy}{ext}"
        if p.exists():
            return p
    return None


def _parse_chunks(policy: str, chunk_max_tokens: int = 512) -> list | None:
    """Parse and chunk a policy document. Returns list of Chunk objects or None."""
    from concorde_policy_mapper.extract.parse import chunk_documents, parse_document

    policy_file = _find_policy_file(policy)
    if not policy_file:
        logger.warning("No policy file for %s", policy)
        return None
    try:
        parsed = parse_document(policy_file)
    except Exception as e:
        logger.warning("Parse error for %s: %s", policy, e)
        return None
    return chunk_documents([parsed], max_tokens=chunk_max_tokens) or None


def _build_examples_from_run(
    policy: str,
    run_dir: Path,
    risk_lookup: dict[str, dict],
) -> list[dspy.Example]:
    """Build examples from a pipeline run's judge calls with full chunk text."""
    gt_path = _GT_DIR / f"{policy}.yaml"
    ext_path = run_dir / policy / "risk-extraction.json"
    if not gt_path.exists() or not ext_path.exists():
        return []

    gt_data = yaml.safe_load(gt_path.read_text())
    gt_ids = set(r["id"] for r in gt_data.get("risks", []))

    ext = json.loads(ext_path.read_text())

    chunk_max_tokens = ext.get("metadata", {}).get("chunk_max_tokens", 512)
    chunks = _parse_chunks(policy, chunk_max_tokens=chunk_max_tokens)
    if not chunks:
        return []

    chunk_texts = {i: c.text for i, c in enumerate(chunks)}

    examples = []

    for call in ext.get("llm_calls", []):
        if call["stage"] != "judge":
            continue

        chunk_index = call["chunk_index"]
        chunk_text = chunk_texts.get(chunk_index)
        if not chunk_text or len(chunk_text) < 50:
            continue

        candidates = []
        for rid in call["risk_ids"]:
            info = risk_lookup.get(rid, {})
            candidates.append({
                "risk_id": rid,
                "risk_name": info.get("name", ""),
                "risk_description": info.get("description", ""),
            })

        if not candidates:
            continue

        expected_verdicts = [
            {"risk_id": c["risk_id"], "relevant": c["risk_id"] in gt_ids}
            for c in candidates
        ]

        has_positive = any(v["relevant"] for v in expected_verdicts)
        has_negative = any(not v["relevant"] for v in expected_verdicts)
        if not has_positive and not has_negative:
            continue

        examples.append(dspy.Example(
            chunk_text=chunk_text,
            candidate_risks=_format_candidates(candidates),
            expected_verdicts=expected_verdicts,
        ).with_inputs("chunk_text", "candidate_risks"))

    return examples


def _build_threshold_examples_from_run(
    policy: str,
    run_dir: Path,
    risk_lookup: dict[str, dict],
    chunks: list | None = None,
    max_per_chunk: int = 10,
) -> list[dspy.Example]:
    """Build examples from threshold-accepted candidates (top-N by CE score).

    These are candidates the judge DOESN'T see in production (they skip the
    judge), but including some as training examples helps the model learn the
    full score range. We use a subset to avoid dominating the dataset.
    """
    gt_path = _GT_DIR / f"{policy}.yaml"
    ext_path = run_dir / policy / "risk-extraction.json"
    if not gt_path.exists() or not ext_path.exists():
        return []

    gt_data = yaml.safe_load(gt_path.read_text())
    gt_ids = set(r["id"] for r in gt_data.get("risks", []))

    ext = json.loads(ext_path.read_text())

    if chunks is None:
        chunk_max_tokens = ext.get("metadata", {}).get("chunk_max_tokens", 512)
        chunks = _parse_chunks(policy, chunk_max_tokens=chunk_max_tokens)
    if not chunks:
        return []

    chunk_texts = {i: c.text for i, c in enumerate(chunks)}

    threshold_by_chunk: dict[int, list[dict]] = {}
    for r in ext["risks"]:
        if r["accepted_by"] != "threshold":
            continue
        for c_info in ext.get("chunks", []):
            if r["risk_id"] in c_info.get("accepted_risk_ids", []):
                threshold_by_chunk.setdefault(c_info["index"], []).append(r)
                break

    examples = []
    for chunk_idx, risks in threshold_by_chunk.items():
        chunk_text = chunk_texts.get(chunk_idx)
        if not chunk_text or len(chunk_text) < 50:
            continue

        sample = risks[:max_per_chunk]
        candidates = []
        for r in sample:
            rid = r["risk_id"]
            info = risk_lookup.get(rid, {})
            candidates.append({
                "risk_id": rid,
                "risk_name": info.get("name", r.get("risk_name", "")),
                "risk_description": info.get("description", r.get("risk_description", "")),
            })

        if not candidates:
            continue

        expected_verdicts = [
            {"risk_id": c["risk_id"], "relevant": c["risk_id"] in gt_ids}
            for c in candidates
        ]

        has_positive = any(v["relevant"] for v in expected_verdicts)
        if not has_positive:
            continue

        examples.append(dspy.Example(
            chunk_text=chunk_text,
            candidate_risks=_format_candidates(candidates),
            expected_verdicts=expected_verdicts,
        ).with_inputs("chunk_text", "candidate_risks"))

    return examples


def load_dataset(
    nexus_base_dir: str,
    run_dir: str | None = None,
    seed: int = 42,
    include_threshold: bool = False,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load DSPy dataset for judge optimization.

    Args:
        nexus_base_dir: Path to ai-atlas-nexus repo.
        run_dir: Path to a battery run with --no-grounding (uses pipeline
            candidates as examples). If None, falls back to cross-encoder mining.
        seed: Random seed.
        include_threshold: Also include threshold-accepted candidates as examples.
    """
    random.seed(seed)

    from ai_atlas_nexus import AIAtlasNexus

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    risk_lookup = {}
    for r in nexus.get_all_risks():
        taxonomy = getattr(r, "isDefinedByTaxonomy", "") or ""
        if taxonomy in _EXCLUDED_TAXONOMIES:
            continue
        risk_lookup[r.id] = {"name": r.name or "", "description": r.description or ""}

    logger.info("Loaded %d risk-level risks from Nexus", len(risk_lookup))

    if run_dir:
        run_path = Path(run_dir)
        logger.info("Building dataset from pipeline run: %s", run_path)

        train, val = [], []
        for split_name, policies, dest in [
            ("train", TRAIN_POLICIES, train),
            ("eval", EVAL_POLICIES, val),
        ]:
            for policy in policies:
                examples = _build_examples_from_run(policy, run_path, risk_lookup)
                if include_threshold:
                    examples.extend(
                        _build_threshold_examples_from_run(policy, run_path, risk_lookup)
                    )
                dest.extend(examples)
                logger.info("  %s: %d examples", policy, len(examples))
            logger.info("%s total: %d examples", split_name, len(dest))

        return train, val

    logger.info("No --run-dir provided, falling back to cross-encoder mining")
    return _load_dataset_legacy(nexus_base_dir, risk_lookup, seed)


def _load_dataset_legacy(
    nexus_base_dir: str,
    risk_lookup: dict[str, dict],
    seed: int,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Legacy dataset builder using cross-encoder hard negative mining."""
    import numpy as np
    from sentence_transformers import CrossEncoder

    from concorde_policy_mapper.extract.parse import chunk_documents, parse_document

    all_risks = [
        {"risk_id": rid, "risk_name": info["name"], "risk_description": info["description"]}
        for rid, info in risk_lookup.items()
    ]

    logger.info("Loading cross-encoder for hard negative mining...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def _evidence_in_chunk(evidence_text: str, chunk_text: str) -> bool:
        ev_norm = " ".join(evidence_text.lower().split())
        chunk_norm = " ".join(chunk_text.lower().split())
        if ev_norm in chunk_norm:
            return True
        ev_words = set(ev_norm.split())
        chunk_words = set(chunk_norm.split())
        if not ev_words:
            return False
        return len(ev_words & chunk_words) / len(ev_words) >= 0.6

    def _build_for_policy(policy):
        gt_path = _GT_DIR / f"{policy}.yaml"
        if not gt_path.exists():
            return []
        gt_data = yaml.safe_load(gt_path.read_text())
        gt_risks = gt_data.get("risks", [])
        if not gt_risks:
            return []

        gt_ids = {r["id"] for r in gt_risks}
        gt_evidence = {r["id"]: r.get("evidence", []) for r in gt_risks}
        gt_names = {r["id"]: r.get("name", "") for r in gt_risks}

        policy_file = None
        for ext in [".md", ".pdf", ".docx", ".html", ".txt"]:
            p = _POLICY_DIR / f"{policy}{ext}"
            if p.exists():
                policy_file = p
                break
        if not policy_file:
            return []

        try:
            parsed = parse_document(policy_file)
        except Exception:
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

            non_gt = [r for r in all_risks if r["risk_id"] not in gt_ids]
            pairs = [(f"{r['risk_name']}: {r['risk_description']}", chunk.text) for r in non_gt]
            raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
            scored = sorted(zip(non_gt, sigmoid_scores), key=lambda x: x[1], reverse=True)
            n_neg = min(5, 12 - len(positives))
            hard_negatives = [r for r, _ in scored[:n_neg]]

            candidates = list(positives) + hard_negatives
            random.shuffle(candidates)

            positive_ids = {c["risk_id"] for c in positives}
            expected_verdicts = [
                {"risk_id": c["risk_id"], "relevant": c["risk_id"] in positive_ids}
                for c in candidates
            ]

            examples.append(dspy.Example(
                chunk_text=chunk.text,
                candidate_risks=_format_candidates(candidates),
                expected_verdicts=expected_verdicts,
            ).with_inputs("chunk_text", "candidate_risks"))

        return examples

    train, val = [], []
    for policy in TRAIN_POLICIES:
        examples = _build_for_policy(policy)
        train.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))
    for policy in EVAL_POLICIES:
        examples = _build_for_policy(policy)
        val.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))

    logger.info("Dataset: %d train, %d eval examples", len(train), len(val))
    return train, val
