"""Build DSPy training/eval examples for the grounding stage from enriched GT.

For each policy document:
1. Parse + chunk (same as pipeline)
2. Match GT evidence to chunks to find positive risks per chunk
3. Mine hard negatives WITHOUT cross-encoder (scores are ~random):
   a. Same-document, other-chunk negatives: GT risks whose evidence is in
      different chunks of the same policy (hard — same domain, wrong chunk)
   b. Random catalog negatives: fill remaining slots from the full risk catalog
   c. Optional pipeline negatives: if --run-dir is given, load
      grounding_filtered_candidates from actual pipeline runs
4. Build dspy.Example with (chunk_text, candidate_risks, expected_verdicts)
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
]
EVAL_POLICIES = [
    "ars", "leicestershire_police", "lse-legreg", "aus-gov", "lenovo",
    "prosus", "new-york-state", "lse-marking", "ebay", "vps",
]

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


def _matching_evidence_texts(
    evidence_list: list[dict], chunk_text: str
) -> list[str]:
    return [
        ev.get("text", "")
        for ev in evidence_list
        if ev.get("text") and _evidence_in_chunk(ev["text"], chunk_text)
    ]


def _mine_other_chunk_negatives(
    chunk_text: str,
    chunk_positive_ids: set[str],
    all_chunk_positives: dict[str, set[str]],
    gt_risks: list[dict],
    risk_lookup: dict[str, dict],
) -> list[dict]:
    """GT risks from the same document whose evidence is in other chunks."""
    doc_gt_ids = {r["id"] for r in gt_risks}
    other_chunk_ids = set()
    for _chunk_key, pos_ids in all_chunk_positives.items():
        other_chunk_ids |= pos_ids
    candidates_ids = (doc_gt_ids & other_chunk_ids) - chunk_positive_ids
    result = []
    for rid in candidates_ids:
        info = risk_lookup.get(rid, {})
        gt_entry = next((r for r in gt_risks if r["id"] == rid), None)
        result.append({
            "risk_id": rid,
            "risk_name": gt_entry.get("name", "") if gt_entry else info.get("name", ""),
            "risk_description": info.get("description", ""),
        })
    return result


def _load_pipeline_negatives(
    run_dir: Path, policy: str
) -> dict[int, list[dict]]:
    """Load grounding_filtered_candidates from a pipeline run, keyed by chunk_index."""
    result_path = run_dir / policy / "risk-extraction.json"
    if not result_path.exists():
        return {}
    data = json.loads(result_path.read_text())
    by_chunk: dict[int, list[dict]] = {}
    for c in data.get("grounding_filtered_candidates", []):
        ci = c.get("chunk_index", -1)
        if ci < 0:
            continue
        by_chunk.setdefault(ci, []).append({
            "risk_id": c["risk_id"],
            "risk_name": c.get("risk_name", ""),
            "risk_description": "",
        })
    return by_chunk


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
    run_dir: Path | None = None,
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

    from asago_policy_mapper.extract.parse import parse_document, chunk_documents

    try:
        parsed = parse_document(policy_file)
    except Exception as e:
        logger.warning("Parse error for %s: %s", policy, e)
        return []
    chunks = chunk_documents([parsed], max_tokens=512)
    if not chunks:
        return []

    chunk_positives_map: dict[int, set[str]] = {}
    chunk_evidence_map: dict[int, dict[str, list[str]]] = {}
    for i, chunk in enumerate(chunks):
        pos_ids = set()
        ev_map: dict[str, list[str]] = {}
        for rid in gt_ids:
            evidence_list = gt_evidence.get(rid, [])
            if not evidence_list:
                continue
            matched_texts = _matching_evidence_texts(evidence_list, chunk.text)
            if matched_texts:
                pos_ids.add(rid)
                ev_map[rid] = matched_texts
        chunk_positives_map[i] = pos_ids
        chunk_evidence_map[i] = ev_map

    pipeline_negs = (
        _load_pipeline_negatives(run_dir, policy) if run_dir else {}
    )

    non_gt_risks = [r for r in all_risks if r["risk_id"] not in gt_ids]

    examples = []
    for i, chunk in enumerate(chunks):
        pos_ids = chunk_positives_map[i]
        if not pos_ids:
            continue

        positives = []
        for rid in pos_ids:
            info = risk_lookup.get(rid, {})
            positives.append({
                "risk_id": rid,
                "risk_name": gt_names.get(rid, info.get("name", "")),
                "risk_description": info.get("description", ""),
            })

        n_neg = min(MAX_NEGATIVES_PER_CHUNK, MAX_CANDIDATES_PER_EXAMPLE - len(positives))

        negatives: list[dict] = []

        other_chunk_negs = _mine_other_chunk_negatives(
            chunk.text, pos_ids, chunk_positives_map, gt_risks, risk_lookup,
        )
        random.shuffle(other_chunk_negs)
        negatives.extend(other_chunk_negs[:n_neg])

        if len(negatives) < n_neg and i in pipeline_negs:
            seen_ids = {n["risk_id"] for n in negatives}
            for pn in pipeline_negs[i]:
                if pn["risk_id"] not in seen_ids and pn["risk_id"] not in pos_ids:
                    info = risk_lookup.get(pn["risk_id"], {})
                    pn["risk_description"] = info.get("description", pn.get("risk_description", ""))
                    negatives.append(pn)
                    seen_ids.add(pn["risk_id"])
                if len(negatives) >= n_neg:
                    break

        if len(negatives) < n_neg:
            seen_ids = pos_ids | {n["risk_id"] for n in negatives}
            pool = [r for r in non_gt_risks if r["risk_id"] not in seen_ids]
            random.shuffle(pool)
            negatives.extend(pool[: n_neg - len(negatives)])

        candidates = list(positives) + negatives
        random.shuffle(candidates)

        if len(candidates) > MAX_CANDIDATES_PER_EXAMPLE:
            positive_ids = {c["risk_id"] for c in positives}
            pos_cands = [c for c in candidates if c["risk_id"] in positive_ids]
            neg_cands = [c for c in candidates if c["risk_id"] not in positive_ids]
            candidates = pos_cands[:MAX_CANDIDATES_PER_EXAMPLE]
            remaining = MAX_CANDIDATES_PER_EXAMPLE - len(candidates)
            candidates.extend(neg_cands[:remaining])

        expected_verdicts = []
        for c in candidates:
            rid = c["risk_id"]
            is_positive = rid in pos_ids
            expected_verdicts.append({
                "risk_id": rid,
                "grounded": is_positive,
                "expected_quotes": chunk_evidence_map[i].get(rid, []) if is_positive else [],
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
    run_dir: str | None = None,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    random.seed(seed)

    from ai_atlas_nexus import AIAtlasNexus

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    risk_lookup: dict[str, dict] = {}
    all_risks: list[dict] = []
    for r in nexus.get_all_risks():
        rid = r.id
        name = r.name or ""
        desc = r.description or ""
        risk_lookup[rid] = {"name": name, "description": desc}
        all_risks.append({
            "risk_id": rid,
            "risk_name": name,
            "risk_description": desc,
        })

    logger.info("Loaded %d risks from Nexus", len(all_risks))

    run_path = Path(run_dir) if run_dir else None

    logger.info("Building train examples from %d policies...", len(TRAIN_POLICIES))
    train = []
    for policy in TRAIN_POLICIES:
        examples = _build_examples_for_policy(
            policy, all_risks, risk_lookup, run_path,
        )
        train.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))

    logger.info("Building eval examples from %d policies...", len(EVAL_POLICIES))
    val = []
    for policy in EVAL_POLICIES:
        examples = _build_examples_for_policy(
            policy, all_risks, risk_lookup, run_path,
        )
        val.extend(examples)
        logger.info("  %s: %d examples", policy, len(examples))

    logger.info("Dataset: %d train, %d eval examples", len(train), len(val))
    return train, val
