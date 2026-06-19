from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

_TAXONOMY_PREFIXES = [
    ("ai-risk-taxonomy-", "ai-risk-taxonomy"),
    ("atlas-", "ibm-risk-atlas"),
    ("credo-", "credo-ucf"),
    ("mit-ai-risk-", "mit-ai-risk-repository"),
    ("mit-ai-causal-", "mit-ai-risk-repository-causal"),
    ("nist-", "nist-ai-rmf"),
    ("ail-", "ailuminate-v1.0"),
    ("granite-", "ibm-granite-guardian"),
    ("llm", "owasp-llm-2.0"),
    ("asi0", "owasp-asi"),
    ("shieldgemma-", "shieldgemma-taxonomy"),
]

_CATEGORY_TAXONOMIES = {"nist-ai-rmf", "owasp-llm-2.0", "ailuminate-v1.0", "owasp-asi"}

_STRONG_PREDICATES = {"skos:exactMatch", "skos:closeMatch", "skos:broadMatch"}


def _sanitise_risk_id(risk_id: str) -> str:
    """Strip trailing name/description appended after a space in malformed Nexus IDs."""
    return risk_id.split(" ")[0].strip()


def _infer_taxonomy(risk_id: str) -> str:
    for prefix, taxonomy in _TAXONOMY_PREFIXES:
        if risk_id.startswith(prefix):
            return taxonomy
    return "unknown"


def _load_risk_to_category_map(
    sssom_path: Path | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Load SSSOM mapping and return {risk_id: {category_taxonomy: {category_ids}}}.

    Only includes mappings with strong predicates (exact/close/broad).
    """
    if sssom_path is None:
        sssom_path = Path(__file__).resolve().parents[3] / "data" / "risk_to_category.sssom.tsv"
    if not sssom_path.exists():
        return {}

    mapping: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with open(sssom_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("subject_id") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            subj, _subj_src, pred, obj, obj_src = parts[:5]
            if pred not in _STRONG_PREDICATES:
                continue
            obj_tax = obj_src.strip()
            if obj_tax in _CATEGORY_TAXONOMIES:
                mapping[subj][obj_tax].add(obj)
    return dict(mapping)


def _derive_categories(
    risk_ids: set[str],
    risk_to_cat: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Derive category-level risk sets from risk-level IDs, per category taxonomy."""
    by_taxonomy: dict[str, set[str]] = defaultdict(set)
    for rid in risk_ids:
        for cat_tax, cat_ids in risk_to_cat.get(rid, {}).items():
            by_taxonomy[cat_tax].update(cat_ids)
    return dict(by_taxonomy)


def _build_taxonomy_map(ext_data: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for r in ext_data.get("risks", []):
        rid = _sanitise_risk_id(r.get("risk_id", ""))
        tax = r.get("taxonomy", "")
        if rid and tax:
            mapping[rid] = tax
    for fc in ext_data.get("grounding_filtered_candidates", []):
        rid = _sanitise_risk_id(fc.get("risk_id", ""))
        tax = fc.get("taxonomy", "")
        if rid and tax and rid not in mapping:
            mapping[rid] = tax
    return mapping


def _compute_prf(matched: int, expected: int, extracted: int) -> tuple[float, float, float]:
    missing = expected - matched
    spurious = extracted - matched
    precision = matched / (matched + spurious) if matched + spurious > 0 else 0.0
    recall = matched / (matched + missing) if matched + missing > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _per_taxonomy_breakdown(
    expected: set[str],
    extracted: set[str],
    matched: set[str],
    taxonomy_map: dict[str, str],
) -> dict[str, dict]:
    all_ids = expected | extracted
    by_tax: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"expected": set(), "extracted": set(), "matched": set()}
    )

    for rid in all_ids:
        tax = taxonomy_map.get(rid) or _infer_taxonomy(rid)
        if rid in expected:
            by_tax[tax]["expected"].add(rid)
        if rid in extracted:
            by_tax[tax]["extracted"].add(rid)
        if rid in matched:
            by_tax[tax]["matched"].add(rid)

    result = {}
    for tax in sorted(by_tax):
        t = by_tax[tax]
        n_exp = len(t["expected"])
        n_ext = len(t["extracted"])
        n_match = len(t["matched"])
        precision, recall, f1 = _compute_prf(n_match, n_exp, n_ext)
        result[tax] = {
            "expected": n_exp,
            "extracted": n_ext,
            "matched": n_match,
            "missing": sorted(t["expected"] - t["matched"]),
            "spurious": sorted(t["extracted"] - t["matched"]),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return result


def evaluate_extraction(
    ground_truth_path: Path,
    extracted_path: Path,
    policy_name: str = "",
    min_recall: float = 0.80,
    min_precision: float = 0.60,
    sssom_path: Path | None = None,
) -> dict:
    gt_data = yaml.safe_load(ground_truth_path.read_text())
    if "risks" in gt_data:
        expected = {_sanitise_risk_id(r["id"]) for r in gt_data["risks"]}
    else:
        expected = {_sanitise_risk_id(str(rid)) for rid in gt_data["risk_ids"]}

    ext_data = json.loads(extracted_path.read_text())
    extracted = {_sanitise_risk_id(r["risk_id"]) for r in ext_data.get("risks", [])}

    matched = expected & extracted
    missing = sorted(expected - extracted)
    spurious = sorted(extracted - expected)

    precision = len(matched) / (len(matched) + len(spurious)) if matched or spurious else 0.0
    recall = len(matched) / (len(matched) + len(missing)) if matched or missing else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    taxonomy_map = _build_taxonomy_map(ext_data)
    per_taxonomy = _per_taxonomy_breakdown(expected, extracted, matched, taxonomy_map)

    risk_to_cat = _load_risk_to_category_map(sssom_path)
    category_eval = _evaluate_categories(expected, extracted, risk_to_cat)

    return {
        "policy": policy_name,
        "total_expected": len(expected),
        "total_extracted": len(extracted),
        "matched": len(matched),
        "matched_ids": sorted(matched),
        "missing": missing,
        "spurious": spurious,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pass": recall >= min_recall and precision >= min_precision,
        "per_taxonomy": per_taxonomy,
        "category_eval": category_eval,
    }


def _evaluate_categories(
    expected_risks: set[str],
    extracted_risks: set[str],
    risk_to_cat: dict[str, dict[str, set[str]]],
) -> dict[str, dict]:
    """Compute category-level precision/recall/F1 per category taxonomy."""
    if not risk_to_cat:
        return {}

    expected_cats = _derive_categories(expected_risks, risk_to_cat)
    extracted_cats = _derive_categories(extracted_risks, risk_to_cat)

    all_taxonomies = sorted(set(expected_cats.keys()) | set(extracted_cats.keys()))
    result = {}
    for tax in all_taxonomies:
        exp = expected_cats.get(tax, set())
        ext = extracted_cats.get(tax, set())
        matched = exp & ext
        missing = sorted(exp - ext)
        spurious = sorted(ext - exp)
        p, r, f = _compute_prf(len(matched), len(exp), len(ext))
        result[tax] = {
            "expected": sorted(exp),
            "extracted": sorted(ext),
            "matched": sorted(matched),
            "missing": missing,
            "spurious": spurious,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
        }
    return result
