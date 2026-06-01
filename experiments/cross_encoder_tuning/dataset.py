"""Build cross-encoder training/eval dataset from enriched ground truth.

For each policy document:
  1. Parse + chunk using the same logic as the extraction pipeline
  2. Match GT evidence spans to chunks (fuzzy substring)
  3. Generate labeled pairs:
     - Positive: (risk_description, chunk_text) where evidence matches
     - Hard negative: spurious/grounding-filtered risks from extraction results
     - Easy negative: random non-GT risks

Output: JSONL files for train and eval splits.

Usage:
    uv run python -m experiments.cross_encoder_tuning.dataset \
        --run-dir risk-landscaper/extract-runs/risk-selected_20260528_211858 \
        --nexus-base-dir /path/to/ai-atlas-nexus \
        --output-dir experiments/cross_encoder_tuning/datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING)

_ROOT = Path(__file__).resolve().parent.parent.parent
GT_DIR = _ROOT / "evals" / "ground_truth"
POLICY_DIR = _ROOT / "policy_examples"

TRAIN_POLICIES = [
    "sap", "cisco-supplier",
    "firstsource", "guy-nhs", "rdash-nhs", "dhs-gov",
    "eu-com", "ovic", "camden-borough-work", "llvm",
    "amadeus", "fs-isac", "gray",
]
EVAL_POLICIES = [
    "ars", "leicestershire_police",
    "lse-legreg", "aus-gov", "lenovo",
    "prosus", "new-york-state", "lse-marking", "ebay", "vps",
    "npcc", "penn", "st-johns", "icrc",
]

HARD_NEG_PER_POSITIVE = 3
EASY_NEG_PER_POSITIVE = 2


def _find_policy_file(policy: str) -> Path | None:
    for ext in [".md", ".pdf", ".docx", ".html", ".txt"]:
        p = POLICY_DIR / f"{policy}{ext}"
        if p.exists():
            return p
    return None


def _load_enriched_gt(gt_path: Path) -> list[dict]:
    data = yaml.safe_load(gt_path.read_text())
    if "risks" in data:
        return data["risks"]
    return []


def _load_extraction_result(run_dir: Path, policy: str) -> dict | None:
    path = run_dir / policy / "risk-extraction.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


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


def _clean_description(name: str, description: str) -> str:
    """Strip tautological prefixes from risk descriptions.

    ai-risk-taxonomy descriptions are all "{Name} is defined as {content}".
    Strip the prefix so the cross-encoder sees the substantive part.
    """
    if not description or not name:
        return description or ""
    prefix = f"{name} is defined as "
    if description.lower().startswith(prefix.lower()):
        rest = description[len(prefix):]
        rest = rest.lstrip("if ").lstrip("whether ").lstrip("the ")
        return rest
    return description


def _format_risk_string(name: str, description: str) -> str | None:
    if not name or not description:
        return None
    cleaned = _clean_description(name, description)
    return f"{name}: {cleaned}"


def process_policy(
    policy: str,
    run_dir: Path,
    risk_lookup: dict[str, dict],
    all_risk_ids: list[str],
    chunk_max_tokens: int = 512,
) -> list[dict]:
    gt_path = GT_DIR / f"{policy}.yaml"
    if not gt_path.exists():
        return []

    gt_risks = _load_enriched_gt(gt_path)
    if not gt_risks:
        return []

    gt_ids = {r["id"] for r in gt_risks}
    gt_evidence = {}
    for r in gt_risks:
        gt_evidence[r["id"]] = r.get("evidence", [])

    policy_file = _find_policy_file(policy)
    if not policy_file:
        print(f"  {policy}: no policy file found", file=sys.stderr)
        return []

    from concorde_policy_mapper.extract.parse import parse_document, chunk_documents
    try:
        parsed = parse_document(policy_file)
    except Exception as e:
        print(f"  {policy}: parse error: {e}", file=sys.stderr)
        return []

    chunks = chunk_documents([parsed], max_tokens=chunk_max_tokens)
    if not chunks:
        return []

    ext_data = _load_extraction_result(run_dir, policy)
    spurious_ids = set()
    filtered_ids = set()
    if ext_data:
        extracted_ids = {r["risk_id"] for r in ext_data.get("risks", [])}
        spurious_ids = extracted_ids - gt_ids
        filtered_ids = {
            fc["risk_id"]
            for fc in ext_data.get("grounding_filtered_candidates", [])
        }

    hard_neg_pool = list((spurious_ids | filtered_ids) - gt_ids)
    easy_neg_pool = [rid for rid in all_risk_ids if rid not in gt_ids and rid not in hard_neg_pool]

    pairs = []
    pos_count = 0

    for chunk in chunks:
        chunk_text = chunk.text

        chunk_positives = []
        for rid in gt_ids:
            evidence_list = gt_evidence.get(rid, [])
            if not evidence_list:
                continue
            matched = any(
                _evidence_in_chunk(ev.get("text", ""), chunk_text)
                for ev in evidence_list
            )
            if matched:
                info = risk_lookup.get(rid, {})
                risk_str = _format_risk_string(info.get("name", ""), info.get("description", ""))
                if not risk_str:
                    continue
                chunk_positives.append(rid)
                pairs.append({
                    "sentence1": risk_str,
                    "sentence2": chunk_text,
                    "label": 1,
                    "metadata": {
                        "policy": policy,
                        "risk_id": rid,
                        "chunk_index": chunk.index,
                        "section": chunk.section,
                        "pair_type": "positive",
                    },
                })
                pos_count += 1

        n_pos = len(chunk_positives)
        if n_pos == 0:
            continue

        n_hard = min(n_pos * HARD_NEG_PER_POSITIVE, len(hard_neg_pool))
        sampled_hard = random.sample(hard_neg_pool, n_hard) if n_hard > 0 else []
        for rid in sampled_hard:
            info = risk_lookup.get(rid, {})
            risk_str = _format_risk_string(info.get("name", ""), info.get("description", ""))
            if not risk_str:
                continue
            pairs.append({
                "sentence1": risk_str,
                "sentence2": chunk_text,
                "label": 0,
                "metadata": {
                    "policy": policy,
                    "risk_id": rid,
                    "chunk_index": chunk.index,
                    "section": chunk.section,
                    "pair_type": "hard_negative",
                },
            })

        n_easy = min(n_pos * EASY_NEG_PER_POSITIVE, len(easy_neg_pool))
        sampled_easy = random.sample(easy_neg_pool, n_easy) if n_easy > 0 else []
        for rid in sampled_easy:
            info = risk_lookup.get(rid, {})
            risk_str = _format_risk_string(info.get("name", ""), info.get("description", ""))
            if not risk_str:
                continue
            pairs.append({
                "sentence1": risk_str,
                "sentence2": chunk_text,
                "label": 0,
                "metadata": {
                    "policy": policy,
                    "risk_id": rid,
                    "chunk_index": chunk.index,
                    "section": chunk.section,
                    "pair_type": "easy_negative",
                },
            })

    print(f"  {policy:25s}: {len(chunks)} chunks, {pos_count} positives, {len(pairs)} total pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build cross-encoder dataset")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--nexus-base-dir", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("risk-landscaper/datasets/cross-encoder"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ai_atlas_nexus import AIAtlasNexus
    nexus = AIAtlasNexus(base_dir=args.nexus_base_dir)
    risk_lookup = {}
    all_risk_ids = []
    for r in nexus.get_all_risks():
        risk_lookup[r.id] = {
            "name": r.name or "",
            "description": r.description or "",
        }
        all_risk_ids.append(r.id)

    for policy_dir in args.run_dir.iterdir():
        rpath = policy_dir / "risk-extraction.json"
        if rpath.exists():
            data = json.loads(rpath.read_text())
            for r in data.get("risks", []):
                if r["risk_id"] not in risk_lookup:
                    risk_lookup[r["risk_id"]] = {
                        "name": r.get("risk_name", ""),
                        "description": r.get("risk_description", ""),
                    }

    print(f"Risk descriptions loaded: {len(risk_lookup)}")
    print(f"Train policies: {len(TRAIN_POLICIES)}")
    print(f"Eval policies: {len(EVAL_POLICIES)}")

    print("\n--- Train ---")
    train_pairs = []
    for policy in TRAIN_POLICIES:
        pairs = process_policy(policy, args.run_dir, risk_lookup, all_risk_ids)
        train_pairs.extend(pairs)

    print("\n--- Eval ---")
    eval_pairs = []
    for policy in EVAL_POLICIES:
        pairs = process_policy(policy, args.run_dir, risk_lookup, all_risk_ids)
        eval_pairs.extend(pairs)

    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    for path, pairs in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

    def _stats(pairs):
        pos = sum(1 for p in pairs if p["label"] == 1)
        neg = sum(1 for p in pairs if p["label"] == 0)
        hard = sum(1 for p in pairs if p["metadata"]["pair_type"] == "hard_negative")
        easy = sum(1 for p in pairs if p["metadata"]["pair_type"] == "easy_negative")
        policies = len({p["metadata"]["policy"] for p in pairs})
        return pos, neg, hard, easy, policies

    print(f"\n{'':25s} {'Total':>6s} {'Pos':>6s} {'Neg':>6s} {'Hard-':>6s} {'Easy-':>6s} {'Policies':>8s}")
    for name, pairs, path in [("Train", train_pairs, train_path), ("Eval", eval_pairs, eval_path)]:
        pos, neg, hard, easy, policies = _stats(pairs)
        print(f"{name:25s} {len(pairs):6d} {pos:6d} {neg:6d} {hard:6d} {easy:6d} {policies:8d}")

    print(f"\nOutput: {train_path} ({len(train_pairs)} pairs)")
    print(f"Output: {eval_path} ({len(eval_pairs)} pairs)")

    meta = {
        "train_policies": TRAIN_POLICIES,
        "eval_policies": EVAL_POLICIES,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "seed": args.seed,
        "hard_neg_ratio": HARD_NEG_PER_POSITIVE,
        "easy_neg_ratio": EASY_NEG_PER_POSITIVE,
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
