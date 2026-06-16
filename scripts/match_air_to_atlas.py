#!/usr/bin/env python3
"""Match AIR 2024 risks to Atlas risks using embeddings + manual overrides.
Generates a cross-mapping YAML file for the mitigation bridge."""
from __future__ import annotations

import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

KG = Path(".venv/lib/python3.12/site-packages/ai_atlas_nexus/data/knowledge_graph")
DATA_DIR = Path("data")

# Group mapping: AIR group ID → list of Atlas group IDs
GROUP_MAP: dict[str, list[str]] = {
    "ai-risk-taxonomy-ai-system-safety-failures": ["ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-ai-system-security-vulnerabilities": ["ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-availability": ["ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-information-security-risks": ["ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-integrity": ["ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-autonomous-unsafe-operation-of-systems": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-criminal-activities": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-supporting-malicious-organized-groups": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-malicious-use": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-weapon-usage-&-development": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-military-and-warfare": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-compromising-societal-trust-in-ai": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-concentration-of-power": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-economic-and-cultural-harms": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-economic-harm": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-environmental-harms": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-erosion-of-social-cohesion": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-freedom-and-autonomy": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-human-autonomy-and-integrity": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-human-replacement-and-displacement": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-information-ecosystem-risks": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-macroeconomic-and-societal-risks": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-political-and-geopolitical-risks": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-psychological-and-societal-harms": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-services/exploitation": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-socioeconomic-and-environmental-harms": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-disempowering-workers": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-content-safety-risks": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-ethical-concerns": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-hate-speech-(inciting/promoting/expressing-hatred)": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-misinformation-and-disinformation": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-over-reliance-on-ai": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-unethical-behavior": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-perpetuating-harmful-beliefs": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-offensive-language": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-celebrating-suffering": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-violent-acts": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-depicting-violence": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-harassment": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-adult-content": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-erotic": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-non-consensual-nudity": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-monetized": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-endangerment,-harm,-or-abuse-of-children": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-child-sexual-abuse": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-suicidal-and-non-suicidal-self-injury": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-discrimination-and-bias": ["ibm-risk-atlas-fairness"],
    "ai-risk-taxonomy-inequality-and-social-justice": ["ibm-risk-atlas-fairness"],
    "ai-risk-taxonomy-discrimination/protected-characteristics-combinations": ["ibm-risk-atlas-fairness"],
    "ai-risk-taxonomy-confidentiality": ["ibm-risk-atlas-privacy", "ibm-risk-atlas-robustness"],
    "ai-risk-taxonomy-data-privacy-risks": ["ibm-risk-atlas-privacy"],
    "ai-risk-taxonomy-privacy-violations/sensitive-data-combinations": ["ibm-risk-atlas-privacy"],
    "ai-risk-taxonomy-intellectual-property": ["ibm-risk-atlas-intellectual-property"],
    "ai-risk-taxonomy-specific-types-of-rights": ["ibm-risk-atlas-intellectual-property", "ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-lack-of-transparency-and-explainability": ["ibm-risk-atlas-explainability", "ibm-risk-atlas-transparency", "ibm-risk-atlas-governance"],
    "ai-risk-taxonomy-legal-and-rights-violations": ["ibm-risk-atlas-legal-compliance"],
    "ai-risk-taxonomy-violation-of-fundamental-rights": ["ibm-risk-atlas-legal-compliance"],
    "ai-risk-taxonomy-other-illegal/unlawful/criminal-activities": ["ibm-risk-atlas-legal-compliance"],
    "ai-risk-taxonomy-governance-and-accountability-failure": ["ibm-risk-atlas-governance"],
    "ai-risk-taxonomy-deception-and-manipulation": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-political-persuasion": ["ibm-risk-atlas-misuse", "ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-influencing-politics": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-deterring-democratic-participation": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-high-risk-financial-activities": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-unfair-market-practices": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-schemes": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-fraud": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-academic-dishonesty": ["ibm-risk-atlas-societal-impact"],
    "ai-risk-taxonomy-mis/disinformation": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-sowing-division": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-misrepresentation": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-types-of-defamation": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-illegal/regulated-substances/goods": ["ibm-risk-atlas-misuse"],
    "ai-risk-taxonomy-advice-in-heavily-regulated-industries": ["ibm-risk-atlas-value-alignment"],
    "ai-risk-taxonomy-automated-decision-making": ["ibm-risk-atlas-fairness", "ibm-risk-atlas-legal-compliance"],
    "ai-risk-taxonomy-disrupting-social-order-(china-unique)": [],
}

# Manual overrides: AIR risk ID → atlas target ID
# Applied when embedding match is wrong or LOW score
MANUAL_OVERRIDES: dict[str, str] = {}

# Group-level overrides: all children in these AIR groups get a fixed Atlas target
GROUP_OVERRIDES: dict[str, str] = {
    "ai-risk-taxonomy-autonomous-unsafe-operation-of-systems": "atlas-dangerous-use",
    "ai-risk-taxonomy-weapon-usage-&-development": "atlas-dangerous-use",
    "ai-risk-taxonomy-military-and-warfare": "atlas-dangerous-use",
    "ai-risk-taxonomy-child-sexual-abuse": "atlas-harmful-output",
    "ai-risk-taxonomy-non-consensual-nudity": "atlas-harmful-output",
    "ai-risk-taxonomy-erotic": "atlas-harmful-output",
    "ai-risk-taxonomy-suicidal-and-non-suicidal-self-injury": "atlas-harmful-output",
    "ai-risk-taxonomy-schemes": "atlas-improper-usage",
    "ai-risk-taxonomy-high-risk-financial-activities": "atlas-improper-usage",
    "ai-risk-taxonomy-unfair-market-practices": "atlas-improper-usage",
}

# Individual risk overrides (for specific wrong matches within otherwise OK groups)
RISK_OVERRIDES: dict[str, str] = {
    # Availability: system impairment is not hallucination
    "ai-risk-taxonomy-system/website-impairment": "atlas-external-resources-attack-agentic",
    "ai-risk-taxonomy-network-disruption": "atlas-external-resources-attack-agentic",
    # Hate speech: wrong embedding matches for some protected characteristics
    "ai-risk-taxonomy-beliefs": "atlas-toxic-output",
    "ai-risk-taxonomy-genetic-information": "atlas-toxic-output",
    # Content safety: embedding picked wrong value-alignment targets
    "ai-risk-taxonomy-persons-(including-murder)": "atlas-harmful-output",
    "ai-risk-taxonomy-suicide": "atlas-harmful-output",
    "ai-risk-taxonomy-building-services-targeting-minors/failure-to-employ-age-gating": "atlas-harmful-output",
    "ai-risk-taxonomy-building-services-to-present-a-persona-of-minor": "atlas-harmful-output",
    "ai-risk-taxonomy-promotion-of-sexual-services": "atlas-harmful-output",
    # Automated decision-making: all should be decision bias
    "ai-risk-taxonomy-financing-eligibility/creditworthiness": "atlas-decision-bias",
    "ai-risk-taxonomy-criminal-justice/predictive-policing": "atlas-decision-bias",
    "ai-risk-taxonomy-adversely-affects-legal-rights": "atlas-decision-bias",
    "ai-risk-taxonomy-employment": "atlas-decision-bias",
    "ai-risk-taxonomy-social-scoring": "atlas-decision-bias",
    "ai-risk-taxonomy-housing-eligibility": "atlas-decision-bias",
    "ai-risk-taxonomy-education-eligibility": "atlas-decision-bias",
    "ai-risk-taxonomy-migration-eligibility": "atlas-decision-bias",
    "ai-risk-taxonomy-insurance-eligibility": "atlas-decision-bias",
    "ai-risk-taxonomy-profiling": "atlas-decision-bias",
    # Criminal org: dangerous use not hallucination
    "ai-risk-taxonomy-criminal-organization": "atlas-dangerous-use",
    # Terrorism/extremism → dangerous use
    "ai-risk-taxonomy-terrorism": "atlas-dangerous-use",
    "ai-risk-taxonomy-extremism": "atlas-dangerous-use",
    # Fraud items → spreading-disinformation (deceptive content)
    "ai-risk-taxonomy-scams": "atlas-spreading-disinformation",
    # Impersonating others → nonconsensual-use
    "ai-risk-taxonomy-impersonating-others": "atlas-nonconsensual-use",
    # Automated social media → non-disclosure
    "ai-risk-taxonomy-automated-social-media-posts": "atlas-non-disclosure",
    # Disempowering workers overrides
    "ai-risk-taxonomy-undermine-workers'-rights": "atlas-human-exploitation",
    "ai-risk-taxonomy-worsen-job-quality": "atlas-human-exploitation",
    "ai-risk-taxonomy-cause-harmful-labor-force-disruptions": "atlas-impact-on-jobs",
}


def load_risks(path: Path) -> list[dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [e for e in raw.get("entries", []) if e.get("type") == "Risk"]


def build_text(entry: dict) -> str:
    name = entry.get("name", entry["id"])
    desc = entry.get("description", "")
    concern = entry.get("concern", "")
    parts = [name]
    if desc:
        parts.append(desc[:300])
    if concern:
        parts.append(concern[:200])
    return ". ".join(parts)


def main():
    air_risks = load_risks(KG / "air_2024_data.yaml")
    atlas_risks = load_risks(KG / "risk_atlas_data.yaml")

    air_by_group: dict[str, list[dict]] = {}
    for r in air_risks:
        gid = r.get("isPartOf", "unknown")
        air_by_group.setdefault(gid, []).append(r)

    atlas_by_group: dict[str, list[dict]] = {}
    for r in atlas_risks:
        gid = r.get("isPartOf", "unknown")
        atlas_by_group.setdefault(gid, []).append(r)

    atlas_by_id = {r["id"]: r for r in atlas_risks}

    print(f"AIR: {len(air_risks)} risks in {len(air_by_group)} groups")
    print(f"Atlas: {len(atlas_risks)} risks in {len(atlas_by_group)} groups")

    # Build embeddings
    all_air_texts = {r["id"]: build_text(r) for r in air_risks}
    all_atlas_texts = {r["id"]: build_text(r) for r in atlas_risks}

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    air_ids = list(all_air_texts.keys())
    atlas_ids = list(all_atlas_texts.keys())

    air_embeddings = model.encode([all_air_texts[i] for i in air_ids], show_progress_bar=True)
    atlas_embeddings = model.encode([all_atlas_texts[i] for i in atlas_ids], show_progress_bar=True)

    sims = cosine_similarity(air_embeddings, atlas_embeddings)
    sim_matrix = {
        air_ids[i]: {atlas_ids[j]: float(sims[i][j]) for j in range(len(atlas_ids))}
        for i in range(len(air_ids))
    }

    # Build mappings
    entries = []
    stats = {"override_group": 0, "override_risk": 0, "embedding": 0, "skipped": 0}
    score_buckets = {"close": 0, "related": 0, "broad": 0}

    for air_group_id, atlas_group_ids in sorted(GROUP_MAP.items()):
        air_children = air_by_group.get(air_group_id, [])
        if not air_children or not atlas_group_ids:
            stats["skipped"] += len(air_children)
            continue

        atlas_candidates = []
        for ag in atlas_group_ids:
            atlas_candidates.extend(atlas_by_group.get(ag, []))
        atlas_cand_ids = [c["id"] for c in atlas_candidates]

        for air_child in air_children:
            air_id = air_child["id"]

            # Check overrides first
            if air_id in RISK_OVERRIDES:
                target_id = RISK_OVERRIDES[air_id]
                method = "manual_risk"
                score = None
                stats["override_risk"] += 1
            elif air_group_id in GROUP_OVERRIDES:
                target_id = GROUP_OVERRIDES[air_group_id]
                method = "manual_group"
                score = None
                stats["override_group"] += 1
            else:
                # Use embedding match within constrained group
                scores = [(aid, sim_matrix[air_id][aid]) for aid in atlas_cand_ids]
                scores.sort(key=lambda x: x[1], reverse=True)
                target_id = scores[0][0]
                score = scores[0][1]
                method = "embedding"
                stats["embedding"] += 1

            # Determine predicate based on score or method
            if method.startswith("manual"):
                predicate = "related_mappings"
                score_buckets["related"] += 1
            elif score >= 0.5:
                predicate = "close_mappings"
                score_buckets["close"] += 1
            elif score >= 0.3:
                predicate = "related_mappings"
                score_buckets["related"] += 1
            else:
                predicate = "broad_mappings"
                score_buckets["broad"] += 1

            entries.append({
                "air_id": air_id,
                "atlas_id": target_id,
                "predicate": predicate,
                "score": score,
                "method": method,
            })

    # Build output YAML in Nexus mapping format
    # Group entries by AIR risk ID
    by_air: dict[str, dict[str, list[str]]] = {}
    for e in entries:
        if e["air_id"] not in by_air:
            by_air[e["air_id"]] = {}
        pred = e["predicate"]
        by_air[e["air_id"]].setdefault(pred, []).append(e["atlas_id"])

    output_entries = []
    for air_id in sorted(by_air.keys()):
        entry = {"id": air_id}
        for pred in ["close_mappings", "related_mappings", "broad_mappings"]:
            if pred in by_air[air_id]:
                entry[pred] = sorted(set(by_air[air_id][pred]))
        output_entries.append(entry)

    output = {"entries": output_entries}

    output_path = DATA_DIR / "air_2024_to_atlas_mappings.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# AIR 2024 → IBM AI Risk Atlas cross-mappings\n"
        "# Auto-generated by scripts/match_air_to_atlas.py\n"
        "# Method: group-constrained embedding similarity (all-MiniLM-L6-v2) + manual overrides\n"
        f"# Total: {len(output_entries)} AIR risks mapped to Atlas risks\n"
        f"# Embedding: {stats['embedding']}, Group overrides: {stats['override_group']}, "
        f"Risk overrides: {stats['override_risk']}, Skipped: {stats['skipped']}\n"
        f"# Predicates: close={score_buckets['close']}, "
        f"related={score_buckets['related']}, broad={score_buckets['broad']}\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nWritten to {output_path}")
    print(f"Stats: {stats}")
    print(f"Score buckets: {score_buckets}")

    # Print summary of what each atlas risk gets mapped from
    atlas_incoming: dict[str, int] = {}
    for e in entries:
        atlas_incoming[e["atlas_id"]] = atlas_incoming.get(e["atlas_id"], 0) + 1

    print(f"\nAtlas risks receiving AIR mappings ({len(atlas_incoming)} unique):")
    for aid, count in sorted(atlas_incoming.items(), key=lambda x: -x[1])[:20]:
        name = atlas_by_id[aid].get("name", aid) if aid in atlas_by_id else aid
        print(f"  {name:45s} ← {count} AIR risks")


if __name__ == "__main__":
    main()
