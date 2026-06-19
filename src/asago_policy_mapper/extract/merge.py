from __future__ import annotations

from collections import defaultdict

from asago_policy_mapper.extract.models import EvidenceSpan, RiskMatch


def merge_matches(
    matches: list[RiskMatch],
    max_evidence: int = 3,
) -> list[RiskMatch]:
    if not matches:
        return []

    groups: dict[str, list[RiskMatch]] = defaultdict(list)
    for m in matches:
        groups[m.risk_id].append(m)

    merged = []
    for risk_id, group in groups.items():
        best = max(group, key=lambda m: m.confidence)

        all_spans: list[EvidenceSpan] = []
        seen_texts: set[str] = set()
        for m in sorted(group, key=lambda m: m.confidence, reverse=True):
            for span in m.evidence:
                if span.text not in seen_texts:
                    seen_texts.add(span.text)
                    all_spans.append(span)

        all_spans.sort(key=lambda s: s.cross_encoder_score, reverse=True)
        capped = all_spans[:max_evidence]

        merged.append(
            best.model_copy(
                update={
                    "confidence": max(m.confidence for m in group),
                    "evidence": capped,
                }
            )
        )

    merged.sort(key=lambda m: m.confidence, reverse=True)
    return merged
