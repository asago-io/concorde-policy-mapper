from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel


class GroundVerdict(BaseModel):
    risk_id: str
    grounded: bool
    confidence: Literal["high", "medium", "low"]
    quotes: list[str]


class GroundRiskEvidence(dspy.Signature):
    """Given a text chunk from a policy document and a list of candidate AI
    risks, determine which risks are actually discussed, addressed, or implied
    by the text.

    For each grounded risk, extract 1-3 direct quotes from the chunk as
    evidence. Quotes must be exact substrings of the provided text.

    Focus on semantic relevance, not exact keyword matches. A passage about
    "unlawful monitoring of individuals" is evidence for a "mass surveillance"
    risk. Only mark a risk as grounded if the text genuinely discusses that
    risk concept — not just because the text is about AI in general.

    Respond with a verdict for EACH candidate risk."""

    chunk_text: str = dspy.InputField(
        desc="Text passage from an AI policy document"
    )
    candidate_risks: str = dspy.InputField(
        desc="Candidate AI risks to evaluate, each with ID, name, and description"
    )
    verdicts: list[GroundVerdict] = dspy.OutputField(
        desc="Grounding verdict with evidence quotes for each candidate risk"
    )
