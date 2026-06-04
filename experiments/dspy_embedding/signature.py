# experiments/dspy_embedding/signature.py
from __future__ import annotations

import dspy


class RetrieveRisks(dspy.Signature):
    """Given a text passage from an AI governance policy document, retrieve AI risk descriptions that are relevant to the concepts, requirements, or concerns discussed in the passage"""

    chunk_text: str = dspy.InputField(
        desc="Text passage from an AI policy document"
    )
    risk_ids: list[str] = dspy.OutputField(
        desc="Retrieved risk IDs"
    )
