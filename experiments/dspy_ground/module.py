from __future__ import annotations

import dspy

from experiments.dspy_ground.signature import GroundRiskEvidence


class RiskGrounder(dspy.Module):
    def __init__(self):
        self.ground = dspy.ChainOfThought(GroundRiskEvidence)

    def forward(self, chunk_text: str, candidate_risks: str):
        return self.ground(
            chunk_text=chunk_text, candidate_risks=candidate_risks
        )
