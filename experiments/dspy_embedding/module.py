# experiments/dspy_embedding/module.py
from __future__ import annotations

import logging

import dspy

from concorde_policy_mapper.extract.index import RiskIndex
from experiments.dspy_embedding.signature import RetrieveRisks

logger = logging.getLogger(__name__)


class EmbeddingRetriever(dspy.Module):
    def __init__(self, index: RiskIndex, top_k: int = 50, rrf_min_score: float = 0.015):
        super().__init__()
        self.retrieve = dspy.Predict(RetrieveRisks)
        self._index = index
        self._top_k = top_k
        self._rrf_min_score = rrf_min_score

    def forward(self, chunk_texts: list[str], risk_ids: list[str] = None):
        instruction = self.retrieve.signature.instructions
        self._index.set_query_instruction(instruction)

        retrieved_ids: set[str] = set()
        for text in chunk_texts:
            candidates = self._index.hybrid_search(
                text,
                top_k=self._top_k,
                rrf_min_score=self._rrf_min_score,
            )
            for c in candidates:
                retrieved_ids.add(c.risk_id)

        return dspy.Prediction(risk_ids=sorted(retrieved_ids))
