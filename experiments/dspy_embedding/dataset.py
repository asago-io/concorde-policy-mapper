"""Build DSPy examples for embedding instruction optimization.

Each example represents one policy document with its chunk texts and
ground truth risk IDs. The module evaluates retrieval recall across
all chunks for each policy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dspy
import yaml

from asago_policy_mapper.extract.parse import chunk_documents, parse_document
from asago_policy_mapper.extract.retrieve import build_padded_text

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


def _find_policy_file(policy: str) -> Path | None:
    for ext in [".md", ".pdf", ".docx", ".html", ".txt"]:
        p = _POLICY_DIR / f"{policy}{ext}"
        if p.exists():
            return p
    return None


def _build_example(
    policy: str,
    chunk_max_tokens: int = 512,
    context_sentences: int = 2,
) -> dspy.Example | None:
    """Build a single DSPy example for one policy document."""
    gt_path = _GT_DIR / f"{policy}.yaml"
    if not gt_path.exists():
        logger.warning("No ground truth for %s", policy)
        return None

    policy_file = _find_policy_file(policy)
    if not policy_file:
        logger.warning("No policy file for %s", policy)
        return None

    gt_data = yaml.safe_load(gt_path.read_text())
    gt_risks = gt_data.get("risks", [])
    if not gt_risks:
        logger.warning("Empty ground truth for %s", policy)
        return None

    risk_ids = [
        r["id"] for r in gt_risks
        if not any(r["id"].startswith(p) for p in _EXCLUDED_TAXONOMIES)
    ]
    if not risk_ids:
        return None

    try:
        parsed = parse_document(policy_file)
    except Exception as e:
        logger.warning("Parse error for %s: %s", policy, e)
        return None

    chunks = chunk_documents([parsed], max_tokens=chunk_max_tokens)
    if not chunks:
        return None

    chunk_texts = [
        build_padded_text(chunks, i, context_sentences=context_sentences)
        for i in range(len(chunks))
    ]

    logger.info("  %s: %d chunks, %d GT risks", policy, len(chunk_texts), len(risk_ids))

    return dspy.Example(
        chunk_texts=chunk_texts,
        risk_ids=risk_ids,
        policy_name=policy,
    ).with_inputs("chunk_texts")


def load_dataset(
    chunk_max_tokens: int = 512,
    context_sentences: int = 2,
    train_policies: list[str] | None = None,
    eval_policies: list[str] | None = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load DSPy dataset for embedding instruction optimization.

    Args:
        chunk_max_tokens: Max tokens per chunk.
        context_sentences: Sentences of context from adjacent chunks.
        train_policies: Override train policy list (for prototyping subsets).
        eval_policies: Override eval policy list.

    Returns:
        (train_examples, eval_examples) — one Example per policy.
    """
    train_list = train_policies or TRAIN_POLICIES
    eval_list = eval_policies or EVAL_POLICIES

    train, val = [], []
    for split_name, policies, dest in [
        ("train", train_list, train),
        ("eval", eval_list, val),
    ]:
        for policy in policies:
            example = _build_example(
                policy,
                chunk_max_tokens=chunk_max_tokens,
                context_sentences=context_sentences,
            )
            if example:
                dest.append(example)
        logger.info("%s: %d policies loaded", split_name, len(dest))

    return train, val
