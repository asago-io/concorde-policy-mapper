"""Embedding instruction optimization via GEPA.

Optimizes the query instruction prefix for instruction-aware embedding
models (e.g. Qwen3-Embedding) by maximizing retrieval recall against
ground truth.

Usage:
    uv run python -m experiments.dspy_embedding \
        --bi-encoder-model https://qwen-embedding.apps.example.com/v1/embeddings \
        --nexus-base-dir /path/to/ai-atlas-nexus \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-a4b-it \
        --auto medium
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import dspy

from concorde_policy_mapper.extract.index import RiskIndex
from experiments.dspy_embedding.dataset import (
    EVAL_POLICIES,
    TRAIN_POLICIES,
    load_dataset,
)
from experiments.dspy_embedding.metric import retrieval_recall_metric
from experiments.dspy_embedding.module import EmbeddingRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"

EXCLUDED_TAXONOMIES = {
    "nist-ai-rmf", "owasp-llm-2.0", "ailuminate-v1.0", "owasp-asi",
    "shieldgemma-taxonomy", "mit-ai-risk-repository-causal", "ibm-granite-guardian",
}


def _score_value(result) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if hasattr(result, "score"):
        return float(result.score)
    return float(result)


def _load_risks(nexus_base_dir: str) -> list:
    from ai_atlas_nexus import AIAtlasNexus

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    risks = []
    for r in nexus.get_all_risks():
        taxonomy = getattr(r, "isDefinedByTaxonomy", "") or ""
        if taxonomy in EXCLUDED_TAXONOMIES:
            continue
        risks.append(r)
    logger.info("Loaded %d risk-level risks from Nexus", len(risks))
    return risks


def _build_index(risks: list, bi_encoder_model: str, query_instruction: str) -> RiskIndex:
    logger.info("Building RiskIndex (no cross-encoder)...")
    index = RiskIndex(
        risks,
        bi_encoder_model=bi_encoder_model,
        cross_encoder_model=None,
        query_instruction=query_instruction,
    )
    logger.info("RiskIndex built: %d risks", index.risk_count)
    return index


def _run_baseline(
    program: EmbeddingRetriever,
    val: list[dspy.Example],
) -> float:
    evaluate = dspy.Evaluate(
        devset=val,
        metric=retrieval_recall_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(program)
    return _score_value(result)


def _run_gepa(
    program: EmbeddingRetriever,
    train: list[dspy.Example],
    val: list[dspy.Example],
    lm: dspy.LM,
    auto: str,
) -> tuple[EmbeddingRetriever, float]:
    optimizer = dspy.GEPA(
        metric=retrieval_recall_metric,
        auto=auto,
        reflection_lm=lm,
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=train, valset=val)

    evaluate = dspy.Evaluate(
        devset=val,
        metric=retrieval_recall_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(optimized)
    return optimized, _score_value(result)


def main():
    parser = argparse.ArgumentParser(
        description="DSPy embedding instruction optimization"
    )
    parser.add_argument("--bi-encoder-model", required=True,
                        help="Remote embedding model URL (e.g. https://qwen.../v1/embeddings)")
    parser.add_argument("--nexus-base-dir", required=True,
                        help="Path to ai-atlas-nexus repo")
    parser.add_argument("--base-url", required=True,
                        help="LLM endpoint for GEPA reflection")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it",
                        help="LLM model for GEPA reflection")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                        help="GEPA optimization intensity (default: medium)")
    parser.add_argument("--train-policies", default=None,
                        help="Comma-separated train policy subset (default: all 13)")
    parser.add_argument("--eval-policies", default=None,
                        help="Comma-separated eval policy subset")
    parser.add_argument("--chunk-max-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K candidates per chunk from hybrid search")
    parser.add_argument("--rrf-min-score", type=float, default=0.015,
                        help="RRF score floor for candidate acceptance")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline evaluation, skip optimization")
    parser.add_argument("--seed-program", default=None,
                        help="Path to a saved DSPy program to use as starting point")
    args = parser.parse_args()

    # Parse policy subsets
    train_policies = args.train_policies.split(",") if args.train_policies else None
    eval_policies = args.eval_policies.split(",") if args.eval_policies else None

    # Configure GEPA reflection LLM
    lm = dspy.LM(
        model=f"openai/{args.model}",
        api_base=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=4096,
    )
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    # Load risks and build index
    risks = _load_risks(args.nexus_base_dir)
    # Use default instruction from RetrievalConfig for initial index
    from concorde_policy_mapper.extract.models import RetrievalConfig
    default_instruction = RetrievalConfig.query_instruction
    index = _build_index(risks, args.bi_encoder_model, default_instruction)

    # Load dataset
    logger.info("Loading dataset...")
    train, val = load_dataset(
        chunk_max_tokens=args.chunk_max_tokens,
        train_policies=train_policies,
        eval_policies=eval_policies,
    )
    logger.info("Train: %d policies, Eval: %d policies", len(train), len(val))

    if not train or not val:
        logger.error("No train or eval examples — check policy files and ground truth")
        sys.exit(1)

    # Build program
    program = EmbeddingRetriever(
        index, top_k=args.top_k, rrf_min_score=args.rrf_min_score,
    )
    if args.seed_program:
        logger.info("Loading seed program from %s", args.seed_program)
        program.load(args.seed_program)

    # Baseline
    logger.info("Running baseline evaluation...")
    baseline_recall = _run_baseline(program, val)
    logger.info("Baseline recall: %.4f", baseline_recall)

    if args.baseline_only:
        print(f"\nBaseline recall: {baseline_recall:.4f}")
        return

    # Optimize
    logger.info("Running GEPA optimization (auto=%s)...", args.auto)
    optimized, optimized_recall = _run_gepa(program, train, val, lm, args.auto)

    improvement = optimized_recall - baseline_recall
    logger.info("Optimized recall: %.4f (improvement: %+.4f)", optimized_recall, improvement)

    # Save results
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    optimized_instructions = ""
    try:
        for name, predictor in optimized.named_predictors():
            if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                optimized_instructions = predictor.signature.instructions
            break
    except Exception:
        pass

    result = {
        "bi_encoder_model": args.bi_encoder_model,
        "reflection_model": args.model,
        "auto": args.auto,
        "baseline_recall": round(baseline_recall * 100, 2),
        "optimized_recall": round(optimized_recall * 100, 2),
        "improvement": round(improvement * 100, 2),
        "train_policies": train_policies or TRAIN_POLICIES,
        "eval_policies": eval_policies or EVAL_POLICIES,
        "n_train": len(train),
        "n_eval": len(val),
        "top_k": args.top_k,
        "rrf_min_score": args.rrf_min_score,
        "chunk_max_tokens": args.chunk_max_tokens,
        "default_instruction": default_instruction,
        "optimized_instruction": optimized_instructions,
        "timestamp": timestamp,
    }

    result_path = _OUTPUT_DIR / f"run_{timestamp}.json"
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", result_path)

    program_path = _OUTPUT_DIR / f"program_{timestamp}.json"
    optimized.save(str(program_path))
    logger.info("Program saved to %s", program_path)

    print(f"\n{'=' * 60}")
    print(f"Baseline recall: {baseline_recall * 100:.2f}%")
    print(f"Optimized recall: {optimized_recall * 100:.2f}%")
    print(f"Improvement: {improvement * 100:+.2f}%")
    if optimized_instructions:
        print(f"\nOptimized instruction:\n{optimized_instructions}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
