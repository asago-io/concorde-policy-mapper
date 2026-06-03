"""Baseline eval + GEPA optimization for the grounding stage.

Usage:
    uv run python -m experiments.dspy_ground \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-a4b-it \
        --nexus-base-dir /path/to/ai-atlas-nexus \
        [--run-dir extract-runs/risk-selected_YYYYMMDD_HHMMSS] \
        [--auto medium]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import dspy

from experiments.dspy_ground.dataset import load_dataset
from experiments.dspy_ground.metric import ground_metric
from experiments.dspy_ground.module import RiskGrounder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


def _score_value(result) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if hasattr(result, "score"):
        return float(result.score)
    return float(result)


def _run_baseline(
    program: RiskGrounder,
    val: list[dspy.Example],
) -> float:
    evaluate = dspy.Evaluate(
        devset=val,
        metric=ground_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(program)
    return _score_value(result)


def _run_gepa(
    program: RiskGrounder,
    train: list[dspy.Example],
    val: list[dspy.Example],
    lm: dspy.LM,
    auto: str,
) -> tuple[RiskGrounder, float]:
    optimizer = dspy.GEPA(
        metric=ground_metric,
        auto=auto,
        reflection_lm=lm,
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=train, valset=val)

    evaluate = dspy.Evaluate(
        devset=val,
        metric=ground_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(optimized)
    return optimized, _score_value(result)


def main():
    parser = argparse.ArgumentParser(description="DSPy grounding optimization")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--nexus-base-dir", required=True)
    parser.add_argument("--run-dir", default=None,
                        help="Battery run directory for pipeline-mined negatives (optional)")
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                        help="GEPA optimization level")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline evaluation, skip optimization")
    args = parser.parse_args()

    lm = dspy.LM(
        model=f"openai/{args.model}",
        api_base=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=4096,
    )
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    logger.info("Loading dataset (GT evidence matching + hard negatives)...")
    train, val = load_dataset(
        nexus_base_dir=args.nexus_base_dir,
        run_dir=args.run_dir,
    )

    logger.info("Train: %d examples, Eval: %d examples", len(train), len(val))

    program = RiskGrounder()

    logger.info("Running baseline evaluation...")
    baseline_score = _run_baseline(program, val)
    logger.info("Baseline combined score: %.4f", baseline_score)

    if args.baseline_only:
        print(f"\nBaseline combined score: {baseline_score:.4f}")
        return

    logger.info("Running GEPA optimization (auto=%s)...", args.auto)
    optimized, optimized_score = _run_gepa(program, train, val, lm, args.auto)
    improvement = optimized_score - baseline_score

    logger.info("Optimized score: %.4f (improvement: %+.4f)", optimized_score, improvement)

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
        "model": args.model,
        "auto": args.auto,
        "baseline_score": round(baseline_score, 2),
        "optimized_score": round(optimized_score, 2),
        "improvement": round(improvement, 2),
        "train_examples": len(train),
        "eval_examples": len(val),
        "run_dir": args.run_dir,
        "optimized_instructions": optimized_instructions,
        "timestamp": timestamp,
    }

    result_path = _OUTPUT_DIR / f"run_{timestamp}.json"
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", result_path)

    program_path = _OUTPUT_DIR / f"program_{timestamp}.json"
    optimized.save(str(program_path))
    logger.info("Program saved to %s", program_path)

    print(f"\n{'='*60}")
    print(f"Baseline score: {baseline_score:.2f}%")
    print(f"Optimized score: {optimized_score:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")
    if optimized_instructions:
        print(f"\nOptimized instructions:\n{optimized_instructions}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
