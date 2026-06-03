"""Baseline eval + prompt optimization for the risk judge.

Supports GEPA (instruction-only) and MIPROv2 (instructions + few-shot demos).

Usage:
    # GEPA (default)
    uv run python -m experiments.dspy_judge \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-a4b-it \
        --run-dir extract-runs/risk-selected_20260602_164503 \
        --nexus-base-dir /path/to/ai-atlas-nexus \
        [--auto medium]

    # MIPROv2 with few-shot
    uv run python -m experiments.dspy_judge \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-a4b-it \
        --run-dir extract-runs/risk-selected_20260602_164503 \
        --nexus-base-dir /path/to/ai-atlas-nexus \
        --optimizer mipro --mipro-demos 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import dspy

from experiments.dspy_judge.dataset import load_dataset
from experiments.dspy_judge.metric import judge_metric
from experiments.dspy_judge.module import RiskJudge

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
    program: RiskJudge,
    val: list[dspy.Example],
) -> float:
    evaluate = dspy.Evaluate(
        devset=val,
        metric=judge_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(program)
    return _score_value(result)


def _run_gepa(
    program: RiskJudge,
    train: list[dspy.Example],
    val: list[dspy.Example],
    lm: dspy.LM,
    auto: str,
) -> tuple[RiskJudge, float]:
    optimizer = dspy.GEPA(
        metric=judge_metric,
        auto=auto,
        reflection_lm=lm,
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=train, valset=val)

    evaluate = dspy.Evaluate(
        devset=val,
        metric=judge_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(optimized)
    return optimized, _score_value(result)


def _run_mipro(
    program: RiskJudge,
    train: list[dspy.Example],
    val: list[dspy.Example],
    lm: dspy.LM,
    auto: str,
    max_demos: int = 3,
) -> tuple[RiskJudge, float]:
    optimizer = dspy.MIPROv2(
        metric=judge_metric,
        auto=auto,
        prompt_model=lm,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
    )
    optimized = optimizer.compile(program, trainset=train, valset=val)

    evaluate = dspy.Evaluate(
        devset=val,
        metric=judge_metric,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    result = evaluate(optimized)
    return optimized, _score_value(result)


def main():
    parser = argparse.ArgumentParser(description="DSPy judge optimization")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--nexus-base-dir", required=True)
    parser.add_argument("--run-dir", default=None,
                        help="Battery run dir with --no-grounding (pipeline-mined candidates)")
    parser.add_argument("--optimizer", default="gepa", choices=["gepa", "mipro"],
                        help="Optimizer to use (default: gepa)")
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                        help="Optimization level")
    parser.add_argument("--mipro-demos", type=int, default=3,
                        help="Max few-shot demos for MIPROv2 (default: 3)")
    parser.add_argument("--seed-program", default=None,
                        help="Path to a saved DSPy program to use as starting point (e.g. GEPA output)")
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

    if args.run_dir:
        logger.info("Loading dataset from pipeline run: %s", args.run_dir)
    else:
        logger.info("Loading dataset (with cross-encoder hard negative mining)...")
    train, val = load_dataset(
        nexus_base_dir=args.nexus_base_dir,
        run_dir=args.run_dir,
    )

    logger.info("Train: %d examples, Eval: %d examples", len(train), len(val))

    program = RiskJudge()
    if args.seed_program:
        logger.info("Loading seed program from %s", args.seed_program)
        program.load(args.seed_program)

    logger.info("Running baseline evaluation...")
    baseline_f1 = _run_baseline(program, val)
    logger.info("Baseline F1: %.4f", baseline_f1)

    if args.baseline_only:
        print(f"\nBaseline F1: {baseline_f1:.4f}")
        return

    if args.optimizer == "mipro":
        logger.info("Running MIPROv2 optimization (auto=%s, demos=%d)...", args.auto, args.mipro_demos)
        optimized, optimized_f1 = _run_mipro(program, train, val, lm, args.auto, args.mipro_demos)
    else:
        logger.info("Running GEPA optimization (auto=%s)...", args.auto)
        optimized, optimized_f1 = _run_gepa(program, train, val, lm, args.auto)

    improvement = optimized_f1 - baseline_f1

    logger.info("Optimized F1: %.4f (improvement: %+.4f)", optimized_f1, improvement)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    optimized_instructions = ""
    optimized_demos = []
    try:
        for name, predictor in optimized.named_predictors():
            if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                optimized_instructions = predictor.signature.instructions
            if hasattr(predictor, "demos") and predictor.demos:
                optimized_demos = [
                    {k: str(v)[:200] for k, v in d.items()} if isinstance(d, dict)
                    else str(d)[:200]
                    for d in predictor.demos
                ]
            break
    except Exception:
        pass

    result = {
        "model": args.model,
        "optimizer": args.optimizer,
        "auto": args.auto,
        "baseline_f1": round(baseline_f1 * 100, 2),
        "optimized_f1": round(optimized_f1 * 100, 2),
        "improvement": round(improvement * 100, 2),
        "train_examples": len(train),
        "eval_examples": len(val),
        "optimized_instructions": optimized_instructions,
        "n_demos": len(optimized_demos),
        "timestamp": timestamp,
    }

    result_path = _OUTPUT_DIR / f"run_{args.optimizer}_{timestamp}.json"
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", result_path)

    program_path = _OUTPUT_DIR / f"program_{args.optimizer}_{timestamp}.json"
    optimized.save(str(program_path))
    logger.info("Program saved to %s", program_path)

    print(f"\n{'='*60}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Baseline F1: {baseline_f1*100:.2f}%")
    print(f"Optimized F1: {optimized_f1*100:.2f}%")
    print(f"Improvement: {improvement*100:+.2f}%")
    if optimized_demos:
        print(f"Few-shot demos: {len(optimized_demos)}")
    if optimized_instructions:
        print(f"\nOptimized instructions:\n{optimized_instructions}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
