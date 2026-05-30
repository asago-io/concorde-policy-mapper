from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.genai
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


@dataclass
class TrackingContext:
    enabled: bool = False
    run_id: str | None = None
    child_run_ids: dict[str, str] = field(default_factory=dict)


def is_tracking_enabled(ctx: TrackingContext) -> bool:
    return ctx.enabled


def init_tracking(
    *,
    enabled: bool = True,
    experiment_name: str = "risk-extraction",
    run_name: str | None = None,
) -> TrackingContext:
    if not enabled or not _MLFLOW_AVAILABLE:
        if enabled and not _MLFLOW_AVAILABLE:
            logger.warning("mlflow not installed — tracking disabled")
        return TrackingContext(enabled=False)

    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        logger.info("MLflow tracking started: experiment=%s run=%s", experiment_name, run.info.run_id)
        return TrackingContext(enabled=True, run_id=run.info.run_id)
    except Exception:
        logger.warning("MLflow tracking failed to initialize — continuing without tracking", exc_info=True)
        return TrackingContext(enabled=False)


def end_tracking(ctx: TrackingContext) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.end_run()
    except Exception:
        logger.warning("MLflow end_run failed", exc_info=True)
