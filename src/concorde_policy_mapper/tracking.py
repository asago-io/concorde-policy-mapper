from __future__ import annotations

import hashlib
import logging
import subprocess
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


def log_params(ctx: TrackingContext, params: dict[str, str]) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_params(params)
    except Exception:
        logger.warning("MLflow log_params failed", exc_info=True)


def log_metrics(ctx: TrackingContext, metrics: dict[str, float]) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception:
        logger.warning("MLflow log_metrics failed", exc_info=True)


def log_artifact(ctx: TrackingContext, path: Path) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception:
        logger.warning("MLflow log_artifact failed for %s", path, exc_info=True)


def log_child_run(
    ctx: TrackingContext,
    *,
    name: str,
    params: dict[str, str],
    metrics: dict[str, float],
    tags: dict[str, str],
    artifacts: list[Path],
) -> None:
    if not ctx.enabled:
        return
    try:
        with mlflow.start_run(run_name=name, nested=True) as child:
            ctx.child_run_ids[name] = child.info.run_id
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            for artifact_path in artifacts:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
    except Exception:
        logger.warning("MLflow child run failed for %s", name, exc_info=True)


_PROMPT_NAMES = ["judge_risk", "ground_evidence", "causal_synthesis"]


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sync_prompts(ctx: TrackingContext, templates_dir: Path) -> dict[str, int]:
    if not ctx.enabled:
        return {}

    prompts_dir = templates_dir / "prompts"
    versions: dict[str, int] = {}
    git_sha = _get_git_sha()

    for name in _PROMPT_NAMES:
        system_file = prompts_dir / f"{name}_system.j2"
        user_file = prompts_dir / f"{name}_user.j2"

        if not user_file.exists():
            continue

        system_text = system_file.read_text() if system_file.exists() else ""
        user_text = user_file.read_text()
        content_hash = hashlib.sha256((system_text + user_text).encode()).hexdigest()

        try:
            existing = mlflow.genai.load_prompt(name)
            if existing.tags.get("content_hash") == content_hash:
                versions[name] = existing.version
                continue
        except Exception:
            pass

        try:
            template = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            prompt = mlflow.genai.register_prompt(
                name=name,
                template=template,
                commit_message=f"git:{git_sha}",
                tags={"content_hash": content_hash},
            )
            versions[name] = prompt.version
            logger.info("Registered prompt %s version %d", name, prompt.version)
        except Exception:
            logger.warning("Failed to register prompt %s", name, exc_info=True)

    return versions
