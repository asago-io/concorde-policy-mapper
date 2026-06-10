import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from concorde_policy_mapper.tracking import (
    TrackingContext,
    end_tracking,
    init_tracking,
    is_tracking_enabled,
    log_artifact,
    log_child_run,
    log_metrics,
    log_params,
    sync_prompts,
)


def test_tracking_disabled_by_flag():
    ctx = init_tracking(enabled=False, experiment_name="test")
    assert not is_tracking_enabled(ctx)
    end_tracking(ctx)


@patch("concorde_policy_mapper.tracking.mlflow")
def test_tracking_enabled_sets_experiment_and_starts_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    assert is_tracking_enabled(ctx)

    mock_mlflow.set_experiment.assert_called_once_with("test-exp")
    mock_mlflow.start_run.assert_called_once_with(run_name="run-1")


@patch("concorde_policy_mapper.tracking.mlflow")
def test_tracking_survives_mlflow_failure(mock_mlflow):
    mock_mlflow.set_experiment.side_effect = Exception("connection refused")

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    assert not is_tracking_enabled(ctx)


@patch("concorde_policy_mapper.tracking.mlflow")
def test_end_tracking_calls_end_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    end_tracking(ctx)
    mock_mlflow.end_run.assert_called_once()


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_params(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_params(ctx, {"model": "gemma", "threshold": "0.7"})
    mock_mlflow.log_params.assert_called_once_with({"model": "gemma", "threshold": "0.7"})


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_params_skipped_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    log_params(ctx, {"model": "gemma"})
    mock_mlflow.log_params.assert_not_called()


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_metrics(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_metrics(ctx, {"recall": 0.85, "precision": 0.9})
    mock_mlflow.log_metrics.assert_called_once_with({"recall": 0.85, "precision": 0.9})


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_artifact(mock_mlflow, tmp_path):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    artifact = tmp_path / "test.json"
    artifact.write_text('{"key": "value"}')

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_artifact(ctx, artifact)
    mock_mlflow.log_artifact.assert_called_once_with(str(artifact))


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_child_run(mock_mlflow):
    parent_run = MagicMock()
    parent_run.info.run_id = "parent123"
    mock_mlflow.start_run.return_value = parent_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")

    child_run = MagicMock()
    child_run.info.run_id = "child456"
    mock_mlflow.start_run.return_value = child_run

    log_child_run(
        ctx,
        name="dhs-gov",
        params={"policy_name": "dhs-gov"},
        metrics={"recall": 0.8, "f1": 0.75},
        tags={"eval_status": "FAIL"},
        artifacts=[],
    )

    calls = mock_mlflow.start_run.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["run_name"] == "dhs-gov"
    assert calls[1].kwargs["nested"] is True


@patch("concorde_policy_mapper.tracking.mlflow")
def test_log_child_run_skipped_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    log_child_run(ctx, name="policy", params={}, metrics={}, tags={}, artifacts=[])
    mock_mlflow.start_run.assert_not_called()


@patch("concorde_policy_mapper.tracking.mlflow")
def test_sync_prompts_registers_new_prompt(mock_mlflow, tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "judge_risk_system.j2").write_text("You are a judge.")
    (prompts_dir / "judge_risk_user.j2").write_text("Judge this: {{ text }}")

    mock_mlflow.genai.load_prompt.side_effect = Exception("not found")
    mock_prompt = MagicMock()
    mock_prompt.version = 1
    mock_mlflow.genai.register_prompt.return_value = mock_prompt

    ctx = TrackingContext(enabled=True, run_id="abc")
    versions = sync_prompts(ctx, prompts_dir.parent)

    assert "judge_risk" in versions
    assert versions["judge_risk"] == 1
    mock_mlflow.genai.register_prompt.assert_called_once()


@patch("concorde_policy_mapper.tracking.mlflow")
def test_sync_prompts_skips_unchanged(mock_mlflow, tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)
    system = "You are a judge."
    user = "Judge this: {{ text }}"
    (prompts_dir / "judge_risk_system.j2").write_text(system)
    (prompts_dir / "judge_risk_user.j2").write_text(user)

    content_hash = hashlib.sha256((system + user).encode()).hexdigest()
    mock_existing = MagicMock()
    mock_existing.version = 3
    mock_existing.tags = {"content_hash": content_hash}
    mock_mlflow.genai.load_prompt.return_value = mock_existing

    ctx = TrackingContext(enabled=True, run_id="abc")
    versions = sync_prompts(ctx, prompts_dir.parent)

    assert versions["judge_risk"] == 3
    mock_mlflow.genai.register_prompt.assert_not_called()


@patch("concorde_policy_mapper.tracking.mlflow")
def test_sync_prompts_returns_empty_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    versions = sync_prompts(ctx, Path("/nonexistent"))
    assert versions == {}
    mock_mlflow.genai.register_prompt.assert_not_called()
