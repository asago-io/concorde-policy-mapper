from unittest.mock import patch, MagicMock
import pytest

from concorde_policy_mapper.tracking import (
    is_tracking_enabled,
    init_tracking,
    end_tracking,
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
