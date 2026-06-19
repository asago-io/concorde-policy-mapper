import json
from types import SimpleNamespace

import pytest

from asago_policy_mapper.debug import (
    _extract_response,
    _slug_from_context,
    configure,
    log_call,
)


@pytest.fixture(autouse=True)
def _reset_debug():
    from asago_policy_mapper import debug

    debug.configure(None)
    yield
    debug.configure(None)


# ---------------------------------------------------------------------------
# _slug_from_context
# ---------------------------------------------------------------------------


def test_slug_from_none():
    assert _slug_from_context(None) == ""


def test_slug_from_empty_dict():
    assert _slug_from_context({}) == ""


def test_slug_from_risk_name():
    assert _slug_from_context({"risk_name": "Model Bias"}) == "-model-bias"


def test_slug_from_policy_concept():
    """policy_concept is checked before risk_name, so it takes priority."""
    ctx = {"policy_concept": "Data Privacy", "risk_name": "Model Bias"}
    assert _slug_from_context(ctx) == "-data-privacy"


def test_slug_truncates_long_names():
    long_name = "a" * 80
    slug = _slug_from_context({"risk_name": long_name})
    # dash prefix + 40 chars max
    assert slug == "-" + "a" * 40
    assert len(slug) == 41


# ---------------------------------------------------------------------------
# _extract_response
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal object with a model_dump method, mimicking a Pydantic model."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


def test_extract_pydantic_model():
    model = _FakeModel({"score": 0.9, "label": "high"})
    result = _extract_response(model)
    assert result == {"score": 0.9, "label": "high"}


def test_extract_list_of_models():
    models = [_FakeModel({"id": 1}), _FakeModel({"id": 2})]
    result = _extract_response(models)
    assert result == [{"id": 1}, {"id": 2}]


def test_extract_list_mixed():
    items = [_FakeModel({"id": 1}), "plain-string", 42]
    result = _extract_response(items)
    assert result == [{"id": 1}, "plain-string", 42]


def test_extract_plain_string():
    assert _extract_response("hello") == "hello"
    assert _extract_response(123) == "123"


# ---------------------------------------------------------------------------
# log_call
# ---------------------------------------------------------------------------


def test_log_call_writes_file(tmp_path):
    configure(tmp_path / "debug_out")
    messages = [{"role": "user", "content": "test prompt"}]
    response = _FakeModel({"answer": "yes"})

    log_call("judge", messages, response)

    files = list((tmp_path / "debug_out").glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "01-judge.json"

    data = json.loads(files[0].read_text())
    assert data["call_number"] == 1
    assert data["stage"] == "judge"
    assert data["messages"] == messages
    assert data["response"] == {"answer": "yes"}
    assert "context" not in data


def test_log_call_no_debug_dir():
    configure(None)
    messages = [{"role": "user", "content": "test"}]
    # Should not raise or write anything
    log_call("judge", messages, "some response")


def test_log_call_with_context(tmp_path):
    configure(tmp_path / "debug_out")
    ctx = {"risk_id": "atlas-001", "risk_name": "Bias"}
    messages = [{"role": "user", "content": "test"}]

    log_call("judge", messages, "resp", context=ctx)

    files = list((tmp_path / "debug_out").glob("*.json"))
    assert len(files) == 1
    # risk_id comes after policy_concept and risk_name in priority,
    # but slug uses the first matching key — here risk_name matches first
    assert "bias" in files[0].name

    data = json.loads(files[0].read_text())
    assert data["context"] == ctx


def test_log_call_with_report():
    configure(None)
    report = SimpleNamespace(events=[])
    messages = [{"role": "user", "content": "prompt"}]
    response = _FakeModel({"result": "ok"})

    log_call("ground", messages, response, report=report)

    assert len(report.events) == 1
    event = report.events[0]
    assert event["stage"] == "ground"
    assert event["event"] == "llm_call"
    assert event["messages"] == messages
    assert event["response"] == {"result": "ok"}
    assert "duration_ms" not in event


def test_log_call_report_with_duration():
    configure(None)
    report = SimpleNamespace(events=[])
    messages = [{"role": "user", "content": "prompt"}]

    log_call(
        "ground",
        messages,
        "resp",
        report=report,
        duration_ms=123.456,
    )

    assert len(report.events) == 1
    assert report.events[0]["duration_ms"] == 123.5


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


def test_configure_creates_dir(tmp_path):
    new_dir = tmp_path / "nested" / "debug"
    assert not new_dir.exists()
    configure(new_dir)
    assert new_dir.is_dir()


def test_configure_none_resets():
    configure(None)  # should not raise
