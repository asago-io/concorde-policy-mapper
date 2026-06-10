import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from instructor.core import IncompleteOutputException, InstructorRetryException

from concorde_policy_mapper.llm import (
    LLMConfig,
    TokenTracker,
    _call_with_retry,
    _extract_response_content,
    _IncompleteOutput,
    _retry_with_validation,
    _strip_titles,
    _track_completion,
    _truncate_messages,
)

# ---------------------------------------------------------------------------
# _strip_titles
# ---------------------------------------------------------------------------


class TestStripTitles:
    def test_removes_title_from_flat_dict(self):
        obj = {"title": "Remove me", "name": "keep"}
        result = _strip_titles(obj)
        assert "title" not in result
        assert result["name"] == "keep"

    def test_removes_title_from_nested_dicts(self):
        obj = {
            "title": "top",
            "properties": {
                "title": "nested",
                "field": {
                    "title": "deep",
                    "type": "string",
                },
            },
        }
        result = _strip_titles(obj)
        assert "title" not in result
        assert "title" not in result["properties"]
        assert "title" not in result["properties"]["field"]
        assert result["properties"]["field"]["type"] == "string"

    def test_removes_title_from_list_of_dicts(self):
        obj = [{"title": "a", "x": 1}, {"title": "b", "y": 2}]
        result = _strip_titles(obj)
        assert all("title" not in d for d in result)
        assert result[0]["x"] == 1
        assert result[1]["y"] == 2

    def test_already_clean_dict(self):
        obj = {"name": "no title here", "value": 42}
        result = _strip_titles(obj)
        assert result == {"name": "no title here", "value": 42}

    def test_non_dict_passthrough(self):
        assert _strip_titles(42) == 42
        assert _strip_titles("hello") == "hello"
        assert _strip_titles(None) is None

    def test_mixed_nested_structure(self):
        obj = {
            "title": "root",
            "items": [
                {"title": "item1", "id": 1},
                {"id": 2},
            ],
            "nested": {"title": "child", "data": [{"title": "deep"}]},
        }
        result = _strip_titles(obj)
        assert "title" not in result
        assert "title" not in result["items"][0]
        assert "title" not in result["nested"]
        assert "title" not in result["nested"]["data"][0]

    def test_mutates_in_place(self):
        obj = {"title": "gone", "keep": "yes"}
        returned = _strip_titles(obj)
        assert returned is obj
        assert "title" not in obj


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


class TestTokenTracker:
    def test_add_with_valid_usage(self):
        tracker = TokenTracker()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        tracker.add(usage)
        assert tracker.prompt_tokens == 10
        assert tracker.completion_tokens == 5
        assert tracker.total_tokens == 15
        assert tracker.calls == 1

    def test_add_accumulates(self):
        tracker = TokenTracker()
        usage1 = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        usage2 = SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        tracker.add(usage1)
        tracker.add(usage2)
        assert tracker.prompt_tokens == 30
        assert tracker.completion_tokens == 15
        assert tracker.total_tokens == 45
        assert tracker.calls == 2

    def test_add_none_usage_is_noop(self):
        tracker = TokenTracker()
        tracker.add(None)
        assert tracker.prompt_tokens == 0
        assert tracker.calls == 0

    def test_add_with_stage(self):
        tracker = TokenTracker()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        tracker.add(usage, stage="judge")
        assert "judge" in tracker.per_stage
        assert tracker.per_stage["judge"]["prompt_tokens"] == 10
        assert tracker.per_stage["judge"]["completion_tokens"] == 5
        assert tracker.per_stage["judge"]["total_tokens"] == 15
        assert tracker.per_stage["judge"]["calls"] == 1

    def test_add_stage_accumulates(self):
        tracker = TokenTracker()
        usage1 = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        usage2 = SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        tracker.add(usage1, stage="ground")
        tracker.add(usage2, stage="ground")
        assert tracker.per_stage["ground"]["calls"] == 2
        assert tracker.per_stage["ground"]["total_tokens"] == 45

    def test_usage_values_handles_missing_attrs(self):
        usage = SimpleNamespace()
        pt, ct, tt = TokenTracker._usage_values(usage)
        assert (pt, ct, tt) == (0, 0, 0)

    def test_usage_values_handles_none_attrs(self):
        usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        pt, ct, tt = TokenTracker._usage_values(usage)
        assert (pt, ct, tt) == (0, 0, 0)

    def test_to_dict(self):
        tracker = TokenTracker()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        tracker.add(usage, stage="judge")
        tracker.record_incident("test_error", "something failed")

        d = tracker.to_dict()
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 5
        assert d["total_tokens"] == 15
        assert d["calls"] == 1
        assert "judge" in d["per_stage"]
        assert len(d["incidents"]) == 1
        assert d["incidents"][0]["kind"] == "test_error"

    def test_record_incident_basic(self):
        tracker = TokenTracker()
        tracker.record_incident("context_overflow", "max_tokens reduced")
        assert len(tracker.incidents) == 1
        assert tracker.incidents[0]["kind"] == "context_overflow"
        assert tracker.incidents[0]["detail"] == "max_tokens reduced"

    def test_record_incident_includes_stage(self):
        tracker = TokenTracker()
        tracker.set_stage("grounding")
        tracker.record_incident("error", "something broke")
        assert tracker.incidents[0]["stage"] == "grounding"

    def test_record_incident_no_stage(self):
        tracker = TokenTracker()
        tracker.record_incident("error", "no stage set")
        assert "stage" not in tracker.incidents[0]

    def test_set_stage(self):
        tracker = TokenTracker()
        tracker.set_stage("judge")
        assert tracker._current_stage == "judge"
        tracker.set_stage(None)
        assert tracker._current_stage is None


# ---------------------------------------------------------------------------
# _track_completion
# ---------------------------------------------------------------------------


class TestTrackCompletion:
    def test_valid_usage_updates_tracker(self):
        tracker = TokenTracker()
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        completion = SimpleNamespace(usage=usage)
        _track_completion(tracker, completion)
        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150
        assert tracker.calls == 1

    def test_no_usage_logs_warning(self, caplog):
        tracker = TokenTracker()
        completion = SimpleNamespace(usage=None)
        with caplog.at_level(logging.WARNING):
            _track_completion(tracker, completion)
        assert tracker._warned_no_usage is True
        assert "no token usage" in caplog.text

    def test_no_usage_warns_only_once(self, caplog):
        tracker = TokenTracker()
        completion = SimpleNamespace(usage=None)
        with caplog.at_level(logging.WARNING):
            _track_completion(tracker, completion)
            caplog.clear()
            _track_completion(tracker, completion)
        assert "no token usage" not in caplog.text

    def test_zero_total_tokens_triggers_warning(self, caplog):
        tracker = TokenTracker()
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        completion = SimpleNamespace(usage=usage)
        with caplog.at_level(logging.WARNING):
            _track_completion(tracker, completion)
        assert tracker._warned_no_usage is True

    def test_no_usage_attr_triggers_warning(self, caplog):
        tracker = TokenTracker()
        completion = object()  # no .usage attribute at all
        with caplog.at_level(logging.WARNING):
            _track_completion(tracker, completion)
        assert tracker._warned_no_usage is True


# ---------------------------------------------------------------------------
# _extract_response_content
# ---------------------------------------------------------------------------


class TestExtractResponseContent:
    def test_valid_completion(self):
        msg = SimpleNamespace(content="Hello world")
        choice = SimpleNamespace(message=msg)
        completion = SimpleNamespace(choices=[choice])
        assert _extract_response_content(completion) == "Hello world"

    def test_empty_choices(self):
        completion = SimpleNamespace(choices=[])
        assert _extract_response_content(completion) is None

    def test_no_choices_attr(self):
        completion = SimpleNamespace()
        assert _extract_response_content(completion) is None

    def test_no_message_attr(self):
        choice = SimpleNamespace()
        completion = SimpleNamespace(choices=[choice])
        assert _extract_response_content(completion) is None

    def test_no_content_attr(self):
        msg = SimpleNamespace()
        choice = SimpleNamespace(message=msg)
        completion = SimpleNamespace(choices=[choice])
        assert _extract_response_content(completion) is None


# ---------------------------------------------------------------------------
# _truncate_messages
# ---------------------------------------------------------------------------


class TestTruncateMessages:
    def test_truncates_long_user_message(self):
        long_content = "x" * 2000
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": long_content},
        ]
        result = _truncate_messages(messages)
        assert result is not None
        # 60% of 2000 = 1200, which is > 500 min
        assert len(result[1]["content"]) == 1200

    def test_short_user_message_returns_none(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "short"},
        ]
        result = _truncate_messages(messages)
        assert result is None

    def test_exactly_min_threshold_returns_none(self):
        messages = [{"role": "user", "content": "x" * 500}]
        result = _truncate_messages(messages)
        assert result is None

    def test_no_user_messages_returns_none(self):
        messages = [{"role": "system", "content": "System only."}]
        result = _truncate_messages(messages)
        assert result is None

    def test_empty_messages_returns_none(self):
        result = _truncate_messages([])
        assert result is None

    def test_does_not_mutate_original(self):
        long_content = "x" * 2000
        messages = [{"role": "user", "content": long_content}]
        original_content = messages[0]["content"]
        _truncate_messages(messages)
        assert messages[0]["content"] == original_content
        assert len(messages[0]["content"]) == 2000

    def test_truncates_longest_user_message(self):
        messages = [
            {"role": "user", "content": "a" * 1000},
            {"role": "user", "content": "b" * 3000},
        ]
        result = _truncate_messages(messages)
        assert result is not None
        # Longest is index 1 (3000 chars), truncated to 60% = 1800
        assert len(result[0]["content"]) == 1000  # unchanged
        assert len(result[1]["content"]) == 1800

    def test_truncation_respects_minimum(self):
        # 60% of 600 = 360, which is < 500, so clamp to 500
        messages = [{"role": "user", "content": "y" * 600}]
        result = _truncate_messages(messages)
        assert result is not None
        assert len(result[0]["content"]) == 500


# ---------------------------------------------------------------------------
# _call_with_retry
# ---------------------------------------------------------------------------


class TestCallWithRetry:
    def _make_config(self):
        return LLMConfig(base_url="http://test", model="test-model")

    def test_success_on_first_attempt(self):
        do_call = MagicMock(return_value=("result", "completion"))
        kwargs = {"messages": [{"role": "user", "content": "hello"}], "max_retries": 0}
        config = self._make_config()

        result = _call_with_retry(do_call, kwargs, config)
        assert result == ("result", "completion")
        do_call.assert_called_once()

    def test_incomplete_output_with_truncatable_messages_retries(self):
        long_content = "x" * 2000
        cause = IncompleteOutputException(last_completion=None)
        do_call = MagicMock(side_effect=[_IncompleteOutput(cause), ("ok", "comp")])
        kwargs = {"messages": [{"role": "user", "content": long_content}], "max_retries": 0}
        config = self._make_config()

        result = _call_with_retry(do_call, kwargs, config)
        assert result == ("ok", "comp")
        assert do_call.call_count == 2

    def test_incomplete_output_short_messages_raises(self):
        cause = IncompleteOutputException(last_completion=None)
        do_call = MagicMock(side_effect=_IncompleteOutput(cause))
        kwargs = {"messages": [{"role": "user", "content": "short"}], "max_retries": 0}
        config = self._make_config()

        with pytest.raises(IncompleteOutputException):
            _call_with_retry(do_call, kwargs, config)

    def test_incomplete_output_records_incident_when_cannot_truncate(self):
        tracker = TokenTracker()
        cause = IncompleteOutputException(last_completion=None)
        do_call = MagicMock(side_effect=_IncompleteOutput(cause))
        kwargs = {"messages": [{"role": "user", "content": "short"}], "max_retries": 0}
        config = self._make_config()

        with pytest.raises(IncompleteOutputException):
            _call_with_retry(do_call, kwargs, config, tracker=tracker)
        assert any(i["kind"] == "output_truncated" for i in tracker.incidents)

    def test_incomplete_output_records_incident_on_retry(self):
        tracker = TokenTracker()
        long_content = "x" * 2000
        cause = IncompleteOutputException(last_completion=None)
        do_call = MagicMock(side_effect=[_IncompleteOutput(cause), ("ok", "comp")])
        kwargs = {"messages": [{"role": "user", "content": long_content}], "max_retries": 0}
        config = self._make_config()

        _call_with_retry(do_call, kwargs, config, tracker=tracker)
        assert any(i["kind"] == "output_truncated" for i in tracker.incidents)

    def test_messages_shortened_on_retry(self):
        long_content = "x" * 2000
        cause = IncompleteOutputException(last_completion=None)

        captured_kwargs = {}

        def side_effect(kw):
            if not captured_kwargs:
                captured_kwargs["first"] = copy.deepcopy(kw)
                raise _IncompleteOutput(cause)
            captured_kwargs["second"] = copy.deepcopy(kw)
            return ("ok", "comp")

        do_call = MagicMock(side_effect=side_effect)
        kwargs = {"messages": [{"role": "user", "content": long_content}], "max_retries": 0}
        config = self._make_config()

        _call_with_retry(do_call, kwargs, config)
        first_len = len(captured_kwargs["first"]["messages"][0]["content"])
        second_len = len(captured_kwargs["second"]["messages"][0]["content"])
        assert second_len < first_len


# ---------------------------------------------------------------------------
# _retry_with_validation
# ---------------------------------------------------------------------------


def _make_retry_exc(msg, failed_attempts=None):
    """Create an InstructorRetryException with the given message and failed attempts."""
    return InstructorRetryException(
        msg,
        n_attempts=1,
        total_usage=0,
        failed_attempts=failed_attempts,
    )


class TestRetryWithValidation:
    def _make_config(self):
        return LLMConfig(base_url="http://test", model="test-model")

    def _make_messages(self):
        return [{"role": "user", "content": "hello"}]

    def test_success_on_first_call(self):
        do_call = MagicMock(return_value=("result", "completion"))
        messages = self._make_messages()
        kwargs = {"messages": messages, "max_tokens": 4096}
        config = self._make_config()

        result = _retry_with_validation(
            do_call,
            kwargs,
            config,
            tracker=None,
            original_messages=copy.deepcopy(messages),
            current_messages=messages,
            max_retries=2,
        )
        assert result == ("result", "completion")
        do_call.assert_called_once()

    def test_incomplete_output_wraps_in_incomplete_output(self):
        cause = IncompleteOutputException(last_completion=None)
        do_call = MagicMock(side_effect=cause)
        messages = self._make_messages()
        kwargs = {"messages": messages, "max_tokens": 4096}
        config = self._make_config()

        with pytest.raises(_IncompleteOutput) as exc_info:
            _retry_with_validation(
                do_call,
                kwargs,
                config,
                tracker=None,
                original_messages=copy.deepcopy(messages),
                current_messages=messages,
                max_retries=2,
            )
        assert exc_info.value.cause is cause

    def test_context_overflow_reduces_max_tokens(self):
        """Context overflow with a valid reduction retries with lower max_tokens."""
        overflow_msg = (
            "maximum context length is 8192 tokens. However, you requested 4096 output tokens and 4000 input tokens"
        )
        exc = _make_retry_exc(overflow_msg)
        do_call = MagicMock(side_effect=[exc, ("ok", "comp")])
        messages = self._make_messages()
        original = copy.deepcopy(messages)
        kwargs = {"messages": copy.deepcopy(messages), "max_tokens": 4096}
        config = self._make_config()

        result = _retry_with_validation(
            do_call,
            kwargs,
            config,
            tracker=None,
            original_messages=original,
            current_messages=messages,
            max_retries=1,
        )
        assert result == ("ok", "comp")
        assert do_call.call_count == 2
        # Expected new_max: 8192 - 4000 - 32 = 4160
        assert kwargs["max_tokens"] == 4160

    def test_context_overflow_records_incident(self):
        overflow_msg = (
            "maximum context length is 8192 tokens. However, you requested 4096 output tokens and 4000 input tokens"
        )
        exc = _make_retry_exc(overflow_msg)
        do_call = MagicMock(side_effect=[exc, ("ok", "comp")])
        messages = self._make_messages()
        original = copy.deepcopy(messages)
        kwargs = {"messages": copy.deepcopy(messages), "max_tokens": 4096}
        config = self._make_config()
        tracker = TokenTracker()

        _retry_with_validation(
            do_call,
            kwargs,
            config,
            tracker=tracker,
            original_messages=original,
            current_messages=messages,
            max_retries=1,
        )
        assert any(i["kind"] == "context_overflow" for i in tracker.incidents)

    def test_validation_error_retries_with_hint(self):
        exc = _make_retry_exc("some validation error")
        do_call = MagicMock(side_effect=[exc, ("ok", "comp")])
        messages = self._make_messages()
        original = copy.deepcopy(messages)
        kwargs = {"messages": copy.deepcopy(messages), "max_tokens": 4096}
        config = self._make_config()

        result = _retry_with_validation(
            do_call,
            kwargs,
            config,
            tracker=None,
            original_messages=original,
            current_messages=messages,
            max_retries=1,
        )
        assert result == ("ok", "comp")
        assert do_call.call_count == 2
        # On retry, messages should have an appended user message with the error hint
        retry_messages = kwargs["messages"]
        assert any(msg["role"] == "user" and "Validation error:" in msg["content"] for msg in retry_messages)

    def test_validation_exhausted_raises(self):
        exc = _make_retry_exc("persistent validation error")
        do_call = MagicMock(side_effect=exc)
        messages = self._make_messages()
        original = copy.deepcopy(messages)
        kwargs = {"messages": copy.deepcopy(messages), "max_tokens": 4096}
        config = self._make_config()
        tracker = TokenTracker()

        with pytest.raises(InstructorRetryException):
            _retry_with_validation(
                do_call,
                kwargs,
                config,
                tracker=tracker,
                original_messages=original,
                current_messages=messages,
                max_retries=2,
            )
        # Should have attempted max_retries + 1 = 3 times
        assert do_call.call_count == 3
        assert any(i["kind"] == "validation_exhausted" for i in tracker.incidents)

    def test_validation_error_appends_failed_content(self):
        """When failed_attempt has completion with content, it is appended as
        an assistant message before the error hint."""
        msg_obj = SimpleNamespace(content="partial JSON output")
        choice = SimpleNamespace(message=msg_obj)
        completion = SimpleNamespace(choices=[choice])
        failed_attempt = SimpleNamespace(
            attempt_number=1,
            exception=ValueError("bad field"),
            completion=completion,
        )
        exc = _make_retry_exc("validation error", failed_attempts=[failed_attempt])
        do_call = MagicMock(side_effect=[exc, ("ok", "comp")])
        messages = self._make_messages()
        original = copy.deepcopy(messages)
        kwargs = {"messages": copy.deepcopy(messages), "max_tokens": 4096}
        config = self._make_config()

        result = _retry_with_validation(
            do_call,
            kwargs,
            config,
            tracker=None,
            original_messages=original,
            current_messages=messages,
            max_retries=1,
        )
        assert result == ("ok", "comp")
        retry_messages = kwargs["messages"]
        # Should contain the failed assistant content followed by the error hint
        assistant_msgs = [m for m in retry_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1]["content"] == "partial JSON output"
        # Error hint should reference the exception from the failed attempt
        user_hints = [m for m in retry_messages if m["role"] == "user" and "Validation error:" in m["content"]]
        assert len(user_hints) == 1
        assert "bad field" in user_hints[0]["content"]
