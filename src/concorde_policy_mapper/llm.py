import copy
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import instructor
from instructor.core import IncompleteOutputException, InstructorRetryException
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_INSTRUCTOR_SCHEMA_OVERHEAD = 500


def _strip_titles(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj.pop("title", None)
        for v in obj.values():
            _strip_titles(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_titles(item)
    return obj


class SlimModel(BaseModel):
    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        return _strip_titles(schema)


_SAFETY_MARGIN = 64
_MIN_OUTPUT_TOKENS = 256


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str = "none"
    temperature: float = 0.3
    max_retries: int = 3
    max_tokens: int = 8192
    max_concurrent: int = 32
    max_context: int = 0

    def __post_init__(self):
        if self.max_context > 0 and self.max_tokens >= self.max_context:
            self.max_tokens = self.max_context - _SAFETY_MARGIN - _INSTRUCTOR_SCHEMA_OVERHEAD


@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    per_stage: dict[str, dict[str, int]] = field(default_factory=dict)
    incidents: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _warned_no_usage: bool = field(default=False, repr=False)
    _current_stage: str | None = field(default=None, repr=False)
    _thread_local: threading.local = field(default_factory=threading.local, repr=False)

    @staticmethod
    def _usage_values(usage) -> tuple[int, int, int]:
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            getattr(usage, "total_tokens", 0) or 0,
        )

    def add(self, usage, stage: str | None = None) -> None:
        if usage is None:
            return
        pt, ct, tt = self._usage_values(usage)
        with self._lock:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_tokens += tt
            self.calls += 1
            if stage:
                s = self.per_stage.setdefault(
                    stage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
                )
                s["prompt_tokens"] += pt
                s["completion_tokens"] += ct
                s["total_tokens"] += tt
                s["calls"] += 1

    def set_stage(self, stage: str | None) -> None:
        with self._lock:
            self._current_stage = stage

    def record_incident(self, kind: str, detail: str) -> None:
        with self._lock:
            incident: dict = {"kind": kind, "detail": detail}
            if self._current_stage:
                incident["stage"] = self._current_stage
            messages = getattr(self._thread_local, "current_messages", None)
            if messages:
                incident["messages"] = copy.deepcopy(messages)
            self.incidents.append(incident)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "per_stage": dict(self.per_stage),
            "incidents": list(self.incidents),
        }


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    total += 2
    return total


def budget_max_tokens(
    messages: list[dict[str, str]],
    max_context: int,
    requested_max_tokens: int,
) -> int:
    if max_context <= 0:
        return requested_max_tokens

    estimated_input = estimate_message_tokens(messages) + _INSTRUCTOR_SCHEMA_OVERHEAD
    available = max_context - estimated_input - _SAFETY_MARGIN

    if available < _MIN_OUTPUT_TOKENS:
        logger.warning(
            "Token budget critically low: estimated_input=%d, max_context=%d, available_for_output=%d (min=%d)",
            estimated_input,
            max_context,
            available,
            _MIN_OUTPUT_TOKENS,
        )
        return _MIN_OUTPUT_TOKENS

    capped = min(requested_max_tokens, available)
    if capped < requested_max_tokens:
        logger.debug(
            "Budget: max_tokens %d -> %d (input~%d, context=%d)",
            requested_max_tokens,
            capped,
            estimated_input,
            max_context,
        )
    return capped


def create_client(
    config: LLMConfig,
    tracker: TokenTracker | None = None,
) -> instructor.Instructor:
    client = instructor.from_openai(
        OpenAI(base_url=config.base_url, api_key=config.api_key),
        mode=instructor.Mode.JSON,
    )
    if tracker is not None:
        _wrap_with_tracking(client, tracker, config)
    return client


_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?(\d+) output tokens.*?(\d+) input tokens",
)


def _extract_reduced_max_tokens(exc: Exception) -> int | None:
    msg = str(exc)
    matches = list(_CONTEXT_LENGTH_RE.finditer(msg))
    if not matches:
        return None
    m = matches[-1]
    context_limit = int(m.group(1))
    input_tokens = int(m.group(3))
    new_max = context_limit - input_tokens - 32
    return new_max if new_max >= 256 else None


def _extract_response_content(completion) -> str | None:
    try:
        return completion.choices[0].message.content
    except (AttributeError, IndexError):
        return None


_TRUNCATION_FACTOR = 0.6
_MIN_USER_MESSAGE_CHARS = 500
_MAX_TRUNCATION_RETRIES = 3


def _truncate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    longest_idx = -1
    longest_len = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            content_len = len(msg.get("content", ""))
            if content_len > longest_len:
                longest_len = content_len
                longest_idx = i
    if longest_idx < 0 or longest_len <= _MIN_USER_MESSAGE_CHARS:
        return None
    truncated = copy.deepcopy(messages)
    new_len = max(int(longest_len * _TRUNCATION_FACTOR), _MIN_USER_MESSAGE_CHARS)
    truncated[longest_idx]["content"] = truncated[longest_idx]["content"][:new_len]
    return truncated


def _apply_budget(kwargs: dict, config: LLMConfig) -> None:
    if config.max_context > 0:
        messages = kwargs.get("messages", [])
        requested = kwargs.get("max_tokens", config.max_tokens)
        kwargs["max_tokens"] = budget_max_tokens(
            messages,
            config.max_context,
            requested,
        )


class _IncompleteOutput(Exception):
    """Signals that validation succeeded but output was truncated."""

    def __init__(self, cause):
        self.cause = cause


def _retry_with_validation(do_call, kwargs, config, tracker, original_messages, current_messages, max_retries):
    for val_attempt in range(max_retries + 1):
        try:
            return do_call(kwargs)
        except IncompleteOutputException as e:
            raise _IncompleteOutput(e) from e
        except InstructorRetryException as e:
            new_max = _extract_reduced_max_tokens(e)
            if new_max is not None:
                old_max = kwargs.get("max_tokens", 8192)
                logger.info("Context overflow, reducing max_tokens %d -> %d and retrying", old_max, new_max)
                if tracker:
                    tracker.record_incident(
                        "context_overflow", f"Context overflow, reducing max_tokens {old_max} -> {new_max}"
                    )
                kwargs["messages"] = copy.deepcopy(original_messages)
                kwargs["max_tokens"] = new_max
                _apply_budget(kwargs, config)
                continue
            if val_attempt < max_retries:
                logger.info(
                    "Validation error (attempt %d/%d), retrying with fresh messages + error hint",
                    val_attempt + 1,
                    max_retries,
                )
                retry_messages = copy.deepcopy(current_messages)
                last_attempt = e.failed_attempts[-1] if e.failed_attempts else None
                if last_attempt and last_attempt.completion:
                    failed_content = _extract_response_content(last_attempt.completion)
                    if failed_content:
                        retry_messages.append({"role": "assistant", "content": failed_content})
                error_text = str(last_attempt.exception) if last_attempt else str(e)
                retry_messages.append(
                    {
                        "role": "user",
                        "content": f"Validation error: {error_text}\nCorrect your JSON response, fix the errors.",
                    }
                )
                kwargs["messages"] = retry_messages
                _apply_budget(kwargs, config)
                continue
            if tracker:
                tracker.record_incident("validation_exhausted", f"Validation retries exhausted: {e}")
            raise
    raise RuntimeError("_retry_with_validation exhausted retries without returning a completion")


def _call_with_retry(
    do_call,
    kwargs: dict,
    config: LLMConfig,
    tracker: TokenTracker | None = None,
) -> tuple:
    original_messages = copy.deepcopy(kwargs.get("messages", []))
    current_messages = kwargs.get("messages", [])
    max_validation_retries = kwargs.pop("max_retries", config.max_retries)
    kwargs["max_retries"] = 0

    for attempt in range(_MAX_TRUNCATION_RETRIES + 1):
        try:
            return _retry_with_validation(
                do_call,
                kwargs,
                config,
                tracker,
                original_messages,
                current_messages,
                max_validation_retries,
            )
        except _IncompleteOutput as e:
            shorter = _truncate_messages(current_messages)
            if shorter is None:
                if tracker:
                    tracker.record_incident(
                        "output_truncated", "Output truncated and prompt cannot be shortened further"
                    )
                raise e.cause
            logger.info(
                "Output truncated (attempt %d/%d), retrying with shorter prompt", attempt + 1, _MAX_TRUNCATION_RETRIES
            )
            if tracker:
                tracker.record_incident(
                    "output_truncated",
                    f"Output truncated (attempt {attempt + 1}/{_MAX_TRUNCATION_RETRIES}), retrying with shorter prompt",
                )
            current_messages = shorter
            kwargs["messages"] = shorter
            _apply_budget(kwargs, config)


def _track_completion(tracker: TokenTracker, completion) -> None:
    usage = getattr(completion, "usage", None)
    if usage is None or (getattr(usage, "total_tokens", 0) or 0) == 0:
        if not tracker._warned_no_usage:
            logger.warning("LLM backend returned no token usage — cost tracking unavailable")
            tracker._warned_no_usage = True
    tracker.add(usage)


def _wrap_with_tracking(client: instructor.Instructor, tracker: TokenTracker, config: LLMConfig) -> None:
    def _do_call(kwargs):
        return client.chat.completions.create_with_completion(**kwargs)

    def tracked_create(**kwargs):
        _apply_budget(kwargs, config)
        tracker._thread_local.current_messages = kwargs.get("messages")
        try:
            result, completion = _call_with_retry(_do_call, kwargs, config, tracker=tracker)
        finally:
            tracker._thread_local.current_messages = None
        _track_completion(tracker, completion)
        return result

    client.chat.completions.create = tracked_create
