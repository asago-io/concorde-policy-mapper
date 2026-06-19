import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_call_counter = 0
_counter_lock = threading.Lock()
_debug_dir: Path | None = None


def configure(debug_dir: Path | None) -> None:
    global _debug_dir, _call_counter
    _debug_dir = debug_dir
    _call_counter = 0
    if _debug_dir:
        _debug_dir.mkdir(parents=True, exist_ok=True)


def _next_call_num() -> int:
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        return _call_counter


def _slug_from_context(context: dict[str, str] | None) -> str:
    if not context:
        return ""
    for key in ("policy_concept", "risk_name", "risk_id"):
        if key in context:
            return "-" + context[key].lower().replace(" ", "-").replace("/", "-")[:40]
    return ""


def _extract_response(response: object) -> dict | list | str:
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]
    if isinstance(response, list):
        return [r.model_dump() if hasattr(r, "model_dump") else r for r in response]
    return str(response)


def log_call(
    stage: str,
    messages: list[dict],
    response,
    *,
    context: dict | None = None,
    report=None,
    duration_ms: float | None = None,
) -> None:
    call_num = _next_call_num()
    slug = _slug_from_context(context)
    response_data = _extract_response(response)

    if _debug_dir is not None:
        entry = {
            "call_number": call_num,
            "stage": stage,
            "messages": messages,
            "response": response_data,
        }
        if context:
            entry["context"] = context

        filename = f"{call_num:02d}-{stage}{slug}.json"
        path = _debug_dir / filename
        path.write_text(json.dumps(entry, indent=2, default=str))
        logger.debug("Debug log written to %s", path)

    if report is not None:
        event: dict = {
            "stage": stage,
            "event": "llm_call",
            "call_number": call_num,
            "messages": messages,
            "response": response_data,
        }
        if context:
            event["context"] = context
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        report.events.append(event)
