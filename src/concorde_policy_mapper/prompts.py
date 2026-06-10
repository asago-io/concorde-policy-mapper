from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


@lru_cache(maxsize=1)
def _env() -> Environment:
    template_dir = Path(__file__).parent / "templates" / "prompts"
    return Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def _try_render(name: str, context: dict[str, Any]) -> str | None:
    try:
        tpl = _env().get_template(name)
    except Exception:
        return None
    return tpl.render(**context)


def render_prompt(prompt_name: str, context: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = _try_render(f"{prompt_name}_system.j2", context)
    if system:
        messages.append({"role": "system", "content": system})
    user = _env().get_template(f"{prompt_name}_user.j2").render(**context)
    messages.append({"role": "user", "content": user})
    return messages
