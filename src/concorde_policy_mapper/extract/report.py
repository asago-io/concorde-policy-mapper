import json
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

_DARK_MODE_SNIPPET: str | None = None
_DARK_TOGGLE = (
    '<button class="dark-toggle" id="rl-dark-toggle" onclick="'
    "document.documentElement.classList.toggle('dark');"
    "var d=document.documentElement.classList.contains('dark');"
    "localStorage.setItem('rl-dark-mode',d);"
    "this.textContent=d?'\\u2600\\uFE0F':'\\uD83C\\uDF19'"
    '"></button>'
    "<script>"
    "document.getElementById('rl-dark-toggle').textContent="
    "document.documentElement.classList.contains('dark')"
    "?'\\u2600\\uFE0F':'\\uD83C\\uDF19';"
    "</script>"
)


def _get_dark_snippet() -> str:
    global _DARK_MODE_SNIPPET
    if _DARK_MODE_SNIPPET is None:
        _DARK_MODE_SNIPPET = (TEMPLATE_DIR / "dark_mode_snippet.html").read_text()
    return _DARK_MODE_SNIPPET


def build_risk_extraction_report(data: dict, output_path: Path) -> Path:
    html = (
        (TEMPLATE_DIR / "risk_extraction_report_template.html")
        .read_text()
        .replace("__REPORT_DATA__", json.dumps(data, default=str))
    )
    html = html.replace("</head>", _get_dark_snippet() + "\n</head>", 1)
    html = html.replace("</body>", _DARK_TOGGLE + "\n</body>", 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
