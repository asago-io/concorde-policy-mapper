no_cross_encoder := ""
rrf_min_score := ""
no_mlflow := ""
no_judge := ""
no_grounding := ""
temperature := ""
top_p := ""
top_k := ""

test:
    uv run pytest tests/ -m "not slow"

test-all:
    uv run pytest tests/

test-slow:
    uv run pytest tests/ -m slow

tidy: format lint type-check

type-check:
    uv run mypy src/asago_policy_mapper/

lint:
    uv run ruff check src/ tests/

format:
    uv run ruff format src/ tests/

run-risk-extract-battery battery base_url="" model="" bi_encoder="" cross_encoder="" jobs="6":
    uv run python run_extract_battery.py {{ battery }} -j {{ jobs }} {{ if base_url != "" { "--base-url " + base_url } else { "" } }} {{ if model != "" { "--model " + model } else { "" } }} {{ if bi_encoder != "" { "--bi-encoder-model " + bi_encoder } else { "" } }} {{ if cross_encoder != "" { "--cross-encoder-model " + cross_encoder } else { "" } }} {{ if no_cross_encoder != "" { "--no-cross-encoder" } else { "" } }} {{ if rrf_min_score != "" { "--rrf-min-score " + rrf_min_score } else { "" } }} {{ if no_mlflow != "" { "--no-mlflow" } else { "" } }} {{ if no_judge != "" { "--no-judge" } else { "" } }} {{ if no_grounding != "" { "--no-grounding" } else { "" } }} {{ if temperature != "" { "--temperature " + temperature } else { "" } }} {{ if top_p != "" { "--top-p " + top_p } else { "" } }} {{ if top_k != "" { "--top-k " + top_k } else { "" } }}
