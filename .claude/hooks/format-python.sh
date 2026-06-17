#!/bin/bash
# Auto-format and lint-fix Python files after edit/write

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // empty')

[[ -z "$FILE_PATH" ]] && exit 0
[[ "$FILE_PATH" == *.py ]] || exit 0

uv run ruff format "$FILE_PATH" 2>/dev/null
uv run ruff check --fix "$FILE_PATH" 2>/dev/null
exit 0
