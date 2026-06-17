#!/bin/bash
# Block destructive shell commands (rm -rf, DROP TABLE (not used in this project), etc.)
# Git operations are intentionally excluded.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if echo "$COMMAND" | grep -qE 'rm -rf |rm -fr '; then
  echo "BLOCK: potentially destructive command detected. Use targeted operations or ask the user." >&2
  exit 2
fi
exit 0