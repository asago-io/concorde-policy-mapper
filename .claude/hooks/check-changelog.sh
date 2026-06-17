#!/bin/bash
# Enforce CHANGELOG.md updates when src/ files are committed.
# Minor/maintenance commits (fix:, chore:, ci:, style:, docs:, test:)
# are auto-exempt. All other commits must include a CHANGELOG.md update.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only applies to git commit commands
echo "$COMMAND" | grep -q 'git commit' || exit 0

# Only when src/ files are staged
STAGED=$(git diff --cached --name-only 2>/dev/null)
echo "$STAGED" | grep -q '^src/' || exit 0

# Pass if CHANGELOG.md is already staged
echo "$STAGED" | grep -q '^CHANGELOG.md$' && exit 0

# Auto-exempt minor/maintenance commits
echo "$COMMAND" | grep -qE '(fix|chore|ci|style|docs|test)[:(]' && exit 0

echo "BLOCK: src/ files are staged but CHANGELOG.md has no staged changes." >&2
echo "Either update CHANGELOG.md and stage it, or use an appropriate" >&2
echo "semantic commit prefix (fix:, chore:, ci:, style:, docs:, test:)" >&2
echo "if the change does not warrant a changelog entry." >&2
exit 2
