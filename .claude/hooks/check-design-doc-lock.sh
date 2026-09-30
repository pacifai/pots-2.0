#!/usr/bin/env bash
# Stop hook: keeps a session from ending its turn with a shared design doc left LOCKED.
# The lock protocol is described in CLAUDE.md under "Shared design docs (multi-session)".
#
# Blocks the stop only when both hold:
#   1. a doc in docs/verification/ has its EDIT STATUS line set to LOCKED, and
#   2. this session edited a doc in docs/verification/ (so the lock is probably its own).
# A lock held by another session never blocks this one. The hook blocks once per stop; if
# the lock is still set on the retry (stop_hook_active), it warns the user and lets the stop through.

set -euo pipefail

input=$(cat)
transcript=$(jq -r '.transcript_path // empty' <<<"$input")
retry=$(jq -r '.stop_hook_active // false' <<<"$input")

docs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/docs/verification"
[[ -d "$docs_dir" ]] || exit 0

locked=()
for f in "$docs_dir"/*.md; do
  status=$(grep -m1 'EDIT STATUS' "$f" || true)
  # "UNLOCKED" contains "LOCKED", so require LOCKED not preceded by N.
  if [[ -n "$status" ]] && grep -Eq '🔒|(^|[^N])LOCKED' <<<"$status"; then
    locked+=("$(basename "$f")")
  fi
done
(( ${#locked[@]} )) || exit 0

[[ -n "$transcript" && -f "$transcript" ]] || exit 0
edited=$(jq -r '
  select(.type == "assistant") | .message.content[]?
  | select(.type == "tool_use")
  | select(
      ((.name | test("^(Edit|MultiEdit|Write|NotebookEdit)$")) and ((.input.file_path // "") | contains("docs/verification/")))
      or (.name == "Bash" and ((.input.command // "")
            | test("docs/verification/|STATUS\\.md|_TASKS\\.md|DECISIONS_|BACKGROUND\\.md|VERIFICATION_PROTOCOL_")
              and test("(sed|perl) +-i|\\btee\\b|(^|[^0-9&])>|\\b(mv|cp|python3?)\\b")))
    )
  | "yes"' "$transcript" 2>/dev/null | head -n1)
[[ "$edited" == "yes" ]] || exit 0

list=$(printf '%s, ' "${locked[@]}"); list=${list%, }

if [[ "$retry" == "true" ]]; then
  jq -n --arg l "$list" '{systemMessage: ("Design doc still LOCKED: " + $l + ". If this session set the lock, release it (set EDIT STATUS back to 🔓 UNLOCKED).")}'
  exit 0
fi

jq -n --arg l "$list" '{
  decision: "block",
  reason: ("Shared design doc left LOCKED: " + $l + ". This session edited docs/verification/ in this conversation. If the lock is yours, set the EDIT STATUS line back to `🔓 UNLOCKED` now. If another session holds it (its lock line names that session), leave it alone and end your turn.")
}'
