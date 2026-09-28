#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd -- "$project_root"

for tool in openspec rg; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required command not found in PATH: $tool" >&2
    exit 127
  fi
done

if [[ ! -f openspec/config.yaml && ! -f openspec/config.yml ]]; then
  echo "OpenSpec project configuration is missing." >&2
  exit 1
fi

shopt -s nullglob globstar
for proposal in openspec/changes/*/proposal.md; do
  [[ "$proposal" == openspec/changes/archive/* ]] && continue
  if ! rg -q '^## Why\r?$' "$proposal"; then
    echo "$proposal must contain the canonical '## Why' section." >&2
    exit 1
  fi
  if ! rg -q '^## What Changes\r?$' "$proposal"; then
    echo "$proposal must contain the canonical '## What Changes' section." >&2
    exit 1
  fi
done

specs=(openspec/specs/**/spec.md)
active_changes=()
for change_path in openspec/changes/*; do
  [[ -d "$change_path" && "$change_path" != openspec/changes/archive ]] || continue
  active_changes+=("$change_path")
done

if (( ${#specs[@]} == 0 && ${#active_changes[@]} == 0 )); then
  echo "No active changes or main specifications to validate."
  exit 0
fi

exec openspec validate --all --strict --no-interactive
