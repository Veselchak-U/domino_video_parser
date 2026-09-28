#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd -- "$project_root"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <change-id>" >&2
  exit 64
fi

change_id="$1"
if [[ ! "$change_id" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ || "$change_id" == archive ]]; then
  echo "Invalid OpenSpec change id: $change_id" >&2
  exit 64
fi

for tool in openspec rg; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required command not found in PATH: $tool" >&2
    exit 127
  fi
done

change_dir="openspec/changes/$change_id"
proposal="$change_dir/proposal.md"
tasks="$change_dir/tasks.md"

if [[ ! -d "$change_dir" || ! -f "$change_dir/.openspec.yaml" ||
      ! -f "$proposal" || ! -f "$tasks" ]]; then
  echo "Incomplete OpenSpec change: $change_dir" >&2
  exit 1
fi

if ! rg -q '^## Why\r?$' "$proposal"; then
  echo "$proposal must contain the canonical '## Why' section." >&2
  exit 1
fi
if ! rg -q '^## What Changes\r?$' "$proposal"; then
  echo "$proposal must contain the canonical '## What Changes' section." >&2
  exit 1
fi
if rg -n '^[[:blank:]]*[-*+] \[ \]' "$tasks"; then
  echo "OpenSpec change has unchecked tasks and cannot be archived." >&2
  exit 1
fi

shopt -s nullglob globstar
planning_files=("$change_dir"/**/*.md)
if rg -ni '\b(TBD|TODO)\b|нужно решить|уточнить' "${planning_files[@]}"; then
  echo "OpenSpec change contains unresolved decisions." >&2
  exit 1
fi

target_name="$change_id"
if [[ ! "$target_name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}- ]]; then
  target_name="$(date +%Y-%m-%d)-$change_id"
fi
archive_dir="openspec/changes/archive/$target_name"
if [[ -e "$archive_dir" ]]; then
  echo "OpenSpec archive already exists: $archive_dir" >&2
  exit 1
fi

bash "$script_dir/run_openspec_validation.sh"
openspec archive "$change_id" --yes

if [[ ! -d "$archive_dir" || -d "$change_dir" ]]; then
  echo "Unexpected archive location or source still present: $archive_dir" >&2
  exit 1
fi

archived_tasks="$archive_dir/tasks.md"
archived_files=("$archive_dir"/**/*.md)
if rg -n '^[[:blank:]]*[-*+] \[ \]' "$archived_tasks" ||
   rg -ni '\b(TBD|TODO)\b|нужно решить|уточнить' "${archived_files[@]}"; then
  echo "Archived OpenSpec change contains unresolved work." >&2
  exit 1
fi

for archived_spec in "$archive_dir"/specs/**/spec.md; do
  relative_spec="${archived_spec#"$archive_dir/specs/"}"
  current_spec="openspec/specs/$relative_spec"
  if [[ ! -f "$current_spec" ]]; then
    if rg -q '^## (ADDED|MODIFIED|RENAMED) Requirements\r?$' "$archived_spec"; then
      echo "Archived capability was not synchronized: $current_spec" >&2
      exit 1
    fi
    continue
  fi
  if rg -ni '\b(TBD|TODO)\b|нужно решить|уточнить' "$current_spec"; then
    echo "Synchronized capability contains unresolved decisions: $current_spec" >&2
    exit 1
  fi
done

bash "$script_dir/run_openspec_validation.sh"
echo "OpenSpec change archived and validated: $archive_dir"
