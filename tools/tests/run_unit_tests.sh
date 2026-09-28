#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../.."
if [[ -x .venv/Scripts/python.exe ]]; then
  exec .venv/Scripts/python.exe -X utf8 -m pytest "$@"
elif [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m pytest "$@"
else
  echo "Create .venv and install .[test] first." >&2
  exit 127
fi
