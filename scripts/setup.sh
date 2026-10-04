#!/bin/sh
# One-time setup: lab venv, JARVIS data, and the pointer the MCP launcher reads.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_features.py
mkdir -p "$HOME/.falsify-labs"
printf '%s\n' "$ROOT" > "$HOME/.falsify-labs/root"
echo "Setup done. Lab root registered: $ROOT"
