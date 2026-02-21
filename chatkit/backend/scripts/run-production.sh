#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Use the shared streamlit-env instead of creating a new venv
VENV_BIN="/data/ec2-user/streamlit-env/bin"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY in your environment before running this script."
  exit 1
fi

echo "Starting ChatKit backend on http://127.0.0.1:8000 ..."
exec "$VENV_BIN/uvicorn" app.main:app --host 127.0.0.1 --port 8000
