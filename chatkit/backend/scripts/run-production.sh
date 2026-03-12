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

# If threevictors is available only in system python, bridge it into PYTHONPATH.
if ! "$VENV_BIN/python" -c "import threevictors" >/dev/null 2>&1; then
  THREEVICTORS_SITE_PACKAGES="$(
    python3 - <<'PY' 2>/dev/null
import os
try:
    import threevictors
except Exception:
    raise SystemExit(1)
print(os.path.dirname(os.path.dirname(threevictors.__file__)))
PY
  )"
  if [ -n "${THREEVICTORS_SITE_PACKAGES:-}" ] && [ -d "$THREEVICTORS_SITE_PACKAGES" ]; then
    export PYTHONPATH="$THREEVICTORS_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    echo "Using threevictors from system site-packages: $THREEVICTORS_SITE_PACKAGES"
  fi
fi

HOST="${CHATKIT_HOST:-127.0.0.1}"
PORT="${CHATKIT_PORT:-8000}"
echo "Starting ChatKit backend on http://${HOST}:${PORT} ..."
exec "$VENV_BIN/uvicorn" app.main:app --host "$HOST" --port "$PORT"
