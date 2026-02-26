#!/usr/bin/env bash

# Simple helper to start the ChatKit backend (similar to cat-lounge UX).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Creating virtual env in $PROJECT_ROOT/.venv ..."
  python -m venv .venv
fi

source .venv/bin/activate

echo "Installing backend deps (editable) ..."
pip install -e . >/dev/null

# If threevictors is installed in system python but not in this venv,
# bridge system site-packages into PYTHONPATH for runtime tool access.
if ! python -c "import threevictors" >/dev/null 2>&1; then
  THREEVICTORS_SITE_PACKAGES="$(
    python3 - <<'PY' 2>/dev/null || true
import importlib.util
import os
spec = importlib.util.find_spec("threevictors")
if spec is not None and spec.origin:
    print(os.path.dirname(os.path.dirname(spec.origin)))
PY
  )"
  if [ -n "${THREEVICTORS_SITE_PACKAGES:-}" ] && [ -d "$THREEVICTORS_SITE_PACKAGES" ]; then
    export PYTHONPATH="$THREEVICTORS_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    echo "Using threevictors from system site-packages: $THREEVICTORS_SITE_PACKAGES"
  fi
fi

# Load env vars from the repo's .env.local (if present) so OPENAI_API_KEY
# does not need to be exported manually.
ENV_FILE="$PROJECT_ROOT/../.env.local"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then
  echo "Sourcing OPENAI_API_KEY from $ENV_FILE"
  # shellcheck disable=SC1090
  set -a
  . "$ENV_FILE"
  set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY in your environment or in .env.local before running this script."
  exit 1
fi

echo "Starting ChatKit backend on http://127.0.0.1:8000 ..."
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
