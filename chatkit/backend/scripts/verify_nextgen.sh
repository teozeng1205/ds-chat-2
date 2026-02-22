#!/usr/bin/env bash

# Unified verification runner for DS Chat Next-Gen.
# Runs unit tests + threevictors connectivity smoke + end-to-end agent smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE="3VDEV"
MODEL="gpt-4.1-mini"
MAX_TURNS=40
SCENARIOS=""
SKIP_UNIT=0
SKIP_CONNECTIVITY=0
SKIP_E2E=0

usage() {
  cat <<'EOF'
Usage: backend/scripts/verify_nextgen.sh [options]

Options:
  --profile <name>       AWS profile for granted credential-process (default: 3VDEV)
  --model <name>         Model for smoke_nextgen_pipeline.py (default: gpt-4.1-mini)
  --max-turns <n>        Max turns per scenario for E2E smoke (default: 40)
  --scenarios <csv>      Optional scenario filter for E2E smoke
  --skip-unit            Skip pytest unit/integration tests
  --skip-connectivity    Skip smoke_threevictors.py
  --skip-e2e             Skip smoke_nextgen_pipeline.py
  -h, --help             Show help

Examples:
  backend/scripts/verify_nextgen.sh
  backend/scripts/verify_nextgen.sh --skip-unit
  backend/scripts/verify_nextgen.sh --scenarios top_site_issues,market_anomalies_distribution
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="${2:-}"
      shift 2
      ;;
    --scenarios)
      SCENARIOS="${2:-}"
      shift 2
      ;;
    --skip-unit)
      SKIP_UNIT=1
      shift
      ;;
    --skip-connectivity)
      SKIP_CONNECTIVITY=1
      shift
      ;;
    --skip-e2e)
      SKIP_E2E=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

cd "$BACKEND_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual env in $BACKEND_ROOT/.venv ..."
  python -m venv .venv
fi

source .venv/bin/activate

echo "Installing backend deps (editable) ..."
pip install -e . >/dev/null

# Bridge system site-packages for threevictors when not installed in venv.
if ! python -c "import threevictors" >/dev/null 2>&1; then
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
  if [[ -n "${THREEVICTORS_SITE_PACKAGES:-}" && -d "$THREEVICTORS_SITE_PACKAGES" ]]; then
    export PYTHONPATH="$THREEVICTORS_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    echo "Using threevictors from system site-packages: $THREEVICTORS_SITE_PACKAGES"
  fi
fi

echo ""
echo "Verification Plan:"
echo "  profile=${PROFILE}"
echo "  model=${MODEL}"
echo "  max_turns=${MAX_TURNS}"
echo "  scenarios=${SCENARIOS:-<all>}"
echo "  run_unit=$((1-SKIP_UNIT))"
echo "  run_connectivity=$((1-SKIP_CONNECTIVITY))"
echo "  run_e2e=$((1-SKIP_E2E))"
echo ""

if [[ "$SKIP_UNIT" -eq 0 ]]; then
  echo "==> Running pytest"
  python -m pytest -q
fi

if [[ "$SKIP_CONNECTIVITY" -eq 0 ]]; then
  echo "==> Running threevictors connectivity smoke"
  python scripts/smoke_threevictors.py --profile "$PROFILE"
fi

if [[ "$SKIP_E2E" -eq 0 ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for E2E smoke."
    exit 1
  fi

  echo "==> Running nextgen E2E smoke"
  E2E_CMD=(
    python scripts/smoke_nextgen_pipeline.py
    --profile "$PROFILE"
    --model "$MODEL"
    --max-turns "$MAX_TURNS"
  )
  if [[ -n "$SCENARIOS" ]]; then
    E2E_CMD+=(--scenarios "$SCENARIOS")
  fi
  "${E2E_CMD[@]}"
fi

echo ""
echo "Verification complete."
