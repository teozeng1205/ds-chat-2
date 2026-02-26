#!/usr/bin/env bash
# Compatibility wrapper for legacy nextgen verification script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/verify_investigation.sh" "$@"
