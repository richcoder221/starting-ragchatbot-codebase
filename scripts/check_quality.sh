#!/usr/bin/env bash
set -euo pipefail

# Run code quality checks for the project.
# Usage:
#   ./scripts/check_quality.sh          # check only (exit 1 if formatting needed)
#   ./scripts/check_quality.sh --fix    # auto-format in place

FIX=false
for arg in "$@"; do
    [[ "$arg" == "--fix" ]] && FIX=true
done

echo "=== Black ==="
if $FIX; then
    uv run black .
else
    uv run black --check --diff .
fi

echo ""
echo "Quality checks passed."
