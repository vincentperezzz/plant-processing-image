#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No .venv. Run: bash deploy/install-pi.sh"
  exit 1
fi
if [[ ! -f "$ROOT/models/best.pt" ]]; then
  echo "Missing models/best.pt. Copy it onto this Pi first."
  exit 1
fi
cd "$ROOT"
exec "$PY" "$ROOT/src/kiosk.py" --fullscreen --lite --camera auto "$@"
