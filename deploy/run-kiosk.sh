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

kiosk_pids() {
  for pid in $(ps -eo pid= 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    if grep -aq "src/kiosk.py" "/proc/$pid/cmdline" 2>/dev/null; then
      echo "$pid"
    fi
  done
}

# A kiosk wedged inside libcamera ignores TERM and keeps holding the sensor,
# which is what turns the next launch into "No camera". Escalate to KILL.
for pid in $(kiosk_pids); do kill "$pid" 2>/dev/null || true; done
for _ in 1 2 3 4 5 6; do
  [ -z "$(kiosk_pids)" ] && break
  sleep 0.5
done
for pid in $(kiosk_pids); do kill -9 "$pid" 2>/dev/null || true; done
sleep 1

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec "$PY" -u "$ROOT/src/kiosk.py" --fullscreen --lite --camera usb --stream-fps 25 --serve --open "$@"
