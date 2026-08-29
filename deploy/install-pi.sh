#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
PY="$ROOT/.venv/bin/python"
EXEC="$PY $ROOT/src/kiosk.py --fullscreen --lite --camera auto"
WHEELS="$ROOT/vendor/wheels"
DEBS="$ROOT/vendor/debs"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this as the desktop user, not root. Use: bash deploy/install-pi.sh"
  exit 1
fi

if [[ ! -f "$ROOT/models/best.pt" ]]; then
  echo "Missing models/best.pt. Copy the whole plant-health-kiosk folder onto this Pi."
  exit 1
fi

shopt -s nullglob
wheel_files=("$WHEELS"/*.whl)
if [[ ${#wheel_files[@]} -eq 0 ]]; then
  echo "Missing vendor/wheels. This pack is not the offline release."
  exit 1
fi

have_tk() { python3 -c "import tkinter" >/dev/null 2>&1; }
have_venv() { python3 -m venv -h >/dev/null 2>&1; }

install_local_debs() {
  local files=("$DEBS"/*.deb)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "Need python3-tk and python3-venv on this OS image (no .deb bundle found)."
    exit 1
  fi
  echo "Installing local .deb packages (no network)..."
  sudo dpkg -i "${files[@]}" || true
  sudo dpkg --configure -a || true
}

if ! have_tk || ! have_venv; then
  install_local_debs
fi
if ! have_tk; then
  echo "tkinter still missing. Flash Raspberry Pi OS Desktop 64-bit, then retry."
  exit 1
fi
if ! have_venv; then
  echo "python3 venv still missing. Flash Raspberry Pi OS Desktop 64-bit, then retry."
  exit 1
fi

python3 -m venv --system-site-packages "$ROOT/.venv"
"$PY" -m pip install --upgrade --no-index --find-links "$WHEELS" pip || true
"$PY" -m pip install --no-index --find-links "$WHEELS" --no-warn-script-location \
  torch torchvision pillow pyyaml "opencv-python>=4.8,<5"

FONTS="$ROOT/vendor/fonts"
if [[ -d "$FONTS" ]]; then
  echo "Installing kiosk fonts (Caprasimo, Figtree)..."
  mkdir -p "$HOME_DIR/.local/share/fonts/plant-health"
  cp -f "$FONTS"/*.ttf "$HOME_DIR/.local/share/fonts/plant-health/"
  fc-cache -f "$HOME_DIR/.local/share/fonts/plant-health" || true
fi

mkdir -p "$HOME_DIR/.config/autostart"
chmod +x "$ROOT/deploy/run-kiosk.sh" "$ROOT/deploy/install-pi.sh" || true
DESK="$HOME_DIR/.config/autostart/plant-health.desktop"
sed -e "s|__EXEC__|$EXEC|g" -e "s|__ROOT__|$ROOT|g" \
  "$ROOT/deploy/plant-health.desktop.in" > "$DESK"

echo
echo "Offline install done."
echo "  $DESK"
echo "Start now:"
echo "  bash $ROOT/deploy/run-kiosk.sh"
echo "Camera: CSI ribbon first (needs Pi OS camera stack already on the image), else USB."
