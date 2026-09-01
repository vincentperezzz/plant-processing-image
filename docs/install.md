# Run and install — Plant Health

PC window = practice booth (same 1024×600 layout as the 7 inch panel).  
Pi = the real kiosk on the robot.

Full product spec: [spec.md](spec.md).

---

## 1. Windows PC (simulation / daily demo)

Need: this repo, `.venv` already set up, a USB webcam, `models/best.pt`.

**One click / one command:**

```text
.\deploy\run-kiosk.ps1
```

If PowerShell blocks scripts:

```text
powershell -ExecutionPolicy Bypass -File .\deploy\run-kiosk.ps1
```

Or double-click `deploy\run-kiosk.bat`.

Same thing by hand:

```text
.\.venv\Scripts\python.exe src\kiosk.py
```

First-time venv (once):

```text
.\deploy\setup-pc.ps1
```

| What you typed | What you get |
| --- | --- |
| `.\deploy\run-kiosk.ps1` | 1024×600; lite if no YOLO-World weights, else YOLO + CLIP |
| `.\deploy\run-kiosk.ps1 -Lite` | Pi diet: green boxes + MobileNet only |
| `.\deploy\run-kiosk.ps1 -Fullscreen` | No window chrome |
| `python src\pi_sim.py` | Same app (old name still works) |

On screen: Live grades as you aim and never saves. The shutter flashes, writes a PNG of the whole view (photo + plant type + health + notes) and stays live. **GALLERY** is the photo album. Pi photos: `/home/admin/Pictures/plant-health` on the microSD. PC photos: `data/scans/`.

Lab extras (not the kiosk):

```text
.\.venv\Scripts\python.exe src\scan_drop.py
.\.venv\Scripts\python.exe src\scan_cli.py path\to\photo.jpg
```

---

## 2. GitHub release pack

Rebuild after code or `best.pt` changes:

```text
.\.venv\Scripts\python.exe deploy\pack-pi.py
```

Upload `dist/release/plant-health-kiosk-*.zip`. Paste `dist/release/GITHUB-RELEASE.md` as the release body.

**On the Pi**

1. Copy the zip (USB stick or `scp`).
2. Unzip into the home folder:

```text
cd ~
unzip plant-health-kiosk-*.zip
cd plant-health-kiosk
```

3. Run the installer (**offline** — torch is in the pack):

```text
bash deploy/install-pi.sh
```

4. Start:

```text
bash deploy/run-kiosk.sh
```

Windows simulation from the same zip: `deploy\setup-pc.ps1` then `deploy\run-kiosk.ps1` (pip once; lite grader).

Inside the zip: `src/`, `models/best.pt`, PC + Pi scripts, `vendor/` ARM wheels. No training photos, no YOLO-World, no CLIP.

---

## 3. Pi hardware (7 inch kiosk)

| Cable | Job |
| --- | --- |
| HDMI | Picture (Pi → Makerlab 7 inch) |
| USB | Finger / touch (panel → Pi) |
| CSI ribbon | Camera V2 NoIR (contacts toward HDMI on a Pi 4) |
| USB webcam | Fine if the ribbon is not ready |

Flash **Raspberry Pi OS Desktop 64-bit**. First boot: user, Wi‑Fi. Confirm the panel is **1024 × 600**.

---

## 4. Install on the Pi (one script)

On the Pi, in the unzipped folder:

```text
cd ~/plant-health-kiosk
bash deploy/install-pi.sh
```

That script:

1. Uses bundled ARM wheels (no Wi‑Fi)
2. Installs Tk from bundled `.deb` files if the OS image lacks it
3. Autostarts the kiosk on desktop login

**Start now** (or reboot):

```text
bash deploy/run-kiosk.sh
```

# Run and install — Plant Health

PC window = practice booth (same 1024×600 layout as the 7 inch panel).  
Pi = the real kiosk on the robot.

Full product spec: [spec.md](spec.md).

---

## 1. Windows PC (simulation / daily demo)

Need: this repo, `.venv` already set up, a USB webcam, `models/best.pt`.

**One click / one command:**

```text
.\deploy\run-kiosk.ps1
```

If PowerShell blocks scripts:

```text
powershell -ExecutionPolicy Bypass -File .\deploy\run-kiosk.ps1
```

Or double-click `deploy\run-kiosk.bat`.

Same thing by hand:

```text
.\.venv\Scripts\python.exe src\kiosk.py
```

First-time venv (once):

```text
.\deploy\setup-pc.ps1
```

| What you typed | What you get |
| --- | --- |
| `.\deploy\run-kiosk.ps1` | 1024×600; lite if no YOLO-World weights, else YOLO + CLIP |
| `.\deploy\run-kiosk.ps1 -Lite` | Pi diet: green boxes + MobileNet only |
| `.\deploy\run-kiosk.ps1 -Fullscreen` | No window chrome |
| `python src\pi_sim.py` | Same app (old name still works) |

On screen: Live grades as you aim and never saves. The shutter flashes, writes a PNG of the whole view (photo + plant type + health + notes) and stays live. **GALLERY** is the photo album. Pi photos: `/home/admin/Pictures/plant-health` on the microSD. PC photos: `data/scans/`.

Lab extras (not the kiosk):

```text
.\.venv\Scripts\python.exe src\scan_drop.py
.\.venv\Scripts\python.exe src\scan_cli.py path\to\photo.jpg
```

---

## 2. GitHub release pack

Rebuild after code or `best.pt` changes:

```text
.\.venv\Scripts\python.exe deploy\pack-pi.py
```

Upload `dist/release/plant-health-kiosk-*.zip`. Paste `dist/release/GITHUB-RELEASE.md` as the release body.

**On the Pi**

1. Copy the zip (USB stick or `scp`).
2. Unzip into the home folder:

```text
cd ~
unzip plant-health-kiosk-*.zip
cd plant-health-kiosk
```

3. Run the installer (**offline** — torch is in the pack):

```text
bash deploy/install-pi.sh
```

4. Start:

```text
bash deploy/run-kiosk.sh
```

Windows simulation from the same zip: `deploy\setup-pc.ps1` then `deploy\run-kiosk.ps1` (pip once; lite grader).

Inside the zip: `src/`, `models/best.pt`, PC + Pi scripts, `vendor/` ARM wheels. No training photos, no YOLO-World, no CLIP.

---

## 3. Pi hardware (7 inch kiosk)

| Cable | Job |
| --- | --- |
| HDMI | Picture (Pi → Makerlab 7 inch) |
| USB | Finger / touch (panel → Pi) |
| CSI ribbon | Camera V2 NoIR (contacts toward HDMI on a Pi 4) |
| USB webcam | Fine if the ribbon is not ready |

Flash **Raspberry Pi OS Desktop 64-bit**. First boot: user, Wi‑Fi. Confirm the panel is **1024 × 600**.

---

## 4. Install on the Pi (one script)

On the Pi, in the unzipped folder:

```text
cd ~/plant-health-kiosk
bash deploy/install-pi.sh
```

That script:

1. Uses bundled ARM wheels (no Wi‑Fi)
2. Installs Tk from bundled `.deb` files if the OS image lacks it
3. Autostarts the kiosk on desktop login

**Start now** (or reboot):

```text
bash deploy/run-kiosk.sh
```

Same as:

```text
.venv/bin/python src/kiosk.py --fullscreen --lite --camera auto
```

| Flag | Meaning |
| --- | --- |
| `--fullscreen` | Fill the 7 inch panel (or connected 1080p display) |
| `--windowed` | 1024×600 window (PC simulator / debug mode) |
| `--lite` | ExG + MobileNet (Recommended on Pi) |
| `--world` | YOLO-World + CLIP — **PC only, will thrash the Pi** |
| `--camera auto` | CSI first, falls back to USB. Or `csi`, `usb`, `0`, `1` |
| `--stream-fps 25` | Target streaming framerate (supports 15–30 FPS with threaded capture) |
| `--stream-width 1024`| Stream resolution width (defaults to 1024×576 HD) |
| `--stream-quality 85`| Stream JPEG compression quality (defaults to 85) |

After install, scan time does **not** need the internet.

---

## 5. If something is wrong

| You see | Cause | Fix |
| --- | --- | --- |
| `No trained model` / missing `best.pt` | Model checkpoint missing | Copy `models/best.pt` from the PC or train via `training/` |
| `No camera` | Hardware disconnected | Reseat CSI ribbon, or plug in a USB webcam and run `--camera usb` |
| Electric Cyan/Blue or Purple cast | Uncalibrated camera profile | Tap **Reset** in Settings to engage dynamic hardware AWB, or tune Red/Green/Blue sliders |
| Low framerate on USB Webcam | Blocking software scaling | Ensure `src/camera.py` uses `_ThreadedUsbCam` with MJPG hardware mode (benchmarked at 15–30 FPS) |
| Autostart missing | Service registration failed | Re-run `bash deploy/install-pi.sh` as the **desktop user**, not root |
| Touch does nothing | Touch cable unplugged | USB from the panel must be in the Pi; HDMI alone is picture only |

Snaps live in `~/Pictures/plant-health` on the Pi (microSD root partition) or `data/scans/` on a PC.

---

## 6. Capture indicator LED (optional)

A small LED on a GPIO pin lights for 3 seconds every time a photo is captured — touchscreen or remote. If nothing is wired up, the kiosk simply runs without it.

| Wire | Where |
| --- | --- |
| BCM 17 (physical pin 11) | 220–330 Ω resistor → LED **anode** |
| GND (physical pin 9) | LED **cathode** |

Indicator LED only — a Pi pin sources about 16 mA. For anything brighter use a relay or transistor.

Test the wiring before the kiosk is involved:

```text
cd ~/plant-health-kiosk
.venv/bin/python -m src.signal_light --pin 17 --hold 1 --count 3
```

It prints the backend it resolved (or why there is none) and exits non-zero if GPIO is unavailable. The `admin` user is already in the `gpio` group, so **no sudo**.

| Flag | Meaning |
| --- | --- |
| `--gpio-pin 17` | BCM pin (env `PLANT_GPIO_PIN` overrides) |
| `--gpio-active-low` | For relay boards that switch on a LOW pin |
| `--no-gpio` | Never touch GPIO |

---

## 7. Phone & PC Web Remote

Watch the live view, tap the shutter, tune color profiles, and browse/download gallery scans from any phone or computer on the **same network**:

```text
.venv/bin/python src/kiosk.py --fullscreen --lite --camera auto --serve --open
```

The kiosk prints the address at startup, and displays it on the **Settings** page:

```text
remote: http://192.168.1.226:8000/  (open mode, no token)
```

Open `http://<pi-ip>:8000/` on any browser.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--serve` | Turn the remote web server on | Off |
| `--open` | Run in open mode (no PIN token required on LAN) | Off |
| `--port 8000` | HTTP listening port | `8000` |
| `--token` | Custom PIN token if not in open mode | Random |
| `--stream-width 1024` | Native MJPEG stream width | `1024` (1024×576) |
| `--stream-quality 85` | JPEG quality | `85` |
| `--stream-fps 20` | Ceiling on encoded frames per second | `12.0` (or `25.0`) |

### Web Remote Features:
- **Symmetrical 2-Column Interface:** 16:9 HD video feed aligned symmetrically with diagnosis status cards and color tuning controls.
- **Gallery Download Button:** Programmatically downloads high-resolution scans as `.png` files with informative filenames (e.g. `plant_scan_1_tomato.png`) and real-time toast feedback.
- **Color Calibration Remote:** Adjust Red, Green, Blue, Saturation, Brightness, and Contrast with live video feedback, saving presets directly to Pi storage.
- **Multi-Client Isolation:** Multiple users can view and control the kiosk without degrading physical screen performance.
