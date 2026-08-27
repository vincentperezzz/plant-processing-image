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

`--lite` is the Pi default: ExG boxes + MobileNet. No YOLO-World, no CLIP.

| Flag | Meaning |
| --- | --- |
| `--fullscreen` | Fill the 7 inch panel |
| `--windowed` | 1024×600 window (debug) |
| `--lite` | ExG + MobileNet (Pi) |
| `--world` | YOLO-World + CLIP — **PC only, will thrash the Pi** |
| `--camera auto` | CSI first, else USB. Or `csi`, `usb`, `0`, `1` |

After install, scan time does **not** need the internet.

---

## 5. If something is wrong

| You see | Fix |
| --- | --- |
| `No trained model` / missing `best.pt` | Copy `models/best.pt` from the PC |
| `No camera` | Reseat CSI ribbon, or `bash deploy/run-kiosk.sh --camera usb` |
| Black NoIR picture | Needs visible light (fill LEDs). NoIR is not night vision without IR lamps |
| Autostart missing | Re-run `bash deploy/install-pi.sh` as the **desktop user**, not root |
| Touch does nothing | USB from the panel must be in the Pi; HDMI alone is picture only |

Snaps live in `~/Pictures/plant-health` on the Pi (microSD root partition) or `data/scans/` on a PC.
