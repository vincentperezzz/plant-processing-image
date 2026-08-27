# Install — Plant Health kiosk

Unzip this release first. You should see `src/`, `models/best.pt`, and `deploy/`.

---

## A. Windows PC (simulation)

Need: Windows 10/11, Python 3.11+, a USB webcam, internet **once** for pip.

1. Open PowerShell in the unzipped folder.

2. First-time setup:

```text
powershell -ExecutionPolicy Bypass -File .\deploy\setup-pc.ps1
```

3. Run the kiosk (1024×600, same layout as the 7 inch panel):

```text
powershell -ExecutionPolicy Bypass -File .\deploy\run-kiosk.ps1
```

Or double-click `deploy\run-kiosk.bat`.

This release runs **lite** (green boxes + MobileNet). That matches the Pi. Shutter saves a PNG of the live view to `data\scans\`. **GALLERY** is the photo album.

| Flag | Meaning |
| --- | --- |
| `-Fullscreen` | No window chrome |
| `-Camera 0` | Force webcam index |

If PowerShell blocks scripts, the `-ExecutionPolicy Bypass` lines above are enough.

---

## B. Raspberry Pi (real kiosk)

Need: Pi 4B 8 GB, 7 inch HDMI 1024×600, Camera V2 NoIR **or** USB webcam. Flash **Raspberry Pi OS Desktop 64-bit** on a PC with Imager. The Pi does **not** need Wi‑Fi.

### Cables

| Cable | Plug |
| --- | --- |
| HDMI | Pi → 7 inch screen (picture) |
| USB | 7 inch screen → Pi (touch). HDMI alone is picture only. |
| CSI ribbon | Camera V2 NoIR. On a Pi 4, contacts toward the HDMI ports. |
| USB webcam | Fine if the ribbon is not ready |

### Copy onto the Pi

USB or `scp` the zip, then:

```text
cd ~
unzip plant-health-kiosk-*.zip
cd plant-health-kiosk
```

Or copy the unzipped `plant-health-kiosk` folder into the home directory.

### Install (offline)

```text
cd ~/plant-health-kiosk
bash deploy/install-pi.sh
```

Run as the **desktop user**, not `root` / not `sudo bash`. It may ask for a password to install local Tk `.deb` files. Torch and OpenCV come from `vendor/wheels` on the USB, not the web.

### Start

```text
bash deploy/run-kiosk.sh
```

Or reboot — the kiosk autostarts on desktop login.

| If you need | Type |
| --- | --- |
| USB camera | `bash deploy/run-kiosk.sh --camera usb` |
| CSI ribbon | `bash deploy/run-kiosk.sh --camera csi` |
| Window not fullscreen | `bash deploy/run-kiosk.sh --windowed` |

---

## If something is wrong

| You see | Fix |
| --- | --- |
| Missing `best.pt` | Unzip / copy the whole folder again |
| No `.venv` on Windows | Run `deploy\setup-pc.ps1` |
| No camera | PC: try `-Camera 0` or `1`. Pi: reseat CSI, or `--camera usb` |
| Black NoIR picture | Needs visible light |
| Autostart missing on Pi | Re-run `bash deploy/install-pi.sh` as the desktop user |
| Touch does nothing | USB from the panel must be in the Pi |

Snaps save in `data/scans/` on whichever machine took them.
