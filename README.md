# Plant Health Scanner

Touchscreen kiosk and web remote that identifies farm crops, grades plant health (**healthy / mild / critical / dead**), and provides actionable agricultural tips.

**Target Hardware:** Raspberry Pi 4B (8 GB RAM) + 7" 1024×600 LCD Touchscreen + Raspberry Pi Camera V2 / USB Webcam.

- **Installation & Setup Guide:** [docs/install.md](docs/install.md)  
- **Product & Architecture Spec:** [docs/spec.md](docs/spec.md)
- **UI Design System & Tokens:** [docs/UI-modernization/README.md](docs/UI-modernization/README.md)

---

## Key Features

1. **Dual Camera Architecture:**
   - **CSI Ribbon Camera (Picamera2):** Real-time hardware ISP capture with dynamic Auto White Balance (AWB) running at 15–20 FPS.
   - **Threaded HD USB Webcam (V4L2):** Zero-latency multi-threaded hardware MJPG capture running at **15–30 FPS** in 1280×720 HD with true natural color balance.
2. **Interactive Touchscreen Kiosk:**
   - **Scan Tab:** Real-time multi-plant bounding box tracking, health diagnosis cards, and single-tap shutter capture.
   - **Gallery Tab:** Review historical scans, inspect health records and tips, or delete scans.
   - **Settings Tab:** Live color calibration sliders (Red, Green, Blue, Saturation, Brightness, Contrast) with persistent `Night` and `Morning` profile storage.
3. **Responsive Web Remote:**
   - Symmetrical 2-column live viewfinder and controls reachable from any phone or PC browser on the local Wi-Fi network.
   - In-browser full-resolution scan downloader (`.png` Blob downloads).

---

## Quickstart

### Windows PC (Simulation / Local Testing)

```powershell
.\deploy\setup-pc.ps1
.\deploy\run-kiosk.ps1
```

Or run directly with the virtual environment:

```powershell
.venv\Scripts\python.exe src\kiosk.py --windowed --lite --serve --open
```

### Raspberry Pi (Physical Kiosk)

```bash
cd ~/plant-health-kiosk
bash deploy/install-pi.sh
bash deploy/run-kiosk.sh
```

Or start the kiosk service with specific camera options:

```bash
# CSI Camera (Default)
.venv/bin/python src/kiosk.py --fullscreen --lite --camera auto --serve --open

# High-FPS USB Webcam
.venv/bin/python src/kiosk.py --fullscreen --lite --camera usb --stream-fps 25 --serve --open
```

