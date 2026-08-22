# Plant Health Scanner

Kiosk that names a farm crop, grades **healthy / mild / critical / dead**, and writes a short tip. Target: Raspberry Pi 4B 8 GB + 7 inch 1024×600 LCD.

**How to run and install:** [docs/install.md](docs/install.md)  
**Product spec:** [docs/spec.md](docs/spec.md)

Training photos are **not** in this repo. GitHub gets the code plus `models/best.pt` (the trained grader). The offline Pi zip lives on [GitHub Releases](../../releases), not in git.

### Windows (simulation)

```text
.\deploy\setup-pc.ps1
.\deploy\run-kiosk.ps1
```

### Raspberry Pi (real kiosk)

Download `plant-health-kiosk-*.zip` from Releases, unzip, then:

```text
cd ~/plant-health-kiosk
bash deploy/install-pi.sh
bash deploy/run-kiosk.sh
```
