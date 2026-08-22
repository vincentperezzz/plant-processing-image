# 7 inch Pi LCD

**Panel:** [Makerlab 7 inch HDMI LCD](https://makerlab.ph/products/7-inch-hdmi-lcd-for-raspberry-pi?_pos=1&_psq=lcd&_psid=3861f3c33&_ss=e&variant=43922119229631)  
**Size:** 7 inch · **1024 × 600** · IPS · capacitive touch (USB) · HDMI video · driver-free on Raspberry Pi OS

HDMI = picture. USB = finger.

Run and install steps live in **[docs/install.md](../install.md)** (PC simulation + Pi script).

On the PC:

```text
.\deploy\run-kiosk.ps1
```

Copy `dist/plant-health-pi.zip` to the Pi, unzip, then:

```text
cd ~/plant-health-pi
bash deploy/install-pi.sh
bash deploy/run-kiosk.sh
```
