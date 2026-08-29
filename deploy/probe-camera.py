#!/usr/bin/env python3
"""Say out loud why the camera did or did not open.

The kiosk collapses every failure into "No camera". This prints the actual
libcamera / OpenCV error, one line per stage, so a bad ribbon, a busy sensor,
and a rejected format stop looking alike.

    .venv/bin/python deploy/probe-camera.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def probe_stack() -> None:
    section("stack")
    for mod in ("picamera2", "libcamera", "cv2", "numpy"):
        try:
            m = __import__(mod)
            print(f"  {mod}: {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            print(f"  {mod}: MISSING ({exc})")


def probe_devices() -> None:
    section("devices")
    vids = sorted(Path("/dev").glob("video*")) + sorted(Path("/dev").glob("media*"))
    print(f"  {', '.join(p.name for p in vids) if vids else 'no /dev/video* or /dev/media*'}")
    try:
        from picamera2 import Picamera2

        info = Picamera2.global_camera_info()
        if not info:
            print("  global_camera_info(): [] -- libcamera sees no sensor")
        for cam in info:
            print(f"  {cam}")
    except Exception as exc:
        print(f"  global_camera_info() failed: {exc}")


def probe_holders() -> None:
    """Anything already holding the sensor makes a fresh open fail."""
    section("who holds the camera")
    found = False
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            continue
        if not cmdline.strip():
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except Exception:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except Exception:
                continue
            if "/dev/video" in target or "/dev/media" in target or "dma_heap" in target:
                print(f"  pid {proc.name}: {cmdline.strip()[:90]}  -> {target}")
                found = True
                break
    if not found:
        print("  nothing -- sensor is free")


def probe_open() -> None:
    section("open_capture")
    from src.camera import open_capture

    pref = sys.argv[1] if len(sys.argv) > 1 else "auto"
    t0 = time.monotonic()
    cap = open_capture(pref)
    print(f"  open_capture({pref!r}) took {time.monotonic() - t0:.1f}s -> {cap!r}")
    if cap is None:
        print("  RESULT: no camera")
        return
    frames = 0
    t0 = time.monotonic()
    shape = None
    while time.monotonic() - t0 < 3.0:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames += 1
            shape = frame.shape
    dt = time.monotonic() - t0
    print(f"  RESULT: {frames} frames in {dt:.1f}s = {frames / dt:.1f} fps, shape={shape}")
    cap.release()


def main() -> None:
    probe_stack()
    probe_devices()
    if sys.platform != "win32":
        probe_holders()
    probe_open()


if __name__ == "__main__":
    main()
