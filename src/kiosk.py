import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk

from src.pi_sim import PiSim


def on_pi() -> bool:
    return platform.machine().lower() in ("aarch64", "armv7l", "armv8l")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plant Health kiosk")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--world", action="store_true")
    parser.add_argument("--camera", default="auto", help="auto, csi, usb, or a camera index")
    args = parser.parse_args(argv)
    root = tk.Tk()
    panel = root.winfo_screenwidth() <= 1280 and root.winfo_screenheight() <= 800
    lite = bool(args.lite or (on_pi() and not args.world))
    fullscreen = bool((args.fullscreen or panel) and not args.windowed)
    PiSim(root, fullscreen=fullscreen, lite=lite, camera=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
