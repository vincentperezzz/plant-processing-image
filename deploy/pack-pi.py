from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
VERSION = "2026.08.22"
NAME = "plant-health-kiosk"
STAGE = DIST / ".pack-stage"
ZIP_NAME = f"{NAME}-{VERSION}.zip"
RELEASE = DIST / "release"
ZIP_PATH = RELEASE / ZIP_NAME
GUIDE = ROOT / "deploy" / "INSTALL.md"
README = ROOT / "deploy" / "README.md"
GH_NOTES = ROOT / "deploy" / "GITHUB-RELEASE.md"

FILES = [
    "requirements-kiosk.txt",
    "data/label_map.yaml",
    "data/plant_dictionary.yaml",
    "models/best.pt",
    "models/meta.json",
    "deploy/install-pi.sh",
    "deploy/run-kiosk.sh",
    "deploy/setup-pc.ps1",
    "deploy/run-kiosk.ps1",
    "deploy/run-kiosk.bat",
    "deploy/plant-health.desktop.in",
    "deploy/plant-health.service.in",
]


def _wipe(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    ckpt = ROOT / "models" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("Missing models/best.pt")
    if not GUIDE.exists():
        raise SystemExit("Missing deploy/INSTALL.md")
    if not README.exists():
        raise SystemExit("Missing deploy/README.md")
    _wipe(STAGE)
    _wipe(DIST / "plant-health-pi")
    _wipe(DIST / "plant-health-pi.zip")
    _wipe(DIST / f"{NAME}.zip")
    _wipe(DIST / ZIP_NAME)
    DIST.mkdir(parents=True, exist_ok=True)
    _wipe(RELEASE)
    RELEASE.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"Missing {rel}")
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    shutil.copy2(README, STAGE / "README.md")
    shutil.copy2(GUIDE, STAGE / "INSTALL.md")
    (STAGE / "VERSION").write_text(VERSION + "\n", encoding="utf-8")
    vendor = ROOT / "vendor"
    wheels = list((vendor / "wheels").glob("*.whl")) if (vendor / "wheels").is_dir() else []
    if not wheels:
        raise SystemExit("Missing vendor/wheels. Run: python deploy/fetch-pi-offline.py")
    for src in vendor.rglob("*"):
        if not src.is_file():
            continue
        dest = STAGE / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for src in sorted((ROOT / "src").glob("*.py")):
        dest = STAGE / "src" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in STAGE.rglob("*"):
            if path.is_file():
                zf.write(path, Path(NAME) / path.relative_to(STAGE))
    shutil.copy2(README, RELEASE / "README.md")
    shutil.copy2(GUIDE, RELEASE / "INSTALL.md")
    if GH_NOTES.exists():
        shutil.copy2(GH_NOTES, RELEASE / "GITHUB-RELEASE.md")
    _wipe(STAGE)
    print("zip", ZIP_PATH, ZIP_PATH.stat().st_size)
    print("release", RELEASE)


if __name__ == "__main__":
    main()
