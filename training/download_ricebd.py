import io
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / ".cache" / "hf"))

from PIL import Image
from tqdm import tqdm

from src.labels import IMAGE_EXTS, normalize_key
from src.paths import RAW, SOURCES

sys.path.insert(0, str(ROOT / "training"))
from download import download_file, mendeley_files

DEST = RAW / "ricebd"
ZIP_NAME = "RiceLeafDiseaseBD.zip"
MENDELEY_ID = "86s4jzj2m4"
MAX_SIDE = 640
CLASS_KEYS = {
    "healthy": "Healthy",
    "blast": "Blast",
    "brown spot": "Brown Spot",
    "leaf smut": "Leaf Smut",
    "rice tungro": "Rice Tungro",
    "tungro": "Rice Tungro",
    "sheath blight": "Sheath Blight",
}


def _count() -> int:
    if not DEST.exists():
        return 0
    return sum(1 for p in DEST.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _class_from_name(name: str) -> str | None:
    parts = Path(name.replace("\\", "/")).parts
    for part in parts:
        key = normalize_key(part)
        if key in CLASS_KEYS:
            return CLASS_KEYS[key]
    blob = normalize_key(name)
    for key, folder in sorted(CLASS_KEYS.items(), key=lambda kv: -len(kv[0])):
        if key in blob:
            return folder
    return None


def fetch_zip() -> Path:
    files = mendeley_files(MENDELEY_ID)
    hit = next((f for f in files if f["filename"] == ZIP_NAME), None)
    if hit is None:
        raise SystemExit(f"No {ZIP_NAME} on Mendeley {MENDELEY_ID}")
    zdir = RAW / ".zips"
    zdir.mkdir(parents=True, exist_ok=True)
    dest = zdir / ZIP_NAME
    url = hit["content_details"]["download_url"]
    expected = int(hit["size"])
    if dest.exists() and dest.stat().st_size != expected:
        print(f"drop bad zip size={dest.stat().st_size} expected={expected}")
        dest.unlink()
    print(f"get {ZIP_NAME} ({expected} bytes)")
    download_file(url, dest)
    if dest.stat().st_size != expected:
        raise SystemExit(f"zip size {dest.stat().st_size} != {expected}")
    return dest


def extract_images(zip_path: Path) -> None:
    marker = DEST / ".extracted"
    if marker.exists() and _count() >= 4000:
        print("ricebd already extracted", _count())
        return
    DEST.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for info in tqdm(members, desc="extract ricebd"):
            name = info.filename.replace("\\", "/")
            if name.startswith("__MACOSX") or "/__MACOSX/" in name:
                continue
            low = name.lower()
            if any(tok in low for tok in ("/labels/", "/label/", ".txt", ".xml", ".yaml", ".json", ".pdf")):
                skipped += 1
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                skipped += 1
                continue
            klass = _class_from_name(name)
            if klass is None:
                skipped += 1
                continue
            out = DEST / klass / f"{saved:05d}.jpg"
            if out.exists():
                saved += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src:
                im = Image.open(io.BytesIO(src.read())).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(out, quality=85)
            saved += 1
    marker.write_text(f"{saved}\n", encoding="utf-8")
    print(f"ricebd saved={saved} skipped={skipped}")


def write_sources() -> None:
    line = (
        f"RiceLeafDiseaseBD field palay {date.today().isoformat()} "
        "https://data.mendeley.com/datasets/86s4jzj2m4/2 CC BY 4.0 "
        "(Leaf Smut + blast/tungro/sheath blight; not UCI holdout)"
    )
    old = SOURCES.read_text(encoding="utf-8") if SOURCES.exists() else "# Dataset sources\n\n"
    if "RiceLeafDiseaseBD" in old:
        return
    SOURCES.write_text(old.rstrip() + "\n- " + line + "\n", encoding="utf-8")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if _count() >= 4000:
        print("ricebd already saved", _count())
        write_sources()
        return
    zip_path = fetch_zip()
    extract_images(zip_path)
    write_sources()
    print("ricebd download done", _count())


if __name__ == "__main__":
    main()
