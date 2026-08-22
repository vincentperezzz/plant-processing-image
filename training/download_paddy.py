import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / ".cache" / "hf"))

from PIL import Image
from tqdm import tqdm

from src.labels import IMAGE_EXTS
from src.paths import RAW, SOURCES

MAX_SIDE = 640
DEST = RAW / "paddydoc"
HF_ID = "Project-AgML/paddy_disease_classification"


def _count() -> int:
    if not DEST.exists():
        return 0
    return sum(1 for p in DEST.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _save(im: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    rgb = im.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb.save(dest, quality=85)


def _label_name(ds, row) -> str:
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label
    feat = ds.features.get("label")
    names = getattr(feat, "names", None)
    if names is not None and not isinstance(label, str):
        return str(names[int(label)])
    for key in ("class_label", "disease", "class"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return "unknown"


def fetch_paddy() -> None:
    if _count() >= 9000:
        print("paddydoc already saved", _count())
        return
    from datasets import concatenate_datasets, load_dataset

    print("== paddydoc field palay ==")
    raw = load_dataset(HF_ID)
    if hasattr(raw, "keys"):
        parts = [raw[k] for k in raw.keys()]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    else:
        ds = raw
    saved = 0
    DEST.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(ds, desc="paddydoc")):
        label = _label_name(ds, row)
        folder = "".join(c if c.isalnum() or c in " -_" else "_" for c in label).strip() or "class"
        out = DEST / folder / f"{i:05d}.jpg"
        img = row.get("image")
        if img is None:
            continue
        _save(img, out)
        saved += 1
    print(f"saved {saved} paddydoc ({_count()} on disk)")


def write_sources() -> None:
    line = (
        f"Paddy Doctor field palay {date.today().isoformat()} "
        "https://huggingface.co/datasets/Project-AgML/paddy_disease_classification "
        "(Petchiammal et al. 2022; dead_heart mapped to dead)"
    )
    old = SOURCES.read_text(encoding="utf-8") if SOURCES.exists() else "# Dataset sources\n\n"
    if "Paddy Doctor" in old:
        return
    SOURCES.write_text(old.rstrip() + "\n- " + line + "\n", encoding="utf-8")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_paddy()
    write_sources()
    print("paddy download done")


if __name__ == "__main__":
    main()
