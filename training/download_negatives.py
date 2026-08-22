import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.paths import DATA

NEGATIVES = DATA / "negatives"
NEG_MANIFEST = DATA / "negatives_manifest.csv"
OUR_HOSTS = ("tomato", "rice", "pepper", "chili", "chilli", "eggplant", "brinjal", "lettuce")
OBJECT_TARGET = 3000
PLANT_PER_CLASS = 30
INDOOR_TARGET = 1400
INDOOR_HITS = (
    "curtain",
    "drape",
    "blinds",
    "t-shirt",
    "tshirt",
    "sweater",
    "hoodie",
    "polo",
    "living room",
    "bedroom",
    "couch",
    "sofa",
    "wardrobe",
    "pillow",
    "blanket",
    "portrait",
    "a man in",
    "a woman in",
    "a person in",
    "wearing a",
    "window shade",
)


def _safe(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return cleaned[:80] or "class"


def fetch_objects() -> int:
    dest = NEGATIVES / "objects"
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.rglob("*.jpg")))
    if existing >= OBJECT_TARGET // 2:
        print(f"objects already saved ({existing})")
        return existing
    from datasets import load_dataset

    ds = None
    for repo, cfg in (("frgfm/imagenette", "160px"), ("zh-plus/tiny-imagenet", None)):
        try:
            ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
            print("loaded", repo)
            break
        except Exception as exc:
            print(repo, "failed:", exc)
    if ds is None:
        print("no object source available")
        return existing
    names = getattr(ds.features.get("label"), "names", None)
    saved = 0
    step = max(1, len(ds) // OBJECT_TARGET)
    for i in tqdm(range(0, len(ds), step), desc="objects"):
        if saved >= OBJECT_TARGET:
            break
        try:
            row = ds[i]
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            continue
        label = str(row.get("label", "obj"))
        if names is not None:
            try:
                label = names[int(row["label"])]
            except (ValueError, TypeError, IndexError):
                pass
        folder = dest / _safe(label)
        folder.mkdir(parents=True, exist_ok=True)
        img.thumbnail((256, 256))
        img.save(folder / f"{i:06d}.jpg", quality=85)
        saved += 1
    print(f"saved {saved} object negatives")
    return saved


def fetch_other_plants() -> int:
    dest = NEGATIVES / "other_plants"
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.rglob("*.jpg")))
    if existing >= 800:
        print(f"other plants already saved ({existing})")
        return existing
    hub = DATA / ".cache" / "hf" / "hub"
    zips = list(hub.rglob("plantwild.zip")) if hub.exists() else []
    if not zips:
        print("plantwild.zip not cached, skipping other-plant negatives")
        return existing
    zip_path = max(zips, key=lambda p: p.stat().st_size)
    counts = defaultdict(int)
    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if "/images/" in n.replace("\\", "/") and not n.endswith("/")]
        for name in tqdm(members, desc="other plants"):
            parts = name.replace("\\", "/").split("/")
            try:
                label = parts[parts.index("images") + 1]
            except (ValueError, IndexError):
                continue
            low = label.lower()
            if any(h in low for h in OUR_HOSTS):
                continue
            if counts[label] >= PLANT_PER_CLASS:
                continue
            try:
                with zf.open(name) as src:
                    img = Image.open(src).convert("RGB")
            except Exception:
                continue
            folder = dest / _safe(label)
            folder.mkdir(parents=True, exist_ok=True)
            img.thumbnail((256, 256))
            img.save(folder / f"{counts[label]:03d}.jpg", quality=85)
            counts[label] += 1
            saved += 1
    print(f"saved {saved} other-plant negatives across {len(counts)} classes")
    return saved


def _caption_blob(row) -> str:
    caps = row.get("caption") or row.get("captions") or row.get("sentences") or row.get("text") or ""
    if isinstance(caps, list):
        parts = []
        for item in caps:
            if isinstance(item, dict):
                parts.append(str(item.get("raw") or item.get("text") or item))
            else:
                parts.append(str(item))
        return " ".join(parts).lower()
    if isinstance(caps, dict):
        return str(caps.get("raw") or caps.get("text") or caps).lower()
    return str(caps).lower()


def _save_indoor(img, dest: Path, idx: int) -> None:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((256, 256))
    folder = dest / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / f"{idx:06d}.jpg", quality=85)


def fetch_indoor() -> int:
    dest = NEGATIVES / "indoor"
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.rglob("*.jpg")))
    if existing >= INDOOR_TARGET // 2:
        print(f"indoor already saved ({existing})")
        return existing
    from datasets import load_dataset

    saved = existing
    skip_rooms = {"greenhouse", "florist"}
    try:
        ds = load_dataset("keremberke/indoor-scene-classification", name="full", split="train")
        print("loaded keremberke/indoor-scene-classification")
        names = getattr(ds.features.get("label"), "names", None)
        per = {}
        for i, row in enumerate(tqdm(ds, desc="indoor rooms")):
            if saved >= INDOOR_TARGET:
                break
            label = str(row.get("label", "room"))
            if names is not None:
                try:
                    label = names[int(row["label"])]
                except (ValueError, TypeError, IndexError):
                    pass
            low = label.lower().replace(" ", "")
            if low in skip_rooms:
                continue
            if per.get(label, 0) >= 40:
                continue
            img = row.get("image")
            if img is None:
                continue
            try:
                folder = dest / _safe(label)
                folder.mkdir(parents=True, exist_ok=True)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((256, 256))
                img.save(folder / f"{per.get(label, 0):04d}.jpg", quality=85)
            except Exception:
                continue
            per[label] = per.get(label, 0) + 1
            saved += 1
    except Exception as exc:
        print("indoor rooms failed:", exc)
    sources = (
        ("lmms-lab/flickr30k", "test"),
        ("lmms-lab/flickr30k", "train"),
    )
    for repo, split in sources:
        if saved >= INDOOR_TARGET:
            break
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            print("loaded", repo, split)
        except Exception as exc:
            print(repo, split, "failed:", exc)
            continue
        for row in tqdm(ds, desc=f"indoor {repo}"):
            if saved >= INDOOR_TARGET:
                break
            blob = _caption_blob(row)
            if not any(hit in blob for hit in INDOOR_HITS):
                continue
            img = row.get("image")
            if img is None:
                continue
            try:
                _save_indoor(img, dest, saved)
            except Exception:
                continue
            saved += 1
    print(f"saved indoor negatives total {saved}")
    return saved


def build_manifest() -> None:
    rows = []
    for path in NEGATIVES.rglob("*.jpg"):
        if "other_plants" in path.parts:
            kind = "other_plants"
        elif "indoor" in path.parts:
            kind = "indoor"
        else:
            kind = "objects"
        rows.append({"path": str(path.resolve()), "kind": kind, "class_folder": path.parent.name})
    df = pd.DataFrame(rows)
    df.to_csv(NEG_MANIFEST, index=False)
    print(df.groupby("kind").size())
    print("rows", len(df), "->", NEG_MANIFEST)


def main() -> None:
    NEGATIVES.mkdir(parents=True, exist_ok=True)
    fetch_objects()
    fetch_other_plants()
    fetch_indoor()
    build_manifest()


if __name__ == "__main__":
    main()
