import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from src.paths import DATA

NEGATIVES = DATA / "negatives"
NEG_MANIFEST = DATA / "negatives_manifest.csv"
OUR_HOSTS = ("tomato", "rice", "pepper", "chili", "chilli", "eggplant", "brinjal", "lettuce")
TARGET_AGRI = 1200
PER_SPECIES = 45

WEED_TAXA = {
    "nutsedge_cyperus": 2714571,
    "wiregrass_eleusine": 2705917,
    "pigweed_amaranthus": 3082329,
    "purslane_portulaca": 3084714,
    "crabgrass_digitaria": 2705445,
    "sourgrass_paspalum": 2705295,
    "barnyardgrass_echinochloa": 5289746,
}


def _safe(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return cleaned[:80] or "weed"


def extract_from_plantwild() -> int:
    dest = NEGATIVES / "agri_noise"
    dest.mkdir(parents=True, exist_ok=True)
    hub = DATA / ".cache" / "hf" / "hub"
    zips = list(hub.rglob("plantwild.zip")) if hub.exists() else []
    if not zips:
        print("plantwild.zip not found")
        return 0

    zip_path = max(zips, key=lambda p: p.stat().st_size)
    counts = defaultdict(int)
    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if "/images/" in n.replace("\\", "/") and not n.endswith("/")]
        for name in tqdm(members, desc="Extracting outdoor garden plants"):
            if saved >= 850:
                break
            parts = name.replace("\\", "/").split("/")
            try:
                label = parts[parts.index("images") + 1]
            except (ValueError, IndexError):
                continue
            low = label.lower()
            if any(h in low for h in OUR_HOSTS):
                continue
            if counts[label] >= PER_SPECIES:
                continue
            try:
                with zf.open(name) as src:
                    img = Image.open(src).convert("RGB")
            except Exception:
                continue

            folder = dest / _safe(label)
            folder.mkdir(parents=True, exist_ok=True)
            img.thumbnail((256, 256))
            out_p = folder / f"{counts[label]:03d}.jpg"
            img.save(out_p, quality=85)
            counts[label] += 1
            saved += 1

    print(f"Extracted {saved} outdoor plant negatives from PlantWild across {len(counts)} classes")
    return saved


def harvest_gbif_weeds() -> int:
    dest = NEGATIVES / "agri_noise" / "farm_weeds"
    dest.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "PlantHealthScanner/1.0 (local negatives training)"})

    for weed_name, taxon_key in WEED_TAXA.items():
        sub_folder = dest / weed_name
        sub_folder.mkdir(parents=True, exist_ok=True)
        existing = len(list(sub_folder.glob("*.jpg")))
        if existing >= 40:
            total_saved += existing
            continue

        url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={taxon_key}&mediaType=StillImage&limit=60"
        try:
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
        except Exception:
            continue

        saved_this = existing
        for occ in results:
            if saved_this >= 40:
                break
            media = occ.get("media", [])
            if not media:
                continue
            id_str = str(occ.get("key", ""))
            img_url = media[0].get("identifier")
            if not img_url:
                continue

            out_file = sub_folder / f"{id_str}.jpg"
            if out_file.exists():
                saved_this += 1
                continue

            try:
                img_resp = session.get(img_url, timeout=8)
                if img_resp.status_code != 200 or len(img_resp.content) < 5000:
                    continue
                im = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                im.thumbnail((256, 256))
                im.save(out_file, quality=85)
                saved_this += 1
                total_saved += 1
            except Exception:
                continue

        print(f"Harvested {saved_this} images of {weed_name}")

    return total_saved


def rebuild_negatives_manifest():
    rows = []
    for p in NEGATIVES.rglob("*.jpg"):
        rel = p.relative_to(NEGATIVES)
        parts = rel.parts
        kind = parts[0]
        class_folder = parts[1] if len(parts) > 1 else "misc"
        rows.append({
            "path": str(p.resolve()),
            "kind": kind,
            "class_folder": class_folder,
        })
    df = pd.DataFrame(rows)
    df.to_csv(NEG_MANIFEST, index=False)
    print(f"\nUpdated {NEG_MANIFEST} with {len(df)} total negative samples:")
    print(df["kind"].value_counts())


def main():
    print("=== Harvesting Agricultural & Field Negatives ===")
    extract_from_plantwild()
    harvest_gbif_weeds()
    rebuild_negatives_manifest()


if __name__ == "__main__":
    main()
