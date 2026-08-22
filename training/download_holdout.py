import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.labels import IMAGE_EXTS, match_label
from src.paths import DATA, EXTERNAL, HOLDOUT_MANIFEST

sys.path.insert(0, str(ROOT / "training"))
from download import curl_get, extract_zip

PER_CLASS = 20
UNKNOWN_PER_HOST = 8
PLANTDOC_KEEP_HOSTS = {"tomato", "pepper", "bell pepper", "bell_pepper"}
PLANTDOC_UNKNOWN_HOSTS = {"apple", "corn", "grape"}
BDVEG_KEEP = ("eggplant", "tomato")


def _safe_folder(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return cleaned[:80] or "class"


def _save_pil(img, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(dest, quality=90)


def _label_from_row(row, names) -> str:
    for key in ("class_label", "label"):
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if isinstance(val, str):
            return val
        if names is not None:
            try:
                return names[int(val)]
            except (ValueError, TypeError, IndexError):
                pass
        return str(val)
    return "unknown"


def _host_from_row(row, label: str) -> str:
    host = str(row.get("host") or "").lower().replace("_", " ").strip()
    if host:
        return host
    low = label.lower()
    for name in ("tomato", "bell_pepper", "bell pepper", "pepper", "apple", "corn", "grape"):
        if low.startswith(name.replace("_", " ")) or low.startswith(name):
            return name.replace("_", " ")
    return ""


def save_hf_samples(repo: str, dest: Path, keep_fn, per_class: int) -> int:
    from datasets import load_dataset

    print(f"== {repo} -> {dest.name} ==")
    ds = load_dataset(repo, split="train")
    feat = ds.features.get("label")
    names = getattr(feat, "names", None)
    counts = defaultdict(int)
    saved = 0
    for i, row in enumerate(tqdm(ds, desc=dest.name)):
        label = _label_from_row(row, names)
        if not keep_fn(row, label, counts):
            continue
        folder = dest / _safe_folder(label)
        out = folder / f"{i:05d}.jpg"
        _save_pil(row["image"], out)
        counts[label] += 1
        saved += 1
        if saved >= 400:
            break
    print(f"saved {saved} into {dest}")
    return saved


def keep_plantdoc(row, label: str, counts: dict) -> bool:
    host = _host_from_row(row, label)
    if host in PLANTDOC_KEEP_HOSTS or any(h in host for h in ("tomato", "pepper")):
        return counts[label] < PER_CLASS
    if any(h in host for h in PLANTDOC_UNKNOWN_HOSTS) or any(
        label.lower().startswith(h) for h in PLANTDOC_UNKNOWN_HOSTS
    ):
        return counts[label] < UNKNOWN_PER_HOST
    return False


def keep_bdveg(row, label: str, counts: dict) -> bool:
    low = label.lower()
    if not any(low.startswith(k) for k in BDVEG_KEEP):
        return False
    if low in {"fresh leaf", "anthracnose", "mosaic virus"}:
        return False
    return counts[label] < PER_CLASS


def fetch_uci_rice() -> None:
    dest = EXTERNAL / "rice"
    marker = dest / ".extracted"
    if marker.exists() and any(dest.rglob("*.jpg")):
        print("uci rice already saved")
        return
    print("== uci rice ==")
    zdir = EXTERNAL / ".zips"
    zdir.mkdir(parents=True, exist_ok=True)
    zip_path = zdir / "rice_leaf_diseases.zip"
    urls = [
        "https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00475/rice_leaf_diseases.zip",
    ]
    last_err = None
    for url in urls:
        try:
            print("get", url)
            curl_get(url, zip_path)
            if zip_path.stat().st_size > 10000:
                last_err = None
                break
        except Exception as exc:
            last_err = exc
    if last_err and (not zip_path.exists() or zip_path.stat().st_size < 10000):
        raise RuntimeError(f"UCI rice download failed: {last_err}")
    extract_zip(zip_path, dest)
    marker.write_text("ok", encoding="utf-8")


def build_manifest() -> pd.DataFrame:
    rows = []
    images = [p for p in EXTERNAL.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    for path in images:
        if ".zips" in path.parts:
            continue
        hit = match_label(path)
        unknown = hit is None
        rows.append(
            {
                "path": str(path.resolve()),
                "crop": "unknown" if unknown else hit["crop"],
                "health": "unknown" if unknown else hit["health"],
                "source": path.parts[path.parts.index("external") + 1] if "external" in path.parts else "holdout",
                "class_folder": path.parent.name,
                "expect_unknown": unknown,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(HOLDOUT_MANIFEST, index=False)
    return df


def write_notes() -> None:
    note = DATA / "holdout-SOURCES.md"
    note.write_text(
        "# Holdout sources (not used in training)\n\n"
        f"Downloaded: {date.today().isoformat()}\n\n"
        "- PlantDoc field photos https://huggingface.co/datasets/geraldmc/plantdoc-full (Singh et al. 2020). Tomato + bell pepper + apple/corn/grape unknown probes.\n"
        "- UCI Rice Leaf Diseases (120 images) https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases — not RiceLeafBD.\n"
        "- Bangladesh vegetable field leaves https://huggingface.co/datasets/Project-AgML/plant_leaf_disease_classification (Hasan et al. 2024). Eggplant + tomato only.\n"
        "- Lettuce skipped (still no kaggle.json).\n",
        encoding="utf-8",
    )


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    save_hf_samples("geraldmc/plantdoc-full", EXTERNAL / "plantdoc", keep_plantdoc, PER_CLASS)
    fetch_uci_rice()
    save_hf_samples(
        "Project-AgML/plant_leaf_disease_classification",
        EXTERNAL / "bdveg",
        keep_bdveg,
        PER_CLASS,
    )
    df = build_manifest()
    write_notes()
    if df.empty:
        raise SystemExit("No holdout images saved.")
    print(df.groupby(["crop", "health"]).size().unstack(fill_value=0))
    print("unknown probes", int(df["expect_unknown"].sum()))
    print("holdout rows", len(df))
    print("manifest", HOLDOUT_MANIFEST)


if __name__ == "__main__":
    main()
