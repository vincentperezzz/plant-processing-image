import io
import random
import subprocess
import sys
import tarfile
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.labels import IMAGE_EXTS, match_label
from src.paths import DATA, EXTERNAL

PER_CLASS = 12
ROBOT_N = 80
WILD_HOSTS = ("tomato", "rice", "palay", "pepper", "chili", "chilli", "sili", "eggplant", "brinjal", "lettuce")
RGB_URL = "https://ndownloader.figshare.com/files/25337984"
MANIFEST = DATA / "realistic_manifest.csv"


def _safe_folder(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return cleaned[:80] or "class"


def fetch_rob2pheno() -> Path:
    dest = EXTERNAL / "rob2pheno"
    dest.mkdir(parents=True, exist_ok=True)
    if len(list(dest.rglob("*.png"))) + len(list(dest.rglob("*.jpg"))) >= ROBOT_N:
        print("rob2pheno already sampled")
        return dest
    zdir = EXTERNAL / ".zips"
    zdir.mkdir(parents=True, exist_ok=True)
    tar_path = zdir / "rob2pheno_rgb.tar.gz"
    if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
        print("== rob2pheno RGB (greenhouse robot, RealSense) ==")
        if tar_path.exists():
            tar_path.unlink()
        subprocess.check_call(
            [
                "curl.exe",
                "-L",
                "-A",
                "Mozilla/5.0",
                "--retry",
                "5",
                "--progress-bar",
                "--output",
                str(tar_path),
                RGB_URL,
            ]
        )
    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.rglob("*")):
        print("extract rob2pheno")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(raw, filter="data")
    files = [p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    random.Random(42).shuffle(files)
    out_dir = dest / "tomato_robot"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src in files:
        if saved >= ROBOT_N:
            break
        im = Image.open(src).convert("RGB")
        im.thumbnail((640, 640))
        im.save(out_dir / f"{saved:03d}.jpg", quality=82)
        saved += 1
    print(f"sampled {saved} robot frames into {out_dir}")
    return dest


def _label_from_row(row, names) -> str:
    for key in ("class_label", "label", "ground_truth"):
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if hasattr(val, "label"):
            val = val.label
        if isinstance(val, dict) and "label" in val:
            val = val["label"]
        if isinstance(val, str):
            return val
        if names is not None:
            try:
                return names[int(val)]
            except (ValueError, TypeError, IndexError):
                pass
        return str(val)
    return "unknown"


def keep_wild(row, label: str, counts: dict) -> bool:
    low = label.lower().replace("_", " ")
    if not any(low.startswith(h) or f" {h} " in f" {low} " for h in WILD_HOSTS):
        return False
    return counts[label] < PER_CLASS


def fetch_plantwild() -> Path:
    dest = EXTERNAL / "plantwild"
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*.jpg")):
        print("plantwild already sampled")
        return dest
    zips = list((DATA / ".cache" / "hf").rglob("plantwild.zip")) if (DATA / ".cache" / "hf").exists() else []
    zips += list(Path.home().joinpath(".cache/huggingface").rglob("plantwild.zip"))
    hub = DATA / ".cache" / "hf" / "hub"
    if hub.exists():
        zips += list(hub.rglob("plantwild.zip"))
    zips = [p for p in zips if p.exists()]
    if not zips:
        from datasets import load_dataset

        print("== plantwild pulling zip ==")
        load_dataset("uqtwei2/PlantWild", split="train")
        zips = list((DATA / ".cache" / "hf" / "hub").rglob("plantwild.zip"))
    if not zips:
        print("plantwild zip missing")
        return dest
    zip_path = max(zips, key=lambda p: p.stat().st_size)
    print("== plantwild from", zip_path, "==")
    import zipfile

    counts = defaultdict(int)
    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if "/images/" in n.replace("\\", "/") and not n.endswith("/")]
        for name in tqdm(members, desc="plantwild zip"):
            parts = name.replace("\\", "/").split("/")
            try:
                label = parts[parts.index("images") + 1]
            except (ValueError, IndexError):
                continue
            dummy = {"class_label": label}
            if not keep_wild(dummy, label, counts):
                continue
            try:
                with zf.open(name) as src:
                    img = Image.open(src).convert("RGB")
            except Exception:
                continue
            folder = dest / _safe_folder(label)
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / f"{counts[label]:03d}.jpg"
            img.save(out, quality=85)
            counts[label] += 1
            saved += 1
    print(f"saved {saved} plantwild images")
    return dest


def build_manifest() -> pd.DataFrame:
    rows = []
    for path in EXTERNAL.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        if ".zips" in path.parts or "_raw" in path.parts:
            continue
        source = None
        if "rob2pheno" in path.parts:
            source = "rob2pheno"
        elif "plantwild" in path.parts:
            source = "plantwild"
        elif "plantdoc" in path.parts:
            source = "plantdoc"
        else:
            continue
        if source == "rob2pheno":
            rows.append(
                {
                    "path": str(path.resolve()),
                    "crop": "tomato",
                    "health": "",
                    "source": source,
                    "class_folder": path.parent.name,
                    "expect_unknown": False,
                    "score_health": False,
                }
            )
            continue
        hit = match_label(path)
        unknown = hit is None
        rows.append(
            {
                "path": str(path.resolve()),
                "crop": "unknown" if unknown else hit["crop"],
                "health": "" if unknown else hit["health"],
                "source": source,
                "class_folder": path.parent.name,
                "expect_unknown": unknown,
                "score_health": not unknown,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    return df


def write_notes() -> None:
    (DATA / "realistic-SOURCES.md").write_text(
        "# Realistic / robot-like holdout\n\n"
        f"Downloaded: {date.today().isoformat()}\n\n"
        "- Rob2Pheno RGB: greenhouse tomato **robot** (RealSense on a trolley, ~0.5 m side view, whole plant, not a filled-frame leaf). https://doi.org/10.4121/13173422\n"
        "- PlantWild: in-the-wild web photos, mixed cameras, off-angle, messy backgrounds. https://huggingface.co/datasets/uqtwei2/PlantWild\n"
        "- PlantDoc already in data/external/plantdoc (field/web, not robot).\n"
        "- No public Pi Camera NoIR farm-robot close-up set exists for palay / sili / lettuce / eggplant.\n",
        encoding="utf-8",
    )


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    fetch_rob2pheno()
    fetch_plantwild()
    df = build_manifest()
    write_notes()
    if df.empty:
        raise SystemExit("No realistic images saved.")
    print(df.groupby(["source", "crop"]).size())
    print("rows", len(df), "manifest", MANIFEST)


if __name__ == "__main__":
    main()
