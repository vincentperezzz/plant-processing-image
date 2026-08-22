import io
import json
import os
import random
import sys
import tarfile
import time
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / ".cache" / "hf"))

from PIL import Image
from tqdm import tqdm

from src.labels import IMAGE_EXTS
from src.paths import RAW, SOURCES

sys.path.insert(0, str(ROOT / "training"))
from download import curl_get, download_file, mendeley_files

MAX_SIDE = 640
ZDIR = RAW / ".zips"
OLID_FILES = (
    "tomato__part_1.zip",
    "tomato__part_2.zip",
    "eggplant__part_1.zip",
    "eggplant__part_2.zip",
    "eggplant__part_3.zip",
)
INAT_TAXA = (
    ("tomato", "Solanum lycopersicum", 500),
    ("sili", "Capsicum annuum", 300),
    ("eggplant", "Solanum melongena", 250),
    ("palay", "Oryza sativa", 200),
    ("lettuce", "Lactuca sativa", 200),
)
ROB2_N = 400
ROB2_URL = "https://ndownloader.figshare.com/files/25337984"


def _thumb(im: Image.Image) -> Image.Image:
    rgb = im.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    return rgb


def _save_rgb(im: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    _thumb(im).save(dest, quality=85)


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def fetch_plantdoc() -> None:
    dest = RAW / "plantdoc"
    if _count_images(dest) >= 400:
        print("plantdoc already saved")
        return
    from datasets import load_dataset

    print("== plantdoc tomato+pepper ==")
    ds = load_dataset("geraldmc/plantdoc-full", split="train")
    saved = 0
    dest.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(ds, desc="plantdoc")):
        host = str(row.get("host") or "").lower()
        label = str(row.get("class_label") or "")
        if "tomato" not in host and "pepper" not in host:
            continue
        folder = "".join(c if c.isalnum() or c in " -_" else "_" for c in label).strip() or "class"
        out = dest / folder / f"{i:05d}.jpg"
        _save_rgb(row["image"], out)
        saved += 1
    print(f"saved {saved} plantdoc")


def fetch_bdveg() -> None:
    dest = RAW / "bdveg"
    if _count_images(dest) >= 200:
        print("bdveg already saved")
        return
    from datasets import load_dataset

    print("== bdveg tomato+eggplant ==")
    ds = load_dataset("Project-AgML/plant_leaf_disease_classification", split="train")
    feat = ds.features.get("label")
    names = getattr(feat, "names", None)
    saved = 0
    dest.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(ds, desc="bdveg")):
        label = row.get("class_label") or row.get("label")
        if not isinstance(label, str):
            label = names[int(label)] if names else str(label)
        low = label.lower()
        if not (low.startswith("tomato") or low.startswith("eggplant")):
            continue
        folder = "".join(c if c.isalnum() or c in " -_" else "_" for c in label).strip() or "class"
        out = dest / folder / f"{i:05d}.jpg"
        _save_rgb(row["image"], out)
        saved += 1
    print(f"saved {saved} bdveg")


def fetch_chili_growth() -> None:
    dest = RAW / "chili_growth"
    if _count_images(dest) >= 800:
        print("chili growth already saved")
        return
    print("== chili growth originals ==")
    files = mendeley_files("w9mr3vf56s")
    hit = next(f for f in files if f["filename"] == "Chili Growth Stage Original Dataset.zip")
    ZDIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZDIR / hit["filename"]
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        download_file(hit["content_details"]["download_url"], zip_path)
    _zip_thumbs(zip_path, dest)
    print("chili growth images", _count_images(dest))


def fetch_olid() -> None:
    dest = RAW / "olid"
    if _count_images(dest) >= 800:
        print("olid already saved")
        return
    print("== OLID-I skipped (zenodo stall); field leaves already in bdveg ==")
    return


def _zip_thumbs(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n
            for n in zf.namelist()
            if Path(n).suffix.lower() in IMAGE_EXTS
            and "__macosx" not in n.lower()
            and "augmented" not in n.lower()
        ]
        for idx, name in enumerate(tqdm(members, desc=zip_path.name[:28])):
            parts = [p for p in Path(name.replace("\\", "/")).parts if p not in (".",)]
            folders = parts[:-1]
            if not folders:
                continue
            class_name = folders[-1]
            crop_name = folders[-2] if len(folders) >= 2 else "class"
            out = dest / crop_name / class_name / f"{zip_path.stem}_{idx:05d}.jpg"
            if out.exists():
                continue
            try:
                with zf.open(name) as src:
                    img = Image.open(io.BytesIO(src.read()))
                    _save_rgb(img, out)
            except Exception:
                continue


def fetch_rob2pheno() -> None:
    dest = RAW / "rob2pheno" / "tomato_robot"
    if _count_images(dest) >= 80:
        print("rob2pheno already saved")
        return
    print("== rob2pheno tomato robot ==")
    from src.paths import EXTERNAL

    fallback = EXTERNAL / "rob2pheno"
    if _count_images(fallback) >= 40:
        dest.mkdir(parents=True, exist_ok=True)
        saved = 0
        for src in fallback.rglob("*"):
            if src.suffix.lower() not in IMAGE_EXTS or not src.is_file():
                continue
            if ".zips" in src.parts or "_raw" in src.parts:
                continue
            _save_rgb(Image.open(src), dest / f"{saved:04d}.jpg")
            saved += 1
        print(f"copied {saved} rob2pheno from external")
        return
    ZDIR.mkdir(parents=True, exist_ok=True)
    tar_path = ZDIR / "rob2pheno_rgb.tar.gz"
    if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
        download_file(ROB2_URL, tar_path)
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if Path(m.name).suffix.lower() in IMAGE_EXTS]
        random.Random(42).shuffle(members)
        for mem in members:
            if saved >= ROB2_N:
                break
            f = tf.extractfile(mem)
            if f is None:
                continue
            try:
                img = Image.open(io.BytesIO(f.read()))
                _save_rgb(img, dest / f"{saved:04d}.jpg")
                saved += 1
            except Exception:
                continue
    print(f"saved {saved} rob2pheno")


def _http_json(url: str) -> dict:
    raw = curl_get(url)
    return json.loads(raw)


def fetch_inat() -> None:
    dest_root = RAW / "inat"
    ua = "PlantHealthScanner/1.0 (local training)"
    for crop, query, cap in INAT_TAXA:
        dest = dest_root / crop
        if _count_images(dest) >= cap:
            print("inat", crop, "already saved")
            continue
        print("== inat", query, "==")
        taxa = _http_json(f"https://api.inaturalist.org/v1/taxa?q={quote(query)}&rank=species")
        results = taxa.get("results") or []
        if not results:
            print("no taxon", query)
            continue
        tid = results[0]["id"]
        dest.mkdir(parents=True, exist_ok=True)
        saved = _count_images(dest)
        page = 1
        while saved < cap and page <= 8:
            url = (
                "https://api.inaturalist.org/v1/observations"
                f"?taxon_id={tid}&quality_grade=research&photos=true"
                "&photo_license=cc0,cc-by&per_page=200"
                f"&page={page}&order_by=id&order=desc"
            )
            payload = _http_json(url)
            obs = payload.get("results") or []
            if not obs:
                break
            for row in obs:
                if saved >= cap:
                    break
                photos = row.get("photos") or []
                if not photos:
                    continue
                photo = photos[0]
                url_img = photo.get("url") or ""
                url_img = url_img.replace("/square.", "/medium.").replace("square.", "medium.")
                if not url_img:
                    continue
                out = dest / f"{row.get('id', saved)}_{photo.get('id', saved)}.jpg"
                if out.exists():
                    saved += 1
                    continue
                try:
                    import subprocess

                    subprocess.check_call(
                        ["curl.exe", "-sL", "-A", ua, "--retry", "3", "--output", str(out), url_img],
                        timeout=60,
                    )
                    im = Image.open(out)
                    _thumb(im).save(out, quality=85)
                    saved += 1
                except Exception:
                    if out.exists():
                        out.unlink()
                    continue
                time.sleep(0.05)
            page += 1
            time.sleep(0.4)
        print("inat", crop, _count_images(dest))


def write_sources() -> None:
    extra = [
        f"wild mix {date.today().isoformat()}",
        "PlantDoc tomato+pepper into train https://huggingface.co/datasets/geraldmc/plantdoc-full CC BY 4.0",
        "Chili growth stage originals https://data.mendeley.com/datasets/w9mr3vf56s/1 CC BY 4.0 (crop-only)",
        "OLID-I tomato+eggplant https://zenodo.org/records/8105154 CC BY 4.0",
        "Bangladesh veg field tomato+eggplant https://huggingface.co/datasets/Project-AgML/plant_leaf_disease_classification",
        "Rob2Pheno greenhouse tomato https://doi.org/10.4121/13173422 (crop-only)",
        "iNaturalist CC0/CC BY whole-plant photos (crop-only)",
    ]
    old = SOURCES.read_text(encoding="utf-8") if SOURCES.exists() else "# Dataset sources\n\n"
    if "wild mix" in old:
        return
    SOURCES.write_text(old.rstrip() + "\n" + "\n".join(f"- {r}" for r in extra) + "\n", encoding="utf-8")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_plantdoc()
    fetch_bdveg()
    fetch_chili_growth()
    fetch_olid()
    fetch_rob2pheno()
    fetch_inat()
    write_sources()
    print("wild download done")


if __name__ == "__main__":
    main()
