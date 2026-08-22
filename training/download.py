import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import shutil
import subprocess
import zipfile
from datetime import date

from tqdm import tqdm

from src.paths import RAW, SOURCES

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

MENDELEY = {
    "chili": {
        "id": "w9mr3vf56s",
        "keep": ["Chili Leaf Disease Original Dataset.zip"],
        "dest": RAW / "chili",
        "license": "CC BY 4.0",
        "url": "https://data.mendeley.com/datasets/w9mr3vf56s/1",
    },
    "eggplant": {
        "id": "d3ypkphghb",
        "keep": ["Eggplant Dataset.zip"],
        "dest": RAW / "eggplant",
        "license": "CC BY 4.0",
        "url": "https://data.mendeley.com/datasets/d3ypkphghb/2",
    },
    "riceleafbd": {
        "id": "kx9rx8p2mz",
        "keep": ["Original Images.zip"],
        "dest": RAW / "riceleafbd",
        "license": "CC BY 4.0",
        "url": "https://data.mendeley.com/datasets/kx9rx8p2mz/1",
    },
}


def curl_get(url: str, dest: Path | None = None) -> Path | bytes:
    ua = HEADERS["User-Agent"]
    if dest is None:
        raw = subprocess.check_output(
            ["curl.exe", "-sL", "-A", ua, "-H", "Accept: application/json", "-H", "Referer: https://data.mendeley.com/", url],
            timeout=120,
        )
        return raw
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-L",
        "-A",
        ua,
        "-H",
        "Referer: https://data.mendeley.com/",
        "--retry",
        "5",
        "-C",
        "-",
        "--progress-bar",
        "--output",
        str(dest),
        url,
    ]
    subprocess.check_call(cmd)
    return dest


def download_file(url: str, dest: Path) -> None:
    curl_get(url, dest)


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for info in tqdm(members, desc=f"extract {zip_path.name}"):
            name = info.filename.replace("\\", "/")
            if name.startswith("__MACOSX") or "/__MACOSX/" in name:
                continue
            if "augmented" in name.lower():
                continue
            parts = [p for p in Path(name).parts if p not in (".",) and "__macosx" not in p.lower()]
            if not parts:
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    marker.write_text("ok", encoding="utf-8")


def mendeley_files(dataset_id: str) -> list[dict]:
    url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
    raw = curl_get(url)
    data = json.loads(raw)
    if "files" not in data:
        raise RuntimeError(f"Mendeley blocked or unexpected payload for {dataset_id}: {raw[:240]!r}")
    return data["files"]


def fetch_mendeley(name: str, spec: dict) -> None:
    print(f"== {name} ==")
    files = mendeley_files(spec["id"])
    wanted = set(spec["keep"])
    hits = [f for f in files if f["filename"] in wanted]
    if not hits:
        hits = [f for f in files if "augmented" not in f["filename"].lower() and "growth stage" not in f["filename"].lower()]
    zdir = RAW / ".zips"
    zdir.mkdir(parents=True, exist_ok=True)
    for f in hits:
        zip_path = zdir / f["filename"]
        url = f["content_details"]["download_url"]
        print(f"get {f['filename']} ({f['size']} bytes)")
        download_file(url, zip_path)
        extract_zip(zip_path, spec["dest"])


def fetch_plantvillage() -> None:
    dest = RAW / "plantvillage"
    marker = dest / ".extracted"
    if marker.exists():
        print("plantvillage already saved")
        return
    print("== plantvillage ==")
    from datasets import load_dataset

    hosts = {"tomato", "pepper"}
    saved = 0
    try:
        ds = load_dataset("geraldmc/plantvillage-full", split="train")
        label_key = "class_label"
        leaf_key = "leaf_id"
        host_key = "host"
    except Exception as exc:
        print("geraldmc failed, trying mohanty/PlantVillage:", exc)
        ds = load_dataset("mohanty/PlantVillage", split="train")
        label_key = "label"
        leaf_key = "leaf_id"
        host_key = "crop"

    dest.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(ds, desc="save pv tomato/pepper")):
        host = str(row.get(host_key, "")).lower()
        label = row[label_key]
        if isinstance(label, str):
            label_s = label
        else:
            feat = ds.features.get(label_key)
            names = getattr(feat, "names", None)
            label_s = names[int(label)] if names else str(label)
        if host_key == "crop":
            ok = host in {"tomato", "pepper,_bell", "pepper"} or "tomato" in label_s.lower() or "pepper" in label_s.lower()
        else:
            ok = any(h in host for h in hosts) or "pepper" in host or "tomato" in label_s.lower() or "pepper" in label_s.lower()
        if not ok:
            continue
        folder = dest / label_s.replace("/", "_")
        folder.mkdir(parents=True, exist_ok=True)
        leaf = str(row.get(leaf_key, i))
        safe_leaf = "".join(c if c.isalnum() or c in "-_." else "_" for c in leaf)[:80]
        out = folder / f"{safe_leaf}_{i}.jpg"
        if out.exists():
            saved += 1
            continue
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out, quality=92)
        saved += 1
    marker.write_text(str(saved), encoding="utf-8")
    print(f"saved {saved} plantvillage images")


def write_sources(rows: list[str]) -> None:
    SOURCES.write_text(
        "# Dataset sources\n\n"
        f"Downloaded: {date.today().isoformat()}\n\n"
        + "\n".join(f"- {r}" for r in rows)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    notes = []
    fetch_mendeley("chili", MENDELEY["chili"])
    notes.append(f"chili originals {MENDELEY['chili']['url']} {MENDELEY['chili']['license']}")
    fetch_mendeley("eggplant", MENDELEY["eggplant"])
    notes.append(f"eggplant {MENDELEY['eggplant']['url']} {MENDELEY['eggplant']['license']}")
    fetch_mendeley("riceleafbd", MENDELEY["riceleafbd"])
    notes.append(f"RiceLeafBD originals {MENDELEY['riceleafbd']['url']} {MENDELEY['riceleafbd']['license']}")
    fetch_plantvillage()
    notes.append("PlantVillage color tomato+pepper https://huggingface.co/datasets/geraldmc/plantvillage-full (upstream Mohanty et al. 2016; treat redistribution under original terms / CC-style openness)")
    notes.append("Lettuce Kaggle set skipped (no kaggle.json on this machine).")
    write_sources(notes)
    print("download done")


if __name__ == "__main__":
    main()
