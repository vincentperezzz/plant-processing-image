"""Rebuild ledger rows for images already in data/raw/insitu/.

Filenames are <occurrenceKey>_<urlhash>.jpg, so the licence and rights holder
are recoverable from GBIF even when the ledger row was lost. Safe to re-run:
occurrence keys already present in the ledger are skipped.

  python training/backfill_insitu_ledger.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv

import requests
from tqdm import tqdm

from download_insitu import CROP_TAXA, DEST, LEDGER, UA, ledger_append

OCC = "https://api.gbif.org/v1/occurrence/{key}"


def existing_paths() -> set:
    """Keyed on path, not occurrence key -- several files can share one
    observation, and each file needs its own licence row."""
    if not LEDGER.exists():
        return set()
    import io

    text = LEDGER.read_text(encoding="utf-8")
    return {r["path"] for r in csv.DictReader(io.StringIO(text)) if r.get("path")}


def main() -> None:
    have = existing_paths()
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    todo = []
    for crop in CROP_TAXA:
        for p in sorted((DEST / crop).glob("*.jpg")):
            key = p.stem.split("_")[0]
            if key.isdigit() and str(p) not in have and str(p.resolve()) not in have:
                todo.append((crop, key, p))
    if not todo:
        print("nothing to backfill")
        return
    print(f"backfilling {len(todo)} images ({len(have)} already recorded)")
    ok = miss = 0
    for crop, key, path in tqdm(todo, desc="backfill"):
        try:
            r = sess.get(OCC.format(key=key), timeout=45)
            r.raise_for_status()
            occ = r.json()
        except Exception:
            miss += 1
            continue
        media = [m for m in (occ.get("media") or []) if (m.get("type") or "StillImage") == "StillImage"]
        m = media[0] if media else {}
        ledger_append(
            {
                "crop": crop,
                "species": occ.get("species") or "",
                "occurrence_key": key,
                "country": occ.get("countryCode") or "",
                "media_url": m.get("identifier") or "",
                "media_license": m.get("license") or "",
                "occurrence_license": occ.get("license") or "",
                "rights_holder": m.get("rightsHolder") or occ.get("rightsHolder") or "",
                "creator": m.get("creator") or "",
                "verdict": "keep",
                "box_frac": "",
                "green_frac": "",
                "dhash": "",
                "path": str(path),
            }
        )
        ok += 1
    print(f"recovered {ok}, unresolved {miss}")
    if miss:
        print("unresolved rows have no licence record -- delete those files or re-check by hand")


if __name__ == "__main__":
    main()
