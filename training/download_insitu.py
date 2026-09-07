"""In-situ whole-plant harvester (GBIF -> data/raw/insitu/<crop>).

The lab sets we already have are filled-frame leaves. The robot sees a whole
plant from standing distance on messy ground. This pulls that view: living
plants photographed in place, gated on how much of the frame the plant fills.

  python training/download_insitu.py --dry-run
  python training/download_insitu.py --license all-cc --cap 1200
  python training/download_insitu.py --source inat --license all-cc --cap 1200

GBIF carries only what iNaturalist exports to it: research-grade, non-captive.
A crop in a field is cultivated, so iNat flags it captive, which forces the
observation to casual grade, which means GBIF never sees it. --source inat goes
straight to the API and picks those up. Both sources share one ledger and one
dedup set, so the second pass cannot re-fetch what the first already kept.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import hashlib
import io
import re
import time
from datetime import date

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.paths import RAW, SOURCES

DEST = RAW / "insitu"
LEDGER = ROOT / "data" / "insitu_ledger.csv"
GBIF = "https://api.gbif.org/v1/occurrence/search"
UA = "PlantHealthScanner/1.0 (local training)"
MAX_SIDE = 640

# Species keys verified against api.gbif.org/v1/species/match on 2026-09-04.
# sili leads with frutescens: siling labuyo is C. frutescens, not annuum, and
# the old iNat pull only ever asked for annuum.
CROP_TAXA = {
    "sili": [("Capsicum frutescens", 8403992), ("Capsicum annuum", 2932944)],
    "eggplant": [("Solanum melongena", 2930617)],
    "palay": [("Oryza sativa", 2703459)],
    "tomato": [("Solanum lycopersicum", 2930137)],
    "lettuce": [("Lactuca sativa", 7403263)],
}

# Tropical Asia first: same sun, same soil, same cultivars as Mindoro.
ASIA = ("PH", "ID", "MY", "TH", "VN", "IN", "BD", "TW", "KH", "LA", "MM", "LK")

LICENSES = {
    "commercial": ("CC0_1_0", "CC_BY_4_0"),
    "all-cc": ("CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"),
}

INAT = "https://api.inaturalist.org/v1/observations"

# The same six species as CROP_TAXA, keyed for iNaturalist instead of GBIF.
# Verified against api.inaturalist.org/v1/taxa on 2026-09-05. Do not reuse the
# GBIF keys here: different taxonomy, unrelated id space. Searching iNat by
# name is not a shortcut either -- q="Lactuca sativa" ranks Lactuca serriola,
# a roadside weed, above the crop.
INAT_TAXA = {
    "sili": [("Capsicum frutescens", 122796), ("Capsicum annuum", 48514)],
    "eggplant": [("Solanum melongena", 63190)],
    "palay": [("Oryza sativa", 61381)],
    "tomato": [("Solanum lycopersicum", 51737)],
    "lettuce": [("Lactuca sativa", 122976)],
}

# iNat filters on place_id, not ISO code, and its autocomplete is too loose to
# resolve them at runtime: "ID" returns Idlib and "TH" returns The Gambia
# before the country. These are the admin_level=0 places for ASIA, resolved
# once on 2026-09-05.
INAT_PLACES = {
    "PH": 6873, "ID": 6966, "MY": 7155, "TH": 6967, "VN": 6847, "IN": 6681,
    "BD": 7154, "TW": 7887, "KH": 7002, "LA": 7001, "MM": 6992, "LK": 7077,
}

INAT_LICENSES = {
    "commercial": ("cc0", "cc-by"),
    "all-cc": ("cc0", "cc-by", "cc-by-nc"),
}

# Framing gate.
#
# The robot photo always contains context: gravel, soil, sky, a pot, a shoe.
# A leaf macro and a filled-frame canopy both read as wall-to-wall foliage, and
# neither teaches the model what a plant looks like standing on ground. So the
# gate is mostly about how much of the frame is NOT vegetation.
CLOSEUP_FRAC = 0.93  # plant box this much of the frame -> nothing but plant
FAR_FRAC = 0.03  # below this the plant is a speck
MAX_GREEN = 0.75  # vegetation coverage above this -> leaf macro or filled canopy
MIN_GREEN = 0.08  # below this there is no plant in the frame at all
MIN_CONTEXT = 0.25  # at least this much of the frame must be non-vegetation
FLAT_BG_STD = 12.0  # backdrop std (0-255) below this -> studio / herbarium sheet
MIN_SAT = 0.10  # dried herbarium sheets and scans are near-grey


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def _prefer_smaller(url: str) -> str:
    """Normalise an iNaturalist photo URL to the 'large' (~1024px) variant.

    It corrects in both directions, because the two sources err opposite ways:
    GBIF republishes multi-MB originals, while the iNat observations API hands
    back 75px square thumbnails that the gate would bin as too_small.

    It is also what makes cross-source dedup free. Both sources serve the same
    photo from the same bucket under the same id, so once the variant is
    normalised the URLs are byte-identical and seen_urls catches the overlap
    before it costs a request.
    """
    for variant in ("/original.", "/square.", "/medium.", "/small.", "/thumb."):
        if variant in url:
            return url.replace(variant, "/large.")
    return url


def _media_rows(occ: dict) -> list[dict]:
    out = []
    for m in occ.get("media") or []:
        ident = m.get("identifier") or ""
        if not ident.lower().startswith("http"):
            continue
        if (m.get("type") or "StillImage") != "StillImage":
            continue
        out.append(m)
    return out


def _license_ok(text: str, allow_nc: bool) -> bool:
    """Media licence can differ from the occurrence licence. Judge the media.

    GBIF publishers are inconsistent here: the field holds a licence URL on
    some records and a credit line like "Some Person (cc-by-nc)" on others, so
    split on every non-alphanumeric or the closing paren hides the "nc".
    """
    if not text:
        return True  # fall back to the occurrence licence, already filtered
    parts = set(re.split(r"[^a-z0-9]+", str(text).lower()))
    if "nd" in parts:
        return False
    if not allow_nc and "nc" in parts:
        return False
    return True


def search(sess, key: int, licenses: tuple, countries: tuple, offset: int, limit: int = 300) -> dict:
    params = [
        ("speciesKey", key),
        ("mediaType", "StillImage"),
        ("basisOfRecord", "HUMAN_OBSERVATION"),
        ("limit", limit),
        ("offset", offset),
    ]
    params += [("license", v) for v in licenses]
    params += [("country", c) for c in countries]
    for attempt in range(4):
        try:
            r = sess.get(GBIF, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {"results": [], "endOfRecords": True, "count": 0}


def _inat_norm(o: dict) -> dict:
    """Reshape one iNat observation into the GBIF occurrence dict harvest reads.

    Only the fields anything downstream actually touches.

    Photos carrying no license_code are all-rights-reserved and are dropped
    here rather than left to _license_ok, which reads an empty licence as
    "inherit the occurrence's" -- correct for GBIF, wrong here. The
    photo_license filter qualifies the observation, not every photo hanging
    off it, so a licensed observation can still carry reserved photos.
    """
    user = o.get("user") or {}
    who = user.get("name") or user.get("login") or ""
    media = []
    for ph in o.get("photos") or []:
        lic = ph.get("license_code") or ""
        if not lic or ph.get("hidden"):
            continue
        media.append({
            "identifier": ph.get("url") or "",
            "type": "StillImage",
            "license": lic,
            "rightsHolder": ph.get("attribution") or "",
            "creator": who,
        })
    # iNat reports where an observation falls as a bag of nested place ids, with
    # no ISO code anywhere on the record. Recover the country for the places we
    # named; the global pass leaves it blank, which the ledger already allows.
    places = set(o.get("place_ids") or [])
    country = next((c for c, pid in INAT_PLACES.items() if pid in places), "")
    return {
        # Namespaced: a bare iNat id is the same shape as a GBIF occurrence key
        # and per_occurrence counts them in one dict.
        "key": f"inat-{o.get('id')}",
        "license": o.get("license_code") or "",
        "countryCode": country,
        "rightsHolder": who,
        "media": media,
    }


def search_inat(sess, taxon: int, licenses: tuple, countries: tuple, cursor: int, limit: int = 200) -> dict:
    """One page of iNat observations, shaped like a GBIF response.

    Pages by id cursor rather than offset because page/per_page stops dead at
    10,000 results and the annuum and lycopersicum pools run well past that.
    """
    params = {
        "taxon_id": taxon,
        "photo_license": ",".join(licenses),
        "per_page": limit,
        "order_by": "id",
        "order": "asc",
    }
    if cursor:
        params["id_above"] = cursor
    if countries:
        pids = [INAT_PLACES[c] for c in countries if c in INAT_PLACES]
        if pids:
            params["place_id"] = ",".join(str(v) for v in pids)
    for attempt in range(4):
        try:
            r = sess.get(INAT, params=params, timeout=60)
            if r.status_code == 200:
                payload = r.json()
                occs = payload.get("results") or []
                time.sleep(1.0)  # iNat asks for <= 60 requests/minute
                return {
                    "results": [_inat_norm(o) for o in occs],
                    "endOfRecords": len(occs) < limit,
                    "count": int(payload.get("total_results") or 0),
                    "cursor": occs[-1]["id"] if occs else cursor,
                }
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {"results": [], "endOfRecords": True, "count": 0, "cursor": cursor}


def pages(sess, source: str, key: int, licenses: tuple, countries: tuple):
    """Yield pages of occurrences from whichever source is selected.

    GBIF walks an offset and iNat walks an id cursor. harvest() should not have
    to know which.
    """
    if source == "inat":
        cursor, end = 0, False
        while not end:
            payload = search_inat(sess, key, licenses, countries, cursor)
            occs = payload.get("results") or []
            end = bool(payload.get("endOfRecords"))
            cursor = payload.get("cursor") or cursor
            if not occs:
                return
            yield occs
    else:
        offset, end = 0, False
        while not end and offset < 100000:
            payload = search(sess, key, licenses, countries, offset)
            occs = payload.get("results") or []
            end = bool(payload.get("endOfRecords")) or not occs
            offset += 300
            if not occs:
                return
            yield occs


def taxa_for(crop: str, source: str) -> list:
    return INAT_TAXA[crop] if source == "inat" else CROP_TAXA[crop]


def licenses_for(source: str, name: str) -> tuple:
    return INAT_LICENSES[name] if source == "inat" else LICENSES[name]


def count(sess, key: int, licenses: tuple, countries: tuple, source: str = "gbif") -> int:
    if source == "inat":
        return int(search_inat(sess, key, licenses, countries, 0, limit=0).get("count") or 0)
    return int(search(sess, key, licenses, countries, 0, limit=0).get("count") or 0)


def _dhash(img: Image.Image) -> str:
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def _saturation(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    return float(np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6).mean())


def _green_frac(img: Image.Image) -> float:
    """Vegetation coverage of the whole frame, same green test as plant_look."""
    arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    green = (g > r * 1.06) & (g > b * 1.04) & (g > 0.16) & (sat > 0.12)
    return float(green.mean())


def _backdrop_std(img: Image.Image, xyxy: tuple) -> float:
    """Std of everything outside the plant box. Flat = studio sheet."""
    arr = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float32)
    w, h = img.size
    x0, y0, x1, y1 = (int(v) for v in xyxy)
    sx, sy = 256.0 / max(w, 1), 256.0 / max(h, 1)
    mask = np.ones((256, 256), dtype=bool)
    ry0, ry1 = max(0, int(y0 * sy)), min(256, int(y1 * sy))
    rx0, rx1 = max(0, int(x0 * sx)), min(256, int(x1 * sx))
    mask[ry0:ry1, rx0:rx1] = False
    if mask.mean() < 0.06:
        return 999.0  # box covers the frame; the closeup gate handles it
    return float(np.mean([arr[:, :, c][mask].std() for c in range(3)]))


def gate(img: Image.Image, finder, max_green: float = MAX_GREEN) -> tuple[str, float, float]:
    """Return (verdict, box_frac, green_frac). 'keep' means robot-like framing."""
    w, h = img.size
    green = _green_frac(img)
    if min(w, h) < 200:
        return "too_small", 0.0, green
    if _saturation(img) < MIN_SAT:
        return "greyish", 0.0, green
    # The decisive test, and it needs no detector: can we see anything but plant?
    if green >= max_green or (1.0 - green) < MIN_CONTEXT:
        return "no_context", 0.0, green
    if finder is None:
        return "keep", -1.0, green
    boxes = finder.find(img)
    if not boxes:
        # YOLO-World at imgsz=320 loses thin wispy plants against gravel -- it
        # misses 3 of our own 13 field photos. Green coverage already proved
        # there is a plant, so trust that rather than the detector.
        return ("keep" if green >= MIN_GREEN else "no_plant_box"), 0.0, green
    fracs = []
    for b in boxes:
        x0, y0, x1, y1 = b["xyxy"]
        fracs.append(max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(w * h))
    frac = max(fracs)
    best = boxes[int(np.argmax(fracs))]
    if frac >= CLOSEUP_FRAC:
        return "closeup", frac, green
    if frac <= FAR_FRAC:
        return "too_far", frac, green
    if _backdrop_std(img, best["xyxy"]) < FLAT_BG_STD:
        return "flat_backdrop", frac, green
    return "keep", frac, green


def load_finder(enabled: bool):
    if not enabled:
        return None
    from src.detect import PlantFinder

    finder = PlantFinder(backend="color")
    finder._load_yolo()
    if finder.backend != "yolo":
        print("!! YOLO-World unavailable; framing gate falls back to ExG boxes")
    return finder


def load_ledger() -> tuple[list, set, set, dict]:
    """Read the ledger fully into memory before anything can append to it.

    Parsing straight off a live handle while rows are being appended to the
    same file makes the reader chase its own writes -- that turned a 400-row
    ledger into 124 MB once. Snapshot the text, then parse.
    """
    rows, hashes, urls, per_occ = [], set(), set(), {}
    if LEDGER.exists():
        text = LEDGER.read_text(encoding="utf-8")
        for row in csv.DictReader(io.StringIO(text)):
            rows.append(row)
            if row.get("dhash"):
                hashes.add(row["dhash"])
            urls.add(row["media_url"])
            if row["verdict"] == "keep":
                k = row["occurrence_key"]
                per_occ[k] = per_occ.get(k, 0) + 1
    return rows, hashes, urls, per_occ


def scan_disk(crops: list) -> dict:
    """dhash -> path for images already harvested.

    A resume must not re-count an image it already has toward the cap, and the
    re-fetched copy is the chance to record the licence the ledger is missing.
    """
    known = {}
    for crop in crops:
        for p in (DEST / crop).glob("*.jpg"):
            try:
                with Image.open(p) as im:
                    known[_dhash(im)] = str(p)
            except Exception:
                continue
    return known


LEDGER_COLS = [
    "crop", "species", "occurrence_key", "country", "media_url", "media_license",
    "occurrence_license", "rights_holder", "creator", "verdict", "box_frac", "green_frac",
    "dhash", "path",
]


def ledger_append(rec: dict) -> None:
    """Append one row, header-on-create, flushed immediately.

    Deliberately append-only: rewriting the whole file on every flush means a
    kill (or two overlapping runs) truncates the licence record for images
    already on disk, which is exactly what happened once.
    """
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    fresh = not LEDGER.exists() or LEDGER.stat().st_size == 0
    with LEDGER.open("a", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LEDGER_COLS, extrasaction="ignore")
        if fresh:
            wr.writeheader()
        wr.writerow({k: rec.get(k, "") for k in LEDGER_COLS})
        f.flush()


def _add(rows: list, rec: dict) -> None:
    rows.append(rec)
    ledger_append(rec)


def _save(img: Image.Image, dest: Path) -> None:
    rgb = img.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb.save(dest, quality=88)


def report(sess, crops: list, licenses: tuple, source: str = "gbif") -> None:
    scope = "captive included" if source == "inat" else "basisOfRecord=HUMAN_OBSERVATION"
    print(f"source: {source}   licence filter: {', '.join(licenses)}   {scope}\n")
    print(f"{'crop':<10}{'species':<26}{'tropical Asia':>14}{'global':>10}")
    for crop in crops:
        for name, key in taxa_for(crop, source):
            asia = count(sess, key, licenses, ASIA, source)
            glob = count(sess, key, licenses, (), source)
            print(f"{crop:<10}{name:<26}{asia:>14}{glob:>10}")
    print(
        "\nCounts are occurrences, not photos, and are pre-gate."
        " Expect roughly 40-60% to survive the framing gate."
    )


def harvest(args) -> None:
    sess = _session()
    licenses = licenses_for(args.source, args.license)
    allow_nc = args.license == "all-cc"
    crops = args.crops or list(CROP_TAXA)

    if args.dry_run:
        report(sess, crops, licenses, args.source)
        return

    finder = load_finder(not args.no_gate)
    rows, seen_hashes, seen_urls, per_occ = load_ledger()
    on_disk = scan_disk(crops)
    seen_hashes |= set(on_disk)
    if on_disk:
        print(f"resuming: {len(on_disk)} images already on disk, {len(rows)} ledger rows")
    tallies = {}

    for crop in crops:
        dest = DEST / crop
        dest.mkdir(parents=True, exist_ok=True)
        kept = len(list(dest.glob("*.jpg")))
        tally = {}
        if args.asia_only:
            passes = [ASIA]
        else:
            passes = [ASIA, ()] if args.asia_first else [()]
        for countries in passes:
            scope = "asia" if countries else "global"
            for name, key in taxa_for(crop, args.source):
                if kept >= args.cap:
                    break
                bar = tqdm(desc=f"{crop}/{scope}/{name.split()[-1]}", unit="img")
                for occs in pages(sess, args.source, key, licenses, countries):
                    if kept >= args.cap:
                        break
                    for occ in occs:
                        if kept >= args.cap:
                            break
                        okey = str(occ.get("key") or "")
                        if per_occ.get(okey, 0) >= args.per_occurrence:
                            continue
                        occ_lic = occ.get("license") or ""
                        for m in _media_rows(occ):
                            if kept >= args.cap:
                                break
                            if per_occ.get(okey, 0) >= args.per_occurrence:
                                break
                            url = _prefer_smaller(m["identifier"])
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            mlic = m.get("license") or ""
                            base = {
                                "crop": crop,
                                "species": name,
                                "occurrence_key": okey,
                                "country": occ.get("countryCode") or "",
                                "media_url": url,
                                "media_license": mlic,
                                "occurrence_license": occ_lic,
                                "rights_holder": m.get("rightsHolder") or occ.get("rightsHolder") or "",
                                "creator": m.get("creator") or "",
                                "path": "",
                            }
                            if not _license_ok(mlic, allow_nc):
                                _add(rows, {**base, "verdict": "license", "box_frac": "", "green_frac": "", "dhash": ""})
                                continue
                            try:
                                r = sess.get(url, timeout=90)
                                r.raise_for_status()
                                img = Image.open(io.BytesIO(r.content))
                                img.load()
                            except Exception:
                                _add(rows, {**base, "verdict": "fetch_error", "box_frac": "", "green_frac": "", "dhash": ""})
                                continue
                            dh = _dhash(img)
                            if dh in seen_hashes:
                                _add(rows, {**base, "verdict": "duplicate", "box_frac": "", "green_frac": "",
                                            "dhash": dh, "path": on_disk.get(dh, "")})
                                continue
                            verdict, frac, green = gate(img, finder, args.max_green)
                            rec = {
                                **base,
                                "verdict": verdict,
                                "box_frac": f"{frac:.4f}",
                                "green_frac": f"{green:.4f}",
                                "dhash": dh,
                            }
                            if verdict != "keep":
                                tally[verdict] = tally.get(verdict, 0) + 1
                                _add(rows, rec)
                                if args.keep_rejects:
                                    rj = DEST / "_rejects" / crop / verdict
                                    rj.mkdir(parents=True, exist_ok=True)
                                    _save(img, rj / f"{okey}_{dh[:10]}.jpg")
                                continue
                            out = dest / f"{okey}_{hashlib.sha1(url.encode()).hexdigest()[:10]}.jpg"
                            _save(img, out)
                            rec["path"] = str(out)
                            _add(rows, rec)
                            seen_hashes.add(dh)
                            per_occ[okey] = per_occ.get(okey, 0) + 1
                            kept += 1
                            bar.update(1)
                            bar.set_postfix(kept=kept, box=f"{frac:.2f}", green=f"{green:.2f}")
                    time.sleep(args.sleep)
                bar.close()
        tally["keep"] = kept
        tallies[crop] = tally

    print("\n== in-situ harvest ==")
    for crop, tally in tallies.items():
        kept = tally.pop("keep", 0)
        drops = " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        print(f"{crop:<10} kept={kept:<6} {drops}")
    print(f"\nledger: {LEDGER}")
    print("next: python training/remap.py && python training/finetune_other.py")
    note_sources(args)


def note_sources(args) -> None:
    if args.source == "inat":
        tag = "iNaturalist in-situ whole-plant harvest"
        line = (
            f"{tag} {date.today().isoformat()} "
            f"(licence={args.license}, all quality grades incl. captive/cultivated, "
            "framing-gated) https://www.inaturalist.org/observations - per-image "
            "licence and rights holder in data/insitu_ledger.csv."
        )
    else:
        tag = "GBIF in-situ whole-plant harvest"
        line = (
            f"{tag} {date.today().isoformat()} "
            f"(licence={args.license}, basisOfRecord=HUMAN_OBSERVATION, framing-gated) "
            "https://www.gbif.org/occurrence/search - per-image licence and rights holder "
            "in data/insitu_ledger.csv. Includes Capsicum frutescens (siling labuyo)."
        )
    old = SOURCES.read_text(encoding="utf-8") if SOURCES.exists() else "# Dataset sources\n\n"
    if tag in old:
        return
    SOURCES.write_text(old.rstrip() + "\n- " + line + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--crops", nargs="*", choices=list(CROP_TAXA), help="default: all five")
    ap.add_argument(
        "--source",
        choices=("gbif", "inat"),
        default="gbif",
        help="gbif is research-grade only; inat adds cultivated plants, which GBIF "
        "never receives (iNat flags them captive -> casual -> never exported). Both "
        "write the same ledger and dedupe against each other",
    )
    ap.add_argument(
        "--license",
        choices=list(LICENSES),
        default="commercial",
        help="commercial = CC0 + CC-BY (safe for a sold box); all-cc adds CC-BY-NC (~8x data, research only)",
    )
    ap.add_argument("--cap", type=int, default=1200, help="kept images per crop")
    ap.add_argument("--per-occurrence", type=int, default=2, help="photos per observation (near-dup guard)")
    ap.add_argument("--asia-first", action="store_true", default=True)
    ap.add_argument("--global-only", dest="asia_first", action="store_false")
    ap.add_argument(
        "--asia-only",
        action="store_true",
        help="skip the global fill pass. Use it for sili: the global pass is dominated by "
        "US desert chiltepin, which is the right species in the wrong environment",
    )
    ap.add_argument("--no-gate", action="store_true", help="skip the framing gate (faster, dirtier)")
    ap.add_argument(
        "--max-green",
        type=float,
        default=MAX_GREEN,
        help="reject frames with more vegetation coverage than this (raise it if the harvest is too thin)",
    )
    ap.add_argument("--keep-rejects", action="store_true", help="save rejects under _rejects/ to audit the gate")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true", help="print available counts, download nothing")
    harvest(ap.parse_args())


if __name__ == "__main__":
    main()
