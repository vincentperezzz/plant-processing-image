import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.paths import LABEL_MAP

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SOURCE_HINTS = (
    ("plantvillage", None),
    ("plantdoc", None),
    ("plantwild", None),
    ("bdveg", None),
    ("chili_growth", "sili"),
    ("chili", "sili"),
    ("eggplant", "eggplant"),
    ("riceleafbd", "palay"),
    ("banglarice", "palay"),
    ("ricebd", "palay"),
    ("paddydoc", "palay"),
    ("rice", "palay"),
    ("lettuce", "lettuce"),
    ("olid", None),
    ("rob2pheno", "tomato"),
    ("inat", None),
)


def normalize_key(text: str) -> str:
    t = text.lower().replace("___", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@lru_cache(maxsize=1)
def load_spec() -> dict:
    with LABEL_MAP.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def crops() -> list[str]:
    return list(load_spec()["crops"])


def health_levels() -> list[str]:
    return list(load_spec()["health"])


def skip_path(path: Path) -> bool:
    s = str(path).lower()
    return any(tok in s for tok in load_spec().get("skip_path_substrings", []))


def infer_source(path: Path) -> tuple[str | None, str | None]:
    parts = [p.lower() for p in path.parts]
    for key, crop in SOURCE_HINTS:
        if key in parts:
            return key, crop
    return None, None


def _candidates(path: Path, source_crop: str | None) -> list[str]:
    names = [path.parent.name]
    if path.parent.parent:
        names.append(path.parent.parent.name)
        names.append(f"{path.parent.parent.name} {path.parent.name}")
    out = []
    seen = set()
    for name in names:
        n = normalize_key(name)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if source_crop:
            prefixed = normalize_key(f"{source_crop} {n}")
            if prefixed not in seen:
                seen.add(prefixed)
                out.append(prefixed)
            if source_crop == "sili":
                for extra in (f"chili {n}", f"pepper {n}"):
                    p = normalize_key(extra)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
            if source_crop == "palay":
                for extra in (f"rice {n}",):
                    p = normalize_key(extra)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
    return out


def match_label(path: Path) -> dict | None:
    spec = load_spec()
    source, source_crop = infer_source(path)
    cands = _candidates(path, source_crop)
    best = None
    best_len = -1
    for entry in spec["maps"]:
        if source_crop and entry["crop"] != source_crop:
            continue
        for alias in entry["match"]:
            na = normalize_key(alias)
            if not na:
                continue
            for cand in cands:
                hit = cand == na or cand.endswith(" " + na)
                if hit and len(na) > best_len:
                    best = entry
                    best_len = len(na)
    return best


def crop_index(name: str) -> int:
    return crops().index(name)


def health_index(name: str) -> int:
    return health_levels().index(name)
