import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.labels import IMAGE_EXTS, infer_source, match_label, skip_path
from src.paths import MANIFEST, RAW


def main() -> None:
    rows = []
    images = [p for p in RAW.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    unmatched = 0
    for path in tqdm(images, desc="remap"):
        if skip_path(path):
            continue
        hit = match_label(path)
        if not hit:
            unmatched += 1
            continue
        source, _crop = infer_source(path)
        health = "" if hit.get("crop_only") or not hit.get("health") else hit["health"]
        group = f"{source}:{path.parent.name}:{path.stem.split('_')[0]}"
        rows.append(
            {
                "path": str(path.resolve()),
                "crop": hit["crop"],
                "health": health,
                "source": source or "unknown",
                "group_id": group,
                "class_folder": path.parent.name,
            }
        )
    if not rows:
        raise SystemExit("No labeled images. Run training/download.py first.")
    df = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(df.groupby(["crop", "health"]).size().unstack(fill_value=0))
    print(f"rows={len(df)} unmatched={unmatched}")


if __name__ == "__main__":
    main()
