import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.detect import find_plants_exg
from src.dictionary import PlantDictionary
from src.paths import DATA, MANIFEST


def grade_image(img: Image.Image, dictionary: PlantDictionary) -> str:
    # Isolate plant foliage to avoid garden dirt / soil interference
    try:
        boxes = find_plants_exg(img)
        if boxes:
            best_b = max(boxes, key=lambda b: (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]))
            x1, y1, x2, y2 = best_b["xyxy"]
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                target_img = img.crop((x1, y1, x2, y2))
            else:
                target_img = img
        else:
            target_img = img
    except Exception:
        target_img = img

    res = dictionary.grade_health(target_img, "lettuce")
    health = res.get("health")
    conf = res.get("confidence", 0.0)
    scores = res.get("scores", {})

    # Require minimum confidence or margin to prevent assigning noise
    if health in ("healthy", "mild", "critical", "dead"):
        # If healthy score is strong, assign healthy
        if scores.get("healthy", 0.0) >= 0.32:
            return "healthy"
        if health == "dead" and scores.get("dead", 0.0) < 0.38:
            # Re-evaluate: if not overwhelmingly dead, check if it's mild or critical
            if scores.get("critical", 0.0) >= scores.get("mild", 0.0):
                return "critical"
            return "mild"
        return health
    return ""


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    print(f"Loaded manifest: {len(df)} rows")
    lettuce_mask = df["crop"] == "lettuce"
    lettuce_count = int(lettuce_mask.sum())
    print(f"Total lettuce images to grade: {lettuce_count}")

    # Backup manifest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DATA / f"manifest_backup_{ts}.csv"
    df.to_csv(backup, index=False)
    print(f"Backed up manifest to {backup}")

    dictionary = PlantDictionary()
    grades = []
    indices = df[lettuce_mask].index

    for idx in tqdm(indices, desc="Grading lettuce with ExG"):
        path_str = df.at[idx, "path"]
        p = Path(path_str)
        if not p.exists():
            grades.append("")
            continue
        try:
            with Image.open(p) as raw_img:
                img = raw_img.convert("RGB")
            grade = grade_image(img, dictionary)
            grades.append(grade)
        except Exception:
            grades.append("")

    df.loc[lettuce_mask, "health"] = grades
    df.to_csv(MANIFEST, index=False)

    print("\nUpdated lettuce health distribution in manifest:")
    print(df[lettuce_mask]["health"].fillna("crop-only").value_counts())
    print(f"\nSaved updated manifest to {MANIFEST}")


if __name__ == "__main__":
    main()
