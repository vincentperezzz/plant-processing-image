import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.infer import Scanner
from src.paths import CKPT, DATA

MANIFEST = DATA / "realistic_manifest.csv"
RESULTS = DATA / "realistic_results.csv"


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Run training/download_realistic.py first.")
    df = pd.read_csv(MANIFEST)
    scanner = Scanner(CKPT)
    recs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="realistic scan"):
        pred = scanner.scan(row.path)
        recs.append(
            {
                "path": row.path,
                "source": row.source,
                "class_folder": row.class_folder,
                "true_crop": row.crop,
                "true_health": row.health if isinstance(row.health, str) else "",
                "expect_unknown": bool(row.expect_unknown),
                "score_health": bool(row.score_health),
                "pred_crop": pred["crop"],
                "pred_health": pred["health"],
                "crop_confidence": pred["crop_confidence"],
                "health_confidence": pred["health_confidence"],
                "pred_unknown": pred["unknown"],
            }
        )
    out = pd.DataFrame(recs)
    out.to_csv(RESULTS, index=False)
    print("\n=== by source ===")
    for source, g in out.groupby("source"):
        known = g[~g["expect_unknown"]]
        crop_ok = (known["true_crop"] == known["pred_crop"]).mean() if len(known) else float("nan")
        unk = g["pred_unknown"].mean()
        conf = g["crop_confidence"].mean()
        print(f"{source}: n={len(g)} crop_acc={crop_ok:.3f} mean_crop_conf={conf:.3f} flagged_unknown={unk:.3f}")
        if len(known):
            for crop, sub in known.groupby("true_crop"):
                print(f"  {crop}: acc={(sub['pred_crop'] == crop).mean():.3f} (n={len(sub)})")
            print("predicted crops")
            print(g["pred_crop"].value_counts().to_string())
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
