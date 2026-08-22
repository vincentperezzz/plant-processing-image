import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.infer import Scanner
from src.paths import CKPT, HOLDOUT_MANIFEST, HOLDOUT_RESULTS


def main() -> None:
    if not HOLDOUT_MANIFEST.exists():
        raise SystemExit("Missing data/holdout_manifest.csv. Run training/download_holdout.py first.")
    df = pd.read_csv(HOLDOUT_MANIFEST)
    scanner = Scanner(CKPT)
    recs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="holdout scan"):
        pred = scanner.scan(row.path)
        recs.append(
            {
                "path": row.path,
                "source": row.source,
                "class_folder": row.class_folder,
                "true_crop": row.crop,
                "true_health": row.health,
                "expect_unknown": bool(row.expect_unknown),
                "pred_crop": pred["crop"],
                "pred_health": pred["health"],
                "crop_confidence": pred["crop_confidence"],
                "health_confidence": pred["health_confidence"],
                "pred_unknown": pred["unknown"],
            }
        )
    out = pd.DataFrame(recs)
    out.to_csv(HOLDOUT_RESULTS, index=False)
    known = out[~out["expect_unknown"]]
    unknown = out[out["expect_unknown"]]
    crop_ok = (known["true_crop"] == known["pred_crop"]).mean() if len(known) else 0.0
    both_ok = (
        (known["true_crop"] == known["pred_crop"]) & (known["true_health"] == known["pred_health"])
    ).mean() if len(known) else 0.0
    print(f"known n={len(known)} crop_acc={crop_ok:.3f} crop+health={both_ok:.3f}")
    if len(known):
        print("by crop")
        g = known.assign(ok=known["true_crop"] == known["pred_crop"])
        print(g.groupby("true_crop")["ok"].agg(["count", "mean"]))
        print("health given crop-correct")
        hit = known[known["true_crop"] == known["pred_crop"]]
        if len(hit):
            h = hit.assign(ok=hit["true_health"] == hit["pred_health"])
            print(h.groupby("true_health")["ok"].agg(["count", "mean"]))
    if len(unknown):
        flagged = unknown["pred_unknown"].mean()
        print(f"unknown probes n={len(unknown)} flagged_unknown={flagged:.3f}")
        print(unknown.groupby(["class_folder", "pred_crop"]).size().unstack(fill_value=0))
    print("wrote", HOLDOUT_RESULTS)


if __name__ == "__main__":
    main()
