from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import pandas as pd

from finetune_other import grouped_split
from src.paths import FIELD_HOLDOUT_GROUPS, FIELD_HOLDOUT_MANIFEST, MANIFEST

FIELD_PALAY_SOURCES = ("paddydoc", "ricebd", "riceleafbd")
EXAM_CAP = 480
SEED = 42


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing {MANIFEST}")
    pos = pd.read_csv(MANIFEST)
    _train, val = grouped_split(pos[["path", "crop", "health", "source", "group_id"]], seed=SEED)
    palay = val[
        (val["crop"] == "palay") & (val["source"].isin(FIELD_PALAY_SOURCES))
    ].copy()
    if palay.empty:
        raise SystemExit("No field palay in the 15% group split. Remap first.")
    groups = sorted(palay["group_id"].dropna().astype(str).unique())
    FIELD_HOLDOUT_GROUPS.parent.mkdir(parents=True, exist_ok=True)
    FIELD_HOLDOUT_GROUPS.write_text("\n".join(groups) + "\n", encoding="utf-8")

    palay["health"] = palay["health"].fillna("").astype(str)
    parts = []
    healths = [h for h in palay["health"].unique() if h != ""]
    if not healths:
        healths = [""]
    per = max(40, EXAM_CAP // max(1, len(healths)))
    for h in healths:
        g = palay[palay["health"] == h]
        n = min(per, len(g))
        parts.append(g.sample(n=n, random_state=SEED) if n < len(g) else g)
    exam = pd.concat(parts, ignore_index=True)
    if len(exam) > EXAM_CAP:
        exam = exam.sample(n=EXAM_CAP, random_state=SEED)
    exam = exam.copy()
    exam["expect_unknown"] = False
    exam["class_folder"] = exam["group_id"]
    exam = exam[["path", "crop", "health", "source", "class_folder", "expect_unknown", "group_id"]]
    exam.to_csv(FIELD_HOLDOUT_MANIFEST, index=False)
    print("locked groups", len(groups), "->", FIELD_HOLDOUT_GROUPS)
    print("exam photos", len(exam), "->", FIELD_HOLDOUT_MANIFEST)
    print(exam.groupby(["source", "health"]).size().to_string())
    print("pool (all val field palay)", len(palay))


if __name__ == "__main__":
    main()
