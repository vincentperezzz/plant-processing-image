import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.infer import Scanner
from src.labels import IMAGE_EXTS
from src.paths import CKPT, DATA, HOLDOUT_MANIFEST, HOLDOUT_RESULTS, MANIFEST, META

REALISTIC_MANIFEST = DATA / "realistic_manifest.csv"
REALISTIC_RESULTS = DATA / "realistic_results.csv"


def _ts(path: Path) -> str:
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _kv(key: str, value: object) -> None:
    print(f"  {key:<22} {value}")


def _load_scanner(*, full: bool) -> Scanner:
    print(f"loading {CKPT.name}  clip={'on' if full else 'off (CNN only)'}")
    return Scanner(CKPT, use_dictionary=full)


def print_report() -> None:
    print("SHIPPED VAL  (same-textbook quiz)")
    print(f"  weights                {CKPT}  {_ts(CKPT)}")
    print(f"  scores                 {META}  {_ts(META)}")
    if META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        val = meta.get("val") or {}
        f1 = val.get("health_f1") or []
        names = meta.get("health") or ["healthy", "mild", "critical", "dead"]
        _kv("crop ID (in-list)", _pct(val.get("id_crop_acc", 0)))
        _kv("junk as other", _pct(val.get("ood_recall", 0)))
        _kv("real crop as other", _pct(val.get("false_other", 0)))
        _kv("health macro-F1", _pct(val.get("health_macro_f1", 0)))
        for name, score in zip(names, f1):
            _kv(f"health F1 {name}", _pct(score))
    else:
        print("  missing models/meta.json")

    print()
    print("TRAIN PILE  (data/manifest.csv)")
    print(f"  written                {_ts(MANIFEST)}")
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        _kv("photos", f"{len(man):,}")
        for crop, n in man["crop"].value_counts().items():
            _kv(str(crop), f"{int(n):,}")
        health = man["health"].fillna("crop-only").value_counts()
        print("  health labels")
        for name, n in health.items():
            _kv(str(name), f"{int(n):,}")
        dead_n = int((man["health"] == "dead").sum())
        if dead_n == 0:
            _kv("dead", "0  (class cannot learn)")

    print()
    print("FIELD CSV  (older exam - not today's best.pt unless dates match)")
    print(f"  holdout csv            {_ts(HOLDOUT_RESULTS)}")
    print(f"  realistic csv          {_ts(REALISTIC_RESULTS)}")
    stale = (
        CKPT.exists()
        and HOLDOUT_RESULTS.exists()
        and HOLDOUT_RESULTS.stat().st_mtime < CKPT.stat().st_mtime
    )
    if stale:
        print("  STALE                   field files are older than best.pt")
        print("                          re-run: python training/test_model.py --rerun holdout")

    if HOLDOUT_RESULTS.exists():
        h = pd.read_csv(HOLDOUT_RESULTS)
        known = h[~h["expect_unknown"].astype(bool)]
        unk = h[h["expect_unknown"].astype(bool)]
        crop_ok = (known["true_crop"] == known["pred_crop"]).mean() if len(known) else 0.0
        both_ok = (
            ((known["true_crop"] == known["pred_crop"]) & (known["true_health"] == known["pred_health"])).mean()
            if len(known)
            else 0.0
        )
        print()
        print(f"  HOLDOUT  n={len(h)}  known={len(known)}")
        _kv("crop acc", _pct(crop_ok))
        _kv("crop+health", _pct(both_ok))
        if len(unk):
            _kv("junk flagged unknown", _pct(unk["pred_unknown"].mean()))
        g = known.assign(ok=known["true_crop"] == known["pred_crop"])
        for crop, sub in g.groupby("true_crop"):
            _kv(str(crop), f"{_pct(sub['ok'].mean())}  n={len(sub)}")

    if REALISTIC_RESULTS.exists():
        r = pd.read_csv(REALISTIC_RESULTS)
        print()
        print(f"  REALISTIC  n={len(r)}")
        for source, grp in r.groupby("source"):
            known = grp[~grp["expect_unknown"].astype(bool)]
            crop_ok = (known["true_crop"] == known["pred_crop"]).mean() if len(known) else float("nan")
            _kv(str(source), f"crop {_pct(crop_ok)}  unknown-flag {_pct(grp['pred_unknown'].mean())}  n={len(grp)}")


def _image_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def print_grade(path: Path, pred: dict) -> None:
    print(path)
    _kv("crop", f"{pred['crop']}  {_pct(pred['crop_confidence'])}")
    _kv("health", f"{pred['health']}  {_pct(pred['health_confidence'])}")
    _kv("reason", pred.get("reason") or "ok")
    _kv("view", pred.get("view") or "")
    if pred.get("named_plant"):
        _kv("named", pred["named_plant"])
    scores = pred.get("crop_scores") or {}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top:
        _kv("crop top-3", "  ".join(f"{name} {_pct(score)}" for name, score in top))
    hscores = pred.get("health_scores") or {}
    if hscores:
        _kv(
            "health probs",
            "  ".join(f"{name} {_pct(score)}" for name, score in hscores.items()),
        )
    if pred.get("tip"):
        _kv("tip", pred["tip"])
    print()


def grade_paths(paths: list[Path], *, full: bool, dump_json: bool) -> None:
    scanner = _load_scanner(full=full)
    for path in paths:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        pred = scanner.scan(path)
        if dump_json:
            print(path)
            print(json.dumps(pred, indent=2))
        else:
            print_grade(path, pred)


def _eval_frame(df: pd.DataFrame, scanner: Scanner, desc: str) -> pd.DataFrame:
    recs = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_k: x
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc):
        src = Path(row.path)
        if not src.exists():
            continue
        pred = scanner.scan(src)
        recs.append(
            {
                "path": row.path,
                "source": getattr(row, "source", ""),
                "class_folder": getattr(row, "class_folder", ""),
                "true_crop": row.crop,
                "true_health": row.health if isinstance(row.health, str) else "",
                "expect_unknown": bool(row.expect_unknown),
                "pred_crop": pred["crop"],
                "pred_health": pred["health"],
                "crop_confidence": pred["crop_confidence"],
                "health_confidence": pred["health_confidence"],
                "pred_unknown": pred["unknown"],
            }
        )
    return pd.DataFrame(recs)


def _print_eval(out: pd.DataFrame, label: str) -> None:
    if out.empty:
        print(f"{label}: no images found")
        return
    known = out[~out["expect_unknown"].astype(bool)]
    unk = out[out["expect_unknown"].astype(bool)]
    crop_ok = (known["true_crop"] == known["pred_crop"]).mean() if len(known) else 0.0
    both_ok = (
        ((known["true_crop"] == known["pred_crop"]) & (known["true_health"] == known["pred_health"])).mean()
        if len(known)
        else 0.0
    )
    print(f"{label}  n={len(out)}  known={len(known)}")
    _kv("crop acc", _pct(crop_ok))
    _kv("crop+health", _pct(both_ok))
    if len(known):
        g = known.assign(ok=known["true_crop"] == known["pred_crop"])
        for crop, sub in g.groupby("true_crop"):
            _kv(str(crop), f"{_pct(sub['ok'].mean())}  n={len(sub)}")
    if len(unk):
        _kv("junk flagged unknown", _pct(unk["pred_unknown"].mean()))
    if "source" in out.columns:
        for source, grp in out.groupby("source"):
            k = grp[~grp["expect_unknown"].astype(bool)]
            acc = (k["true_crop"] == k["pred_crop"]).mean() if len(k) else float("nan")
            _kv(str(source), f"crop {_pct(acc)}  n={len(grp)}")


def rerun(kind: str, *, full: bool, write: bool) -> None:
    scanner = _load_scanner(full=full)
    if kind == "holdout":
        if not HOLDOUT_MANIFEST.exists():
            raise SystemExit("Missing data/holdout_manifest.csv")
        df = pd.read_csv(HOLDOUT_MANIFEST)
        out = _eval_frame(df, scanner, "holdout")
        _print_eval(out, "HOLDOUT NOW")
        dest = HOLDOUT_RESULTS
    else:
        if not REALISTIC_MANIFEST.exists():
            raise SystemExit("Missing data/realistic_manifest.csv")
        df = pd.read_csv(REALISTIC_MANIFEST)
        out = _eval_frame(df, scanner, "realistic")
        _print_eval(out, "REALISTIC NOW")
        dest = REALISTIC_RESULTS
    if write:
        out.to_csv(dest, index=False)
        print("wrote", dest)


def sample_manifest(n: int, *, full: bool, seed: int) -> None:
    if not MANIFEST.exists():
        raise SystemExit("Missing data/manifest.csv")
    man = pd.read_csv(MANIFEST)
    rows = man.sample(n=min(n, len(man)), random_state=seed)
    scanner = _load_scanner(full=full)
    ok = 0
    for row in rows.itertuples(index=False):
        path = Path(row.path)
        if not path.exists():
            print(f"missing {path}")
            continue
        pred = scanner.scan(path)
        hit = pred["crop"] == row.crop
        ok += int(hit)
        mark = "OK" if hit else "MISS"
        health_t = row.health if isinstance(row.health, str) and row.health else "crop-only"
        print(f"{mark}  true={row.crop}/{health_t}  pred={pred['crop']}/{pred['health']}  {path.name}")
    print(f"crop hits {ok}/{len(rows)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Terminal tester for models/best.pt")
    p.add_argument("path", nargs="?", help="one image to grade")
    p.add_argument("--dir", dest="folder", help="folder of images")
    p.add_argument("--report", action="store_true", help="print saved scores (default if no image)")
    p.add_argument("--rerun", choices=["holdout", "realistic"], help="score current best.pt on a field set")
    p.add_argument("--write", action="store_true", help="overwrite the field CSV after --rerun")
    p.add_argument("--sample", type=int, metavar="N", help="grade N random train photos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="also load CLIP dictionary (slow)")
    p.add_argument("--json", action="store_true", help="print raw JSON for a photo")
    args = p.parse_args()

    if args.rerun:
        rerun(args.rerun, full=args.full, write=args.write)
        return
    if args.sample:
        sample_manifest(args.sample, full=args.full, seed=args.seed)
        return
    paths: list[Path] = []
    if args.folder:
        paths = _image_paths(Path(args.folder))
        if not paths:
            raise SystemExit(f"No images in {args.folder}")
    elif args.path:
        paths = [Path(args.path)]
    if paths:
        grade_paths(paths, full=args.full, dump_json=args.json)
        return
    print_report()


if __name__ == "__main__":
    main()
