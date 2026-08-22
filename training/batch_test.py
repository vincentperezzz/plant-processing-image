from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import Scanner
from src.paths import CKPT, HOLDOUT_MANIFEST, MANIFEST

CROPS = ("eggplant", "lettuce", "palay", "sili", "tomato")
HEALTH = ("healthy", "mild", "critical", "dead")
REALISTIC_MANIFEST = ROOT / "data" / "realistic_manifest.csv"
RESULTS = ROOT / "test-results"


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "  n/a"
    return f"{100.0 * n / d:5.1f}%"


def _bar(n: int, d: int, width: int = 20) -> str:
    if d <= 0:
        return " " * width
    fill = int(round(width * n / d))
    fill = min(width, max(0, fill))
    return "#" * fill + "-" * (width - fill)


def _load_set(kind: str) -> pd.DataFrame:
    if kind == "holdout":
        path = HOLDOUT_MANIFEST
    elif kind == "realistic":
        path = REALISTIC_MANIFEST
    elif kind == "train":
        path = MANIFEST
    else:
        raise SystemExit(f"Unknown set {kind}")
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    df = pd.read_csv(path)
    if "expect_unknown" not in df.columns:
        df["expect_unknown"] = False
    if "health" not in df.columns:
        df["health"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    df["expect_unknown"] = df["expect_unknown"].fillna(False).astype(bool)
    df["health"] = df["health"].where(df["health"].notna(), "")
    df["health"] = df["health"].astype(str).replace({"nan": "", "None": ""})
    return df


def _scan_rows(df: pd.DataFrame, scanner: Scanner) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_k):
            return x

    recs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="grading", unit="photo"):
        src = Path(row.path)
        if not src.exists():
            recs.append(
                {
                    "file": src.name,
                    "path": str(src),
                    "source": getattr(row, "source", ""),
                    "true_crop": row.crop,
                    "true_health": row.health,
                    "expect_unknown": bool(row.expect_unknown),
                    "pred_crop": "missing",
                    "pred_health": "missing",
                    "crop_confidence": 0.0,
                    "health_confidence": 0.0,
                    "unknown": True,
                    "reason": "missing_file",
                }
            )
            continue
        pred = scanner.scan(src)
        recs.append(
            {
                "file": src.name,
                "path": str(src),
                "source": getattr(row, "source", ""),
                "true_crop": row.crop if isinstance(row.crop, str) else "",
                "true_health": row.health if isinstance(row.health, str) else "",
                "expect_unknown": bool(row.expect_unknown),
                "pred_crop": pred.get("crop") or "unknown",
                "pred_health": pred.get("health") or "unknown",
                "crop_confidence": float(pred.get("crop_confidence") or 0),
                "health_confidence": float(pred.get("health_confidence") or 0),
                "unknown": bool(pred.get("unknown")),
                "reason": pred.get("reason") or "ok",
            }
        )
    return pd.DataFrame(recs)


def _scoreboard_text(out: pd.DataFrame, title: str) -> str:
    buf = io.StringIO()
    w = buf.write
    w("\n")
    w(title + "\n")
    w("=" * 64 + "\n")
    junk = out[out["expect_unknown"]]
    known = out[~out["expect_unknown"]]
    crop_hit = int((known["true_crop"] == known["pred_crop"]).sum()) if len(known) else 0
    both = known[
        (known["true_crop"] == known["pred_crop"])
        & (known["true_health"] != "")
        & (known["true_health"] == known["pred_health"])
    ]
    health_n = int((known["true_health"] != "").sum())
    junk_hit = int(junk["unknown"].sum()) if len(junk) else 0
    w(f"  photos                 {len(out)}\n")
    w(f"  crop name              {crop_hit}/{len(known)}   {_pct(crop_hit, len(known))}  {_bar(crop_hit, len(known))}\n")
    w(f"  crop + health          {len(both)}/{health_n}   {_pct(len(both), health_n)}  {_bar(len(both), health_n)}\n")
    if len(junk):
        w(f"  junk as unknown        {junk_hit}/{len(junk)}   {_pct(junk_hit, len(junk))}  {_bar(junk_hit, len(junk))}\n")
    w("\n")
    w("BY CROP  (name only)\n")
    w(f"  {'crop':<12} {'hit':>10}  {'pct':>7}  \n")
    for crop in CROPS:
        g = known[known["true_crop"] == crop]
        if g.empty:
            continue
        hit = int((g["true_crop"] == g["pred_crop"]).sum())
        w(f"  {crop:<12} {hit:>4}/{len(g):<5}  {_pct(hit, len(g))}  {_bar(hit, len(g))}\n")
    extra = known[~known["true_crop"].isin(CROPS)]
    if len(extra):
        hit = int((extra["true_crop"] == extra["pred_crop"]).sum())
        w(f"  {'other-labeled':<12} {hit:>4}/{len(extra):<5}  {_pct(hit, len(extra))}\n")
    w("\n")
    w("BY HEALTH  (crop name already right)\n")
    named_ok = known[known["true_crop"] == known["pred_crop"]]
    w(f"  {'grade':<12} {'hit':>10}  {'pct':>7}  \n")
    for grade in HEALTH:
        g = named_ok[named_ok["true_health"] == grade]
        if g.empty:
            continue
        hit = int((g["true_health"] == g["pred_health"]).sum())
        w(f"  {grade:<12} {hit:>4}/{len(g):<5}  {_pct(hit, len(g))}  {_bar(hit, len(g))}\n")
    w("\n")
    w("CONFUSION  true \\ pred\n")
    labels = [c for c in CROPS if c in set(known["true_crop"]) | set(known["pred_crop"])]
    if "unknown" in set(known["pred_crop"]):
        labels = labels + ["unknown"]
    header = "  " + f"{'':>10}" + "".join(f"{lab[:7]:>8}" for lab in labels)
    w(header + "\n")
    for true in [c for c in CROPS if c in set(known["true_crop"])]:
        g = known[known["true_crop"] == true]
        cells = []
        for pred in labels:
            n = int((g["pred_crop"] == pred).sum())
            cells.append(f"{n:>8}" if n else f"{'.':>8}")
        w("  " + f"{true:>10}" + "".join(cells) + "\n")
    w("\n")
    misses = known[known["true_crop"] != known["pred_crop"]]
    w(f"NAME MISSES  {len(misses)}\n")
    if misses.empty:
        w("  none\n")
    else:
        show = misses.head(25)
        for row in show.itertuples(index=False):
            w(
                f"  {row.true_crop}/{row.true_health or '-':<8} -> {row.pred_crop}/{row.pred_health}"
                f"  {row.file}\n"
            )
        leftover = len(misses) - len(show)
        if leftover > 0:
            w(f"  ... {leftover} more\n")
    w("\n")
    return buf.getvalue()


def _print_scoreboard(out: pd.DataFrame, title: str) -> str:
    text = _scoreboard_text(out, title)
    sys.stdout.write(text)
    return text


def _export(out: pd.DataFrame, report: str, kind: str, header: str) -> tuple[Path, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    stem = f"{stamp}-{kind}"
    txt_path = RESULTS / f"{stem}.txt"
    csv_path = RESULTS / f"{stem}.csv"
    txt_path.write_text(header + report, encoding="utf-8")
    out.to_csv(csv_path, index=False)
    return txt_path, csv_path


def _print_verbose(out: pd.DataFrame) -> None:
    print(f"{'FILE':<28} {'TRUE':<22} {'PRED':<22} {'CROP':^4} {'HLTH':^4}")
    print("-" * 90)
    for row in out.itertuples(index=False):
        if row.expect_unknown:
            crop_ok = bool(row.unknown) or row.pred_crop not in CROPS
            health_ok = None
            true_txt = "other / unknown"
        else:
            crop_ok = row.true_crop == row.pred_crop
            health_ok = (
                row.true_health == row.pred_health if row.true_health else None
            )
            true_txt = f"{row.true_crop} / {row.true_health or '-'}"
        pred_txt = f"{row.pred_crop} / {row.pred_health}"
        cmark = "OK" if crop_ok else "NO"
        hmark = "-" if health_ok is None else ("OK" if health_ok else "NO")
        print(f"{row.file:<28} {true_txt:<22} {pred_txt:<22} {cmark:^4} {hmark:^4}")
    print()


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(
        description="Batch-grade labeled photos from disk (holdout / realistic / train)."
    )
    p.add_argument(
        "--set",
        choices=["holdout", "realistic", "train"],
        default="holdout",
        help="which labeled pile to feed (default: holdout exam)",
    )
    p.add_argument("--sample", type=int, metavar="N", help="random subset of N photos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="CNN + CLIP (slow, PC only)")
    p.add_argument("--verbose", action="store_true", help="print every photo")
    p.add_argument("--export", dest="export", action="store_true", default=True)
    p.add_argument("--no-export", dest="export", action="store_false")
    args = p.parse_args()
    if not CKPT.exists():
        raise SystemExit(f"Missing {CKPT}")
    df = _load_set(args.set)
    if args.sample:
        n = min(args.sample, len(df))
        df = df.sample(n=n, random_state=args.seed)
    lite = not args.full
    header = (
        f"BATCH TEST\n"
        f"  set        {args.set}  ({len(df)} photos)\n"
        f"  grader     {CKPT}  {'CNN only' if lite else 'CNN + CLIP'}\n"
        f"  when       {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    print()
    print(header, end="")
    print()
    scanner = Scanner(CKPT, use_dictionary=not lite)
    out = _scan_rows(df, scanner)
    if args.verbose:
        _print_verbose(out)
    report = _print_scoreboard(out, f"SCOREBOARD  {args.set}")
    if args.export:
        txt_path, csv_path = _export(out, report, args.set, header + "\n")
        print(f"wrote  {txt_path}")
        print(f"wrote  {csv_path}")
        print()


if __name__ == "__main__":
    main()
