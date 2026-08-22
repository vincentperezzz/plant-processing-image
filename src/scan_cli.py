import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import Scanner
from src.labels import IMAGE_EXTS
from src.paths import CKPT


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _print_grade(path: Path, pred: dict) -> None:
    print(path)
    print(f"  crop                   {pred['crop']}  {_pct(pred['crop_confidence'])}")
    print(f"  health                 {pred['health']}  {_pct(pred['health_confidence'])}")
    print(f"  reason                 {pred.get('reason') or 'ok'}")
    print(f"  view                   {pred.get('view') or ''}")
    if pred.get("named_plant"):
        print(f"  named                  {pred['named_plant']}")
    if pred.get("tip"):
        print(f"  tip                    {pred['tip']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Grade a leaf photo in the terminal")
    p.add_argument("path", nargs="?", help="image file")
    p.add_argument("--dir", dest="folder", help="folder of images")
    p.add_argument("--lite", action="store_true", help="CNN only, no CLIP")
    p.add_argument("--json", action="store_true", help="raw JSON")
    args = p.parse_args()
    paths = []
    if args.folder:
        folder = Path(args.folder)
        paths = sorted(x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS)
    elif args.path:
        paths = [Path(args.path)]
    else:
        raise SystemExit(
            "Usage: python src/scan_cli.py leaf.jpg\n"
            "       python src/scan_cli.py --dir folder\n"
            "Scoreboard: python training/test_model.py"
        )
    scanner = Scanner(CKPT, use_dictionary=not args.lite)
    for path in paths:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        pred = scanner.scan(path)
        if args.json:
            print(path.name)
            print(json.dumps(pred, indent=2))
        else:
            _print_grade(path, pred)


if __name__ == "__main__":
    main()
